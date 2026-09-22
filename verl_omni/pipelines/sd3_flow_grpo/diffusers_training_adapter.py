# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SD3.5 training-side adapter for diffusers-based FlowGRPO."""

from typing import Optional

import torch
from diffusers import ModelMixin
from tensordict import TensorDict
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

__all__ = ["StableDiffusion3FlowGRPO"]


def _calculate_shift(
    image_seq_len: int,
    base_image_seq_len: int = 256,
    max_image_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
) -> float:
    m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    b = base_shift - m * base_image_seq_len
    return image_seq_len * m + b


def _sd3_image_seq_len(height: int, width: int, vae_scale_factor: int = 8) -> int:
    patch_size = 2
    return (int(height) // vae_scale_factor // patch_size) * (int(width) // vae_scale_factor // patch_size)


@DiffusionModelBase.register("StableDiffusion3Pipeline", algorithm="flow_grpo")
class StableDiffusion3FlowGRPO(DiffusionModelBase):
    """Training adapter for SD3.5 FlowGRPO.

    SD3 conditions the MMDiT transformer on a joint CLIP+T5 prompt-embedding
    sequence plus a pooled CLIP projection. Rollout returns both tensors; the
    adapter rebuilds the transformer inputs from them during policy log-prob
    recomputation with the same SDE math as the rollout scheduler.
    """

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path=model_config.local_path,
            subfolder="scheduler",
        )
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler: FlowMatchSDEDiscreteScheduler, model_config: DiffusionModelConfig, device: str):
        scheduler_kwargs = {}
        if scheduler.config.get("use_dynamic_shifting", None):
            scheduler_kwargs["mu"] = _calculate_shift(
                _sd3_image_seq_len(model_config.pipeline.height, model_config.pipeline.width),
                scheduler.config.get("base_image_seq_len", 256),
                scheduler.config.get("max_image_seq_len", 4096),
                scheduler.config.get("base_shift", 0.5),
                scheduler.config.get("max_shift", 1.16),
            )
        scheduler.set_timesteps(model_config.pipeline.num_inference_steps, device=device, **scheduler_kwargs)

    @staticmethod
    def build_transformer_inputs(
        *,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
    ) -> dict:
        """Create the ``SD3Transformer2DModel`` keyword arguments.

        No text attention mask is passed: the rollout pipeline attends over
        the full padded prompt sequence, and the training forward must match
        it exactly for log-prob consistency.
        """
        return {
            "hidden_states": latents,
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
            "timestep": timesteps,
            "return_dict": False,
        }

    @classmethod
    def prepare_model_inputs(
        cls,
        module: ModelMixin,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict, Optional[dict]]:
        if "pooled_prompt_embeds" not in micro_batch:
            raise KeyError("SD3 FlowGRPO requires `pooled_prompt_embeds` from rollout.")

        selected_latents = latents[:, step]
        selected_timesteps = timesteps[:, step]

        model_inputs = cls.build_transformer_inputs(
            latents=selected_latents,
            timesteps=selected_timesteps,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=micro_batch["pooled_prompt_embeds"],
        )

        guidance_scale = model_config.pipeline.guidance_scale
        if guidance_scale is None:
            guidance_scale = 0.0
        if guidance_scale > 1.0:
            if negative_prompt_embeds is None:
                raise ValueError("SD3 CFG requires negative prompt embeds when guidance_scale > 1.")
            if "negative_pooled_prompt_embeds" not in micro_batch:
                raise KeyError("SD3 CFG requires `negative_pooled_prompt_embeds` from rollout.")
            negative_model_inputs = cls.build_transformer_inputs(
                latents=selected_latents,
                timesteps=selected_timesteps,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=micro_batch["negative_pooled_prompt_embeds"],
            )
        else:
            negative_model_inputs = None

        return model_inputs, negative_model_inputs

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module: ModelMixin,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ):
        assert scheduler_inputs is not None
        latents = scheduler_inputs["all_latents"]
        timesteps = scheduler_inputs["all_timesteps"]

        noise_pred = cls.forward(module, model_config, model_inputs)
        guidance_scale = model_config.pipeline.guidance_scale
        if guidance_scale is not None and guidance_scale > 1.0:
            if negative_model_inputs is None:
                raise ValueError("SD3 CFG requires negative model inputs when guidance_scale > 1.")
            neg_noise_pred = cls.forward(module, model_config, negative_model_inputs)
            noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

        _, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = scheduler.sample_previous_step(
            sample=latents[:, step].float(),
            model_output=noise_pred.float(),
            timestep=timesteps[:, step],
            noise_level=model_config.algo.noise_level,
            prev_sample=latents[:, step + 1].float(),
            sde_type=model_config.algo.sde_type,
            return_logprobs=True,
            return_sqrt_dt=True,
        )
        return log_prob, prev_sample_mean, std_dev_t, sqrt_dt

    @classmethod
    def forward_and_sample_window(
        cls,
        module: ModelMixin,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        *,
        all_latents: torch.Tensor,
        all_timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run every SDE-window step in a single batched forward + scheduler call.

        Folds the window axis ``W`` into the batch dimension so the transformer
        executes one large matmul instead of ``W`` small ones, and the reverse-SDE
        log-prob is computed in one ``sample_previous_step`` call. This is the
        batched counterpart of :meth:`forward_and_sample_previous_step`; the
        returned tensors carry the window axis explicitly so callers can stack
        them along ``dim=1`` exactly as the per-step path did.

        Args:
            all_latents: ``(B, W+1, ...)`` trajectory; step ``s`` uses
                ``all_latents[:, s]`` as the sample and ``all_latents[:, s+1]`` as
                the realised ``prev_sample``.
            all_timesteps: ``(B, W)`` per-step timesteps.
            prompt_embeds / pooled_prompt_embeds: positive prompt conditioning,
                shapes ``(B, L, D)`` and ``(B, Dp)``.
            negative_*: optional CFG negative conditioning, same shapes.

        Returns:
            ``(log_prob, prev_sample_mean, std_dev_t, sqrt_dt)`` with window axis
            restored: ``(B, W)``, ``(B, W, ...)``, ``(B, W, ...)``, ``(B, W)``.
        """
        # all_latents: (B, W+1, ...); sample = [:, :-1], prev_sample = [:, 1:]
        sample = all_latents[:, :-1]
        prev_sample = all_latents[:, 1:]
        batch_size, window = sample.shape[0], sample.shape[1]
        # Flatten (B, W, ...) -> (B*W, ...) for the transformer + scheduler.
        sample_flat = sample.reshape(batch_size * window, *sample.shape[2:])
        prev_sample_flat = prev_sample.reshape(batch_size * prev_sample.shape[1], *prev_sample.shape[2:])
        timestep_flat = all_timesteps.reshape(batch_size * window)

        # Expand prompt conditioning along the window axis: (B, L, D) -> (B*W, L, D).
        prompt_embeds_flat = prompt_embeds.repeat_interleave(window, dim=0)
        pooled_flat = pooled_prompt_embeds.repeat_interleave(window, dim=0)

        model_inputs = cls.build_transformer_inputs(
            latents=sample_flat,
            timesteps=timestep_flat,
            prompt_embeds=prompt_embeds_flat,
            pooled_prompt_embeds=pooled_flat,
        )
        noise_pred = cls.forward(module, model_config, model_inputs)

        guidance_scale = model_config.pipeline.guidance_scale
        if guidance_scale is not None and guidance_scale > 1.0:
            if negative_prompt_embeds is None or negative_pooled_prompt_embeds is None:
                raise ValueError("SD3 CFG requires negative prompt embeds when guidance_scale > 1.")
            neg_prompt_flat = negative_prompt_embeds.repeat_interleave(window, dim=0)
            neg_pooled_flat = negative_pooled_prompt_embeds.repeat_interleave(window, dim=0)
            negative_model_inputs = cls.build_transformer_inputs(
                latents=sample_flat,
                timesteps=timestep_flat,
                prompt_embeds=neg_prompt_flat,
                pooled_prompt_embeds=neg_pooled_flat,
            )
            neg_noise_pred = cls.forward(module, model_config, negative_model_inputs)
            noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

        _, log_prob_flat, prev_sample_mean_flat, std_dev_t_flat, sqrt_dt_flat = scheduler.sample_previous_step(
            sample=sample_flat.float(),
            model_output=noise_pred.float(),
            timestep=timestep_flat,
            noise_level=model_config.algo.noise_level,
            prev_sample=prev_sample_flat.float(),
            sde_type=model_config.algo.sde_type,
            return_logprobs=True,
            return_sqrt_dt=True,
        )

        # Restore the window axis.
        log_prob = log_prob_flat.reshape(batch_size, window)
        prev_sample_mean = prev_sample_mean_flat.reshape(batch_size, window, *prev_sample_mean_flat.shape[1:])
        std_dev_t = std_dev_t_flat.reshape(batch_size, window, *std_dev_t_flat.shape[1:])
        sqrt_dt = sqrt_dt_flat.reshape(batch_size, window)
        return log_prob, prev_sample_mean, std_dev_t, sqrt_dt

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

import os
from typing import Any, Literal

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.qwen_image import QwenImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

from .common import apply_true_cfg, build_img_shapes

__all__ = ["QwenImagePipelineWithLogProb"]


def _maybe_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _coalesce_not_none(value, default):
    return default if value is None else value


@VllmOmniPipelineBase.register("QwenImagePipeline", algorithm="flow_grpo")
class QwenImagePipelineWithLogProb(QwenImagePipeline):
    """Rollout pipeline for Qwen-Image that captures per-step log-probabilities.

    Extends :class:`~vllm_omni.diffusion.models.qwen_image.QwenImagePipeline`
    with a custom SDE-based scheduler and additional output fields required
    for RL training (e.g. FlowGRPO).  In addition to the final generated image
    the pipeline returns all intermediate latents, their log-probabilities,
    and the corresponding timesteps.

    Registered under ``"QwenImagePipeline"`` for vllm-omni rollout dispatch.
    """

    # SDE/FlowGRPO knobs read from ``sampling_params.extra_args``.
    _SDE_DEFAULTS: dict[str, Any] = {
        "noise_level": 0.7,
        "sde_window_size": None,
        "sde_window_range": (0, 5),
        "sde_type": "sde",
        "logprobs": True,
    }

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)

        self.scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

    # ------------------------------------------------------------------ #
    # Prompt encoding (token-id input contract)
    # ------------------------------------------------------------------ #

    def _get_qwen_prompt_embeds(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
    ):
        dtype = dtype or self.text_encoder.dtype

        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)

        prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        attention_mask = attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask
        drop_idx = self.prompt_template_encode_start_idx
        encoder_hidden_states = self.text_encoder(
            input_ids=prompt_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype)

        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
    ):
        """Encode prompt token IDs into dense embeddings.

        Args:
            prompt_ids: Token IDs of shape ``(B, L)`` or ``(L,)``.
            attention_mask: Boolean mask of shape ``(B, L)``; inferred as
                all-ones when ``None``.
            num_images_per_prompt: Embeddings are repeated this many times.
            prompt_embeds: Pre-computed embeddings; when provided
                *prompt_ids* is ignored.
            prompt_embeds_mask: Attention mask for pre-computed
                *prompt_embeds*.
            max_sequence_length: Embeddings are truncated to this length.

        Returns:
            ``(prompt_embeds, prompt_embeds_mask)`` of shape
            ``(B * num_images_per_prompt, L, D)`` and
            ``(B * num_images_per_prompt, L)``.
        """
        prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        attention_mask = (
            attention_mask.unsqueeze(0) if attention_mask is not None and attention_mask.ndim == 1 else attention_mask
        )

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt_ids, attention_mask=attention_mask)

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            prompt_embeds_mask = prompt_embeds_mask.repeat_interleave(num_images_per_prompt, dim=0)

        return prompt_embeds, prompt_embeds_mask

    # ------------------------------------------------------------------ #
    # Parent-class hooks
    # ------------------------------------------------------------------ #

    def _encode_prompts(
        self,
        *,
        prompt=None,  # unused; superseded by prompt_ids
        negative_prompt=None,
        height=None,
        width=None,
        num_images_per_prompt=1,
        max_sequence_length=1024,
        true_cfg_scale=4.0,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        callback_on_step_end_tensor_inputs=None,
        # Extra kwargs forwarded through ``_prepare_generation_context``:
        prompt_ids=None,
        prompt_mask=None,
        negative_prompt_ids=None,
        negative_prompt_mask=None,
        **_unused,
    ):
        """Override of the parent hook to encode pre-tokenized prompt ids.

        Returns ``(prompt_embeds, prompt_embeds_mask, negative_prompt_embeds,
        negative_prompt_embeds_mask, do_true_cfg)``.
        """
        if isinstance(prompt_ids, list):
            prompt_ids = torch.tensor(prompt_ids, device=self.device)
        if isinstance(negative_prompt_ids, list):
            negative_prompt_ids = torch.tensor(negative_prompt_ids, device=self.device)

        has_neg_prompt = negative_prompt_ids is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        self.check_cfg_parallel_validity(true_cfg_scale, has_neg_prompt)

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt_ids=prompt_ids,
            attention_mask=prompt_mask,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt_ids=negative_prompt_ids,
                attention_mask=negative_prompt_mask,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None

        return (
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            do_true_cfg,
        )

    # ------------------------------------------------------------------ #
    # SDE helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def _resolve_sde_extra_args(cls, extra_args: dict | None, defaults: dict | None) -> dict:
        """Merge ``sampling_params.extra_args`` over the kwarg defaults."""
        extra_args = extra_args or {}
        defaults = defaults or {}
        return {
            key: _coalesce_not_none(
                extra_args.get(key),
                _coalesce_not_none(defaults.get(key), cls._SDE_DEFAULTS[key]),
            )
            for key in cls._SDE_DEFAULTS
        }

    def _resolve_sde_window(
        self,
        timesteps: torch.Tensor,
        sde_window_size: int | None,
        sde_window_range: tuple[int, int],
        generator: torch.Generator | None,
    ) -> tuple[int, int]:
        """Resolve the ``(start, end)`` SDE window indices."""
        if sde_window_size is None:
            return (0, len(timesteps) - 1)
        start = torch.randint(
            sde_window_range[0],
            sde_window_range[1] - sde_window_size + 1,
            (1,),
            generator=generator,
            device=self.device,
        ).item()
        return (start, start + sde_window_size)

    # ------------------------------------------------------------------ #
    # Diffusion loop
    # ------------------------------------------------------------------ #

    def diffuse(
        self,
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
        latents,
        img_shapes,
        txt_seq_lens,
        negative_txt_seq_lens,
        timesteps,
        do_true_cfg,
        guidance,
        true_cfg_scale,
        noise_level,
        sde_window,
        sde_type,
        generator,
        logprobs,
    ):
        """Run the full SDE diffusion loop and collect per-step rollout data.

        Iterates over all timesteps, optionally applying True-CFG guidance,
        and collects latents and log-probabilities within the SDE window.

        Returns:
            tuple: ``(latents, all_latents, all_log_probs, all_timesteps)``
                with *all_latents* of shape ``(B, W+1, ...)`` (W = SDE
                window length), *all_log_probs* of shape ``(B, W)`` or
                ``None`` when *logprobs* is ``False``, and *all_timesteps*
                of shape ``(B, W)``.
        """
        all_latents = []
        all_log_probs = []
        all_timesteps = []
        self.scheduler.set_begin_index(0)
        for i, timestep_value in enumerate(timesteps):
            if self.interrupt:
                continue

            if i < sde_window[0]:
                cur_noise_level = 0.0
            elif i == sde_window[0]:
                cur_noise_level = noise_level
                all_latents.append(latents)
            elif i > sde_window[0] and i < sde_window[1]:
                cur_noise_level = noise_level
            else:
                cur_noise_level = 0.0

            self._current_timestep = timestep_value
            timestep = timestep_value.expand(latents.shape[0]).to(device=latents.device, dtype=latents.dtype)

            self.transformer.do_true_cfg = do_true_cfg
            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states_mask=prompt_embeds_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                attention_kwargs=self.attention_kwargs,
                return_dict=False,
            )[0]
            if do_true_cfg:
                neg_noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=negative_prompt_embeds_mask,
                    encoder_hidden_states=negative_prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=negative_txt_seq_lens,
                    attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = apply_true_cfg(noise_pred, neg_noise_pred, true_cfg_scale)

            latents, log_prob, _, _ = self.scheduler.step(
                noise_pred,
                timestep_value,
                latents,
                generator=generator,
                noise_level=cur_noise_level,
                sde_type=sde_type,
                return_logprobs=logprobs,
                return_dict=False,
            )

            if i >= sde_window[0] and i < sde_window[1]:
                all_latents.append(latents)
                all_log_probs.append(log_prob)
                all_timesteps.append(timestep_value)

        all_latents = torch.stack(all_latents, dim=1)
        all_log_probs = torch.stack(all_log_probs, dim=1) if all_log_probs and all_log_probs[0] is not None else None
        all_timesteps = torch.stack(all_timesteps).unsqueeze(0).expand(latents.shape[0], -1)
        return latents, all_latents, all_log_probs, all_timesteps

    # ------------------------------------------------------------------ #
    # Public entrypoint
    # ------------------------------------------------------------------ #

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt_ids: torch.Tensor | list[int] | None = None,
        prompt_mask: torch.Tensor | None = None,
        negative_prompt_ids: torch.Tensor | list[int] | None = None,
        negative_prompt_mask: torch.Tensor | None = None,
        true_cfg_scale: float = 4.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end_tensor_inputs: tuple[str, ...] = ("latents",),
        max_sequence_length: int = 512,
        noise_level: float = 0.7,
        sde_window_size: int | None = None,
        sde_window_range: tuple[int, int] = (0, 5),
        sde_type: Literal["sde", "cps"] = "sde",
        logprobs: bool = True,
    ) -> DiffusionOutput:
        """End-to-end image generation with rollout data collection.

        Encodes the prompt, prepares latents, runs the SDE diffusion loop via
        :meth:`diffuse`, and decodes the final latents through the VAE.
        Sampling parameters in *req* take precedence over keyword arguments.
        """
        # ---- Resolve sampling-parameter overrides ---------------------- #
        sampling_params = req.sampling_params
        height = sampling_params.height or self.default_sample_size * self.vae_scale_factor
        width = sampling_params.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = sampling_params.num_inference_steps or num_inference_steps
        max_sequence_length = sampling_params.max_sequence_length or max_sequence_length
        true_cfg_scale = _coalesce_not_none(sampling_params.true_cfg_scale, true_cfg_scale)
        req_num_outputs = getattr(sampling_params, "num_outputs_per_prompt", None)
        if req_num_outputs and req_num_outputs > 0:
            num_images_per_prompt = req_num_outputs

        sde_extra = self._resolve_sde_extra_args(
            sampling_params.extra_args,
            defaults={
                "noise_level": noise_level,
                "sde_window_size": sde_window_size,
                "sde_window_range": sde_window_range,
                "sde_type": sde_type,
                "logprobs": logprobs,
            },
        )

        generator = sampling_params.generator or generator
        if generator is None and sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling_params.seed)

        # ---- Dummy / warm-up path -------------------------------------- #
        if prompt_ids is None and prompt_embeds is None:
            return DiffusionOutput(output=None, custom_output={})

        # ---- Delegate to parent for the shared generation context ------ #
        # ``prompt_ids``/``prompt_mask``/negatives flow through as
        # ``**encode_kwargs`` to our :meth:`_encode_prompts` override.
        ctx = super()._prepare_generation_context(
            prompt=None,
            negative_prompt=None,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            true_cfg_scale=true_cfg_scale,
            max_sequence_length=max_sequence_length,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            latents=latents,
            attention_kwargs=attention_kwargs,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_mask=negative_prompt_mask,
        )

        # ---- Resolve SDE window over the parent-built timesteps -------- #
        sde_window = self._resolve_sde_window(
            ctx["timesteps"],
            sde_extra["sde_window_size"],
            sde_extra["sde_window_range"],
            generator,
        )

        # ---- Run the SDE diffusion loop -------------------------------- #
        latents, all_latents, all_log_probs, all_timesteps = self.diffuse(
            ctx["prompt_embeds"],
            ctx["prompt_embeds_mask"],
            ctx["negative_prompt_embeds"],
            ctx["negative_prompt_embeds_mask"],
            ctx["latents"],
            ctx["img_shapes"],
            ctx["txt_seq_lens"],
            ctx["negative_txt_seq_lens"],
            ctx["timesteps"],
            ctx["do_true_cfg"],
            ctx["guidance"],
            true_cfg_scale,
            sde_extra["noise_level"],
            sde_window,
            sde_extra["sde_type"],
            generator,
            sde_extra["logprobs"],
        )

        self._current_timestep = None
        decoded = self._decode_latents(latents, height, width, output_type or "pil")

        return DiffusionOutput(
            output=_maybe_to_cpu(decoded.output),
            custom_output={
                "all_latents": _maybe_to_cpu(all_latents),
                "all_log_probs": _maybe_to_cpu(all_log_probs),
                "all_timesteps": _maybe_to_cpu(all_timesteps),
                "prompt_embeds": _maybe_to_cpu(ctx["prompt_embeds"]),
                "prompt_embeds_mask": _maybe_to_cpu(ctx["prompt_embeds_mask"]),
                "negative_prompt_embeds": _maybe_to_cpu(ctx["negative_prompt_embeds"]),
                "negative_prompt_embeds_mask": _maybe_to_cpu(ctx["negative_prompt_embeds_mask"]),
            },
        )

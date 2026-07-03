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

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.qwen_image import QwenImagePipeline
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.pipelines.utils import split_diffusion_output_by_request

from .common import QwenImageTokenIdPromptMixin, apply_true_cfg, build_img_shapes, coalesce_not_none

__all__ = ["QwenImagePipelineWithLogProb"]


def _collate_prompt_tokens(
    prompts: list,
    ids_field: str,
    mask_field: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """Collect per-request 1-D token-id sequences and right-pad to a common length.

    Returns ``(ids, attention_mask)`` of shape ``(N, max_len)`` on ``device``, or
    ``(None, None)`` when no request provides ``ids_field``. Each request's own
    ``mask_field`` (if present) supplies valid positions; padding gets mask 0.
    """
    ids_per_req: list[list[int]] = []
    mask_per_req: list[list[int]] = []
    provided = False
    for prompt in prompts:
        if not isinstance(prompt, dict):
            ids_per_req.append(None)  # type: ignore[arg-type]
            mask_per_req.append(None)  # type: ignore[arg-type]
            continue
        raw_ids = prompt.get(ids_field)
        if raw_ids is None:
            ids_per_req.append(None)  # type: ignore[arg-type]
            mask_per_req.append(None)  # type: ignore[arg-type]
            continue
        provided = True
        ids_list = raw_ids.tolist() if isinstance(raw_ids, torch.Tensor) else list(raw_ids)
        raw_mask = prompt.get(mask_field)
        if raw_mask is None:
            mask_list = [1] * len(ids_list)
        else:
            mask_list = raw_mask.tolist() if isinstance(raw_mask, torch.Tensor) else list(raw_mask)
            mask_list = [int(bool(m)) for m in mask_list]
        ids_per_req.append(ids_list)
        mask_per_req.append(mask_list)

    if not provided:
        return None, None

    max_len = max(len(x) for x in ids_per_req if x is not None)
    num_reqs = len(ids_per_req)
    ids = torch.zeros(num_reqs, max_len, dtype=torch.long)
    mask = torch.zeros(num_reqs, max_len, dtype=torch.long)
    for i, (x, m) in enumerate(zip(ids_per_req, mask_per_req)):
        if x is None:
            continue
        ids[i, : len(x)] = torch.tensor(x, dtype=torch.long)
        mask[i, : len(m)] = torch.tensor(m, dtype=torch.long)
    return ids.to(device), mask.to(device)


@VllmOmniPipelineBase.register("QwenImagePipeline", algorithm="flow_grpo")
class QwenImagePipelineWithLogProb(QwenImageTokenIdPromptMixin, QwenImagePipeline):
    """Rollout pipeline for Qwen-Image that captures per-step log-probabilities.

    Extends :class:`~vllm_omni.diffusion.models.qwen_image.QwenImagePipeline`
    with a custom SDE-based scheduler and additional output fields required
    for RL training (e.g. FlowGRPO).  In addition to the final generated image
    the pipeline returns all intermediate latents, their log-probabilities,
    and the corresponding timesteps.

    Registered under ``"QwenImagePipeline"`` for vllm-omni rollout dispatch.

    The request-batch contract (``supports_request_batch = True``) lets the
    vLLM-Omni scheduler coalesce compatible concurrent rollout requests into a
    single fused ``forward``. Per-request prompts are right-padded to a common
    length, the SDE window is shared across the wave, and the resulting
    tensors are sliced back to one :class:`DiffusionOutput` per request.
    """

    supports_request_batch = True

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

        Iterates over all timesteps, optionally applying True-CFG guidance, and
        collects latents and log-probabilities within the SDE window.

        Args:
            prompt_embeds (torch.Tensor): Positive prompt embeddings.
            prompt_embeds_mask (torch.Tensor): Attention mask for *prompt_embeds*.
            negative_prompt_embeds (torch.Tensor): Negative prompt embeddings for CFG.
            negative_prompt_embeds_mask (torch.Tensor): Attention mask for
                *negative_prompt_embeds*.
            latents (torch.Tensor): Initial noisy latents.
            img_shapes (list): Per-sample image shapes used by the transformer.
            txt_seq_lens (list[int]): Sequence lengths for positive prompt embeddings.
            negative_txt_seq_lens (list[int]): Sequence lengths for negative prompt embeddings.
            timesteps (torch.Tensor): Scheduler timestep sequence.
            do_true_cfg (bool): Whether to apply True-CFG guidance.
            guidance (torch.Tensor | None): Guidance scale tensor, or ``None``.
            true_cfg_scale (float): Classifier-free guidance scale.
            noise_level (float): SDE noise injection magnitude within the window.
            sde_window (tuple[int, int]): ``(start, end)`` step indices defining
                where SDE noise is injected and rollout data is collected.
            sde_type (str): SDE variant; one of ``"sde"`` or ``"cps"``.
            generator (torch.Generator | None): Optional random generator for
                reproducibility.
            logprobs (bool): Whether to compute and return per-step log-probabilities.

        Returns:
            tuple: A 4-tuple of
                ``(latents, all_latents, all_log_probs, all_timesteps)`` where
                *all_latents* has shape ``(B, W+1, ...)``
                (W = SDE-window length), *all_log_probs* has shape ``(B, W)``
                or ``None`` when *logprobs* is ``False``, and *all_timesteps*
                has shape ``(B, W)``.
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
                all_latents.append(latents.float())
            elif i > sde_window[0] and i < sde_window[1]:
                cur_noise_level = noise_level
            else:
                cur_noise_level = 0.0

            self._current_timestep = timestep_value
            # Broadcast timestep to match batch size
            timestep = timestep_value.expand(latents.shape[0]).to(device=latents.device, dtype=latents.dtype)

            # Cast to model dtype for transformer forward (scheduler returns float32).
            x = latents.to(self.transformer.img_in.weight.dtype)

            self.transformer.do_true_cfg = do_true_cfg
            # Forward pass for positive prompt (or unconditional if no CFG)
            noise_pred = self.transformer(
                hidden_states=x,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states_mask=prompt_embeds_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                attention_kwargs=self.attention_kwargs,
                return_dict=False,
            )[0]
            # Forward pass for negative prompt (CFG)
            if do_true_cfg:
                neg_noise_pred = self.transformer(
                    hidden_states=x,
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

            # compute the previous noisy sample x_t -> x_t-1
            latents, log_prob, _, _ = self.scheduler.step(
                noise_pred.to(torch.float32),
                timestep_value,
                latents.to(torch.float32),
                generator=generator,
                noise_level=cur_noise_level,
                sde_type=sde_type,
                return_logprobs=logprobs,
                return_dict=False,
            )

            # Save fp32 trajectory BEFORE casting to model dtype, so the
            # trainer recomputes log-probs on full-precision latents.
            if i >= sde_window[0] and i < sde_window[1]:
                all_latents.append(latents.to(torch.float32))
                all_log_probs.append(log_prob)
                all_timesteps.append(timestep_value)

        all_latents = torch.stack(all_latents, dim=1)
        all_log_probs = torch.stack(all_log_probs, dim=1) if all_log_probs and all_log_probs[0] is not None else None
        all_timesteps = torch.stack(all_timesteps).unsqueeze(0).expand(latents.shape[0], -1)
        return latents, all_latents, all_log_probs, all_timesteps

    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        """End-to-end batched image generation with rollout data collection.

        Encodes prompts, prepares latents, runs the SDE diffusion loop via
        :meth:`diffuse`, decodes the final latents through the VAE, and slices
        the batched result into one :class:`DiffusionOutput` per request.

        The vLLM-Omni scheduler groups only compatible requests (same shape,
        CFG, step count, LoRA) into one wave, so sampling parameters are read
        from the first request and applied to the whole batch. Per-request
        prompts are right-padded to a common length. The SDE window is shared
        across the wave; initial latent noise stays per-request via each
        request's own generator (created from its seed by the runner before
        this call).

        Returns:
            list[DiffusionOutput]: One per request, each carrying the decoded
            *output* image slice and a *custom_output* dict with keys
            ``"all_latents"``, ``"all_log_probs"``, ``"all_timesteps"``,
            ``"prompt_embeds"``, ``"prompt_embeds_mask"``,
            ``"negative_prompt_embeds"``, and ``"negative_prompt_embeds_mask"``.
        """
        prompts = req.prompts
        sampling_params_list = req.sampling_params_list
        common = sampling_params_list[0]

        height = common.height or self.default_sample_size * self.vae_scale_factor
        width = common.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = common.num_inference_steps or 50
        sigmas = common.sigmas
        max_sequence_length = common.max_sequence_length or 512
        output_type = common.output_type or "pil"
        true_cfg_scale = coalesce_not_none(common.true_cfg_scale, 4.0)
        if common.guidance_scale_provided:
            guidance_scale = common.guidance_scale
        else:
            guidance_scale = 1.0

        extra = common.extra_args or {}
        noise_level = coalesce_not_none(extra.get("noise_level", None), 0.7)
        sde_window_size = coalesce_not_none(extra.get("sde_window_size", None), None)
        sde_window_range = coalesce_not_none(extra.get("sde_window_range", None), (0, 5))
        sde_type = coalesce_not_none(extra.get("sde_type", None), "sde")
        logprobs = coalesce_not_none(extra.get("logprobs", None), True)

        req_num_outputs = getattr(common, "num_outputs_per_prompt", None)
        num_images_per_prompt = req_num_outputs if req_num_outputs and req_num_outputs > 0 else 1

        prompt_token_ids, prompt_mask = _collate_prompt_tokens(
            prompts, "prompt_token_ids", "prompt_mask", self.device
        )
        negative_prompt_ids, negative_prompt_mask = _collate_prompt_tokens(
            prompts, "negative_prompt_ids", "negative_prompt_mask", self.device
        )

        # Warmup / dummy run with no usable prompts: one empty output per request.
        if prompt_token_ids is None:
            return [DiffusionOutput(output=None, custom_output={}) for _ in range(req.num_reqs)]

        batch_size = prompt_token_ids.shape[0]
        has_neg_prompt = negative_prompt_ids is not None
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        # Per-request generators are created from seeds by the runner before
        # this call. Collate into one generator per generated image so initial
        # latent noise stays per-request seeded. The SDE window and SDE noise
        # injection share a single generator across the wave (GRPO-consistent).
        generator = req.collate_request_generators(num_images_per_prompt, None)
        if generator is None and common.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(common.seed)
        # prepare_latents accepts a list whose length matches the image count;
        # unwrap to a single generator when there is only one image to preserve
        # the legacy single-request RNG path exactly.
        latents_generator = generator[0] if isinstance(generator, list) and len(generator) == 1 else generator
        window_generator = generator[0] if isinstance(generator, list) else generator

        self._guidance_scale = guidance_scale
        self._attention_kwargs = None
        self._current_timestep = None
        self._interrupt = False

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt_ids=prompt_token_ids,
            attention_mask=prompt_mask,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt_ids=negative_prompt_ids,
                attention_mask=negative_prompt_mask,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None

        num_channels_latents = self.transformer.in_channels // 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            latents_generator,
            None,
        )
        img_shapes = build_img_shapes(height, width, batch_size, self.vae_scale_factor)

        timesteps, num_inference_steps = self.prepare_timesteps(num_inference_steps, sigmas, latents.shape[1])
        self._num_timesteps = len(timesteps)

        if self.transformer.guidance_embeds:
            guidance = torch.full([1], guidance_scale, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )

        if sde_window_size is not None:
            start = torch.randint(
                sde_window_range[0],
                sde_window_range[1] - sde_window_size + 1,
                (1,),
                generator=window_generator,
                device=self.device,
            ).item()
            end = start + sde_window_size
            sde_window = (start, end)
        else:
            sde_window = (0, len(timesteps) - 1)

        latents, all_latents, all_log_probs, all_timesteps = self.diffuse(
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
            window_generator,
            logprobs,
        )

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]

        batch_result = DiffusionOutput(
            output=image,
            custom_output={
                "all_latents": all_latents,
                "all_log_probs": all_log_probs,
                "all_timesteps": all_timesteps,
                "prompt_embeds": prompt_embeds,
                "prompt_embeds_mask": prompt_embeds_mask,
                "negative_prompt_embeds": negative_prompt_embeds,
                "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
            },
            to_cpu=True,
        )
        return split_diffusion_output_by_request(
            batch_result,
            num_reqs=req.num_reqs,
            num_outputs_per_prompt=num_images_per_prompt,
        )

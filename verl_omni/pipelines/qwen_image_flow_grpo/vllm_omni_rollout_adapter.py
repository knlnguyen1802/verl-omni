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
from vllm_omni.diffusion.worker.utils import DiffusionRequestState

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
        prompt: str | list[str] | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
        prompt_name: str = "prompt",
        prompts: list | None = None,
        prompt_ids: torch.Tensor | list[int] | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        """Encode text or pre-tokenized prompts into dense embeddings.

        Override of :meth:`QwenImagePipeline.encode_prompt` that accepts
        pre-tokenized ``prompt_ids`` in addition to raw text. When called from
        :meth:`QwenImagePipeline._prepare_generation_context`, ``prompts``
        (the raw OmniPromptType list) is forwarded; this override extracts
        ``prompt_ids`` / ``prompt_mask`` from the structured entries so the
        rollout pipeline can consume tokens produced upstream by the trainer
        without re-tokenising text. Plain text entries (or the engine's dummy
        warm-up run) fall back to tokenisation via the Qwen chat template.

        Args:
            prompt: Ignored; the parent ``_prepare_generation_context`` passes
                pre-extracted text here but this override sources tokens from
                ``prompts`` (or the explicit ``prompt_ids`` kwarg).
            num_images_per_prompt: Repetition count along the batch dim.
            prompt_embeds: Pre-computed embeddings; bypasses encoding when given.
            prompt_embeds_mask: Attention mask for ``prompt_embeds``.
            max_sequence_length: Maximum embedding sequence length.
            prompt_name: ``"prompt"`` or ``"negative_prompt"`` field selector.
            prompts: Raw OmniPromptType list forwarded by the base pipeline.
            prompt_ids: Optional pre-tokenized ids (used by :meth:`forward`).
            attention_mask: Optional mask for ``prompt_ids``.
        """
        del prompt  # text path handled via prompts list / dummy fallback

        if prompt_embeds is None and prompt_ids is None and prompts:
            prompt_ids, attention_mask = self._extract_ids_from_prompts(prompts, prompt_name)

        if prompt_embeds is None and prompt_ids is None:
            return None, None

        if isinstance(prompt_ids, list):
            prompt_ids = torch.tensor(prompt_ids, device=self.device)
        if prompt_ids is not None and prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
                prompt_ids, attention_mask=attention_mask
            )

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            prompt_embeds_mask = prompt_embeds_mask.repeat_interleave(num_images_per_prompt, dim=0)

        return prompt_embeds, prompt_embeds_mask

    def _extract_ids_from_prompts(self, prompts, prompt_name):
        """Pull ``prompt_ids`` / ``prompt_mask`` for ``prompt_name`` out of *prompts*.

        Falls back to tokenising the matching text field via the Qwen chat
        template (covers the engine's dummy warm-up run, which always submits
        a text prompt). Returns ``(None, None)`` when the field is absent
        (e.g. negative side with no negative prompt configured).
        """
        if not prompts:
            return None, None
        p0 = prompts[0]
        is_negative = prompt_name == "negative_prompt"
        ids_key = "negative_prompt_ids" if is_negative else "prompt_ids"
        mask_key = "negative_prompt_mask" if is_negative else "prompt_mask"
        text_key = "negative_prompt" if is_negative else "prompt"

        if isinstance(p0, dict):
            ids = p0.get(ids_key)
            mask = p0.get(mask_key)
            if ids is None and p0.get(text_key):
                text = p0[text_key]
                txt = [
                    self.prompt_template_encode.format(t)
                    for t in ([text] if isinstance(text, str) else text)
                ]
                tokens = self.tokenizer(
                    txt, padding=True, truncation=False, return_tensors="pt"
                ).to(self.device)
                ids, mask = tokens.input_ids, tokens.attention_mask
            return ids, mask
        if isinstance(p0, str) and not is_negative:
            txt = [self.prompt_template_encode.format(p0)]
            tokens = self.tokenizer(
                txt, padding=True, truncation=False, return_tensors="pt"
            ).to(self.device)
            return tokens.input_ids, tokens.attention_mask
        return None, None

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
                all_latents.append(latents)
            elif i > sde_window[0] and i < sde_window[1]:
                cur_noise_level = noise_level
            else:
                cur_noise_level = 0.0

            self._current_timestep = timestep_value
            # Broadcast timestep to match batch size
            timestep = timestep_value.expand(latents.shape[0]).to(device=latents.device, dtype=latents.dtype)

            self.transformer.do_true_cfg = do_true_cfg
            # Forward pass for positive prompt (or unconditional if no CFG)
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
            # Forward pass for negative prompt (CFG)
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

            # compute the previous noisy sample x_t -> x_t-1
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

    def step_scheduler(
        self,
        state: DiffusionRequestState,
        noise_pred: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        """One scheduler step that mirrors the per-iter body of :meth:`diffuse`.

        Mirrors :meth:`forward`'s SDE noise / log-prob bookkeeping on the
        step-execution path so that ``step_execution=True`` does not silently
        drop ``all_latents`` / ``all_log_probs`` / ``all_timesteps``. SDE state
        (window, noise level, generator, accumulators) is lazily initialised on
        the first call so the subclass does not need to override
        ``prepare_encode``.
        """
        del kwargs
        if self.interrupt:
            return

        if state.step_index == 0:
            self._init_sde_state(state)

        i = state.step_index
        timestep_value = state.timesteps[i]
        sde_window = state.sde_window

        if i < sde_window[0]:
            cur_noise_level = 0.0
        elif i == sde_window[0]:
            cur_noise_level = state.noise_level
            state.all_latents.append(state.latents)
        elif i > sde_window[0] and i < sde_window[1]:
            cur_noise_level = state.noise_level
        else:
            cur_noise_level = 0.0

        new_latents, log_prob, _, _ = state.scheduler.step(
            noise_pred,
            timestep_value,
            state.latents,
            generator=state.sampling.generator,
            noise_level=cur_noise_level,
            sde_type=state.sde_type,
            return_logprobs=state.logprobs,
            return_dict=False,
        )
        state.latents = new_latents

        if i >= sde_window[0] and i < sde_window[1]:
            state.all_latents.append(state.latents)
            state.all_log_probs.append(log_prob)
            state.all_timesteps.append(timestep_value)

        state.step_index += 1

    def _init_sde_state(self, state: DiffusionRequestState) -> None:
        """Populate SDE / rollout fields on *state* from sampling extra_args.

        Invoked from :meth:`step_scheduler` on the first step so the subclass
        no longer needs to override ``prepare_encode`` purely for SDE/log-prob
        bookkeeping. Mirrors the knob resolution previously done in the removed
        ``prepare_encode``.
        """
        sampling = state.sampling
        extra = sampling.extra_args or {}
        noise_level = _coalesce_not_none(extra.get("noise_level", None), 0.7)
        sde_window_size = _coalesce_not_none(extra.get("sde_window_size", None), None)
        sde_window_range = _coalesce_not_none(extra.get("sde_window_range", None), (0, 5))
        sde_type = _coalesce_not_none(extra.get("sde_type", None), "sde")
        logprobs = _coalesce_not_none(extra.get("logprobs", None), True)

        # Backfill a deterministic generator from sampling.seed so the per-step
        # path draws from the same RNG stream as ``forward()``.
        generator = sampling.generator
        if generator is None and sampling.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling.seed)
            sampling.generator = generator

        if sde_window_size is not None:
            start = torch.randint(
                sde_window_range[0],
                sde_window_range[1] - sde_window_size + 1,
                (1,),
                generator=generator,
                device=self.device,
            ).item()
            sde_window = (start, start + sde_window_size)
        else:
            sde_window = (0, len(state.timesteps) - 1)

        state.sde_window = sde_window
        state.noise_level = noise_level
        state.sde_type = sde_type
        state.logprobs = logprobs
        state.all_latents = []
        state.all_log_probs = []
        state.all_timesteps = []

    def post_decode(
        self,
        state: DiffusionRequestState,
        **kwargs: Any,
    ) -> DiffusionOutput:
        """Decode final latents, package rollout trajectory, and move to CPU.

        In ``step_execution`` mode the worker ships the returned
        :class:`DiffusionOutput` across an inter-process MessageQueue to the
        ``vLLMOmniHttpServer`` actor.  We must (a) move tensors to CPU so the
        receiving process does not initialise a stray CUDA context on GPU 0,
        and (b) populate ``custom_output`` with the trajectory fields that
        :meth:`forward` produces, so downstream consumers
        (``vllm_omni_async_server.generate`` ->
        ``embeds_padding_2_no_padding``) receive real tensors rather than
        ``None`` (which becomes a non-tensor ``LinkedList`` in the
        ``TensorDict`` and breaks ``mask.shape[0]``).
        """
        output = super().post_decode(state, **kwargs)
        if not isinstance(output, DiffusionOutput):
            return output

        all_latents = state.all_latents
        all_log_probs = state.all_log_probs
        all_timesteps = state.all_timesteps

        stacked_latents = torch.stack(all_latents, dim=1) if all_latents else None
        stacked_log_probs = (
            torch.stack(all_log_probs, dim=1) if all_log_probs and all_log_probs[0] is not None else None
        )
        stacked_timesteps = (
            torch.stack(all_timesteps).unsqueeze(0).expand(state.latents.shape[0], -1) if all_timesteps else None
        )

        output.output = _maybe_to_cpu(output.output)
        output.custom_output = {
            "all_latents": _maybe_to_cpu(stacked_latents),
            "all_log_probs": _maybe_to_cpu(stacked_log_probs),
            "all_timesteps": _maybe_to_cpu(stacked_timesteps),
            "prompt_embeds": _maybe_to_cpu(state.prompt_embeds),
            "prompt_embeds_mask": _maybe_to_cpu(state.prompt_embeds_mask),
            "negative_prompt_embeds": _maybe_to_cpu(state.negative_prompt_embeds),
            "negative_prompt_embeds_mask": _maybe_to_cpu(state.negative_prompt_embeds_mask),
        }
        return output

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
        :meth:`diffuse`, and decodes the final latents through the VAE.  Sampling
        parameters in *req* take precedence over the keyword arguments.

        Args:
            req (OmniDiffusionRequest): Rollout request containing prompts and
                :class:`~vllm_omni.diffusion.data.OmniDiffusionSamplingParams`.
            prompt_ids (torch.Tensor | list[int], *optional*): Token IDs for
                the positive prompt.
            prompt_mask (torch.Tensor, *optional*): Attention mask for *prompt_ids*.
            negative_prompt_ids (torch.Tensor | list[int], *optional*): Token IDs
                for the negative prompt used in True-CFG.
            negative_prompt_mask (torch.Tensor, *optional*): Attention mask for
                *negative_prompt_ids*.
            true_cfg_scale (float): Classifier-free guidance scale; CFG is
                disabled when ``<= 1``.
            height (int, *optional*): Output image height in pixels.
            width (int, *optional*): Output image width in pixels.
            num_inference_steps (int): Number of denoising steps.
            sigmas (list[float], *optional*): Custom sigmas for the scheduler.
            guidance_scale (float): Distilled guidance scale embedded in the
                transformer (``guidance_embeds`` mode).
            num_images_per_prompt (int): Number of images to generate per prompt.
            generator (torch.Generator | list[torch.Generator], *optional*):
                Random generator(s) for reproducibility.
            latents (torch.Tensor, *optional*): Pre-generated initial latents;
                sampled from a Gaussian when ``None``.
            prompt_embeds (torch.Tensor, *optional*): Pre-computed positive
                prompt embeddings; bypasses the text encoder.
            prompt_embeds_mask (torch.Tensor, *optional*): Attention mask for
                pre-computed *prompt_embeds*.
            negative_prompt_embeds (torch.Tensor, *optional*): Pre-computed
                negative prompt embeddings.
            negative_prompt_embeds_mask (torch.Tensor, *optional*): Attention
                mask for *negative_prompt_embeds*.
            output_type (str, *optional*): Format of the returned image;
                ``"latent"`` returns raw latents, otherwise the VAE-decoded image.
            attention_kwargs (dict, *optional*): Extra keyword arguments forwarded
                to the attention layers.
            callback_on_step_end_tensor_inputs (tuple[str, ...]): Names of tensors
                to expose in the step-end callback.
            max_sequence_length (int): Maximum prompt embedding sequence length.
            noise_level (float): SDE noise injection magnitude within the window.
            sde_window_size (int, *optional*): Number of SDE steps; when ``None``
                the full timestep range is used.
            sde_window_range (tuple[int, int]): ``(start, end)`` range from which
                the SDE window start position is randomly sampled.
            sde_type (str): SDE variant; ``"sde"`` or ``"cps"``.
            logprobs (bool): Whether to compute per-step log-probabilities.

        Returns:
            DiffusionOutput: Contains the decoded *output* image and a
                *custom_output* dict with keys ``"all_latents"``,
                ``"all_log_probs"``, ``"all_timesteps"``, ``"prompt_embeds"``,
                ``"prompt_embeds_mask"``, ``"negative_prompt_embeds"``, and
                ``"negative_prompt_embeds_mask"``.
        """
        custom_prompt = req.prompts[0] if req.prompts else {}
        if isinstance(custom_prompt, dict):
            prompt_ids = custom_prompt.get("prompt_ids", prompt_ids)
            prompt_mask = custom_prompt.get("prompt_mask", prompt_mask)
            negative_prompt_ids = custom_prompt.get("negative_prompt_ids", negative_prompt_ids)
            negative_prompt_mask = custom_prompt.get("negative_prompt_mask", negative_prompt_mask)

        sampling_params = req.sampling_params
        height = sampling_params.height or self.default_sample_size * self.vae_scale_factor
        width = sampling_params.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = sampling_params.num_inference_steps or num_inference_steps
        max_sequence_length = sampling_params.max_sequence_length or max_sequence_length

        noise_level = _coalesce_not_none(sampling_params.extra_args.get("noise_level", None), noise_level)
        sde_window_size = _coalesce_not_none(sampling_params.extra_args.get("sde_window_size", None), sde_window_size)
        sde_window_range = _coalesce_not_none(
            sampling_params.extra_args.get("sde_window_range", None), sde_window_range
        )
        sde_type = _coalesce_not_none(sampling_params.extra_args.get("sde_type", None), sde_type)
        logprobs = _coalesce_not_none(sampling_params.extra_args.get("logprobs", None), logprobs)

        generator = sampling_params.generator or generator
        if generator is None and sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling_params.seed)
        true_cfg_scale = _coalesce_not_none(sampling_params.true_cfg_scale, true_cfg_scale)
        req_num_outputs = getattr(sampling_params, "num_outputs_per_prompt", None)
        if req_num_outputs and req_num_outputs > 0:
            num_images_per_prompt = req_num_outputs

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt_ids is not None:
            if isinstance(prompt_ids, list):
                prompt_ids = torch.tensor(prompt_ids, device=self.device)
            batch_size = prompt_ids.shape[0] if prompt_ids.ndim == 2 else 1
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            # Both prompt_ids and prompt_embeds are None (e.g. during warmup/dummy run).
            # Return a minimal dummy output to avoid crashing.
            return DiffusionOutput(output=None, custom_output={})

        if isinstance(negative_prompt_ids, list):
            negative_prompt_ids = torch.tensor(negative_prompt_ids, device=self.device)

        has_neg_prompt = negative_prompt_ids is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )

        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
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

        num_channels_latents = self.transformer.in_channels // 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
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
                generator=generator,
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
            generator,
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

        return DiffusionOutput(
            output=_maybe_to_cpu(image),
            custom_output={
                "all_latents": _maybe_to_cpu(all_latents),
                "all_log_probs": _maybe_to_cpu(all_log_probs),
                "all_timesteps": _maybe_to_cpu(all_timesteps),
                "prompt_embeds": _maybe_to_cpu(prompt_embeds),
                "prompt_embeds_mask": _maybe_to_cpu(prompt_embeds_mask),
                "negative_prompt_embeds": _maybe_to_cpu(negative_prompt_embeds),
                "negative_prompt_embeds_mask": _maybe_to_cpu(negative_prompt_embeds_mask),
            },
        )

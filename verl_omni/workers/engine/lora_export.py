# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Shared actor-to-rollout weight export for the FSDP engines.

One implementation of the export contract (`get_per_tensor_param`):

- merge mode (``model.lora.merge=True``): stream merged (base + adapter)
  weights with ``peft_config=None``, materializing every tensor while
  ``merged_lora_context`` is still open and restoring the actor afterwards;
- adapter mode: ship full base weights once (``base_sync_done=False``), then
  adapter-only tensors plus the ``peft_config`` on every later sync;
- non-LoRA: plain full weights, no ``peft_config``.

Engines parameterize naming and collection details, not logic: the diffusers
engine adds the ``transformer.`` prefix and the DiT prefix walker, the omni
engine renames base layers for a LoRA-enabled vLLM and post-processes QAT.
"""

import logging
from contextlib import nullcontext

import torch
from torch.distributed.tensor import DTensor
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    merged_lora_context,
    normalize_peft_param_name,
    offload_fsdp_model_to_cpu,
    replace_lora_wrapper,
)
from verl.utils.model import convert_weight_keys

from verl_omni.utils.config import resolve_lora_config
from verl_omni.utils.fsdp_utils import collect_lora_params

logger = logging.getLogger(__name__)


class LoRAExportMixin:
    """Backend-agnostic weight export shared by the FSDP engines.

    Mixin contract: ``self.module``, ``self.model_config``,
    ``self._is_offload_param`` and ``self._uses_fsdp2_cpu_offload_policy``
    must exist (both DiffusersFSDPEngine and OmniFSDPEngine provide them).
    """

    def _adapter_context(self, adapter_name: str | None):
        """Select the adapter to export. Named-policy engines override this."""
        return nullcontext()

    @staticmethod
    def _cast_export_tensor(param: torch.Tensor, device: torch.device, clone_plain: bool = False) -> torch.Tensor:
        """Materialize one exported tensor.

        DTensors are gathered to full tensors and floating-point ones cast to
        bf16 (the sync wire dtype); integer tensors keep their dtype. Plain
        tensors pass through untouched unless ``clone_plain``: the merged path
        must clone because returned tensors otherwise alias module storage
        that ``merged_lora_context`` restores on exit.
        """
        if isinstance(param, DTensor):
            tensor = param.to(device, non_blocking=True).full_tensor()
            if tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
                tensor = tensor.to(dtype=torch.bfloat16, non_blocking=True)
            return tensor
        return param.detach().clone() if clone_plain else param

    def export_for_sync(
        self,
        layered_summon: bool = False,
        base_sync_done: bool = False,
        adapter_name: str | None = None,
        *,
        name_prefix: str = "",
        rename_base_layers: bool = False,
        collect_kwargs: dict | None = None,
    ):
        """Export (name, tensor) pairs plus the peft_config for one sync."""
        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)

        # FSDP2 CPUOffloadPolicy owns CPU<->GPU placement; calling model.to(device) here
        # fails the _apply tensor swap on CPU-resident params (verl#5995). The
        # per-DTensor .to(device).full_tensor() below still produces sync tensors.
        if not self._uses_fsdp2_cpu_offload_policy:
            load_fsdp_model_to_gpu(self.module)

        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        collect_kwargs = dict(collect_kwargs or {})
        peft_config = None
        merge_lora = resolve_lora_config(self.model_config).merge

        peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
        if hasattr(peft_model, "peft_config"):  # LoRA
            if not merge_lora:
                peft_config = peft_model.peft_config.get(adapter_name or "default", None)
                with self._adapter_context(adapter_name):
                    params = collect_lora_params(
                        module=self.module,
                        layered_summon=layered_summon,
                        base_sync_done=base_sync_done,
                        adapter_name=adapter_name or "default",
                        **collect_kwargs,
                    )
                if rename_base_layers and not base_sync_done:
                    # First sync into a LoRA-enabled vLLM: base keys must carry
                    # the ``base_layer`` segment the LoRA-wrapped modules expect.
                    params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
            else:
                if adapter_name not in (None, "default"):
                    # merged_lora_context merges the active ("default") adapter only;
                    # silently exporting it for a named adapter would sync the
                    # wrong policy.
                    raise ValueError(
                        "model.lora.merge=True exports the active 'default' adapter only; "
                        f"got adapter_name={adapter_name!r}."
                    )
                return self._merged_export(name_prefix=name_prefix), None
        else:
            params = self.module.state_dict()

        params = convert_weight_keys(params, peft_model)

        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

        if peft_config is not None and base_sync_done:
            per_tensor_param = params.items()
        else:
            device = get_device_id()  # used when fsdp2 set cpu_offload_policy
            per_tensor_param = ((name, self._cast_export_tensor(param, device)) for name, param in params.items())

        if name_prefix:
            per_tensor_param = ((f"{name_prefix}{name}", tensor) for name, tensor in per_tensor_param)
        peft_config_dict = peft_config.to_dict() if peft_config is not None else None
        return per_tensor_param, peft_config_dict

    def _merged_export(self, *, name_prefix: str = ""):
        """Stream merged (base + LoRA) weights for rollout weight sync.

        ``state_dict()`` returns tensors that alias the live FSDP parameter
        storage, and ``merged_lora_context`` restores the un-merged base
        weights when it exits. The context therefore must stay open until the
        consumer has materialized every tensor; ``_cast_export_tensor`` clones
        plain tensors in this path for the same reason. Consuming a state_dict
        captured inside the context after the context has exited would
        silently send base weights without the adapters.
        """
        device = get_device_id()
        try:
            with merged_lora_context(self.module, backup_adapters=True):
                params = normalize_peft_param_name(self.module.state_dict())
                params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))
                for name, param in params.items():
                    yield (f"{name_prefix}{name}", self._cast_export_tensor(param, device, clone_plain=True))
        finally:
            log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.module)
            log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

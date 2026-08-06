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
"""Reusable PEFT/LoRA adapter lifecycle helpers for training engines."""

import logging
from contextlib import contextmanager, nullcontext

import torch
from peft import LoraConfig
from verl.utils.py_functional import convert_to_regular_types

logger = logging.getLogger(__name__)


def _lora_config_from_dict(config_dict: dict):
    """Build a ``LoraConfig`` from an ``adapter_config.json`` dict.

    ``LoraConfig.from_dict`` is not exposed by every PEFT version, so try it
    first and fall back to explicit construction using only the fields the
    installed ``LoraConfig`` actually accepts (introspected from its
    constructor signature).
    """
    from peft import LoraConfig

    from_dict = getattr(LoraConfig, "from_dict", None)
    if from_dict is not None:
        try:
            return from_dict(config_dict)
        except (AttributeError, TypeError, ValueError, KeyError):
            pass

    import inspect

    candidate = {
        "r": int(config_dict.get("r", 32)),
        "lora_alpha": int(config_dict.get("lora_alpha", 64)),
        "target_modules": config_dict.get("target_modules", "all-linear"),
        "lora_dropout": float(config_dict.get("lora_dropout", 0.0)),
        "bias": config_dict.get("bias", "none"),
        "fan_in_fan_out": bool(config_dict.get("fan_in_fan_out", False)),
        "modules_to_save": config_dict.get("modules_to_save", None),
        "init_lora_weights": config_dict.get("init_lora_weights", "gaussian"),
    }
    # Forward optional fields only when present in the JSON AND supported by
    # the installed LoraConfig (avoids unknown-kwarg errors on older PEFT).
    for key in ("use_rslora", "use_dora", "lora_bias", "layer_replication", "runtimes"):
        if key in config_dict:
            candidate[key] = config_dict[key]

    sig_params = inspect.signature(LoraConfig.__init__).parameters
    kwargs = {k: v for k, v in candidate.items() if k in sig_params}
    return LoraConfig(**kwargs)


def load_peft_adapter_into(module, adapter_path: str, adapter_name: str = "default") -> None:
    """Load a pretrained PEFT LoRA adapter directory into ``module``.

    Works with both module flavours used by verl-omni engines:

    - PEFT ``PeftModel`` (e.g. non-diffusers backends) which exposes ``load_adapter``;
    - diffusers models using ``PeftAdapterMixin`` (e.g. ``SD3Transformer2DModel`` on
      diffusers>=0.37), which expose ``add_adapter`` / ``set_adapter`` / ``use_adapter``
      but *not* PEFT's ``load_adapter``.

    For the ``PeftAdapterMixin`` path we mirror ``NonDiffusersModelBase.load_lora_adapter``:
    read ``adapter_config.json``, inject the adapter, then copy the
    ``adapter_model.safetensors`` weights into the freshly created parameters.
    Mismatched keys are warned about but do not raise.
    """
    if hasattr(module, "load_adapter"):
        # PEFT PeftModel path (or a diffusers version that exposes load_adapter).
        module.load_adapter(adapter_path, adapter_name=adapter_name)
        return

    import json
    import os

    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file as safetensors_load_file

    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.isfile(adapter_config_path):
        raise FileNotFoundError(f"LoRA adapter config not found at {adapter_config_path}")
    if not os.path.isfile(adapter_weights_path):
        raise FileNotFoundError(f"LoRA adapter weights not found at {adapter_weights_path}")

    if not hasattr(module, "add_adapter"):
        raise AttributeError(
            f"Module {type(module).__name__} supports neither PEFT load_adapter nor "
            "add_adapter; cannot load pretrained LoRA adapter."
        )

    with open(adapter_config_path) as f:
        lora_config = _lora_config_from_dict(json.load(f))
    module.add_adapter(lora_config, adapter_name=adapter_name)

    adapter_state_dict = safetensors_load_file(adapter_weights_path)
    current_state = get_peft_model_state_dict(module, adapter_name=adapter_name)

    # PEFT checkpoints may carry a ``base_model.model.`` prefix; strip it to align
    # with the module's own parameter names (diffusers saves use no prefix).
    loadable = {}
    for key, value in adapter_state_dict.items():
        norm_key = key[len("base_model.model.") :] if key.startswith("base_model.model.") else key
        if norm_key in current_state:
            loadable[norm_key] = value

    missing = [k for k in current_state if k not in loadable]
    unexpected = [
        k
        for k in adapter_state_dict
        if not k.startswith("base_model.model.") and k not in current_state and k not in loadable
    ]
    if missing:
        logger.warning(
            "LoRA adapter %r: %d keys in model but not in checkpoint; they keep initial values.",
            adapter_name,
            len(missing),
        )
    if unexpected:
        logger.warning(
            "LoRA adapter %r: %d keys in checkpoint but not in model; they are ignored.",
            adapter_name,
            len(unexpected),
        )

    for key, value in loadable.items():
        current_state[key].copy_(value)


class LoRAAdapterMixin:
    """Backend-agnostic helpers for named PEFT/LoRA policy adapters."""

    def _build_lora_module(self, module):
        lora_adapter_path = getattr(self.model_config, "lora_adapter_path", None)
        policy_state_adapters = tuple(getattr(self.model_config, "policy_state_adapters", ("default",)))
        extra_adapters = tuple(adapter for adapter in policy_state_adapters if adapter not in ("default", "reference"))
        if lora_adapter_path is not None:
            from verl.utils.fs import copy_to_local

            print(f"Loading pre-trained LoRA adapter to from: {lora_adapter_path}")
            local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.model_config.use_shm)

            load_peft_adapter_into(module, local_adapter_path, adapter_name="default")
            peft_config = getattr(module, "peft_config", {}).get("default", None)
            for adapter_name in extra_adapters:
                if peft_config is not None and adapter_name not in getattr(module, "peft_config", {}):
                    module.add_adapter(peft_config, adapter_name=adapter_name)
        else:
            lora_config = {
                "r": self.model_config.lora_rank,
                "lora_alpha": self.model_config.lora_alpha,
                "init_lora_weights": self.model_config.lora_init_weights,
                "target_modules": convert_to_regular_types(self.model_config.target_modules),
                "target_parameters": convert_to_regular_types(self.model_config.target_parameters),
                "exclude_modules": convert_to_regular_types(self.model_config.exclude_modules),
                "bias": "none",
            }
            module.add_adapter(LoraConfig(**lora_config), adapter_name="default")
            for adapter_name in extra_adapters:
                module.add_adapter(LoraConfig(**lora_config), adapter_name=adapter_name)

        if "default" in policy_state_adapters and hasattr(module, "set_adapter"):
            module.set_adapter("default")

        # On-policy distillation (OPD): load frozen teacher LoRA adapter(s) onto the same
        # backbone. Teachers are activated only transiently during teacher inference
        # (see ``use_adapter``); their parameters never receive gradients.
        #
        # Multi-task OPD (DiffusionOPD, arXiv:2605.15055): ``teacher_adapters`` is a list of
        # ``{"name", "path", "guidance_scale"}`` dicts; each teacher is loaded as a frozen
        # ``teacher_<name>`` PEFT adapter. The single-teacher ``teacher_adapter_path`` /
        # ``teacher_adapter_name`` pair remains supported for backward compatibility (SOPD).
        teacher_adapters = getattr(self.model_config, "teacher_adapters", None) or []
        if teacher_adapters:
            teacher_adapters = list(teacher_adapters)
        elif getattr(self.model_config, "teacher_adapter_path", None) is not None:
            teacher_adapters = [
                {
                    "name": getattr(self.model_config, "teacher_adapter_name", "teacher"),
                    "path": self.model_config.teacher_adapter_path,
                    "guidance_scale": None,
                }
            ]

        if teacher_adapters:
            from verl.utils.fs import copy_to_local

            teacher_names = [t.get("name", "teacher") for t in teacher_adapters]
            if len(set(teacher_names)) != len(teacher_names):
                raise ValueError(f"Teacher adapter names must be unique, got {teacher_names}.")
            for teacher in teacher_adapters:
                teacher_adapter_name = teacher.get("name", "teacher")
                if teacher_adapter_name in ("default", "reference"):
                    raise ValueError(
                        f"teacher_adapter_name {teacher_adapter_name!r} collides with reserved names; "
                        "use a distinct name (e.g. 'teacher')."
                    )
                teacher_adapter_path = teacher.get("path")
                local_teacher_path = copy_to_local(teacher_adapter_path, use_shm=self.model_config.use_shm)
                load_peft_adapter_into(module, local_teacher_path, adapter_name=teacher_adapter_name)
                # Freeze the teacher adapter parameters in-place.
                for n, p in module.named_parameters():
                    if teacher_adapter_name in n.split("."):
                        p.requires_grad_(False)
                logger.info(
                    "OPD: loaded frozen teacher LoRA adapter %r from %s (adapter_name=%r)",
                    teacher_adapter_name, teacher_adapter_path, teacher_adapter_name,
                )
            # Restore the student ("default") adapter as active.
            if hasattr(module, "set_adapter"):
                module.set_adapter("default")

        lora_dtype = getattr(self.model_config, "lora_dtype", None)
        if lora_dtype is not None:
            from peft.tuners.tuners_utils import BaseTunerLayer
            from verl.utils.torch_dtypes import PrecisionType

            target_dtype = PrecisionType.to_dtype(lora_dtype)
            for name, param in module.named_parameters():
                if param.requires_grad:
                    orig_dtype = param.dtype
                    param.data = param.data.to(target_dtype)
                    logger.debug("LoRA param %s: %s -> %s", name, orig_dtype, param.dtype)

            for submodule in module.modules():
                if isinstance(submodule, BaseTunerLayer):
                    submodule.cast_input_dtype_enabled = False

        return module

    @contextmanager
    def _adapter_state_context(self):
        """Open writable adapter parameter access (FSDP summon when applicable)."""
        from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
        from verl.utils.memory_utils import aggressive_empty_cache

        from verl_omni.utils.fsdp_utils import fsdp_summon_full_params

        is_fsdp_module = fsdp_version(self.module) in (1, 2)
        is_offload_param = getattr(self, "_is_offload_param", False)
        origin_module_device = next(self.module.parameters()).device.type
        if is_fsdp_module and (is_offload_param or origin_module_device == "cpu"):
            load_fsdp_model_to_gpu(self.module)

        ctx = fsdp_summon_full_params(self.module, writeback=True) if is_fsdp_module else nullcontext()
        try:
            with ctx:
                try:
                    yield
                finally:
                    self._set_adapter("default")
        finally:
            if is_offload_param:
                offload_fsdp_model_to_cpu(self.module)
                aggressive_empty_cache(force_sync=True)

    def _set_adapter(self, name: str):
        module = getattr(self.module, "_fsdp_wrapped_module", self.module)
        if not hasattr(module, "set_adapter"):
            raise AttributeError(f"Module does not support set_adapter({name!r})")
        module.set_adapter(name)

    @contextmanager
    def use_adapter(self, name: str):
        """Temporarily select a named PEFT adapter.

        ``"reference"`` is a logical policy state (see ``policy_state_adapters``)
        that runs with all LoRA adapters disabled, not a registered PEFT adapter.
        """
        if name == "reference":
            with self.disable_adapter():
                yield
        else:
            self._set_adapter(name)
            try:
                yield
            finally:
                self._set_adapter("default")

    def _active_adapter_trainable_params(self, adapter_name: str) -> list[torch.nn.Parameter]:
        peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
        if not hasattr(peft_model, "set_adapter"):
            raise AttributeError("Module does not support PEFT adapter selection.")
        peft_model.set_adapter(adapter_name)
        return list(filter(lambda param: param.requires_grad, peft_model.parameters()))

    def copy_adapter(self, source: str = "default", target: str = "old") -> None:
        """Copy LoRA state between named policy adapters."""
        with self._adapter_state_context(), torch.no_grad():
            source_params = self._active_adapter_trainable_params(source)
            target_params = self._active_adapter_trainable_params(target)
            if len(source_params) != len(target_params) or not source_params:
                raise ValueError(
                    f"Adapter copy {source!r} -> {target!r} found mismatched params: "
                    f"{len(source_params)} vs {len(target_params)}"
                )
            for source_param, target_param in zip(source_params, target_params, strict=True):
                target_param.copy_(source_param)

    def ema_update_adapter(self, source: str = "default", target: str = "old", decay: float = 0.0) -> None:
        """EMA-update target adapter parameters from source adapter parameters."""
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"Adapter EMA decay must be in [0, 1], got {decay}.")
        with self._adapter_state_context(), torch.no_grad():
            source_params = self._active_adapter_trainable_params(source)
            target_params = self._active_adapter_trainable_params(target)
            if len(source_params) != len(target_params) or not source_params:
                raise ValueError(
                    f"Adapter EMA {source!r} -> {target!r} found mismatched params: "
                    f"{len(source_params)} vs {len(target_params)}"
                )
            for source_param, target_param in zip(source_params, target_params, strict=True):
                target_param.lerp_(source_param, 1.0 - decay)

    @contextmanager
    def disable_adapter(self):
        """Temporarily disable all PEFT adapters."""
        try:
            self.module.disable_adapters()
            yield
        finally:
            self.module.enable_adapters()

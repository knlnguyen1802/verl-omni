# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""CPU tests for the shared LoRA export contract (LoRAExportMixin).

Both FSDP engines (diffusers and omni) delegate get_per_tensor_param to the
same mixin; these tests pin the contract itself: naming parameterization,
base-then-adapter sync ordering, merge-mode guard, and the clone/cast policy
that keeps the merged stream valid after the actor restore.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import verl_omni.workers.engine.lora_export as lora_export
from verl_omni.workers.engine.lora_export import LoRAExportMixin


class _ToyModel(torch.nn.Module):
    def __init__(self, with_peft: bool = True):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False)
        if with_peft:
            self.peft_config = {"default": SimpleNamespace(to_dict=lambda: {"r": 8})}


def _make_stub(module, lora_config: dict, adapter_context=None) -> LoRAExportMixin:
    engine = object.__new__(LoRAExportMixin)
    engine.module = module
    engine._is_offload_param = False
    engine._uses_fsdp2_cpu_offload_policy = False
    engine.model_config = SimpleNamespace(
        lora=lora_config,
        lora_rank=0,
        lora_alpha=16,
        lora_adapter_path=None,
        policy_state_adapters=("default",),
    )
    if adapter_context is not None:
        engine._adapter_context = adapter_context
    return engine


def _patch_sync_helpers(monkeypatch, merged_context=None):
    monkeypatch.setattr(lora_export, "log_gpu_memory_usage", MagicMock())
    monkeypatch.setattr(lora_export, "load_fsdp_model_to_gpu", MagicMock())
    monkeypatch.setattr(lora_export, "offload_fsdp_model_to_cpu", MagicMock())
    monkeypatch.setattr(lora_export, "get_device_id", lambda: torch.device("cpu"))
    monkeypatch.setattr(lora_export, "normalize_peft_param_name", lambda state: state)
    monkeypatch.setattr(lora_export, "convert_weight_keys", lambda state, model: state)
    if merged_context is not None:
        monkeypatch.setattr(lora_export, "merged_lora_context", merged_context)


DIFFUSERS_KWARGS = {"name_prefix": "transformer.", "collect_kwargs": {"is_diffusers": True, "layer_prefixes": []}}
OMNI_KWARGS = {"rename_base_layers": True}


def test_non_lora_export_full_weights_with_engine_specific_naming(monkeypatch):
    module = _ToyModel(with_peft=False)
    module.proj.weight.detach().fill_(3.0)
    _patch_sync_helpers(monkeypatch)

    diffusers_engine = _make_stub(module, lora_config={})
    omni_engine = _make_stub(module, lora_config={})

    diffusers_weights, diffusers_cfg = diffusers_engine.export_for_sync(base_sync_done=True, **DIFFUSERS_KWARGS)
    omni_weights, omni_cfg = omni_engine.export_for_sync(base_sync_done=True, **OMNI_KWARGS)

    diffusers_weights = dict(diffusers_weights)
    omni_weights = dict(omni_weights)
    assert diffusers_cfg is None and omni_cfg is None
    assert set(diffusers_weights) == {"transformer.proj.weight"}
    assert set(omni_weights) == {"proj.weight"}
    # Plain fp32 tensors pass through uncast and uncloned on the base path
    # (state_dict tensors alias the parameter storage).
    assert omni_weights["proj.weight"].data_ptr() == module.proj.weight.data_ptr()
    assert omni_weights["proj.weight"].dtype == torch.float32


def test_merge_mode_rejects_named_adapter_for_both_engines(monkeypatch):
    _patch_sync_helpers(monkeypatch, merged_context=MagicMock())
    module = _ToyModel()

    for kwargs in (DIFFUSERS_KWARGS, OMNI_KWARGS):
        engine = _make_stub(module, lora_config={"merge": True})
        with pytest.raises(ValueError, match="adapter_name='old'"):
            engine.export_for_sync(base_sync_done=True, adapter_name="old", **kwargs)
        _, peft_config = engine.export_for_sync(base_sync_done=True, adapter_name="default", **kwargs)
        assert peft_config is None


def test_merged_export_clones_before_actor_restore(monkeypatch):
    module = _ToyModel()
    original = module.proj.weight.detach().clone()

    @contextmanager
    def merged_context(actor, backup_adapters):
        assert backup_adapters
        with torch.no_grad():
            module.proj.weight.fill_(2.0)  # merged weights live while the context is open
        try:
            yield
        finally:
            with torch.no_grad():
                module.proj.weight.copy_(original)  # actor restored on exit

    _patch_sync_helpers(monkeypatch, merged_context=merged_context)
    engine = _make_stub(module, lora_config={"merge": True})

    merged_stream, _ = engine.export_for_sync(name_prefix="transformer.")
    weights = dict(merged_stream)
    exported = weights["transformer.proj.weight"]

    torch.testing.assert_close(exported, torch.full_like(original, 2.0))
    # The stream must not alias module storage: the restore already ran.
    assert exported.data_ptr() != module.proj.weight.data_ptr()
    torch.testing.assert_close(module.proj.weight, original)


def test_adapter_mode_base_sync_then_adapter_only(monkeypatch):
    module = _ToyModel()
    adapter_weight = torch.zeros(8, 4)
    collect = MagicMock(return_value={"proj.lora_A.weight": adapter_weight})
    monkeypatch.setattr(lora_export, "collect_lora_params", collect)
    _patch_sync_helpers(monkeypatch)
    monkeypatch.setattr(lora_export, "replace_lora_wrapper", lambda name, peft_config: f"renamed:{name}")

    engine = _make_stub(module, lora_config={})

    # First sync: full base weights, base_layer rename applied for LoRA-enabled vLLM.
    collect.return_value = {"proj.weight": module.proj.weight}
    base_params, peft_config = engine.export_for_sync(base_sync_done=False, **OMNI_KWARGS)
    assert dict(base_params) == {"renamed:proj.weight": module.proj.weight}
    assert peft_config == {"r": 8}

    # Steady state: adapter-only tensors pass through untouched, config still attached.
    collect.return_value = {"proj.lora_A.weight": adapter_weight}
    adapter_params, peft_config = engine.export_for_sync(base_sync_done=True, **DIFFUSERS_KWARGS)
    weights = dict(adapter_params)
    assert set(weights) == {"transformer.proj.lora_A.weight"}
    assert weights["transformer.proj.lora_A.weight"].data_ptr() == adapter_weight.data_ptr()
    assert peft_config == {"r": 8}

    collect.assert_called_with(
        module=module,
        layered_summon=False,
        base_sync_done=True,
        adapter_name="default",
        is_diffusers=True,
        layer_prefixes=[],
    )


def test_adapter_context_hook_selects_named_adapter(monkeypatch):
    _patch_sync_helpers(monkeypatch)
    module = _ToyModel()
    entered = []

    @contextmanager
    def recording_context(adapter_name):
        entered.append(adapter_name)
        yield

    engine = _make_stub(module, lora_config={}, adapter_context=recording_context)
    monkeypatch.setattr(lora_export, "collect_lora_params", MagicMock(return_value={}))

    engine.export_for_sync(base_sync_done=True, adapter_name="old", **DIFFUSERS_KWARGS)
    assert entered == ["old"]

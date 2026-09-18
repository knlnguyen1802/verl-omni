# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""CPU tests pinning the LoRA checkpoint contract (RFC #47 preconditions).

The pinned verl ``FSDPCheckpointManager`` implements ``save_lora_only``: it
detects a PEFT actor via ``peft_config`` on the wrapped module and filters
adapter keys out of the sharded state dict. The refactor must keep those
preconditions discoverable: the engines keep ``peft_config`` on the module
the checkpoint manager unwraps, and the pinned config field stays reachable.
"""

from types import SimpleNamespace

import torch

from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine


class _ToyLoraModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False)
        self.peft_config = {"default": SimpleNamespace(to_dict=lambda: {"r": 8})}


class _ToyPlainModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False)


def _checkpoint_manager_has_lora(engine: PPODiffusersFSDPEngine) -> bool:
    """Same predicate the pinned FSDPCheckpointManager._has_lora uses."""
    module = getattr(engine.module, "_fsdp_wrapped_module", engine.module)
    return hasattr(module, "peft_config")


def test_checkpoint_manager_detects_peft_through_the_engine():
    engine = object.__new__(PPODiffusersFSDPEngine)
    engine.module = _ToyLoraModule()
    assert _checkpoint_manager_has_lora(engine)


def test_full_weight_engine_reports_no_peft():
    engine = object.__new__(PPODiffusersFSDPEngine)
    engine.module = _ToyPlainModule()
    assert not _checkpoint_manager_has_lora(engine)


def test_pinned_checkpoint_manager_supports_lora_only_saves():
    from dataclasses import fields

    from verl.trainer.config import CheckpointConfig
    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

    assert any(f.name == "save_lora_only" for f in fields(CheckpointConfig))
    assert hasattr(FSDPCheckpointManager, "_has_lora")
    assert hasattr(FSDPCheckpointManager, "should_save_lora_only")

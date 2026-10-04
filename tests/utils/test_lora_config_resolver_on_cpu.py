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
"""CPU tests for the LoRA config resolver (verl_omni.utils.config)."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import verl_omni
from verl_omni.utils.config import (
    LoraConfigConflictError,
    LoRASettings,
    UnknownLoraKeyError,
    resolve_lora_config,
)

_CONFIG_DIR = Path(verl_omni.__file__).parent / "trainer" / "config"


def _model_config(**overrides):
    base = {
        "lora_rank": 0,
        "lora_alpha": 16,
        "lora_adapter_path": None,
        "policy_state_adapters": ["default"],
        "lora": {"merge": False},
    }
    base.update(overrides)
    return OmegaConf.create(base)


def test_default_config_resolves_disabled():
    settings = resolve_lora_config(_model_config())
    assert settings == LoRASettings(
        rank=0, alpha=16, adapter_path=None, merge=False, adapters=("default",), enabled=False
    )


def test_flat_rank_enables_lora():
    settings = resolve_lora_config(_model_config(lora_rank=32))
    assert settings.rank == 32
    assert settings.enabled


def test_nested_rank_wins_when_flat_is_zero():
    settings = resolve_lora_config(_model_config(lora={"rank": 32, "merge": False}))
    assert settings.rank == 32
    assert settings.enabled


def test_equal_rank_in_both_spellings_is_accepted():
    settings = resolve_lora_config(_model_config(lora_rank=32, lora={"rank": 32, "merge": False}))
    assert settings.rank == 32


def test_conflicting_rank_spellings_raise():
    with pytest.raises(LoraConfigConflictError, match="lora_rank=32"):
        resolve_lora_config(_model_config(lora_rank=32, lora={"rank": 16, "merge": False}))


def test_merge_is_read_from_nested_block_only():
    assert resolve_lora_config(_model_config(lora={"merge": True})).merge
    assert not resolve_lora_config(_model_config()).merge


def test_adapter_path_conflict_raises():
    with pytest.raises(LoraConfigConflictError, match="adapter_path"):
        resolve_lora_config(_model_config(lora_adapter_path="/a", lora={"merge": False, "adapter_path": "/b"}))


def test_unknown_nested_key_raises():
    with pytest.raises(UnknownLoraKeyError, match="does_not_exist"):
        resolve_lora_config(_model_config(lora={"merge": False, "does_not_exist": 1}))


def test_megatron_pinned_keys_are_tolerated():
    nested = {
        "type": "lora",
        "alpha": 32,
        "dropout": 0.0,
        "target_modules": ["linear_qkv"],
        "exclude_modules": [],
        "dropout_position": "pre",
        "lora_A_init_method": "xavier",
        "lora_B_init_method": "zero",
        "a2a_experimental": False,
        "dtype": None,
        "adapter_path": None,
        "freeze_vision_model": True,
        "freeze_vision_projection": True,
        "freeze_language_model": True,
    }
    settings = resolve_lora_config(_model_config(lora={"merge": False, **nested}))
    assert settings.merge is False


def test_policy_state_adapters_default_forced_first():
    assert resolve_lora_config(_model_config(policy_state_adapters=["old", "default"])).adapters == ("default", "old")
    assert resolve_lora_config(_model_config(policy_state_adapters=["old"])).adapters == ("default", "old")
    assert resolve_lora_config(_model_config(policy_state_adapters=["old", "old", "reference"])).adapters == (
        "default",
        "old",
        "reference",
    )


def test_unknown_policy_state_raises():
    with pytest.raises(ValueError, match="banana"):
        resolve_lora_config(_model_config(policy_state_adapters=["banana"]))


def test_resolver_accepts_dataclass_style_access():
    config = SimpleNamespace(
        lora_rank=8,
        lora_alpha=16,
        lora_adapter_path=None,
        policy_state_adapters=("default", "old"),
        lora={"merge": True},
    )
    settings = resolve_lora_config(config)
    assert settings.rank == 8
    assert settings.merge
    assert settings.adapters == ("default", "old")


def test_non_mapping_nested_lora_raises():
    with pytest.raises(ValueError, match="mapping"):
        resolve_lora_config(_model_config(lora=32))


@pytest.mark.parametrize(
    "generated_yaml",
    ["_generated_omni_trainer.yaml", "_generated_omni_megatron_trainer.yaml", "_generated_diffusion_trainer.yaml"],
)
def test_generated_configs_resolve_with_pinned_megatron_keys(generated_yaml):
    """Composed trainer configs still carry the Megatron-pinned nested keys (verl
    injects them via its defaults chain); the resolver must tolerate and read them."""
    config = OmegaConf.load(_CONFIG_DIR / generated_yaml)
    settings = resolve_lora_config(config.actor_rollout_ref.model)
    assert settings.enabled is False
    assert settings.merge is False
    assert config.actor_rollout_ref.model.lora_rank == 0


def test_omni_model_yaml_nested_block_trims_to_merge():
    source = OmegaConf.load(_CONFIG_DIR / "omni" / "model" / "omni_model.yaml")
    assert set(source.lora.keys()) == {"merge"}

# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""CPU tests for the shared actor LoRA config resolver."""

import os

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl_omni.utils.config import LoRASettings, resolve_lora_config


def _model(**overrides):
    base = {
        "lora_rank": 0,
        "lora_alpha": 64,
        "lora": {"merge": False},
        "lora_adapter_path": None,
        "policy_state_adapters": ["default"],
    }
    base.update(overrides)
    return OmegaConf.create(base)


def test_flat_rank_enables_lora():
    settings = resolve_lora_config(_model(lora_rank=64))
    assert settings == LoRASettings(rank=64, alpha=64, adapters=("default",))
    assert settings.enabled


def test_nested_rank_is_the_legacy_spelling():
    settings = resolve_lora_config(_model(lora={"rank": 32}))
    assert settings.rank == 32
    assert settings.enabled


def test_nested_rank_zero_falls_back_to_flat():
    settings = resolve_lora_config(_model(lora={"rank": 0}, lora_rank=16))
    assert settings.rank == 16


def test_conflicting_ranks_raise():
    with pytest.raises(ValueError, match="Conflicting LoRA rank.*lora.rank=32.*lora_rank=64"):
        resolve_lora_config(_model(lora={"rank": 32}, lora_rank=64))


def test_equal_nested_and_flat_rank_is_accepted():
    settings = resolve_lora_config(_model(lora={"rank": 32}, lora_rank=32))
    assert settings.rank == 32


def test_adapter_path_enables_lora():
    settings = resolve_lora_config(_model(lora_adapter_path="/tmp/adapter"))
    assert settings.enabled and settings.adapter_path == "/tmp/adapter"


def test_disabled_by_default():
    settings = resolve_lora_config(_model())
    assert not settings.enabled
    assert settings.rank == 0
    assert not settings.merge
    assert settings.adapter_path is None


def test_merge_flag_is_resolved():
    settings = resolve_lora_config(_model(lora={"merge": True}, lora_rank=64))
    assert settings.enabled and settings.merge


def test_unknown_nested_key_raises():
    with pytest.raises(ValueError, match="Unknown actor_rollout_ref.model.lora keys: \\['bogus'\\].*lora.merge"):
        resolve_lora_config(_model(lora={"merge": False, "bogus": 1}))


def test_verl_injected_megatron_keys_are_tolerated_but_unread():
    injected = {
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
    settings = resolve_lora_config(_model(lora={"merge": False, "rank": 0, **injected}))
    assert not settings.enabled and not settings.merge


def test_adapters_are_normalized_to_tuple():
    settings = resolve_lora_config(_model(policy_state_adapters=["default", "old"]))
    assert settings.adapters == ("default", "old")


def test_attribute_style_config_is_supported():
    class ModelConfig:
        lora_rank = 8
        lora_alpha = 16
        lora = {"merge": True}
        lora_adapter_path = None
        policy_state_adapters = ("default",)

    settings = resolve_lora_config(ModelConfig())
    assert settings.rank == 8 and settings.alpha == 16 and settings.merge


def test_non_mapping_lora_raises():
    with pytest.raises(ValueError, match="must be a mapping"):
        resolve_lora_config(_model(lora="lora"))


@pytest.mark.parametrize(
    "config_name",
    ["diffusion_trainer", "omni_trainer"],
)
def test_composed_trainer_defaults_resolve_clean(config_name):
    config_dir = os.path.abspath("verl_omni/trainer/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name=config_name)
    settings = resolve_lora_config(config.actor_rollout_ref.model)
    assert not settings.enabled
    assert settings.rank == 0
    assert settings.adapter_path is None
    assert not settings.merge
    assert settings.adapters == ("default",)


# Note: the worker config dataclasses (DiffusionModelConfig / OmniModelConfig)
# are frozen and their __post_init__ validates the model path against the HF
# hub, so they cannot be instantiated standalone on CPU. The resolver reads
# them through the same getattr path covered by
# test_attribute_style_config_is_supported.

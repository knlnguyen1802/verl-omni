# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Fail-fast validation shared by VeRL-Omni trainer entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Nested ``model.lora:`` keys verl-omni consumes. ``merge`` selects the weight
# sync mode; ``rank`` is honored only as the legacy spelling of ``lora_rank``.
# The rest of the nested block is Megatron-only upstream and no verl-omni code
# reads it — but verl's pinned ``ppo_trainer`` -> ``hf_model`` default merges
# those keys into every composed config, and verl-omni cannot remove pinned
# keys without forking verl's config tree, so they are tolerated. Any other
# nested key raises.
_VERL_INJECTED_LORA_KEYS = frozenset(
    {
        "type",
        "alpha",
        "dropout",
        "target_modules",
        "exclude_modules",
        "dropout_position",
        "lora_A_init_method",
        "lora_B_init_method",
        "a2a_experimental",
        "dtype",
        "adapter_path",
        "freeze_vision_model",
        "freeze_vision_projection",
        "freeze_language_model",
    }
)
_LORA_NESTED_KEYS = frozenset({"merge", "rank"}) | _VERL_INJECTED_LORA_KEYS


@dataclass(frozen=True)
class LoRASettings:
    """Resolved actor LoRA settings; the single reader of both spellings."""

    rank: int = 0
    alpha: int = 0
    adapter_path: str | None = None
    merge: bool = False
    adapters: tuple[str, ...] = ("default",)

    @property
    def enabled(self) -> bool:
        """True when the actor trains adapters (rank > 0 or a warm-start path)."""
        return self.rank > 0 or self.adapter_path is not None


def resolve_lora_config(model_config: Any) -> LoRASettings:
    """Resolve the actor LoRA settings from flat and nested config spellings.

    verl-omni reads exactly one nested ``lora:`` key — ``merge`` — plus the
    legacy ``lora.rank`` spelling; every other knob is a flat ``model`` field
    (``lora_rank``, ``lora_alpha``, ``target_modules``, ...). The remaining
    nested keys are injected by verl's pinned config and tolerated but unread;
    anything outside that set raises instead of being silently ignored, and
    conflicting ``lora.rank`` / ``lora_rank`` values raise instead of the
    nested one silently winning.
    """
    nested = _select(model_config, "lora", None)
    if nested is None:
        nested = {}
    try:
        nested_keys = set(nested.keys())
    except AttributeError as exc:
        raise ValueError(f"actor_rollout_ref.model.lora must be a mapping, got {type(nested).__name__}.") from exc
    unknown = sorted(nested_keys - _LORA_NESTED_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown actor_rollout_ref.model.lora keys: {unknown}. verl-omni reads only "
            "lora.merge (weight sync mode) and the legacy lora.rank spelling; other LoRA "
            "knobs are flat model fields (lora_rank, lora_alpha, target_modules, ...). "
            "Megatron-only keys injected by the pinned verl config are tolerated but "
            "unread; if you just bumped the verl pin, update "
            "_VERL_INJECTED_LORA_KEYS in verl_omni/utils/config.py."
        )
    nested_rank = int(_select(nested, "rank", 0) or 0)
    flat_rank = int(_select(model_config, "lora_rank", 0) or 0)
    if nested_rank > 0 and flat_rank > 0 and nested_rank != flat_rank:
        raise ValueError(
            f"Conflicting LoRA rank: actor_rollout_ref.model.lora.rank={nested_rank} vs "
            f"actor_rollout_ref.model.lora_rank={flat_rank}. Set exactly one."
        )
    return LoRASettings(
        rank=nested_rank if nested_rank > 0 else flat_rank,
        alpha=int(_select(model_config, "lora_alpha", 0) or 0),
        adapter_path=_select(model_config, "lora_adapter_path", None),
        merge=bool(_select(nested, "merge", False)),
        adapters=tuple(_select(model_config, "policy_state_adapters", ("default",)) or ("default",)),
    )


def _select(config: Any, path: str, default: Any = None) -> Any:
    value = config
    for part in path.split("."):
        if value is None:
            return default
        if hasattr(value, "get"):
            value = value.get(part, default)
        else:
            value = getattr(value, part, default)
    return default if value is None else value


def validate_config(config: Any) -> None:
    """Validate configuration values that otherwise trigger silent fallbacks."""
    if _select(config, "actor_rollout_ref.actor.enable_timestep_staging", False):
        sp_size = _select(config, "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size", 1)
        if sp_size != 1:
            raise ValueError("Timestep staging requires ulysses_sequence_parallel_size=1.")

    resume_mode = _select(config, "trainer.resume_mode")
    valid_resume_modes = ("disable", "auto", "resume_path")
    if resume_mode not in valid_resume_modes:
        raise ValueError(f"Unknown trainer.resume_mode={resume_mode!r}. Available options: {list(valid_resume_modes)}.")
    if resume_mode == "resume_path" and not _select(config, "trainer.resume_from_path"):
        raise ValueError("trainer.resume_from_path must be set when trainer.resume_mode='resume_path'.")

    total_steps = _select(config, "trainer.total_training_steps")
    if total_steps is not None:
        try:
            total_steps = int(total_steps)
        except (TypeError, ValueError) as exc:
            raise ValueError("trainer.total_training_steps must be a positive integer or null.") from exc
        if total_steps <= 0:
            raise ValueError("trainer.total_training_steps must be a positive integer or null.")

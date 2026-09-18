# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Typed contract for named LoRA policy states and the weight-sync decision.

Two contracts that used to live as bare strings and inline conditionals:

- Adapter scopes: ``policy`` (the training adapter), ``old`` (DiffusionNFT
  old-policy state) and ``reference`` (run with all adapters disabled — the
  in-actor reference policy). The batch-metadata key for the reference case
  stays verl's pinned ``no_lora_adapter`` boolean: the omni v1 path runs
  verl's own trainer, which writes it, so the wire format cannot change.
- The LoRA sync plan: merge vs adapter mode decides what a sync ships and
  how far the rollout sleeps. One function, one table.
"""

from dataclasses import dataclass
from typing import Any, Mapping


class AdapterScope:
    """Named LoRA policy states.

    ``REFERENCE`` is a pseudo state implemented as ``disable_adapter()`` —
    it is not a registered PEFT adapter.
    """

    POLICY = "policy"
    OLD = "old"
    REFERENCE = "reference"

    NAMED = frozenset({POLICY, OLD, REFERENCE})

    @classmethod
    def validate(cls, scope: str) -> str:
        if scope not in cls.NAMED:
            raise ValueError(f"Unknown adapter scope {scope!r}. Available: {sorted(cls.NAMED)}.")
        return scope


# Batch-metadata key for the in-actor reference policy. Key and boolean value
# are fixed by verl's pinned trainer; do not rename.
REFERENCE_FLAG = "no_lora_adapter"


def adapters_disabled(metadata: Mapping[str, Any]) -> bool:
    """True when the batch must run with all LoRA adapters disabled."""
    return bool(metadata.get(REFERENCE_FLAG, False))


@dataclass(frozen=True)
class LoraSyncPlan:
    """What a rollout weight sync ships, and how far the rollout sleeps.

    For adapter mode the rollout keeps base weights resident (sleep level 1);
    the first sync ships the full base weights (the rollout was seeded with
    ``load_format=dummy``) and every later sync ships adapter tensors only.
    Merge and full-weight runs ship full weights every sync and sleep at
    level 2.
    """

    adapter_only: bool
    needs_base_sync: bool
    sleep_level: int


def resolve_lora_sync_plan(*, lora_enabled: bool, merge: bool, base_sync_done: bool) -> LoraSyncPlan:
    """Resolve the sync decision table for one worker from its sync knobs."""
    if lora_enabled and not merge:
        return LoraSyncPlan(
            adapter_only=base_sync_done,
            needs_base_sync=not base_sync_done,
            sleep_level=1,
        )
    return LoraSyncPlan(adapter_only=False, needs_base_sync=False, sleep_level=2)

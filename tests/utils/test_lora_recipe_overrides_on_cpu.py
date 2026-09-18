# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Recipe guard: nested ``model.lora.`` overrides in example scripts.

verl-omni reads exactly two nested ``lora:`` keys — ``merge`` (sync mode)
and the legacy ``rank`` spelling. Every other LoRA knob is a flat
``actor_rollout_ref.model.`` field, and the shared resolver
(verl_omni.utils.config.resolve_lora_config) rejects unknown nested keys at
launch. This test keeps example scripts inside that contract so a recipe
never ships a silent no-op override again (a ``lora.dropout`` override
survived unnoticed until phase 1 of the LoRA refactor, RFC #55).
"""

import re
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
NESTED_LORA_OVERRIDE = re.compile(r"actor_rollout_ref\.model\.lora\.(\w+)\s*=")
# ``merge`` selects the sync mode; ``rank`` is the honored legacy spelling.
ALLOWED_NESTED_KEYS = {"merge", "rank"}


def _recipe_scripts() -> list[Path]:
    return sorted(EXAMPLES_DIR.rglob("*.sh"))


def test_nested_lora_overrides_stay_inside_the_config_contract():
    violations = []
    for script in _recipe_scripts():
        for lineno, line in enumerate(script.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for match in NESTED_LORA_OVERRIDE.finditer(stripped):
                key = match.group(1)
                if key not in ALLOWED_NESTED_KEYS:
                    violations.append(
                        f"{script.relative_to(EXAMPLES_DIR.parent)}:{lineno}: "
                        f"actor_rollout_ref.model.lora.{key} is not read by verl-omni "
                        f"(allowed nested keys: {sorted(ALLOWED_NESTED_KEYS)}); "
                        "use the flat model fields (lora_rank, lora_alpha, target_modules, ...)"
                    )
    assert not violations, "Dead or unknown nested lora overrides:\n" + "\n".join(violations)


def test_recipes_do_not_set_conflicting_rank_spellings():
    conflicts = []
    for script in _recipe_scripts():
        text = script.read_text(encoding="utf-8", errors="replace")
        sets_nested_rank = re.search(r"actor_rollout_ref\.model\.lora\.rank\s*=", text)
        sets_flat_rank = re.search(r"actor_rollout_ref\.model\.lora_rank\s*=", text)
        if sets_nested_rank and sets_flat_rank:
            conflicts.append(
                f"{script.relative_to(EXAMPLES_DIR.parent)} sets both lora.rank and lora_rank; "
                "the resolver raises on conflicting values"
            )
    assert not conflicts, "\n".join(conflicts)

# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""CPU tests for the adapter-scope contract and the LoRA sync decision table."""

import pytest

from verl_omni.utils.adapter_scope import (
    REFERENCE_FLAG,
    AdapterScope,
    adapters_disabled,
    resolve_lora_sync_plan,
)


class TestAdapterScope:
    def test_named_scopes(self):
        assert AdapterScope.NAMED == {"policy", "old", "reference"}

    def test_validate_accepts_known_scopes(self):
        for scope in AdapterScope.NAMED:
            assert AdapterScope.validate(scope) == scope

    def test_validate_rejects_unknown_scope(self):
        with pytest.raises(ValueError, match="Unknown adapter scope 'default_0'.*Available"):
            AdapterScope.validate("default_0")


class TestReferenceFlag:
    def test_flag_key_is_the_pinned_verl_boolean(self):
        # The omni v1 path runs verl's own trainer, which writes this exact key.
        assert REFERENCE_FLAG == "no_lora_adapter"

    def test_adapters_disabled_reads_metadata(self):
        assert adapters_disabled({"no_lora_adapter": True})
        assert not adapters_disabled({"no_lora_adapter": False})
        assert not adapters_disabled({})

    def test_adapters_disabled_tolerates_truthy_strings(self):
        assert adapters_disabled({"no_lora_adapter": "True"})


class TestLoraSyncPlan:
    def test_adapter_mode_first_sync_ships_base_and_sleeps_light(self):
        plan = resolve_lora_sync_plan(lora_enabled=True, merge=False, base_sync_done=False)
        assert plan is not None
        assert plan.adapter_only is False
        assert plan.needs_base_sync is True
        assert plan.sleep_level == 1

    def test_adapter_mode_steady_state_ships_adapter_only(self):
        plan = resolve_lora_sync_plan(lora_enabled=True, merge=False, base_sync_done=True)
        assert plan.adapter_only is True
        assert plan.needs_base_sync is False
        assert plan.sleep_level == 1

    def test_merge_mode_ships_full_weights_and_sleeps_deep(self):
        plan = resolve_lora_sync_plan(lora_enabled=True, merge=True, base_sync_done=False)
        assert plan.adapter_only is False
        assert plan.needs_base_sync is False
        assert plan.sleep_level == 2

    def test_full_weight_training_sleeps_deep(self):
        plan = resolve_lora_sync_plan(lora_enabled=False, merge=False, base_sync_done=True)
        assert plan.adapter_only is False
        assert plan.sleep_level == 2

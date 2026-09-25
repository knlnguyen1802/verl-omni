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
"""CPU tests for verl_omni.utils.config validation."""

import pytest
from omegaconf import OmegaConf

from verl_omni.utils.config import validate_config


def _config(**trainer):
    return OmegaConf.create({"trainer": {"resume_mode": "disable", **trainer}})


def test_validate_config_rejects_unknown_resume_mode():
    with pytest.raises(ValueError, match="Available options"):
        validate_config(_config(resume_mode="resumee"))


def test_validate_config_requires_resume_path():
    with pytest.raises(ValueError, match="resume_from_path"):
        validate_config(_config(resume_mode="resume_path"))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("sp_size", [1, 2])
@pytest.mark.parametrize("as_dict", [False, True])
def test_validate_config_timestep_staging(enabled, sp_size, as_dict):
    config = _config()
    config.actor_rollout_ref = {
        "actor": {"enable_timestep_staging": enabled, "fsdp_config": {"ulysses_sequence_parallel_size": sp_size}}
    }
    if as_dict:
        config = OmegaConf.to_container(config)
    if enabled and sp_size != 1:
        with pytest.raises(ValueError, match="sequence_parallel_size=1"):
            validate_config(config)
    else:
        validate_config(config)


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2", "veomni", "megatron"])
@pytest.mark.parametrize("enabled", [False, True])
def test_validate_config_no_sync_gradient_accumulation(strategy, enabled):
    config = _config()
    config.actor_rollout_ref = {"actor": {"strategy": strategy, "use_no_sync_for_gradient_accumulation": enabled}}
    if enabled and strategy not in ("fsdp", "fsdp2"):
        with pytest.raises(ValueError, match="fsdp or fsdp2"):
            validate_config(config)
    else:
        validate_config(config)


def _separate_async_config(param_offload, sync_step, strategy="fsdp2", trainer_mode="separate_async"):
    config = _config()
    config.trainer.v1 = {"trainer_mode": trainer_mode, "separate_async": {"parameter_sync_step": sync_step}}
    config.actor_rollout_ref = {"actor": {"strategy": strategy, "fsdp_config": {"param_offload": param_offload}}}
    return config


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2", "veomni", "megatron"])
@pytest.mark.parametrize("param_offload", [False, True])
@pytest.mark.parametrize("sync_step", [1, 8])
def test_validate_config_rejects_offloaded_decoupled_ppo_snapshots(param_offload, sync_step, strategy):
    # verl-project/verl-omni#645: the plain separate_async mode runs the
    # decoupled-PPO snapshot dance on verl's raw DetachActorWorker, whose
    # fsdp2 save helper returns storage-sharing views for offloaded
    # (CPU-resident) shards — training silently freezes at
    # parameter_sync_step>1. omni_separate_async is exempt (its worker
    # clones aliased slots) and so is fsdp1 (its save helper already copies).
    config = _separate_async_config(param_offload, sync_step, strategy)
    if param_offload and sync_step > 1 and strategy in ("fsdp2", "veomni"):
        with pytest.raises(ValueError, match="silently reset every"):
            validate_config(config)
    else:
        validate_config(config)


def test_validate_config_allows_omni_separate_async_offload():
    # The omni worker owns snapshot storage, so offload stays a valid (if
    # slower) choice there.
    validate_config(_separate_async_config(True, 8, trainer_mode="omni_separate_async"))

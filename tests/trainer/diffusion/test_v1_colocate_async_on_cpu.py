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
"""CPU tests for diffusion V1 colocate_async mode config and validation."""

import os

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync

import verl_omni
from verl_omni.trainer.diffusion.diffusion_trainer_utils import validate_distillation_config
from verl_omni.trainer.diffusion.v1.trainer_base import (
    PolicyGradientDiffusionTrainerV1,
    get_diffusion_trainer_cls,
)
from verl_omni.trainer.diffusion.v1.trainer_separate_async import PolicyGradientDiffusionTrainerV1SeparateAsync
from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(verl_omni.__file__)), "trainer", "config")


class _ColocateAsyncConfigProbe(PolicyGradientDiffusionTrainerV1):
    """Concrete base trainer used to exercise colocate_async config validation."""

    def on_step_end(self):
        return

    def on_sample_end(self):
        return


def _compose_config(*overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="diffusion_trainer", overrides=list(overrides))


def _colocate_config(*overrides):
    return _compose_config("trainer.use_v1=true", "trainer.v1.trainer_mode=colocate_async", *overrides)


class TestColocateAsyncConfig:
    def test_composes_with_expected_defaults(self):
        config = _colocate_config()
        assert config.trainer.v1.colocate_async.num_warmup_batches == 1
        assert config.trainer.v1.colocate_async.parameter_sync_step == 1


class TestTrainerRegistry:
    def test_known_modes_resolve(self):
        assert get_diffusion_trainer_cls("sync") is PolicyGradientDiffusionTrainerV1Sync
        assert get_diffusion_trainer_cls("separate_async") is PolicyGradientDiffusionTrainerV1SeparateAsync

    def test_unknown_mode_lists_available(self):
        with pytest.raises(ValueError, match="Unknown diffusion trainer 'bogus'.*Available:.*separate_async.*sync"):
            get_diffusion_trainer_cls("bogus")


class TestColocateAsyncModeValidation:
    def test_default_config_constructs(self):
        trainer = _ColocateAsyncConfigProbe(_colocate_config())
        assert trainer.trainer_mode == "colocate_async"
        assert trainer.parameter_sync_step == 1

    def test_multi_update_cycle_not_implemented(self):
        with pytest.raises(NotImplementedError, match="parameter_sync_step=1"):
            _ColocateAsyncConfigProbe(_colocate_config("trainer.v1.colocate_async.parameter_sync_step=2"))

    @pytest.mark.parametrize("value", [0, -1])
    def test_non_positive_parameter_sync_step_raises(self, value):
        with pytest.raises(ValueError, match="parameter_sync_step must be a positive integer"):
            _ColocateAsyncConfigProbe(_colocate_config(f"trainer.v1.colocate_async.parameter_sync_step={value}"))

    def test_non_integer_parameter_sync_step_raises(self):
        config = _colocate_config()
        with open_dict(config):
            config.trainer.v1.colocate_async.parameter_sync_step = "two"
        with pytest.raises(ValueError, match="parameter_sync_step must be a positive integer"):
            _ColocateAsyncConfigProbe(config)

    def test_other_modes_skip_colocate_validation(self):
        trainer = object.__new__(_ColocateAsyncConfigProbe)
        trainer.trainer_mode = "separate_async"
        trainer.parameter_sync_step = 4
        trainer._validate_trainer_mode_config()


class TestReplayBufferSelection:
    def test_colocate_async_uses_async_replay_buffer(self):
        trainer = _ColocateAsyncConfigProbe(_colocate_config())
        assert isinstance(trainer.replay_buffer, ReplayBufferAsync)
        assert trainer.replay_buffer.refill_fn == trainer._add_prompts_to_generate

    def test_sync_still_uses_sync_replay_buffer(self):
        trainer = _ColocateAsyncConfigProbe(_compose_config("trainer.use_v1=true", "trainer.v1.trainer_mode=sync"))
        assert type(trainer.replay_buffer) is ReplayBuffer


class TestDropIncompleteGroupsGate:
    @pytest.mark.parametrize("mode", ["colocate_async", "separate_async"])
    def test_rejected_outside_sync(self, mode):
        with pytest.raises(ValueError, match="drop_incomplete_groups is only supported with trainer_mode='sync'"):
            _ColocateAsyncConfigProbe(
                _compose_config(
                    "trainer.use_v1=true",
                    f"trainer.v1.trainer_mode={mode}",
                    "trainer.v1.sampler.drop_incomplete_groups=true",
                )
            )

    def test_sync_still_accepts(self):
        trainer = _ColocateAsyncConfigProbe(
            _compose_config(
                "trainer.use_v1=true",
                "trainer.v1.trainer_mode=sync",
                "trainer.v1.sampler.drop_incomplete_groups=true",
            )
        )
        assert type(trainer.replay_buffer) is ReplayBuffer


class TestExactRefill:
    def test_colocate_async_forces_single_prompt_generation_batches(self):
        trainer = object.__new__(_ColocateAsyncConfigProbe)
        trainer.trainer_mode = "colocate_async"
        trainer.config = OmegaConf.create(
            {
                "data": {"train_batch_size": 8, "gen_batch_size": 8},
                "trainer": {"v1": {"sampler": {"drop_incomplete_groups": False}}},
            }
        )
        assert trainer._generation_batch_size() == 1


class TestDistillationModeGate:
    def test_one_step_off_rejected_for_colocate_async(self):
        config = _colocate_config(
            "distillation.enabled=true",
            "distillation.teacher_models.teacher_model.model_path=/ckpt/teacher",
            "actor_rollout_ref.actor.diffusion_loss.loss_mode=distill_kl",
            "distillation.scheduler=one_step_off",
            "distillation.nnodes=1",
            "distillation.n_gpus_per_node=1",
        )
        with pytest.raises(ValueError, match="one_step_off"):
            validate_distillation_config(config)

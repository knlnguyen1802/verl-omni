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
"""CPU tests for diffusion V1 colocate_async mode config, validation, and trainer hooks."""

import os
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from transfer_queue import KVBatchMeta
from verl import DataProto
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync

import verl_omni
from verl_omni.trainer.diffusion.diffusion_trainer_utils import validate_distillation_config
from verl_omni.trainer.diffusion.v1.trainer_base import (
    PolicyGradientDiffusionTrainerV1,
    get_diffusion_trainer_cls,
)
from verl_omni.trainer.diffusion.v1.trainer_colocate_async import PolicyGradientDiffusionTrainerV1ColocateAsync
from verl_omni.trainer.diffusion.v1.trainer_separate_async import PolicyGradientDiffusionTrainerV1SeparateAsync
from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync
from verl_omni.workers.rollout.diffusion_llm_server import DiffusionWholeSampleRetryLLMServerClient

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
        assert get_diffusion_trainer_cls("colocate_async") is PolicyGradientDiffusionTrainerV1ColocateAsync
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


class _FakeCheckpointManager:
    def __init__(self):
        self.events = []

    def update_weights(self, global_steps=None):
        self.events.append(("update_weights", global_steps))

    def abort_replicas(self):
        self.events.append(("abort",))

    def sleep_replicas(self):
        self.events.append(("sleep",))

    def resume_generation_replicas(self):
        self.events.append(("resume",))


def _bare_colocate_trainer(num_warmup_batches=1):
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1ColocateAsync)
    trainer.config = OmegaConf.create(
        {"trainer": {"v1": {"colocate_async": {"num_warmup_batches": num_warmup_batches}}}}
    )
    trainer.checkpoint_manager = _FakeCheckpointManager()
    trainer.timing_raw = {}
    trainer.global_steps = 0
    return trainer


class TestColocateAsyncTrainer:
    def test_constructs_from_composed_config(self):
        trainer = PolicyGradientDiffusionTrainerV1ColocateAsync(_colocate_config())
        assert trainer.trainer_mode == "colocate_async"
        assert isinstance(trainer.replay_buffer, ReplayBufferAsync)

    def test_get_llm_client_returns_whole_sample_retry_client(self):
        trainer = _bare_colocate_trainer()
        seen = {}

        class FakeServerManager:
            def get_client(self, client_cls=None, **kwargs):
                seen["client_cls"] = client_cls
                return "client"

        trainer.llm_server_manager = FakeServerManager()
        assert trainer.get_llm_client() == "client"
        assert seen["client_cls"] is DiffusionWholeSampleRetryLLMServerClient

    def test_on_init_end_pushes_initial_weights(self):
        trainer = _bare_colocate_trainer()
        trainer.global_steps = 3
        trainer.on_init_end()
        assert trainer.checkpoint_manager.events == [("update_weights", 3)]

    @pytest.mark.parametrize("num_warmup_batches", [0, 1, 4])
    def test_on_train_begin_submits_configured_warmup_batches(self, num_warmup_batches):
        trainer = _bare_colocate_trainer(num_warmup_batches)
        calls = []
        trainer._add_batch_to_generate = lambda: calls.append(1)
        trainer.on_train_begin()
        assert len(calls) == num_warmup_batches

    def test_on_sample_end_aborts_before_sleeping(self):
        trainer = _bare_colocate_trainer()
        trainer.on_sample_end()
        assert trainer.checkpoint_manager.events == [("abort",), ("sleep",)]

    def test_on_step_end_updates_weights_before_resuming(self):
        trainer = _bare_colocate_trainer()
        trainer.global_steps = 7
        trainer.on_step_end()
        assert trainer.checkpoint_manager.events == [("update_weights", 7), ("resume",)]

    def test_full_hook_lifecycle_against_fakes(self):
        """init -> warmup -> abort/sleep -> update/resume, in order."""
        trainer = _bare_colocate_trainer(num_warmup_batches=2)
        feeds = []
        trainer._add_batch_to_generate = lambda: feeds.append("feed")

        trainer.on_init_end()
        trainer.on_train_begin()
        trainer.global_steps = 1
        trainer.on_sample_end()
        trainer.on_step_end()

        assert trainer.checkpoint_manager.events == [
            ("update_weights", 0),
            ("abort",),
            ("sleep",),
            ("update_weights", 1),
            ("resume",),
        ]
        assert feeds == ["feed", "feed"]


def _run_reward_path(trainer, monkeypatch):
    """Drive the colocated-reward section of ``_train_sampled_batch`` with stubs."""
    monkeypatch.setattr(
        "verl_omni.trainer.diffusion.v1.trainer_base.put_dataproto_fields_to_tq",
        lambda *args, **kwargs: None,
    )
    trainer._compute_reward_colocate = lambda data: DataProto.from_dict(
        tensors={"rm_scores": torch.zeros(len(data), 1)}
    )
    trainer._compute_old_log_prob = lambda data: DataProto.from_dict(
        tensors={"old_log_probs": torch.zeros(len(data), 2)}
    )
    trainer._compute_advantage = lambda data: data
    trainer._update_actor = lambda data: DataProto.from_single_dict(data={}, meta_info={"metrics": {}})

    data = DataProto.from_dict(tensors={"responses": torch.zeros(2, 3)})
    batch_meta = KVBatchMeta(partition_id="train", keys=["a", "b"], tags=[{}, {}])
    trainer._train_sampled_batch({}, trainer.timing_raw, batch_meta, data=data)


class TestColocatedRewardModeGate:
    def _bare_trainer(self, mode, ppo_mini_batch_size):
        trainer = object.__new__(PolicyGradientDiffusionTrainerV1ColocateAsync)
        trainer.trainer_mode = mode
        trainer.config = OmegaConf.create(
            {
                "algorithm": {},
                "actor_rollout_ref": {
                    "actor": {"ppo_mini_batch_size": ppo_mini_batch_size},
                    "rollout": {"n": 1},
                },
            }
        )
        trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=None)
        trainer.use_rm = True
        trainer._is_direct_preference = False
        trainer.use_reference_policy = False
        trainer.use_teacher_policy = False
        trainer.checkpoint_manager = _FakeCheckpointManager()
        trainer.timing_raw = {}
        trainer.global_steps = 1
        trainer.actor_rollout_wg = SimpleNamespace()  # no _query_dispatch_info -> dp_size 1
        return trainer

    def test_colocate_async_skips_mid_cycle_sleep_and_weight_sync(self, monkeypatch):
        trainer = self._bare_trainer("colocate_async", ppo_mini_batch_size=1)
        _run_reward_path(trainer, monkeypatch)
        assert trainer.checkpoint_manager.events == []

    def test_separate_async_keeps_mid_cycle_sleep_and_weight_sync(self, monkeypatch):
        trainer = self._bare_trainer("separate_async", ppo_mini_batch_size=2)
        _run_reward_path(trainer, monkeypatch)
        assert trainer.checkpoint_manager.events == [("sleep",), ("update_weights", 1)]

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
"""Separate-async v1 policy-gradient diffusion trainer.

Mirrors upstream ``verl.trainer.ppo.v1.trainer_separate_async.PPOTrainerSeparateAsync``
hook semantics, adapted to verl-omni diffusion rollout:

1. Trainer and rollout are separate. The colocated rollout replicas may switch
   to rollout mode when idle; a standalone rollout manager handles the bulk of
   generation traffic on dedicated GPUs.
2. Partial rollout is enabled: the trainer overproduces rollout work, the
   replay buffer samples complete prompt groups, and unfinished requests are
   aborted when switching to trainer mode. Aborted diffusion samples are
   retried as whole samples by ``DiffusionWholeSampleRetryLLMServerClient``.
3. Weight synchronization from actor to standalone rollout uses a non-naive
   checkpoint backend (nccl/nixl/mooncake/...); the colocated rollout still uses
   the naive in-place backend.

Diffusion-specific compute (reward, old/ref log-prob, Flow-GRPO advantage,
actor update, metrics, dumping) lives in ``PolicyGradientDiffusionTrainerV1``;
this subclass only defines the mode lifecycle hooks and the standalone rollout
wiring.
"""

import logging
import os
from enum import Enum

import ray
from omegaconf import DictConfig

from verl.checkpoint_engine import CheckpointEngineManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.workers.rollout.llm_server import LLMServerManager

from verl_omni.trainer.diffusion.v1.trainer_base import (
    PolicyGradientDiffusionTrainerV1,
    register_diffusion_trainer,
)
from verl_omni.workers.checkpoint_engine import OmniCheckpointEngineManager
from verl_omni.workers.rollout.diffusion_llm_server import DiffusionWholeSampleRetryLLMServerClient

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class HybridEngineMode(Enum):
    TRAINER = "trainer"
    ROLLOUT = "rollout"


@register_diffusion_trainer("separate_async")
class PolicyGradientDiffusionTrainerV1SeparateAsync(PolicyGradientDiffusionTrainerV1):
    """Asynchronous policy-gradient diffusion trainer (v1) with separate rollout.

    Hook behavior:

    - ``on_init_end``: update weights on both the standalone and colocated
      checkpoint managers.
    - ``on_train_begin``: enqueue ``num_warmup_batches`` prompt batches.
    - ``on_validate_begin``: switch to rollout mode if currently training.
    - ``on_sample_begin``: switch to rollout mode if training and the switch
      strategy says so.
    - ``on_sample_end``: switch to trainer mode (abort + sleep colocated
      replicas, remove them from the standalone load balancer).
    - ``on_step_end``: keep the colocated rollout in sync with the actor every
      step (mirroring sync mode), and every ``parameter_sync_step`` also push
      actor weights into the standalone rollout replicas.
    """

    def __init__(self, config: DictConfig):
        train_batch_size = config.data.train_batch_size
        ppo_mini_batch_size = config.actor_rollout_ref.actor.ppo_mini_batch_size
        assert train_batch_size == ppo_mini_batch_size, (
            f"train_batch_size must equal ppo_mini_batch_size in separate async training, "
            f"got {train_batch_size} and {ppo_mini_batch_size}"
        )
        assert config.actor_rollout_ref.rollout.nnodes > 0, (
            "actor_rollout_ref.rollout.nnodes must be > 0 in separate async training"
        )
        assert config.actor_rollout_ref.rollout.n_gpus_per_node > 0, (
            "actor_rollout_ref.rollout.n_gpus_per_node must be > 0 in separate async training"
        )
        assert config.actor_rollout_ref.rollout.checkpoint_engine.backend != "naive", (
            "please use nccl/nixl/mooncake/... backend for separate async training"
        )

        super().__init__(config)

        # Do NOT force ``bypass_mode=True`` here. Sync mode (the convergent
        # reference) leaves ``bypass_mode`` at its config default (``False``),
        # which recomputes ``old_log_probs`` with the current actor for a 3-policy
        # PPO ratio (π_rollout, π_old, π_θ). Forcing ``bypass_mode=True`` would
        # instead set ``old_log_probs := rollout_log_probs`` (2-policy) and
        # silently change the algorithm regardless of the user's config. We
        # respect the config so separate async can behave exactly like sync
        # mode; users who want the 2-policy bypass can still set it explicitly.

    # ------------------------------ setup ------------------------------

    def _init_online_rollout_stack(self, actor_rollout_resource_pool):
        """Build colocated rollout stack (naive ckpt) + standalone rollout stack.

        Overridden so the colocated checkpoint manager uses the naive in-place
        backend (actor and colocated rollout share GPUs), while a second
        standalone ``LLMServerManager`` / ``CheckpointEngineManager`` pair uses
        the configured non-naive backend for trainer -> standalone weight sync.
        """
        from verl.trainer.ppo.utils import Role

        from verl_omni.reward_loop import OmniRewardLoopManager

        resource_pool = (
            self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        )
        self.reward_loop_manager = OmniRewardLoopManager(config=self.config, rm_resource_pool=resource_pool)
        self.enable_agent_reward_loop = (
            not self.use_rm or self.config.reward.reward_model.enable_resource_pool
        )

        # Colocated rollout replicas (share GPUs with the actor).
        self.llm_server_manager = LLMServerManager.create(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
        )
        colocated_ckpt_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        colocated_ckpt_config.backend = "naive"
        self.checkpoint_manager = CheckpointEngineManager(
            config=colocated_ckpt_config,
            actor_wg=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )

    def _setup(self):
        super()._setup()

        # Standalone rollout replicas on dedicated GPUs. start_rank skips the
        # colocated replica ranks to avoid Ray named-actor collisions.
        hybrid_num_replicas = len(self.llm_server_manager.rollout_replicas)
        self.standalone_server_manager: LLMServerManager = LLMServerManager.create(
            config=self.config, start_rank=hybrid_num_replicas
        )

        # Non-naive checkpoint engine for trainer -> standalone rollout weight sync.
        # Uses the verl-omni subclass so the actor's LoRA ``peft_config`` is
        # delivered out-of-band to the standalone rollout workers (verl's
        # ``CheckpointEngineWorker.update_weights`` does not forward it as a
        # kwarg to ``update_weights_from_ipc``).
        standalone_ckpt_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.standalone_checkpoint_manager = OmniCheckpointEngineManager(
            config=standalone_ckpt_config,
            actor_wg=self.actor_rollout_wg,
            replicas=self.standalone_server_manager.get_replicas(),
        )

        # Hybrid engine starts in trainer mode: colocated replicas stay slept
        # (freeing their GPU memory for the actor update) and the standalone
        # rollout replicas serve all generation traffic. The colocated replicas
        # only join the standalone load balancer when ``switch_to_rollout`` is
        # called (currently gated by ``should_switch_to_rollout``).
        self.current_mode = HybridEngineMode.TRAINER
        # Track whether colocated replicas have been slept via switch_to_trainer.
        # wake_up can only be called on replicas that were previously slept;
        # calling it on never-slept replicas triggers a CUDA cumem error.
        self._colocated_slept = False

    # ------------------------------ client handles ------------------------------

    def get_llm_client(self):
        """Get the diffusion whole-sample-retry client backed by the standalone rollout."""
        return self.standalone_server_manager.get_client(
            client_cls=DiffusionWholeSampleRetryLLMServerClient
        )

    # ------------------------------ lifecycle hooks ------------------------------

    def on_init_end(self):
        # Push actor weights into both standalone and colocated rollout replicas.
        logger.warning(
            "LORA_SYNC_PROOF separate_async on_init_end: sending weights to BOTH "
            "standalone and colocated checkpoint managers (global_steps=%s)",
            self.global_steps,
        )
        self._log_actor_lora_checksum(tag="before-standalone-init")
        self.standalone_checkpoint_manager.update_weights(self.global_steps)
        self._log_actor_lora_checksum(tag="before-colocated-init")
        self.checkpoint_manager.update_weights(self.global_steps)

    def on_train_begin(self):
        num_warmup_batches = self.config.trainer.v1.separate_async.num_warmup_batches
        for _ in range(num_warmup_batches):
            self._add_batch_to_generate()
        logger.info(f"Added {num_warmup_batches} warmup batches to the agent loop manager")

    def on_validate_begin(self):
        if self.current_mode == HybridEngineMode.TRAINER and self._colocated_slept:
            # Only wake up colocated replicas if they were actually slept (i.e.
            # switch_to_trainer ran at least once during training). Calling
            # wake_up on never-slept replicas triggers a CUDA error in the
            # cumem allocator ("invalid argument" at create_and_map) because
            # there are no offloaded handles to restore.
            #
            # When should_switch_to_rollout() returns False (the current
            # default), colocated replicas are never slept during training, so
            # validation should just use the standalone replicas — which are
            # already serving and are the ones get_llm_client() routes to.
            logger.info("Switching hybrid engine to rollout mode for validation")
            self.switch_to_rollout()
        else:
            logger.info(
                "Skipping colocated rollout switch for validation "
                "(colocated replicas never slept; using standalone replicas only)"
            )

    def on_validate_end(self):
        if self.current_mode == HybridEngineMode.ROLLOUT:
            logger.info("Switching hybrid engine back to trainer mode after validation")
            self.switch_to_trainer()

    def on_sample_begin(self):
        if self.current_mode == HybridEngineMode.TRAINER and self.should_switch_to_rollout():
            logger.info("Switching hybrid engine to rollout mode for generation")
            self.switch_to_rollout()

    def on_sample_end(self):
        if self.current_mode == HybridEngineMode.ROLLOUT:
            logger.info("Switching hybrid engine to trainer mode for training")
            self.switch_to_trainer()

    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw, color="red"):
            # Mirror sync mode exactly: keep the colocated rollout replicas in
            # sync with the freshly-trained actor every step via the naive
            # in-place backend. This is the same call sync mode makes in its
            # ``on_step_end``. The colocated replicas stay slept (they are not
            # used for generation while ``should_switch_to_rollout`` returns
            # False), but keeping their weights fresh preserves sync-mode
            # parity and ensures they are ready if a switch is ever triggered.
            logger.warning(
                "LORA_SYNC_PROOF separate_async on_step_end: sending weights to "
                "COLOCATED (non-standalone) checkpoint manager (global_steps=%s)",
                self.global_steps,
            )
            self._log_actor_lora_checksum(tag="before-colocated-step")
            self.checkpoint_manager.update_weights(self.global_steps)
            if self.global_steps % self.config.trainer.v1.separate_async.parameter_sync_step == 0:
                # Push freshly-trained actor weights into standalone rollout replicas.
                # The standalone manager pulls from the same actor worker group as the
                # colocated manager, so both receive identical weights each sync.
                logger.warning(
                    "LORA_SYNC_PROOF separate_async on_step_end: sending weights to "
                    "STANDALONE checkpoint manager (global_steps=%s)",
                    self.global_steps,
                )
                self._log_actor_lora_checksum(tag="before-standalone-step")
                self.standalone_checkpoint_manager.update_weights(self.global_steps)

    # ------------------------------ diagnostics ------------------------------

    def _log_actor_lora_checksum(self, tag: str):
        """Fetch and log a checksum of the actor's LoRA adapter weights.

        Both the colocated and standalone checkpoint managers pull from the same
        actor worker group, so the checksum logged before each call must match.
        Mismatched checksums across the two calls would mean the actor weights
        changed between the two syncs (which should not happen within a single
        ``on_step_end`` / ``on_init_end``). Returns silently when the actor is not
        running a LoRA adapter.
        """
        try:
            results = self.actor_rollout_wg.get_lora_weight_checksum()
        except Exception as e:
            logger.warning("LORA_SYNC_PROOF [%s] get_lora_weight_checksum failed: %s", tag, e)
            return
        checksums = [r for r in (results or []) if r is not None]
        if not checksums:
            logger.warning("LORA_SYNC_PROOF [%s] actor reports no LoRA adapter (full-weight sync)", tag)
            return
        # All actor ranks share the same adapter metadata; the per-rank sum may
        # differ under FSDP sharding, so log every rank's checksum for inspection.
        for rank, ck in enumerate(checksums):
            logger.warning(
                "LORA_SYNC_PROOF [%s] rank=%s num_lora_tensors=%s sum=%.6f "
                "first_lora_a=%s first_lora_b=%s last=%s",
                tag,
                rank,
                ck.get("num_lora_tensors"),
                ck.get("sum", 0.0),
                ck.get("first_lora_a"),
                ck.get("first_lora_b"),
                ck.get("last_name"),
            )

    # ------------------------------ mode switching ------------------------------

    def switch_to_rollout(self):
        # Wake colocated replicas (their weights were offloaded by sleep),
        # sync fresh actor weights, and let them serve generation again.
        self.checkpoint_manager.wake_up_replicas()
        self._colocated_slept = False
        self.checkpoint_manager.update_weights(self.global_steps)
        self.checkpoint_manager.resume_generation_replicas()
        self.add_replicas_to_balancer()
        self.current_mode = HybridEngineMode.ROLLOUT

    def switch_to_trainer(self):
        # Remove colocated replicas from the balancer and free their memory for training.
        self.remove_replicas_from_balancer()
        self.checkpoint_manager.abort_replicas()
        self.checkpoint_manager.sleep_replicas()
        self._colocated_slept = True
        self.current_mode = HybridEngineMode.TRAINER

    def add_replicas_to_balancer(self):
        global_load_balancer = self.standalone_server_manager.global_load_balancer
        servers = dict(
            zip(self.llm_server_manager.server_addresses, self.llm_server_manager.server_handles, strict=True)
        )
        ray.get(global_load_balancer.add_servers.remote(servers))

    def remove_replicas_from_balancer(self):
        global_load_balancer = self.standalone_server_manager.global_load_balancer
        ray.get(global_load_balancer.remove_servers.remote(self.llm_server_manager.server_addresses))

    def should_switch_to_rollout(self):
        # TODO: implement a switch strategy based on replay buffer state / switch overhead.
        return False

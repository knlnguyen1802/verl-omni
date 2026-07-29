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
"""verl-omni extensions to verl's ``CheckpointEngineManager``.

The standalone (non-naive) weight-sync path in verl goes through
``CheckpointEngineWorker.update_weights`` -> ``ServerAdapter.update_weights``
-> ``update_weights_from_ipc``, and unlike the colocated path it does not
forward ``peft_config``/``base_sync_done`` kwargs. When the actor trains a
(non-merged) LoRA adapter it ships only the adapter deltas, so the rollout
must apply them via ``add_lora``; without ``peft_config`` the rollout falls
back to ``load_weights`` and raises ``KeyError`` on ``*.lora_A.weight``.

``OmniCheckpointEngineManager`` closes that gap by delivering the actor's
``peft_config`` out-of-band: it fetches it (collective-free) from the actor
worker group and stashes it on each rollout worker extension via
``collective_rpc("set_pending_lora_peft_config")`` before the NCCL broadcast.
``update_weights_from_ipc`` consumes the stash when its ``peft_config`` kwarg
is absent.
"""
import logging
import os

import ray
from verl.checkpoint_engine import CheckpointEngineManager
from verl.utils.ray_utils import auto_await

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class OmniCheckpointEngineManager(CheckpointEngineManager):
    """``CheckpointEngineManager`` subclass that forwards the actor's LoRA
    ``peft_config`` to standalone rollout replicas for separate-async NCCL
    weight sync.

    Drop-in replacement for ``CheckpointEngineManager``: it only overrides
    ``update_weights`` to (a) fetch ``peft_config`` from the actor and stash
    it on the rollout workers, then (b) delegate to the parent. The colocated
    naive path is unchanged (no stash is written when ``backend == "naive"``).
    """

    @auto_await
    async def update_weights(self, global_steps: int = None):
        """Fetch the actor's LoRA ``peft_config`` and stash it on the rollout
        workers before delegating to the parent ``update_weights``.

        For non-naive backends only; the naive (colocated) path passes
        ``peft_config`` directly through kwargs and needs no out-of-band
        delivery.
        """
        if self.backend != "naive":
            peft_config = self._fetch_actor_lora_peft_config()
            # Keep the parent manager's per-run cache aligned with the current
            # actor state. Without this, a transient startup miss can pin the
            # parent path to full-weight mode for the whole run.
            self._lora_peft_config = peft_config
            logger.warning(
                "LORA_SYNC_PROOF manager route backend=%s global_steps=%s mode=%s",
                self.backend,
                global_steps,
                "adapter_only" if peft_config is not None else "full_weight",
            )
            await self._push_lora_peft_config_to_replicas(peft_config)
        await super().update_weights(global_steps=global_steps)

    async def _push_lora_peft_config_to_replicas(self, peft_config: dict | None) -> None:
        """Fetch ``peft_config`` from the actor (collective-free) and stash it
        on every standalone rollout replica's worker extension.

        No-op when the actor is not training a LoRA adapter (``peft_config`` is
        ``None``), in which case the rollout takes the full-weight path as
        before.
        """
        # Stash on each replica's worker extension via collective_rpc. The
        # method name must match ``set_pending_lora_peft_config`` on
        # ``vLLMOmniColocateWorkerExtension`` (inherited by the NPU variant).
        futures = [
            replica.server_handle.collective_rpc.remote(
                "set_pending_lora_peft_config",
                kwargs={"peft_config": peft_config},
            )
            for replica in self.replicas
            if replica.server_handle is not None
        ]
        if futures:
            ray.get(futures)
        if peft_config is None:
            logger.debug("cleared pending LoRA peft_config on %d standalone rollout replica(s)", len(futures))
            logger.warning(
                "LORA_SYNC_PROOF manager push replicas=%d action=clear_pending_lora_config",
                len(futures),
            )
        else:
            logger.debug("pushed LoRA peft_config to %d standalone rollout replica(s)", len(futures))
            logger.warning(
                "LORA_SYNC_PROOF manager push replicas=%d action=set_pending_lora_config rank=%s target_modules=%s",
                len(futures),
                peft_config.get("r", "unknown"),
                peft_config.get("target_modules", "unknown"),
            )

    def _fetch_actor_lora_peft_config(self):
        """Return the actor's LoRA ``peft_config`` dict, or ``None``.

        ``get_lora_peft_config`` is registered ``ONE_TO_ALL`` and reads the
        peft_model metadata without summoning FSDP params, so every actor rank
        returns the same dict (non-actor ranks return ``None``). The worker-group
        proxy is blocking and has already called ``ray.get`` before returning,
        so its result must not be resolved a second time. We take the first
        non-``None`` result. Any failure is logged and treated as "not a LoRA
        run" so the rollout falls back to the full-weight path.
        """
        try:
            results = self.actor_wg.get_lora_peft_config()
        except Exception as e:  # noqa: BLE001 - tolerate backend/registration differences
            logger.warning("get_lora_peft_config failed (%s); assuming non-LoRA run", e)
            return None
        for result in results or []:
            if result is not None:
                return result
        return None

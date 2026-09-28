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
import ray
from verl.checkpoint_engine import CheckpointEngineManager, CheckpointEngineRegistry
from verl.utils.ray_utils import auto_await

# verl's CheckpointEngineWorker gates the "delta_sharded" backend to sglang
# rollouts (its own consumer rides the sglang custom weight loader / the vLLM
# weight-transfer engine of newer pins). verl-omni's vllm_omni rollout consumes
# the same DeltaFlush wire in its own worker extension, so the identical engine
# class is registered under a verl-omni name: verl's worker then constructs it
# through CheckpointEngineRegistry untouched -- no subclass, no gate widening.
# Importing this module performs the registration (see verl_omni/__init__.py).
try:
    from verl.checkpoint_engine import DeltaShardedCheckpointEngine

    if DeltaShardedCheckpointEngine is not None:
        CheckpointEngineRegistry.register("omni_delta_sharded")(DeltaShardedCheckpointEngine)
except ImportError:  # verl records the failure; Registry.get reports it on use
    pass


class OmniCheckpointEngineManager(CheckpointEngineManager):
    """``CheckpointEngineManager`` subclass that forwards the actor's LoRA
    ``peft_config`` to standalone rollout replicas for separate-async NCCL
    weight sync.
    """

    @auto_await
    async def update_weights(self, global_steps: int = None):
        """Fetch the actor's LoRA ``peft_config`` and stash it on the rollout
        workers before delegating to the parent ``update_weights``.
        """
        if self.backend != "naive":
            peft_config = self._fetch_actor_lora_peft_config()
            self._lora_peft_config = peft_config
            await self._push_lora_peft_config_to_replicas(peft_config)
        await super().update_weights(global_steps=global_steps)

    async def _push_lora_peft_config_to_replicas(self, peft_config: dict | None) -> None:
        """Fetch ``peft_config`` from the actor (collective-free) and stash it
        on every standalone rollout replica's worker extension.

        """
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

    def _fetch_actor_lora_peft_config(self):
        """Return the actor's LoRA ``peft_config`` dict, or ``None``."""
        if not hasattr(self.actor_wg, "get_lora_peft_config"):
            return None
        results = self.actor_wg.get_lora_peft_config()
        for result in results or []:
            if result is not None:
                return result
        return None

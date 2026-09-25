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
from typing import Callable, Optional

import torch
from omegaconf import DictConfig
from torch.distributed.tensor import DTensor
from verl.experimental.separation.engine_workers import DetachActorWorker
from verl.workers.config import DistillationConfig

from verl_omni.workers.engine_workers import ActorRolloutRefWorker


def _own_snapshot_storage(save_handler: Callable) -> Callable:
    """Wrap an fsdp2 ``save_to_cpu`` handler so every snapshot owns its storage.

    Pinned verl's ``fsdp2_sharded_save_to_cpu`` moves shards with
    ``detach().cpu()``: a real copy for GPU-resident parameters, but a
    storage-sharing view when ``param_offload=true`` leaves them CPU-resident.
    The decoupled-PPO dance then does ``restore_model_from_cpu(0)`` — an
    in-place ``copy_`` into the live storage — which silently rewrites the
    aliased slot and resets every local update to the cycle-start policy
    (verl-project/verl-omni#645). Cloning only slots that share storage with
    the live parameter keeps the healthy GPU-resident path copy-free and
    becomes a no-op once verl fixes the helper.
    """

    def save(model: torch.nn.Module):
        cpu_sharded_state, global_spec = save_handler(model)
        live_params = dict(model.named_parameters())
        for name, (shard, spec) in cpu_sharded_state.items():
            param = live_params.get(name)
            if param is None:
                continue
            local = param._local_tensor if isinstance(param, DTensor) else param.data
            if shard.untyped_storage().data_ptr() == local.untyped_storage().data_ptr():
                cpu_sharded_state[name] = (shard.clone(), spec)
        return cpu_sharded_state, global_spec

    return save


class OmniDetachActorWorker(ActorRolloutRefWorker, DetachActorWorker):
    """``DetachActorWorker`` routed through verl-omni's ``ActorRolloutRefWorker``.

    The omni worker comes first in the MRO so its LoRA-aware weight sync
    (adapter-only send, ``get_lora_peft_config``) wins over the upstream
    methods; ``DetachActorWorker`` contributes the CPU save/restore used by
    decoupled PPO.
    """

    def __init__(
        self, config: DictConfig, role: str, distillation_config: Optional[DistillationConfig] = None, **kwargs
    ):
        ActorRolloutRefWorker.__init__(self, config, role, distillation_config=distillation_config, **kwargs)
        self._strategy_handlers = None

    def _get_strategy_handlers(self):
        if self._strategy_handlers is None:
            save_handler, restore_handler = super()._get_strategy_handlers()
            # fsdp2 helpers are the ones with the view-aliasing hazard (#645);
            # fsdp1's already copies, and the megatron handlers copy by name.
            if self.config.actor.strategy in ("fsdp2", "veomni"):
                save_handler = _own_snapshot_storage(save_handler)
            self._strategy_handlers = (save_handler, restore_handler)
        return self._strategy_handlers

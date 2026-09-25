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
"""Snapshots taken by ``OmniDetachActorWorker`` must own their storage.

Pinned verl's ``fsdp2_sharded_save_to_cpu`` moves shards with
``detach().cpu()``, which is a storage-sharing view for CPU-resident
parameters (the steady state under ``param_offload=true``). The decoupled-PPO
dance then calls ``restore_model_from_cpu(0)`` — an in-place ``copy_`` into
the live storage — which silently rewrites the aliased slot, so every local
update resets to the cycle-start policy and training freezes with healthy-
looking metrics (verl-project/verl-omni#645).

These tests replay that dance in-process on a CPU DTensor model against the
real verl helpers, and pin the ``_own_snapshot_storage`` wrap that
``OmniDetachActorWorker`` installs for the fsdp2/veomni strategies.
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from verl.utils.fsdp_utils import fsdp2_sharded_load_from_cpu, fsdp2_sharded_save_to_cpu

from verl_omni.workers.omni_engine_workers import OmniDetachActorWorker, _own_snapshot_storage


@pytest.fixture()
def cpu_dist_group(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def _cpu_dtensor_model() -> nn.Module:
    """Tiny module shaped like an offloaded FSDP2 shard: a DTensor weight
    (both save branches) plus a plain bias, all CPU-resident."""
    model = nn.Linear(8, 8)
    mesh = init_device_mesh("cpu", (1,))
    model.weight = nn.Parameter(DTensor.from_local(model.weight.data, mesh, [Shard(0)]))
    return model


def _fill(model: nn.Module, value: float) -> None:
    with torch.no_grad():
        model.weight._local_tensor.fill_(value)
        model.bias.data.fill_(value)


def _shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def _live_shards(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: (param._local_tensor if isinstance(param, DTensor) else param.data)
        for name, param in model.named_parameters()
    }


def test_snapshot_owns_storage(cpu_dist_group):
    model = _cpu_dtensor_model()
    save = _own_snapshot_storage(fsdp2_sharded_save_to_cpu)

    snapshot, _ = save(model)

    for name, (shard, _) in snapshot.items():
        assert not _shares_storage(shard, _live_shards(model)[name])
        assert torch.equal(shard, _live_shards(model)[name].cpu())


def test_decoupled_ppo_dance_keeps_slot_n(cpu_dist_group):
    # The trainer_separate_async dance at local_trigger_step >= 1:
    # save(1) -> restore(0) -> ... -> restore(1). restore(0) copies the
    # cycle-start weights into the live storage in place; an aliased slot 1
    # would be rewritten through the alias and restore(1) would install the
    # old policy instead of the trained one.
    model = _cpu_dtensor_model()
    save = _own_snapshot_storage(fsdp2_sharded_save_to_cpu)

    _fill(model, 0.0)
    slot0, spec0 = save(model)
    _fill(model, 1.0)
    slot1, spec1 = save(model)

    fsdp2_sharded_load_from_cpu(model, slot0, spec0)
    assert torch.equal(model.weight._local_tensor, torch.full_like(model.weight._local_tensor, 0.0))

    fsdp2_sharded_load_from_cpu(model, slot1, spec1)
    assert torch.equal(model.weight._local_tensor, torch.full_like(model.weight._local_tensor, 1.0))


def test_wrapper_passthrough_when_storage_already_owned(cpu_dist_group):
    # Healthy path (GPU-resident params): the verl helper returns real copies,
    # and the wrap must not clone them again.
    model = _cpu_dtensor_model()
    owned = {name: (shard.detach().clone(), None) for name, shard in _live_shards(model).items()}
    fake_copy = lambda m: (owned, next(p._spec for p in m.parameters() if isinstance(p, DTensor)))

    snapshot, _ = _own_snapshot_storage(fake_copy)(model)

    for name, (shard, _) in owned.items():
        assert snapshot[name][0] is shard


@pytest.mark.parametrize("strategy", ["fsdp2", "veomni"])
def test_get_strategy_handlers_wraps_fsdp2_save(cpu_dist_group, strategy):
    worker = object.__new__(OmniDetachActorWorker)
    worker.config = OmegaConf.create({"actor": {"strategy": strategy}})
    worker._strategy_handlers = None

    save_handler, restore_handler = worker._get_strategy_handlers()

    assert save_handler is not fsdp2_sharded_save_to_cpu
    assert restore_handler is fsdp2_sharded_load_from_cpu


def test_get_strategy_handlers_leaves_fsdp1_unwrapped(cpu_dist_group):
    from verl.utils.fsdp_utils import fsdp1_sharded_load_from_cpu, fsdp1_sharded_save_to_cpu

    worker = object.__new__(OmniDetachActorWorker)
    worker.config = OmegaConf.create({"actor": {"strategy": "fsdp"}})
    worker._strategy_handlers = None

    save_handler, restore_handler = worker._get_strategy_handlers()

    assert save_handler is fsdp1_sharded_save_to_cpu
    assert restore_handler is fsdp1_sharded_load_from_cpu

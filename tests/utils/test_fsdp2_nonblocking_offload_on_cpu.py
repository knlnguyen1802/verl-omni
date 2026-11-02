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
"""Run real FSDP2 repadding against a deferred D2H copy on a two-rank CPU mesh."""

from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from verl_omni.utils.fsdp_utils import _shard_repadded_on_cpu, offload_fsdp_model_to_cpu

WORLD_SIZE = 2
UNFINISHED_COPY_VALUE = -999.0


def _build_model(mesh):
    model = torch.nn.Module()
    # Dim 0 of 7 chunks into 4 + 3 rows, so only rank 1 holds a short shard that FSDP2 repads.
    model.uneven = torch.nn.Parameter(torch.arange(7 * 3, dtype=torch.float32).reshape(7, 3))
    model.even = torch.nn.Parameter(torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3))
    rank = dist.get_rank()
    expected = {name: param.detach().clone().chunk(WORLD_SIZE)[rank] for name, param in model.named_parameters()}
    fully_shard(model, mesh=mesh)
    return model, expected


@contextmanager
def _deferred_copies():
    """Leave non-blocking DTensor copies unwritten until the context exits, like an unfinished D2H."""
    original_to = torch.Tensor.to
    pending = []
    non_blocking_by_shape = {}

    def deferred_to(self, *args, **kwargs):
        if not isinstance(self, DTensor):
            return original_to(self, *args, **kwargs)
        # Module.to passes (device, dtype, non_blocking) positionally.
        non_blocking = kwargs.get("non_blocking", len(args) == 3 and args[2])
        non_blocking_by_shape[tuple(self.shape)] = non_blocking
        if not non_blocking:
            return original_to(self, *args, **kwargs)
        source = self.to_local()
        destination = torch.full_like(source, UNFINISHED_COPY_VALUE)
        pending.append(lambda: destination.copy_(source))
        return DTensor(destination, self._spec, requires_grad=self.requires_grad)

    torch.Tensor.to = deferred_to
    try:
        yield non_blocking_by_shape
    finally:
        torch.Tensor.to = original_to
    for copy in pending:
        copy()


def _local_values(model):
    return {name: param.to_local().detach().clone() for name, param in model.named_parameters()}


def _assert_exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _worker(rank, init_method):
    dist.init_process_group(backend="gloo", init_method=init_method, rank=rank, world_size=WORLD_SIZE)
    try:
        mesh = init_device_mesh("cpu", (WORLD_SIZE,))

        # Negative control: the same deferred copy under a plain non-blocking move corrupts the short shard.
        model, expected = _build_model(mesh)
        with _deferred_copies():
            model.to("cpu", non_blocking=True)
        corrupted = _local_values(model)
        _assert_exact(corrupted["even"], expected["even"])
        if rank == 1:
            assert (corrupted["uneven"] == UNFINISHED_COPY_VALUE).all(), "the control must expose the repad race"
        else:
            _assert_exact(corrupted["uneven"], expected["uneven"])

        model, expected = _build_model(mesh)
        with _deferred_copies() as non_blocking_by_shape:
            offload_fsdp_model_to_cpu(model, empty_cache=False)
        # Rank 0 holds the padded size, so only rank 1 gives up the asynchronous path.
        assert non_blocking_by_shape == {(7, 3): rank == 0, (8, 3): True}
        for name, values in _local_values(model).items():
            _assert_exact(values, expected[name])

        # The predicate must agree with the size FSDP2 itself pads to, on every parameter.
        for fsdp_param in model._get_fsdp_state()._fsdp_param_groups[0].fsdp_params:
            param = fsdp_param.sharded_param
            assert _shard_repadded_on_cpu(param) == (param.to_local().size() != fsdp_param.padded_sharded_param_size)
    finally:
        dist.destroy_process_group()


def test_offload_blocks_only_on_shards_fsdp2_repads(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=WORLD_SIZE, join=True)


def test_plain_tensors_are_not_repadded():
    assert not _shard_repadded_on_cpu(torch.zeros(3))

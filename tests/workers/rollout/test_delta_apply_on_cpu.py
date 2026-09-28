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
"""CPU checks for the rollout side of ``omni_delta_sharded`` weight sync.

The wire is verl's ``ServerAdapter._update_delta_weights`` protocol: one
``update_verl_delta_weights`` RPC per flush, streaming that flush's sentinel
tensors. At this verl pin the send loop lives in verl-omni's
``VLLMOmniServerAdapter`` (the pin predates verl#7227); these tests drive both
ends on CPU. The diffusers engine's shard export (seed + steady deltas) is
encoded into wire flushes exactly as the delta checkpoint engine encodes them
(``verl.checkpoint_engine.delta_sync.encode``), then sent through the adapter's
loop and applied through the extension and verl's delta loader into a toy
rollout model whose ``load_weights`` lands on ``param.copy_`` like vllm's
loaders. Only the ZMQ/NCCL transport is faked.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from verl.checkpoint_engine.delta_sync.encode import DeltaParam, checksum
from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

import verl_omni.workers.rollout.base  # noqa: F401  -- rollout class registration side effect
from verl_omni.workers.config.diffusion import DiffusionModelConfig
from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine
from verl_omni.workers.rollout.vllm_rollout.utils import vLLMOmniColocateWorkerExtension

SPEC_NAME = "__delta_spec__"
POSITIONS_NAME = "__positions__"
VALUES_NAME = "__values__"

# verl's ServerAdapter._update_delta_weights RPC protocol: the method names it
# collective_rpcs on the rollout worker, in call order.
DELTA_PROTOCOL_METHODS = (
    "init_weight_transfer_engine",
    "start_weight_update",
    "update_verl_delta_weights",
    "finish_weight_update",
)

# ---------------------------------------------------------------------------
# Trainer side: real diffusers engine export over a toy DiT
# ---------------------------------------------------------------------------


class _ToyDiT(torch.nn.Module):
    _checkpoint_conversion_mapping = {"^transformer_blocks": "blocks"}

    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=False) for _ in range(2)])


def _make_engine(module) -> PPODiffusersFSDPEngine:
    engine = object.__new__(PPODiffusersFSDPEngine)
    engine.module = module
    engine._is_offload_param = False
    engine._uses_fsdp2_cpu_offload_policy = False
    model_config = object.__new__(DiffusionModelConfig)
    object.__setattr__(model_config, "lora", {})
    engine.model_config = model_config
    return engine


def _patch_engine_helpers(monkeypatch):
    import verl_omni.workers.engine.fsdp.diffusers_impl as diffusers_impl

    monkeypatch.setattr(diffusers_impl, "log_gpu_memory_usage", MagicMock())
    monkeypatch.setattr(diffusers_impl, "load_fsdp_model_to_gpu", MagicMock())
    monkeypatch.setattr(diffusers_impl, "offload_fsdp_model_to_cpu", MagicMock())
    monkeypatch.setattr(diffusers_impl, "get_device_id", lambda: torch.device("cpu"))


def _seed_export(engine) -> list:
    """The delta engine's seed: full export, every floating tensor cast to bf16."""
    full, _ = engine.get_per_tensor_param()
    return [(name, t.to(torch.bfloat16) if t.is_floating_point() else t) for name, t in full]


# ---------------------------------------------------------------------------
# Rollout side: toy model whose load_weights lands on copy_ like vllm's loaders
# ---------------------------------------------------------------------------


class _ToyRolloutModel:
    def __init__(self, state: dict):
        self.state = state

    def load_weights(self, weights):
        for name, tensor in weights:
            self.state[name].copy_(tensor)


# ---------------------------------------------------------------------------
# Wire encoding, mirroring the delta engine's flush assembly (world-1: gather is identity)
# ---------------------------------------------------------------------------


def _spec_tensor(spec: dict) -> torch.Tensor:
    return torch.frombuffer(bytearray(json.dumps(spec).encode()), dtype=torch.uint8)


def _encode_dense_flush(named_params: list) -> list:
    params, val_pieces = [], []
    val_off = 0
    for name, tensor in named_params:
        flat = tensor.reshape(-1)
        n = flat.numel()
        params.append(
            DeltaParam(
                name=name,
                dtype=str(flat.dtype).replace("torch.", ""),
                shape=list(tensor.shape),
                pos_start=0,
                pos_end=0,
                pos_width=4,
                val_start=val_off,
                val_end=val_off + n,
            )
        )
        val_pieces.append(flat)
        val_off += n
    values = torch.cat(val_pieces)
    empty_pos = torch.empty(0, dtype=torch.uint8)
    spec = {
        "encoding": "dense",
        "verify": False,
        "is_last": True,
        "params": [vars(p) for p in params],
        "checksum": checksum(empty_pos, values),
    }
    return [(SPEC_NAME, _spec_tensor(spec)), (VALUES_NAME, values)]


def _encode_indices_flush(deltas) -> list:
    params, idx_pieces, val_pieces = [], [], []
    pos_off = val_off = 0
    for slots, dtype_str, counts, hf_idx, hf_val, _pg in deltas:
        off = 0
        for (name, shape), count in zip(slots, counts.tolist(), strict=True):
            idx = hf_idx[off : off + count]
            val = hf_val[off : off + count]
            off += count
            if count == 0:
                continue
            idx_pieces.append(idx.to(torch.int32))
            val_pieces.append(val)
            params.append(
                DeltaParam(
                    name=name,
                    dtype=dtype_str,
                    shape=list(shape),
                    pos_start=pos_off,
                    pos_end=pos_off + count * 4,
                    pos_width=4,
                    val_start=val_off,
                    val_end=val_off + count,
                )
            )
            pos_off += count * 4
            val_off += count
    values = torch.cat(val_pieces) if val_pieces else torch.empty(0, dtype=torch.bfloat16)
    positions = torch.cat(idx_pieces).view(torch.uint8) if idx_pieces else torch.empty(0, dtype=torch.uint8)
    spec = {"encoding": "indices", "params": [vars(p) for p in params], "checksum": checksum(positions, values)}
    return [(SPEC_NAME, _spec_tensor(spec)), (POSITIONS_NAME, positions), (VALUES_NAME, values)]


def _run_delta_flush(worker, buckets: list, monkeypatch):
    """Drive one update_verl_delta_weights RPC with a fake bucketed receiver."""

    class _FakeReceiver:
        def __init__(self, zmq_handle, device, use_shm):
            pass

        def receive_weights(self, on_bucket_received):
            for i, bucket in enumerate(buckets):
                on_bucket_received(bucket, i == len(buckets) - 1)

    monkeypatch.setattr(bucketed_weight_transfer, "BucketedWeightReceiver", _FakeReceiver)
    vLLMOmniColocateWorkerExtension.update_verl_delta_weights(worker, {})


def _delta_worker(target):
    return SimpleNamespace(
        device=torch.device("cpu"),
        _get_zmq_handle=lambda: "ipc:///tmp/test-delta.sock",
        _get_standard_weight_model_and_config=lambda: None,
        model_runner=SimpleNamespace(pipeline=target),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_seed_plus_two_deltas_bit_exact(monkeypatch):
    """End-to-end-style: seed export + two delta rounds reconstruct the reference state."""
    torch.manual_seed(0)
    module = _ToyDiT()
    _patch_engine_helpers(monkeypatch)
    engine = _make_engine(module)

    # Session 1: the seed streams the full export values-only into dummy weights.
    seed_params = _seed_export(engine)
    target = _ToyRolloutModel({name: torch.zeros_like(t) for name, t in seed_params})
    _run_delta_flush(_delta_worker(target), [_encode_dense_flush(seed_params)], monkeypatch)
    for name, tensor in seed_params:
        assert torch.equal(target.state[name], tensor), name
    engine.prime_delta_snapshots()

    # Session 2: perturb, export delta round 1, apply.
    with torch.no_grad():
        module.blocks[0].weight.view(-1)[3] += 0.5
    deltas, _ = engine.get_per_tensor_param_delta_shard()
    _run_delta_flush(_delta_worker(target), [_encode_indices_flush(deltas)], monkeypatch)

    # Session 3: perturb again, export delta round 2, apply.
    with torch.no_grad():
        module.blocks[1].weight.view(-1)[7] -= 1.0
    deltas, _ = engine.get_per_tensor_param_delta_shard()
    _run_delta_flush(_delta_worker(target), [_encode_indices_flush(deltas)], monkeypatch)

    reference = dict(_seed_export(engine))
    for name, ref in reference.items():
        assert torch.equal(target.state[name].view(torch.int16), ref.view(torch.int16)), name


def test_flush_split_across_buckets(monkeypatch):
    """One flush's tensors may land in different buckets; the payload must hold."""
    torch.manual_seed(0)
    module = _ToyDiT()
    _patch_engine_helpers(monkeypatch)
    engine = _make_engine(module)

    seed_params = _seed_export(engine)
    target = _ToyRolloutModel({name: torch.zeros_like(t) for name, t in seed_params})
    _run_delta_flush(_delta_worker(target), [_encode_dense_flush(seed_params)], monkeypatch)
    engine.prime_delta_snapshots()

    with torch.no_grad():
        module.blocks[0].weight.view(-1)[1] += 0.25
    deltas, _ = engine.get_per_tensor_param_delta_shard()
    spec, positions, values = _encode_indices_flush(deltas)

    _run_delta_flush(_delta_worker(target), [[spec, positions], [values]], monkeypatch)

    reference = dict(_seed_export(engine))
    for name, ref in reference.items():
        assert torch.equal(target.state[name], ref), name


def test_delta_apply_rejects_fused_moe_rollout(monkeypatch):
    from verl_omni.workers.rollout.vllm_rollout import npu_utils
    from verl_omni.workers.rollout.vllm_rollout import utils as rollout_utils

    monkeypatch.setattr(npu_utils, "_is_npu_platform", lambda: False)

    class _FakeRoutedExperts(torch.nn.Module):
        pass

    import vllm.model_executor.layers.fused_moe.routed_experts as routed_experts_module

    monkeypatch.setattr(routed_experts_module, "RoutedExperts", _FakeRoutedExperts)
    assert not rollout_utils._model_has_fused_moe(torch.nn.Linear(2, 2))
    model = torch.nn.Module()
    model.moe = _FakeRoutedExperts()
    assert rollout_utils._model_has_fused_moe(model)

    worker = SimpleNamespace(
        device=torch.device("cpu"),
        _get_zmq_handle=lambda: "ipc:///tmp/test-delta.sock",
        _get_standard_weight_model_and_config=lambda: (model, SimpleNamespace()),
    )
    with pytest.raises(NotImplementedError, match="fused-MoE"):
        vLLMOmniColocateWorkerExtension.update_verl_delta_weights(worker, {})


def test_checksum_mismatch_raises(monkeypatch):
    torch.manual_seed(0)
    module = _ToyDiT()
    _patch_engine_helpers(monkeypatch)
    engine = _make_engine(module)

    seed_params = _seed_export(engine)
    target = _ToyRolloutModel({name: torch.zeros_like(t) for name, t in seed_params})
    _run_delta_flush(_delta_worker(target), [_encode_dense_flush(seed_params)], monkeypatch)
    engine.prime_delta_snapshots()

    with torch.no_grad():
        module.blocks[0].weight.view(-1)[2] += 0.5
    deltas, _ = engine.get_per_tensor_param_delta_shard()
    spec, positions, values = _encode_indices_flush(deltas)
    corrupted = values[1].clone()
    corrupted[0] = 42.0

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _run_delta_flush(_delta_worker(target), [[spec, positions, (VALUES_NAME, corrupted)]], monkeypatch)


def test_delta_apply_rejects_npu_platform(monkeypatch):
    from verl_omni.workers.rollout.vllm_rollout import npu_utils

    monkeypatch.setattr(npu_utils, "_is_npu_platform", lambda: True)
    with pytest.raises(NotImplementedError, match="Ascend NPU"):
        vLLMOmniColocateWorkerExtension.update_verl_delta_weights(_delta_worker(_ToyRolloutModel({})), {})


def test_registry_resolves_omni_server_adapter():
    """The delta wire needs the omni adapter at this pin; it must stay a thin
    subclass of verl's ServerAdapter so the named_tensors path is verl's."""
    from verl.workers.rollout.base import get_rollout_class
    from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter

    from verl_omni.workers.rollout.vllm_rollout.server_adapter import VLLMOmniServerAdapter

    rollout_cls = get_rollout_class("vllm_omni", "async")
    assert rollout_cls is VLLMOmniServerAdapter
    assert issubclass(rollout_cls, ServerAdapter)


def test_omni_delta_sharded_registers_verl_delta_engine():
    """The omni backend is an alias of verl's own DeltaShardedCheckpointEngine,
    so verl's unmodified CheckpointEngineWorker constructs it through the
    registry (no subclass, no sglang-gate widening). The alias tracks verl's
    own backend: without the transport deps (e.g. CPU envs without cupy) both
    names are absent and asking for either fails closed."""
    from verl.checkpoint_engine import CheckpointEngineRegistry, DeltaShardedCheckpointEngine

    if DeltaShardedCheckpointEngine is None:
        with pytest.raises(ValueError, match="not registered"):
            CheckpointEngineRegistry.get("omni_delta_sharded")
    else:
        assert CheckpointEngineRegistry.get("omni_delta_sharded") is DeltaShardedCheckpointEngine


def test_extension_implements_verl_delta_protocol():
    """verl's adapter RPCs exactly these methods on a delta_flush sync; the omni
    worker extension (AR and diffusion) must answer every one of them."""
    for name in DELTA_PROTOCOL_METHODS:
        assert callable(getattr(vLLMOmniColocateWorkerExtension, name, None)), name


def test_ar_strategy_routes_delta_protocol_to_weight_sync_stages():
    """Broadcasting a delta receive to non-weight-sync AR stages would block on a
    ZMQ handle nobody sends to; every protocol RPC must hit the sync stages."""
    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy import ARStrategy

    strategy = SimpleNamespace(_weight_sync_stage_ids=[2])
    for name in ("set_pending_lora_peft_config", "update_weights_from_ipc", *DELTA_PROTOCOL_METHODS):
        assert ARStrategy.collective_rpc_stage_ids(strategy, name) == [2], name


# ---------------------------------------------------------------------------
# Sender side: VLLMOmniServerAdapter._update_delta_weights (verl's RPC protocol)
# ---------------------------------------------------------------------------


def _send_loop_adapter(worker, monkeypatch, calls, sent_buckets, apply_flushes=True):
    from verl_omni.workers.rollout.vllm_rollout.server_adapter import VLLMOmniServerAdapter

    adapter = object.__new__(VLLMOmniServerAdapter)
    adapter.use_shm = False
    adapter.zmq_handle = "ipc:///tmp/test-delta.sock"
    adapter.config = SimpleNamespace(checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=64))
    adapter._delta_weight_transfer_engine_initialized = False
    adapter.replica_rank = 0
    adapter.rollout_rank = 0
    adapter._has_server = False

    class _FakeSender:
        def __init__(self, zmq_handle, bucket_size_mb, use_shm):
            assert use_shm is False, "delta flushes must stream over CUDA IPC, not shm"

        async def async_send_weights(self, tensors):
            sent_buckets.append(list(tensors))

    monkeypatch.setattr(bucketed_weight_transfer, "BucketedWeightSender", _FakeSender)

    async def _execute_method(method, non_block=False, timeout=None, args=(), kwargs=None):
        calls.append(method)
        if method == "update_verl_delta_weights" and apply_flushes:

            async def _apply():
                # Runs when the sender awaits the RPC future, i.e. after this
                # flush's buckets were delivered -- one bucket per fake flush.
                _run_delta_flush(worker, [sent_buckets[-1]], monkeypatch)

            return _apply()
        return None

    adapter._execute_method = _execute_method
    adapter._ensure_server_handle = lambda: False
    return adapter


def test_adapter_send_loop_applies_seed_then_delta(monkeypatch):
    """Seed and delta streams both ride the per-flush RPC protocol; the init
    handshake fires once across syncs and the rollout state ends bit-exact."""
    torch.manual_seed(0)
    module = _ToyDiT()
    _patch_engine_helpers(monkeypatch)
    engine = _make_engine(module)

    seed_params = _seed_export(engine)
    target = _ToyRolloutModel({name: torch.zeros_like(t) for name, t in seed_params})
    calls, sent = [], []
    adapter = _send_loop_adapter(_delta_worker(target), monkeypatch, calls, sent)

    asyncio.run(adapter._update_delta_weights([(_encode_dense_flush(seed_params), True)], global_steps=1))
    assert calls == [
        "init_weight_transfer_engine",
        "start_weight_update",
        "update_verl_delta_weights",
        "finish_weight_update",
    ]
    engine.prime_delta_snapshots()

    with torch.no_grad():
        module.blocks[0].weight.view(-1)[3] += 0.5
    deltas, _ = engine.get_per_tensor_param_delta_shard()
    calls.clear()
    asyncio.run(adapter._update_delta_weights([(_encode_indices_flush(deltas), True)], global_steps=2))
    # Second sync: no re-init handshake; one RPC per flush.
    assert calls == ["start_weight_update", "update_verl_delta_weights", "finish_weight_update"]
    assert len(sent) == 2, "one bucketed stream per flush, never a shared one"

    reference = dict(_seed_export(engine))
    for name, ref in reference.items():
        assert torch.equal(target.state[name].view(torch.int16), ref.view(torch.int16)), name


def test_adapter_send_loop_empty_stream_skips_protocol(monkeypatch):
    calls, sent = [], []
    adapter = _send_loop_adapter(_delta_worker(_ToyRolloutModel({})), monkeypatch, calls, sent)
    asyncio.run(adapter._update_delta_weights([], global_steps=1))
    assert calls == [] and sent == []


def test_adapter_send_loop_rejects_data_after_is_last(monkeypatch):
    calls, sent = [], []
    adapter = _send_loop_adapter(_delta_worker(_ToyRolloutModel({})), monkeypatch, calls, sent, apply_flushes=False)
    first = [("__dummy__", torch.zeros(1))]
    with pytest.raises(ValueError, match="data after is_last"):
        asyncio.run(adapter._update_delta_weights([(first, True), (first, False)], global_steps=1))


def test_adapter_send_loop_rejects_stream_without_is_last(monkeypatch):
    calls, sent = [], []
    adapter = _send_loop_adapter(_delta_worker(_ToyRolloutModel({})), monkeypatch, calls, sent, apply_flushes=False)
    first = [("__dummy__", torch.zeros(1))]
    with pytest.raises(ValueError, match="ended without is_last"):
        asyncio.run(adapter._update_delta_weights([(first, False)], global_steps=1))

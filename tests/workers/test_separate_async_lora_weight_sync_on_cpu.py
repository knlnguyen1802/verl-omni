"""CPU tests for separate-async LoRA weight-sync routing in verl-omni."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from verl.checkpoint_engine import CheckpointEngineManager

from verl_omni.workers.checkpoint_engine import OmniCheckpointEngineManager
from verl_omni.workers.engine_workers import ActorRolloutRefWorker


class _FakeCollectiveRpc:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def remote(self, method_name: str, kwargs: dict):
        self.calls.append((method_name, kwargs))
        return {"ok": True}


def _make_fake_replica():
    rpc = _FakeCollectiveRpc()
    server_handle = SimpleNamespace(collective_rpc=rpc)
    return SimpleNamespace(server_handle=server_handle), rpc


def _make_manager(actor_returns: list[dict | None]):
    manager = object.__new__(OmniCheckpointEngineManager)
    manager.backend = "nccl"
    manager._lora_peft_config = None
    manager.actor_wg = SimpleNamespace(get_lora_peft_config=lambda: actor_returns)
    replica_a, rpc_a = _make_fake_replica()
    replica_b, rpc_b = _make_fake_replica()
    manager.replicas = [replica_a, replica_b]
    return manager, (rpc_a, rpc_b)


def test_separate_async_manager_pushes_lora_config_and_updates_parent(monkeypatch):
    manager, rpcs = _make_manager([None, {"r": 8, "lora_alpha": 16}, None])
    parent_calls = []

    async def _fake_parent_update(self, global_steps=None):
        parent_calls.append((self._lora_peft_config, global_steps))

    monkeypatch.setattr(CheckpointEngineManager, "update_weights", _fake_parent_update)
    monkeypatch.setattr("verl_omni.workers.checkpoint_engine.ray.get", lambda refs: refs)

    manager.update_weights(global_steps=17)

    assert manager._lora_peft_config == {"r": 8, "lora_alpha": 16}
    for rpc in rpcs:
        assert rpc.calls == [("set_pending_lora_peft_config", {"peft_config": {"r": 8, "lora_alpha": 16}})]
    assert parent_calls == [({"r": 8, "lora_alpha": 16}, 17)]


def test_separate_async_manager_clears_stale_lora_config_when_actor_has_none(monkeypatch):
    manager, rpcs = _make_manager([None, None])
    manager._lora_peft_config = {"stale": True}
    parent_calls = []

    async def _fake_parent_update(self, global_steps=None):
        parent_calls.append((self._lora_peft_config, global_steps))

    monkeypatch.setattr(CheckpointEngineManager, "update_weights", _fake_parent_update)
    monkeypatch.setattr("verl_omni.workers.checkpoint_engine.ray.get", lambda refs: refs)

    manager.update_weights(global_steps=23)

    assert manager._lora_peft_config is None
    for rpc in rpcs:
        assert rpc.calls == [("set_pending_lora_peft_config", {"peft_config": None})]
    assert parent_calls == [(None, 23)]


def test_get_lora_peft_config_returns_none_when_merge_enabled():
    worker = object.__new__(ActorRolloutRefWorker)
    worker.role = "actor"
    worker.peft_merge = True
    worker.actor = SimpleNamespace(engine=SimpleNamespace(module=SimpleNamespace()))
    assert worker.get_lora_peft_config() is None


def test_update_weights_from_ipc_consumes_pending_lora_and_adds_dummy_lora(monkeypatch):
    pytest.importorskip("vllm_omni")
    from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID
    from verl_omni.workers.rollout.vllm_rollout.utils import (
        OmniTensorLoRARequest,
        vLLMOmniColocateWorkerExtension,
    )

    class _FakeReceiver:
        def __init__(self, zmq_handle, device, use_shm):  # noqa: ARG002
            pass

        def receive_weights(self, on_bucket_received):
            on_bucket_received([("transformer.block0.lora_A.weight", torch.ones(2, 2))])
            on_bucket_received([("transformer.block0.lora_B.weight", torch.full((2, 2), 2.0))])

    monkeypatch.setattr(
        "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer.BucketedWeightReceiver",
        _FakeReceiver,
    )

    worker = object.__new__(vLLMOmniColocateWorkerExtension)
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker._pending_lora_peft_config = {"r": 8, "target_modules": ["to_q", "to_k"]}
    worker._get_zmq_handle = lambda: "ipc:///tmp/fake.sock"
    worker._get_standard_weight_model_and_config = lambda: None

    removed = []
    added = []
    worker.remove_lora = lambda lora_id: removed.append(lora_id)
    worker.add_lora = lambda req: added.append(req)
    worker.load_weights = lambda weights: (_ for _ in ()).throw(AssertionError("load_weights should not be used"))

    worker.update_weights_from_ipc(peft_config=None, base_sync_done=False, use_shm=False)

    assert worker._pending_lora_peft_config is None
    assert removed == [VLLM_LORA_INT_ID]
    assert len(added) == 1
    assert isinstance(added[0], OmniTensorLoRARequest)
    assert added[0].peft_config["r"] == 8
    assert set(added[0].lora_tensors.keys()) == {
        "transformer.block0.lora_A.weight",
        "transformer.block0.lora_B.weight",
    }
    assert torch.equal(added[0].lora_tensors["transformer.block0.lora_A.weight"], torch.ones(2, 2))
    assert torch.equal(
        added[0].lora_tensors["transformer.block0.lora_B.weight"],
        torch.full((2, 2), 2.0),
    )

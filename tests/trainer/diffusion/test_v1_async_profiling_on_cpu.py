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
"""CPU tests for the always-on async profiling of issue #712.

Covers the sample-wait decomposition of ``ProfiledReplayBufferAsync``, the
rollout-side TQ write-tag aggregation, the trainer-side payload byte count,
and the per-phase weight-sync timing mixin (against a stub manager, so no Ray
or engine backend is needed).
"""

import asyncio
import time
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from verl.trainer.ppo.v1 import replay_buffer as replay_buffer_module

from verl_omni.trainer.diffusion.v1.async_profiling import (
    ProfiledReplayBufferAsync,
    aggregate_tq_write_stats,
    dataproto_payload_bytes,
)
from verl_omni.workers.checkpoint_engine import (
    TimedCheckpointEngineManager,
    WeightSyncPhaseTimerMixin,
)


class _FakeTransferQueue:
    """Minimal TransferQueue stand-in covering the replay buffer's read path."""

    def __init__(self, items):
        self.items = {"train": items, "val": {}}

    def kv_list(self):
        return deepcopy(self.items)

    def kv_clear(self, *, partition_id, keys):
        for key in keys:
            self.items.setdefault(partition_id, {}).pop(key, None)

    def add_finished_group(self, uid: str, trajectories: int = 1, global_steps: int = 1):
        self.items["train"][uid] = {"is_prompt": True, "status": "finished", "global_steps": global_steps}
        for session_id in range(trajectories):
            self.items["train"][f"{uid}_{session_id}_0"] = {
                "is_prompt": False,
                "global_steps": global_steps,
                "seq_len": 1,
            }


def _make_profiled_buffer(poll_interval: float = 0.01) -> ProfiledReplayBufferAsync:
    return ProfiledReplayBufferAsync(
        trainer_mode="separate_async",
        trainer_config={},
        max_off_policy_threshold=8,
        max_off_policy_strategy="drop",
        sampler_kwargs={},
        poll_interval=poll_interval,
    )


def _patch_transfer_queue(monkeypatch, fake_tq):
    # Both verl's replay_buffer and verl-omni resolve `tq` to the same
    # transfer_queue module object, so patching it once covers the buffer.
    monkeypatch.setattr(replay_buffer_module.tq, "kv_list", fake_tq.kv_list)
    monkeypatch.setattr(replay_buffer_module.tq, "kv_clear", fake_tq.kv_clear)


def test_sample_wait_decomposition_segments_sum_to_total(monkeypatch):
    """The four segments + total must be consistent after a polling sample."""
    fake_tq = _FakeTransferQueue({})
    _patch_transfer_queue(monkeypatch, fake_tq)

    def finish_group_after_delay():
        time.sleep(0.05)
        fake_tq.add_finished_group("uida")

    import threading

    finisher = threading.Thread(target=finish_group_after_delay)
    finisher.start()

    buffer = _make_profiled_buffer(poll_interval=0.01)
    meta, _eviction = buffer.sample(global_steps=1, partition_id="train", batch_size=1)
    finisher.join()

    timing = buffer.last_sample_timing
    assert len(meta.keys) == 1
    # All accounting keys present and non-negative.
    for key in (
        "sample_wait/poll_sleep",
        "sample_wait/metadata_sync",
        "sample_wait/eviction",
        "sample_wait/no_sampleable",
        "sample_wait/total",
        "sample_wait/polls",
    ):
        assert key in timing, f"missing {key} in {sorted(timing)}"
        assert timing[key] >= 0.0
    # The decomposition must cover the wall-clock window within timer slack.
    segments = sum(timing[f"sample_wait/{s}"] for s in ("poll_sleep", "metadata_sync", "eviction", "no_sampleable"))
    assert segments == pytest.approx(timing["sample_wait/total"], abs=0.05)
    # We polled at least once before the group appeared.
    assert timing["sample_wait/polls"] >= 1.0
    # Buffer-depth telemetry for the polls we waited through.
    assert timing["sample_wait/buffer_finished_max"] >= 0.0


def test_sample_wait_no_poll_when_data_ready(monkeypatch):
    """A ready buffer must report zero poll sleeps and no residual wait."""
    fake_tq = _FakeTransferQueue({})
    _patch_transfer_queue(monkeypatch, fake_tq)
    fake_tq.add_finished_group("uida")

    buffer = _make_profiled_buffer()
    buffer.sample(global_steps=1, partition_id="train", batch_size=1)

    timing = buffer.last_sample_timing
    assert timing["sample_wait/polls"] == 0.0
    assert timing["sample_wait/poll_sleep"] == 0.0
    assert timing["sample_wait/no_sampleable"] == pytest.approx(0.0, abs=0.05)


def test_incidental_metadata_syncs_are_not_profiled(monkeypatch):
    """get_sampleable_count runs outside the profiling window by design."""
    fake_tq = _FakeTransferQueue({})
    _patch_transfer_queue(monkeypatch, fake_tq)
    fake_tq.add_finished_group("uida")

    buffer = _make_profiled_buffer()
    buffer.get_sampleable_count(1, "train")  # syncs metadata, must not accumulate
    assert buffer.last_sample_timing == {}


def test_aggregate_tq_write_stats_sums_tags():
    meta = SimpleNamespace(
        tags=[
            {"tq_put_s": 0.5, "tq_payload_bytes": 100},
            {"tq_put_s": 1.5, "tq_payload_bytes": 300},
            {"status": "success"},  # rows from older writers are skipped
            {"tq_put_s": 1.0, "tq_payload_bytes": 0},
        ]
    )
    stats = aggregate_tq_write_stats(meta)
    assert stats["tq/put/rows"] == 3.0
    assert stats["tq/put/seconds"] == pytest.approx(3.0)
    assert stats["tq/put/bytes"] == 400.0
    assert stats["tq/put/bytes_per_second"] == pytest.approx(400.0 / 3.0)


def test_aggregate_tq_write_stats_empty_and_untagged():
    assert aggregate_tq_write_stats(SimpleNamespace(tags=[])) == {
        "tq/put/rows": 0.0,
        "tq/put/seconds": 0.0,
        "tq/put/bytes": 0.0,
    }
    # Missing tags attribute (sync-mode metas) degrades to zeros, not an error.
    stats = aggregate_tq_write_stats(SimpleNamespace())
    assert stats["tq/put/bytes"] == 0.0


def test_dataproto_payload_bytes_counts_tensor_batch():
    from tensordict import TensorDict

    batch = TensorDict(
        {
            "latents": torch.zeros(4, 8, 8, dtype=torch.float32),
            "ids": torch.zeros(4, 16, dtype=torch.int64),
        },
        batch_size=4,
    )
    data = SimpleNamespace(batch=batch, non_tensor_batch={"uid": ["a"] * 4})
    # 4*8*8*4 + 4*16*8 = 1024 + 512
    assert dataproto_payload_bytes(data) == 1536


# -- weight-sync phase timer -------------------------------------------------


class _StubManagerBase:
    """Stand-in for ``CheckpointEngineManager`` with the pipeline's shape."""

    def __init__(self):
        self.replicas = [SimpleNamespace(), SimpleNamespace()]
        self.calls = []

    async def abort_replicas(self):
        self.calls.append("abort")
        await asyncio.sleep(0.01)

    async def release_kv_cache_replicas(self):
        self.calls.append("kv_release")
        await asyncio.sleep(0.01)

    async def build_process_group(self, rollout):
        self.calls.append("topology")
        await asyncio.sleep(0.01)

    async def resume_kv_cache_replicas(self):
        self.calls.append("kv_resume")
        await asyncio.sleep(0.01)

    async def resume_generation_replicas(self):
        self.calls.append("generation_resume")
        await asyncio.sleep(0.01)

    async def update_weights(self, global_steps=None):
        # The real pipeline's call order: every phase method is invoked through
        # self, so the mixin's overrides intercept, and the inline transfer
        # window is whatever sleep we model here.
        await self.abort_replicas()
        await self.release_kv_cache_replicas()
        await self.build_process_group(rollout=None)
        await asyncio.sleep(0.02)  # worker-group build + transfer + finalize
        await self.resume_kv_cache_replicas()
        await self.resume_generation_replicas()
        return {"engine/changed_ratio": 0.25}


class _StubTimedManager(WeightSyncPhaseTimerMixin, _StubManagerBase):
    async def update_weights(self, global_steps=None):
        return await self._profiled_update_weights(_StubManagerBase.update_weights, global_steps)


def test_weight_sync_phase_timer_decomposes_pipeline():
    manager = _StubTimedManager()
    metrics = asyncio.run(manager.update_weights(global_steps=3))

    # Engine metrics keep their own keys; phase timings are added alongside.
    assert metrics["engine/changed_ratio"] == 0.25
    for phase in ("abort", "kv_release", "topology", "kv_resume", "generation_resume"):
        assert metrics[f"weight_sync/phase/{phase}_s"] == pytest.approx(0.01, abs=0.008)
    # Residual window holds the un-wrapped inline phases (modeled as 0.02s).
    assert metrics["weight_sync/phase/transfer_and_finalize_s"] == pytest.approx(0.02, abs=0.01)
    assert metrics["weight_sync/total_s"] == pytest.approx(0.07, abs=0.02)
    assert metrics["weight_sync/replicas"] == 2.0
    # The phases ran through the mixin's overrides in pipeline order.
    assert manager.calls == ["abort", "kv_release", "topology", "kv_resume", "generation_resume"]


def test_timed_manager_sync_call_style_still_works():
    """Phase methods keep the dual sync/async call contract of @auto_await."""
    manager = _StubTimedManager()
    # Calling the coroutine function via asyncio.run from a sync context.
    manager.abort_replicas()  # not awaited: auto_await runs it in a thread pool
    assert manager.calls == ["abort"]


def test_timed_checkpoint_engine_manager_is_constructible_subclass():
    # The Ray-facing classes must still subclass verl's manager for isinstance
    # checks and the trainer's duck typing.
    from verl.checkpoint_engine import CheckpointEngineManager

    assert issubclass(TimedCheckpointEngineManager, CheckpointEngineManager)

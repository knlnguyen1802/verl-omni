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
"""Always-on accounting for the async diffusion V1 trainers (issue #712).

Three complementary pieces feed the per-step metrics dict:

- :class:`ProfiledReplayBufferAsync` decomposes ``sample()`` wall time into
  poll-sleep, TransferQueue metadata sync, eviction, and residual
  no-sampleable-data wait. The trainer reads ``last_sample_timing`` after each
  sample and merges it into ``timing_raw``, so every segment shows up as a
  ``timing_s/sample_wait/*`` metric with no new export channel.
- :func:`aggregate_tq_write_stats` sums the per-row ``tq_put_s`` /
  ``tq_payload_bytes`` tags the rollout-side TQ writer records, attributing
  rollout-side transfer cost and bytes to the batch that actually pays for
  them (the sampled one).
- :func:`dataproto_payload_bytes` measures the trainer-side read payload so
  ``tq/get/bytes`` pairs with ``timing_s/tq_get`` for a GB/s figure.

Everything here is metadata-only: timers that were already being taken (or
tags that were already being written) promoted to metrics, no extra TQ round
trips, no behavior change.
"""

import time
from collections import defaultdict

from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync

# Tag keys written by ``DiffusionAgentLoopWorkerTQ._write_trajectories_to_tq``
# and summed by :func:`aggregate_tq_write_stats`.
TQ_PUT_SECONDS_TAG = "tq_put_s"
TQ_PAYLOAD_BYTES_TAG = "tq_payload_bytes"


class ProfiledReplayBufferAsync(ReplayBufferAsync):
    """``ReplayBufferAsync`` that accounts for where ``sample()`` time went.

    The upstream poll loop sleeps a fixed ``poll_interval`` between
    ``kv_list`` metadata syncs, evicts stale/filtered groups, and re-checks
    sampleability — all inside one opaque wait. This subclass times each
    segment by wrapping the private helpers the loop is built from:

    - ``sample_wait/metadata_sync`` — ``kv_list`` round trips (the metadata
      sync cost of polling itself);
    - ``sample_wait/poll_sleep`` — quantization sleep between polls;
    - ``sample_wait/eviction`` — stale/DAPO/failure eviction passes;
    - ``sample_wait/no_sampleable`` — residual: waiting because not enough
      groups were finished yet (the true generation-bound idle).

    ``last_sample_timing`` (seconds per segment, plus poll count and the
    finished-group depth seen while waiting) is refreshed on every
    ``wait_for_sampleable`` / ``sample`` call. Accumulation only happens
    inside those calls, so incidental metadata syncs from
    ``get_sampleable_count`` (hybrid switch decisions) are not mixed in.

    The queue-depth snapshots answer "was the trainer waiting because the
    buffer was drained or because everything was still running" — pending vs
    running vs finished counts per poll — which is the difference between a
    generation-throughput problem and a scheduling one.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._profiling_active = False
        self._segment_timings: dict[str, float] = defaultdict(float)
        self._poll_depths: list[tuple[int, int, int]] = []
        self.last_sample_timing: dict[str, float] = {}

    # -- timed segments ----------------------------------------------------

    def _sync_metadata_from_transfer_queue(self):
        if not self._profiling_active:
            return super()._sync_metadata_from_transfer_queue()
        start = time.perf_counter()
        try:
            return super()._sync_metadata_from_transfer_queue()
        finally:
            self._segment_timings["metadata_sync"] += time.perf_counter() - start

    def _evict_terminal_groups(self, global_steps, partition_id, eviction_reasons):
        if not self._profiling_active:
            return super()._evict_terminal_groups(global_steps, partition_id, eviction_reasons)
        start = time.perf_counter()
        try:
            return super()._evict_terminal_groups(global_steps, partition_id, eviction_reasons)
        finally:
            self._segment_timings["eviction"] += time.perf_counter() - start

    def _wait_for_next_poll(self, partition_id, last_debug_time):
        if not self._profiling_active:
            return super()._wait_for_next_poll(partition_id, last_debug_time)
        self._poll_depths.append(
            (
                len(self.pending_keys[partition_id]),
                len(self.running_keys[partition_id]),
                len(self.finished_keys[partition_id]),
            )
        )
        start = time.perf_counter()
        try:
            return super()._wait_for_next_poll(partition_id, last_debug_time)
        finally:
            self._segment_timings["poll_sleep"] += time.perf_counter() - start

    # -- profiling window --------------------------------------------------

    def _begin_profile_window(self) -> None:
        self._segment_timings = defaultdict(float)
        self._poll_depths = []
        self._profiling_active = True

    def _end_profile_window(self, total_seconds: float) -> None:
        self._profiling_active = False
        # Report every segment every time (0.0 when absent) so dashboards see
        # a constant metric schema across steps regardless of what ran.
        segment_names = ("poll_sleep", "metadata_sync", "eviction", "no_sampleable")
        timings = {
            f"sample_wait/{name}": self._segment_timings.get(name, 0.0) for name in segment_names
        }
        measured = sum(timings.values())
        timings["sample_wait/no_sampleable"] = max(0.0, total_seconds - measured)
        timings["sample_wait/total"] = total_seconds
        timings["sample_wait/polls"] = float(len(self._poll_depths))
        if self._poll_depths:
            finished = [depth[2] for depth in self._poll_depths]
            timings["sample_wait/buffer_finished_mean"] = sum(finished) / len(finished)
            timings["sample_wait/buffer_finished_min"] = float(min(finished))
            timings["sample_wait/buffer_finished_max"] = float(max(finished))
            pending = [depth[0] for depth in self._poll_depths]
            running = [depth[1] for depth in self._poll_depths]
            timings["sample_wait/buffer_pending_mean"] = sum(pending) / len(pending)
            timings["sample_wait/buffer_running_mean"] = sum(running) / len(running)
        self.last_sample_timing = timings

    def wait_for_sampleable(self, global_steps: int, partition_id: str, target_count: int):
        self._begin_profile_window()
        start = time.perf_counter()
        try:
            return super().wait_for_sampleable(global_steps, partition_id, target_count)
        finally:
            self._end_profile_window(time.perf_counter() - start)

    def sample(self, global_steps: int, partition_id: str, batch_size: int):
        self._begin_profile_window()
        start = time.perf_counter()
        try:
            return super().sample(global_steps, partition_id, batch_size)
        finally:
            self._end_profile_window(time.perf_counter() - start)


def aggregate_tq_write_stats(batch_meta) -> dict[str, float]:
    """Sum rollout-side TQ write cost recorded in row tags for one batch.

    ``DiffusionAgentLoopWorkerTQ`` stamps every written row with the wall time
    of its ``kv_batch_put`` and the payload bytes it carried. Rows written by
    older writers (or tag-only prompt rows filtered out before this point)
    simply lack the keys and are skipped, so the aggregation degrades to zero
    rather than error.
    """
    put_seconds = 0.0
    payload_bytes = 0
    tagged_rows = 0
    for tag in getattr(batch_meta, "tags", None) or []:
        if not isinstance(tag, dict):
            continue
        seconds = tag.get(TQ_PUT_SECONDS_TAG)
        if seconds is not None:
            put_seconds += float(seconds)
            tagged_rows += 1
        payload_bytes += int(tag.get(TQ_PAYLOAD_BYTES_TAG, 0) or 0)
    metrics: dict[str, float] = {
        "tq/put/rows": float(tagged_rows),
        "tq/put/seconds": put_seconds,
        "tq/put/bytes": float(payload_bytes),
    }
    if put_seconds > 0 and payload_bytes > 0:
        metrics["tq/put/bytes_per_second"] = payload_bytes / put_seconds
    return metrics


def dataproto_payload_bytes(data) -> int:
    """Payload bytes of a diffusion ``DataProto`` read back from TransferQueue.

    Counts the tensor batch only: non-tensor columns (uid, data_source, ...)
    ride the tag/metadata channel, not the payload transfer the RFC's byte
    counters are about.
    """
    total = 0
    batch = getattr(data, "batch", None)
    if batch is None:
        return 0
    for value in batch.values():
        if hasattr(value, "element_size"):
            total += value.numel() * value.element_size()
    return total

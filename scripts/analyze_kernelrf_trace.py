#!/usr/bin/env python
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
"""Summarize ``kernelrf:`` site costs from torch-profiler chrome traces.

Pairs with ``verl_omni.utils.kernels.site_marker``: run a profiled training
step (``global_profiler.tool=torch``, see ``docs/perf/profiler.md``) with
``VERL_OMNI_KERNEL_RF_MARKERS`` listing the sites of interest, then point this
script at the exported chrome trace(s). It attributes, per site:

- ``cpu_ms``  — time spent inside the marker's CPU span (enqueue cost),
- ``gpu_ms``  — duration of CUDA kernels / copies launched inside the span,
- ``launches``— kernel/copy launch count,
- ``d2h``     — device-to-host copy count (data-dependent-shape smell),
- ``syncs`` / ``sync_ms`` — blocking sync-API time inside the span.

GPU work is attributed via kineto correlation ids to the innermost enclosing
``kernelrf:`` span on the launching thread, so nested markers (e.g.
``S4_sigma_index`` inside ``S1_sde_step``) do not double-count.

Usage::

    python scripts/analyze_kernelrf_trace.py actor_step.json [more.json ...]
    python scripts/analyze_kernelrf_trace.py --self-test   # no torch needed

This supports the profiling plan in ``docs/perf/kernel_replacement_rfc.md``
(Phase 1, layer L2).
"""

import argparse
import json
import sys

MARKER_PREFIX = "kernelrf:"
# Runtime-API calls that block the CPU until GPU work completes. Their wall time
# inside a marker span is the site's sync-stall cost. Non-blocking variants
# (cudaMemcpyAsync, cudaEventRecord, cudaStreamWaitEvent's async cousin
# cudaStreamWaitEvent is blocking-wait only on host timeline: kept) excluded.
def _is_blocking_sync_api(name: str) -> bool:
    return "Synchronize" in name or name in ("cudaMemcpy", "cudaStreamWaitEvent", "cudaMalloc", "cudaFree")
DEVICE_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
RUNTIME_CATS = {"cuda_runtime", "cuda_driver"}


def _site_of(name: str) -> str | None:
    return name[len(MARKER_PREFIX) :] if name.startswith(MARKER_PREFIX) else None


def _innermost_site(ranges, pid, tid, t):
    """Site of the innermost marker span on (pid, tid) containing time t."""
    best = None
    best_dur = None
    for start, end, site, rpid, rtid in ranges:
        if rpid != pid or rtid != tid or not (start <= t <= end):
            continue
        dur = end - start
        if best_dur is None or dur < best_dur:
            best, best_dur = site, dur
    return best


def analyze_trace(events):
    """Return {site: stats} plus unattributed totals for one trace's events."""
    ranges = []  # (start, end, site, pid, tid)
    runtime = []  # (mid, pid, tid, name, dur, correlation)
    device = []  # (correlation, dur, is_d2h)
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat", "")
        name = ev.get("name", "")
        args = ev.get("args") or {}
        site = _site_of(name)
        if site:
            ranges.append((ev["ts"], ev["ts"] + ev["dur"], site, ev["pid"], ev["tid"]))
        elif cat in RUNTIME_CATS:
            mid = ev["ts"] + ev["dur"] / 2.0
            runtime.append((mid, ev["pid"], ev["tid"], name, ev["dur"], args.get("correlation")))
        elif cat in DEVICE_CATS:
            is_d2h = cat == "gpu_memcpy" and "DtoH" in name
            device.append((args.get("correlation"), ev["dur"], is_d2h))

    corr_to_site = {}
    sync_stats = {}  # site -> [count, us]
    for mid, pid, tid, name, dur, corr in runtime:
        site = _innermost_site(ranges, pid, tid, mid)
        if site is None:
            continue
        if _is_blocking_sync_api(name):
            cnt, us = sync_stats.setdefault(site, [0, 0.0])
            sync_stats[site] = [cnt + 1, us + dur]
        if corr is not None:
            corr_to_site[corr] = site

    stats = {}
    for corr, dur, is_d2h in device:
        site = corr_to_site.get(corr)
        if site is None:
            continue
        s = stats.setdefault(site, {"calls": 0, "cpu_ms": 0.0, "gpu_ms": 0.0, "launches": 0, "d2h": 0})
        s["gpu_ms"] += dur / 1000.0
        s["launches"] += 1
        s["d2h"] += 1 if is_d2h else 0
    for start, end, site, _pid, _tid in ranges:
        s = stats.setdefault(site, {"calls": 0, "cpu_ms": 0.0, "gpu_ms": 0.0, "launches": 0, "d2h": 0})
        s["calls"] += 1
        s["cpu_ms"] += (end - start) / 1000.0
    for site, (cnt, us) in sync_stats.items():
        s = stats.setdefault(site, {"calls": 0, "cpu_ms": 0.0, "gpu_ms": 0.0, "launches": 0, "d2h": 0})
        s.setdefault("syncs", 0)
        s.setdefault("sync_ms", 0.0)
        s["syncs"] = s.get("syncs", 0) + cnt
        s["sync_ms"] = s.get("sync_ms", 0.0) + us / 1000.0
    for s in stats.values():
        s.setdefault("syncs", 0)
        s.setdefault("sync_ms", 0.0)
        s["gpu_ms"] = round(s["gpu_ms"], 3)
        s["cpu_ms"] = round(s["cpu_ms"], 3)
        s["sync_ms"] = round(s["sync_ms"], 3)
    return stats


def merge(dst, src):
    empty = {"calls": 0, "cpu_ms": 0.0, "gpu_ms": 0.0, "launches": 0, "d2h": 0, "syncs": 0, "sync_ms": 0.0}
    for site, s in src.items():
        d = dst.setdefault(site, dict(empty))
        for k, v in s.items():
            d[k] = d.get(k, 0) + v
    return dst


def render(stats):
    order = sorted(stats, key=lambda k: stats[k]["gpu_ms"] + stats[k]["sync_ms"], reverse=True)
    header = f"{'site':<28}{'calls':>7}{'cpu_ms':>10}{'gpu_ms':>10}{'launches':>10}{'d2h':>6}{'syncs':>7}{'sync_ms':>9}"
    rows = [header, "-" * len(header)]
    for site in order:
        s = stats[site]
        rows.append(
            f"{site:<28}{s['calls']:>7}{s['cpu_ms']:>10.3f}{s['gpu_ms']:>10.3f}{s['launches']:>10}{s['d2h']:>6}"
            f"{s['syncs']:>7}{s['sync_ms']:>9.3f}"
        )
    return "\n".join(rows)


def _self_test():
    """Synthetic kineto-style trace; asserts attribution rules end to end."""
    # Timeline (us). Threads: python tid=1, CUDA runtime tid=2, device pid=10.
    # S1_sde_step spans 0..100 and contains a nested S4_sigma_index span 0..20.
    # - kernel A (corr 1) launched at t=5  -> innermost S4 (nested, no double count)
    # - kernel B (corr 2) launched at t=50 -> S1
    # - blocking cudaMemcpy at t=80 (corr 3, DtoH device copy) -> S1, sync stall
    # - kernel C (corr 4) launched outside any span -> unattributed (dropped)
    def ev(cat, name, ts, dur, pid, tid, corr=None):
        args = {"correlation": corr} if corr is not None else {}
        return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur, "pid": pid, "tid": tid, "args": args}

    events = [
        ev("cpu_op", "kernelrf:S1_sde_step", 0, 100, pid=1, tid=1),
        ev("cpu_op", "kernelrf:S4_sigma_index", 0, 20, pid=1, tid=1),
        ev("cuda_runtime", "cudaLaunchKernel", 5, 2, pid=1, tid=1, corr=1),
        ev("cuda_runtime", "cudaLaunchKernel", 50, 2, pid=1, tid=1, corr=2),
        ev("cuda_runtime", "cudaMemcpyAsync", 80, 15, pid=1, tid=1, corr=3),
        ev("cuda_runtime", "cudaStreamSynchronize", 95, 4, pid=1, tid=1),
        ev("cuda_runtime", "cudaLaunchKernel", 200, 2, pid=1, tid=1, corr=4),
        ev("kernel", "vectorized_elementwise", 6, 30, pid=10, tid=1, corr=1),
        ev("kernel", "reduce_kernel", 55, 10, pid=10, tid=1, corr=2),
        ev("gpu_memcpy", "Memcpy DtoH (Pinned -> Pageable)", 82, 12, pid=10, tid=1, corr=3),
        ev("kernel", "orphan_kernel", 205, 5, pid=10, tid=1, corr=4),
    ]
    stats = analyze_trace(events)

    assert stats["S4_sigma_index"]["launches"] == 1, stats  # kernel A -> innermost S4
    assert stats["S4_sigma_index"]["gpu_ms"] == 0.030
    s1 = stats["S1_sde_step"]
    assert s1["launches"] == 2, s1  # kernel B + DtoH copy
    assert abs(s1["gpu_ms"] - (10 + 12) / 1000) < 1e-6
    assert s1["d2h"] == 1
    assert s1["syncs"] == 1 and s1["sync_ms"] == 0.004  # cudaStreamSynchronize
    assert abs(s1["cpu_ms"] - 0.1) < 1e-6 and s1["calls"] == 1
    assert sum(s["launches"] for s in stats.values()) == 3  # orphan kernel dropped

    print(render(stats))
    print("\nSELF-TEST PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", nargs="*", help="chrome trace .json file(s) from torch profiler")
    parser.add_argument("--self-test", action="store_true", help="run the built-in synthetic-trace assertions")
    parser.add_argument("--json", dest="json_out", help="also write merged stats as JSON to this path")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.traces:
        parser.error("give at least one trace file, or --self-test")

    stats = {}
    for path in args.traces:
        with open(path, encoding="utf-8") as f:
            trace = json.load(f)
        events = trace["traceEvents"] if isinstance(trace, dict) else trace
        merge(stats, analyze_trace(events))
        print(f"# {path}")
    print(render(stats))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    sys.exit(main())

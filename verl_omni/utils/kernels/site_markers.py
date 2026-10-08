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

"""Profiler markers for candidate kernel-replacement hot-path sites.

Usage at a candidate site:

    from verl_omni.utils.kernels import site_marker

    with site_marker("S1_sde_step"):
        ...hot path...

The context manager is a zero-cost no-op unless the site id is listed in
``VERL_OMNI_KERNEL_RF_MARKERS`` (comma-separated). Enabled ranges appear as
``kernelrf:<site_id>`` CPU-op spans in torch-profiler chrome traces *and* as
NVTX ranges for nsys, so one wrapper serves both analysis paths. Summarize
traces with ``scripts/analyze_kernelrf_trace.py``.

Two auxiliary env vars (both default off):

- ``VERL_OMNI_KERNEL_RF_SYNC_DEBUG=warn|error`` — calls
  ``torch.cuda.set_sync_debug_mode`` once at import so hidden CPU-GPU
  syncs report (or raise) with the enclosing ``kernelrf:`` range visible in
  the traceback / profiler span.
- ``VERL_OMNI_KERNEL_RF_MARKERS`` — see above.

Site ids and their anchors are catalogued in
``docs/perf/kernel_replacement_rfc.md`` (Appendix A).
"""

import contextlib
import os

import torch

_RF_SITES = frozenset(s.strip() for s in os.environ.get("VERL_OMNI_KERNEL_RF_MARKERS", "").split(",") if s.strip())

_SYNC_DEBUG = os.environ.get("VERL_OMNI_KERNEL_RF_SYNC_DEBUG")
if _SYNC_DEBUG and torch.cuda.is_available():
    # Deliberate import-time side effect: sync debugging must be enabled before
    # the training step runs, and site_markers is imported by the wrapped hot
    # paths at module load. No-op (and cheap) when the env var is unset.
    torch.cuda.set_sync_debug_mode(_SYNC_DEBUG)


@contextlib.contextmanager
def site_marker(site_id: str):
    """Label a candidate hot-path site in profiler traces and NVTX.

    No-op (zero cost) unless ``site_id`` is listed in
    ``VERL_OMNI_KERNEL_RF_MARKERS`` and a CUDA device is present.
    """
    if site_id not in _RF_SITES or not torch.cuda.is_available():
        yield
        return
    torch.cuda.nvtx.range_push(f"kernelrf:{site_id}")
    try:
        with torch.profiler.record_function(f"kernelrf:{site_id}"):
            yield
    finally:
        torch.cuda.nvtx.range_pop()

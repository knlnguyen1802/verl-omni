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
"""CPU tests for diffusion hybrid sleep: skip Orchestrator reset_* and unsafe CuMem."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from verl_omni.workers.rollout.vllm_rollout import utils as rollout_utils
from verl_omni.workers.rollout.vllm_rollout.utils import (
    is_diffusion_pipeline_worker,
    should_skip_unsafe_cumem_sleep,
    vLLMOmniColocateWorkerExtension,
)
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer


class AsyncOmni:
    """Name must be AsyncOmni so ``_uses_async_omni_orchestrator`` matches."""

    def __init__(self):
        self.sleep = AsyncMock()
        self.reset_prefix_cache = AsyncMock()
        self.reset_mm_cache = AsyncMock()
        self.reset_encoder_cache = AsyncMock()
        self.request_states = {}
        self.output_processor = object()
        self.abort = AsyncMock()


def _diffusion_server(*, free_cache_engine=True, node_rank=0):
    server = object.__new__(vLLMOmniHttpServer)
    server._ar_mode = False
    server.node_rank = node_rank
    server.config = SimpleNamespace(free_cache_engine=free_cache_engine)
    server.engine = AsyncOmni()
    server._invalidate_lora_request_cache = lambda: None
    return server


def test_uses_async_omni_orchestrator_by_type_name():
    server = _diffusion_server()
    assert server._uses_async_omni_orchestrator()


def test_uses_async_omni_orchestrator_by_stage_clients():
    server = object.__new__(vLLMOmniHttpServer)
    server.engine = SimpleNamespace(engine=SimpleNamespace(stage_clients=[object()]))
    assert server._uses_async_omni_orchestrator()


def test_uses_async_omni_orchestrator_false_for_plain_engine():
    server = object.__new__(vLLMOmniHttpServer)
    server.engine = SimpleNamespace()
    assert not server._uses_async_omni_orchestrator()


@pytest.mark.asyncio
async def test_sleep_does_not_reset_orchestrator_caches():
    server = _diffusion_server()

    await vLLMOmniHttpServer.sleep(server)

    server.engine.sleep.assert_awaited_once_with(level=1)
    server.engine.reset_prefix_cache.assert_not_called()
    server.engine.reset_mm_cache.assert_not_called()
    server.engine.reset_encoder_cache.assert_not_called()


@pytest.mark.asyncio
async def test_sleep_hybrid_does_not_reset_encoder_cache():
    server = _diffusion_server()

    await vLLMOmniHttpServer._sleep_hybrid(server)

    server.engine.sleep.assert_awaited_once_with(level=1)
    server.engine.reset_encoder_cache.assert_not_called()


@pytest.mark.asyncio
async def test_sleep_skipped_when_free_cache_engine_false():
    server = _diffusion_server(free_cache_engine=False)

    await vLLMOmniHttpServer.sleep(server)

    server.engine.sleep.assert_not_called()


@pytest.mark.asyncio
async def test_abort_uses_omni_path_even_if_output_processor_exists():
    server = _diffusion_server()

    result = await vLLMOmniHttpServer.abort_all_requests(server, reset_prefix_cache=True)

    assert result["drained"] is True
    assert result["aborted_count"] == 0
    server.engine.abort.assert_not_called()
    server.engine.reset_mm_cache.assert_not_called()


def test_torchvision_cudart_detected_from_proc_maps(tmp_path, monkeypatch):
    maps = tmp_path / "maps"
    maps.write_text(
        "7f0000000000-7f0000001000 r-xp 00000000 00:00 0 "
        "/usr/lib/python3/dist-packages/torchvision.libs/libcudart.faf08d9a.so.13\n"
    )
    monkeypatch.setattr(rollout_utils, "_PROC_SELF_MAPS", str(maps))
    monkeypatch.setattr(rollout_utils.sys, "modules", {})

    worker = SimpleNamespace(od_config=object(), rank=0, stage_id=0)
    assert is_diffusion_pipeline_worker(worker)
    assert should_skip_unsafe_cumem_sleep(worker)


def test_ar_worker_does_not_skip_cumem_sleep(tmp_path, monkeypatch):
    maps = tmp_path / "maps"
    maps.write_text(
        "7f0000000000-7f0000001000 r-xp 00000000 00:00 0 "
        "/usr/lib/python3/dist-packages/torchvision.libs/libcudart.faf08d9a.so.13\n"
    )
    monkeypatch.setattr(rollout_utils, "_PROC_SELF_MAPS", str(maps))
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=object()), rank=0)

    assert not is_diffusion_pipeline_worker(worker)
    assert not should_skip_unsafe_cumem_sleep(worker)


def test_diffusion_worker_without_torchvision_does_not_skip(monkeypatch):
    monkeypatch.setattr(rollout_utils, "torchvision_bundled_cudart_is_mapped", lambda: False)
    worker = SimpleNamespace(
        od_config=object(),
        model_runner=SimpleNamespace(pipeline=object()),
        rank=0,
    )

    assert is_diffusion_pipeline_worker(worker)
    assert not should_skip_unsafe_cumem_sleep(worker)


def test_handle_sleep_task_skips_parent_when_torchvision_cudart_mapped(monkeypatch):
    worker = object.__new__(vLLMOmniColocateWorkerExtension)
    worker.od_config = object()
    worker.rank = 0
    worker.stage_id = 0
    monkeypatch.setattr(rollout_utils, "should_skip_unsafe_cumem_sleep", lambda _w: True)

    ack = vLLMOmniColocateWorkerExtension.handle_sleep_task(worker, {"task_id": "t1", "level": 1})

    assert ack is not None
    assert ack.status == "SUCCESS"
    assert ack.freed_bytes == 0
    assert ack.metadata["skipped"] == "torchvision_libcudart"


def test_sleep_returns_zero_when_torchvision_cudart_mapped(monkeypatch):
    worker = object.__new__(vLLMOmniColocateWorkerExtension)
    worker.od_config = object()
    monkeypatch.setattr(rollout_utils, "should_skip_unsafe_cumem_sleep", lambda _w: True)

    assert vLLMOmniColocateWorkerExtension.sleep(worker, level=1) == 0

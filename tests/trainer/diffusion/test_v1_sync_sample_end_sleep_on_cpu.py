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
"""CPU tests for v1 sync sample-end drain (no abort-then-sleep)."""

import asyncio
from types import SimpleNamespace

import pytest

from verl_omni.agent_loop.diffusion_agent_loop_tq import DiffusionAgentLoopWorkerTQ
from verl_omni.trainer.diffusion.v1 import trainer_base as trainer_base_module
from verl_omni.trainer.diffusion.v1.trainer_base import PolicyGradientDiffusionTrainerV1
from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync


class _CheckpointManager:
    def __init__(self):
        self.calls: list[str] = []

    def abort_replicas(self):
        self.calls.append("abort")

    def sleep_replicas(self):
        self.calls.append("sleep")


def test_sync_on_sample_end_waits_then_sleeps_without_abort():
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1Sync)
    trainer.checkpoint_manager = _CheckpointManager()
    trainer.agent_loop_manager = SimpleNamespace(agent_loop_workers=[])

    trainer.on_sample_end()

    assert trainer.checkpoint_manager.calls == ["sleep"]


def test_wait_for_tq_background_tasks_invokes_worker_remote(monkeypatch):
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1)
    called: list[str] = []

    class _Wait:
        def remote(self):
            called.append("wait")
            return "future"

    trainer.agent_loop_manager = SimpleNamespace(
        agent_loop_workers=[SimpleNamespace(wait_for_background_tasks=_Wait())]
    )
    monkeypatch.setattr(trainer_base_module.ray, "get", lambda futures: called.append(tuple(futures)))

    trainer._wait_for_tq_background_tasks()

    assert called == ["wait", ("future",)]


def test_wait_for_tq_background_tasks_noops_without_manager():
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1)
    trainer.agent_loop_manager = None
    trainer._wait_for_tq_background_tasks()


@pytest.mark.asyncio
async def test_tq_worker_wait_for_background_tasks_drains_pending():
    worker_cls = DiffusionAgentLoopWorkerTQ.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    worker.background_tasks = set()
    settled = []

    async def _slow():
        await asyncio.sleep(0.01)
        settled.append("done")

    task = asyncio.create_task(_slow())
    worker.background_tasks.add(task)
    task.add_done_callback(worker.background_tasks.discard)

    await worker.wait_for_background_tasks()

    assert settled == ["done"]
    assert not worker.background_tasks

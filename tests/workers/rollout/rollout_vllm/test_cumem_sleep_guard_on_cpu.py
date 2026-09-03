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
"""CPU checks: diffusion sleep offloads via PyTorch, not CuMem."""

from types import SimpleNamespace

import torch

from verl_omni.workers.rollout.vllm_rollout.utils import vLLMOmniColocateWorkerExtension


class _FakeBase:
    def sleep(self, level: int = 1):
        self.slept = level
        return 42

    def wake_up(self, tags=None):
        self.woke = tags
        return True


class _Worker(vLLMOmniColocateWorkerExtension, _FakeBase):
    pass


def _make_worker(pipeline=None, model=None):
    worker = object.__new__(_Worker)
    worker.model_runner = SimpleNamespace(pipeline=pipeline, model=model)
    worker.device = torch.device("cpu")
    return worker


def test_diffusion_sleep_does_not_call_cumem():
    pipeline = torch.nn.Linear(2, 2)
    worker = _make_worker(pipeline=pipeline)

    assert vLLMOmniColocateWorkerExtension.sleep(worker, 1) == 0
    assert not hasattr(worker, "slept")
    assert vLLMOmniColocateWorkerExtension.wake_up(worker, ["weights"]) is True
    assert not hasattr(worker, "woke")


def test_ar_worker_still_uses_cumem_sleep():
    worker = _make_worker(model=object())

    assert vLLMOmniColocateWorkerExtension.sleep(worker, 1) == 42
    assert worker.slept == 1
    assert vLLMOmniColocateWorkerExtension.wake_up(worker, ["weights"]) is True
    assert worker.woke == ["weights"]

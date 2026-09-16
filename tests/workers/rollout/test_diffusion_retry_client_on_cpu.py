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
"""CPU contract tests for DiffusionWholeSampleRetryLLMServerClient.

The colocate_async and separate_async diffusion trainers route rollout through
this client. Aborted samples must be retried as whole samples (no token-append
resume exists for diffusion), the weight-version span must survive onto the
final output's extra_fields, and the retry count must be visible for abort-rate
metrics.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from verl.workers.rollout.llm_server import LLMServerClient

from verl_omni.workers.rollout.diffusion_llm_server import DiffusionWholeSampleRetryLLMServerClient


def _client(max_retries=50):
    client = DiffusionWholeSampleRetryLLMServerClient.__new__(DiffusionWholeSampleRetryLLMServerClient)
    client.max_retries = max_retries
    client.retry_wait_s = 0.0
    return client


def _output(stop_reason, global_steps):
    return SimpleNamespace(stop_reason=stop_reason, extra_fields={"global_steps": global_steps})


async def test_retry_records_version_span_and_count():
    outputs = [_output("aborted", 3), _output("completed", 4)]
    client = _client()
    with patch.object(LLMServerClient, "generate", new=AsyncMock(side_effect=outputs)) as mock_gen:
        final = await client.generate("req-1", prompt_ids=[1, 2], sampling_params={})

    # Whole-sample retry: the same unmodified prompt ids are re-sent.
    assert mock_gen.call_count == 2
    assert mock_gen.call_args_list[0].kwargs["prompt_ids"] == [1, 2]
    assert mock_gen.call_args_list[1].kwargs["prompt_ids"] == [1, 2]
    assert final.stop_reason == "completed"
    assert final.extra_fields["min_global_steps"] == 3
    assert final.extra_fields["max_global_steps"] == 4
    assert final.extra_fields["retry_count"] == 1


async def test_single_attempt_has_zero_retries():
    client = _client()
    with patch.object(LLMServerClient, "generate", new=AsyncMock(side_effect=[_output("completed", 7)])) as mock_gen:
        final = await client.generate("req-1", prompt_ids=[1], sampling_params={})

    assert mock_gen.call_count == 1
    assert final.extra_fields["min_global_steps"] == 7
    assert final.extra_fields["max_global_steps"] == 7
    assert final.extra_fields["retry_count"] == 0


async def test_gives_up_after_max_retries_and_returns_last_output():
    client = _client(max_retries=3)
    with patch.object(LLMServerClient, "generate", new=AsyncMock(side_effect=[_output("aborted", 2)] * 3)) as mock_gen:
        final = await client.generate("req-1", prompt_ids=[1], sampling_params={})

    assert mock_gen.call_count == 3
    assert final.stop_reason == "aborted"
    assert final.extra_fields["min_global_steps"] == 2
    assert final.extra_fields["max_global_steps"] == 2
    assert final.extra_fields["retry_count"] == 2


async def test_missing_global_steps_stays_none_for_writer_fallback():
    client = _client()
    output = SimpleNamespace(stop_reason="completed", extra_fields={})
    with patch.object(LLMServerClient, "generate", new=AsyncMock(side_effect=[output])):
        final = await client.generate("req-1", prompt_ids=[1], sampling_params={})

    assert final.extra_fields["min_global_steps"] is None
    assert final.extra_fields["max_global_steps"] is None
    assert final.extra_fields["retry_count"] == 0

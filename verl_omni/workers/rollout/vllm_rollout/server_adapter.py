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
"""vLLM-Omni server adapter: verl's vLLM ``ServerAdapter`` plus the delta wire format."""

import logging
import os
from typing import Generator

import torch
from verl.workers.rollout.vllm_rollout import ServerAdapter

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class VLLMOmniServerAdapter(ServerAdapter):
    """verl's vLLM ``ServerAdapter`` plus the ``delta_flush`` wire format.

    The ``omni_delta_sharded`` checkpoint engine's ``receive_weights`` yields
    per-flush sparse payloads already resident on this worker's GPU. Each flush
    is its own ``update_verl_delta_weights`` RPC with its own bucketed stream,
    so flush boundaries never share bucket metadata and ``is_last`` stays with
    its flush. The receive side is the omni worker extension
    (:class:`vLLMOmniColocateWorkerExtension`); the send loop below mirrors
    verl's ``ServerAdapter._update_delta_weights`` protocol (verl#7227) so this
    override can be deleted once the verl pin carries that commit. The
    ``named_tensors`` path is verl's, untouched.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._delta_weight_transfer_engine_initialized = False

    @torch.no_grad()
    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int = None,
        wire_format: str = "named_tensors",
        **kwargs,
    ):
        if wire_format != "delta_flush":
            return await super().update_weights(weights, global_steps=global_steps, wire_format=wire_format, **kwargs)
        return await self._update_delta_weights(weights, global_steps=global_steps)

    async def _update_delta_weights(
        self,
        weights,
        *,
        global_steps: int | None,
    ) -> None:
        """Send one delta weight update as a stream of DeltaFlush payloads."""

        from verl.workers.rollout.utils import ensure_async_iterator
        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender

        if self.use_shm:
            raise NotImplementedError("omni_delta_sharded with vllm-omni requires colocated CUDA IPC")

        flushes = ensure_async_iterator(weights)
        try:
            first_item = await anext(flushes)
        except StopAsyncIteration:
            # A steady sync with no changed BF16 values sends only a terminal
            # marker: weights are unchanged, so advance the step tag without
            # touching the rollout or the KV cache.
            if global_steps is not None and self._ensure_server_handle():
                await self.server_handle.set_global_steps.remote(global_steps)
            return

        first_named_tensors, saw_last = first_item
        if not self._delta_weight_transfer_engine_initialized:
            await self._execute_method(
                "init_weight_transfer_engine",
                kwargs={"init_info": {}},
            )
            self._delta_weight_transfer_engine_initialized = True

        await self._execute_method("start_weight_update")

        async def send_flush(flush_tensors: list[tuple[str, torch.Tensor]]) -> None:
            receiver_future = await self._execute_method(
                "update_verl_delta_weights",
                non_block=True,
                kwargs={"update_info": {}},
            )
            sender = BucketedWeightSender(
                zmq_handle=self.zmq_handle,
                bucket_size_mb=self.config.checkpoint_engine.update_weights_bucket_megabytes,
                use_shm=False,
            )
            await sender.async_send_weights(iter(flush_tensors))
            if receiver_future is not None:
                await receiver_future

        await send_flush(list(first_named_tensors))
        async for named_tensors, is_last in flushes:
            if saw_last:
                raise ValueError("DeltaFlush stream yielded data after is_last=True")
            saw_last = is_last
            await send_flush(list(named_tensors))

        if not saw_last:
            raise ValueError("DeltaFlush stream ended without is_last=True")

        await self._execute_method("finish_weight_update")

        if self._has_server:
            await self.server_handle.clear_kv_cache.remote()
            if global_steps is not None:
                await self.server_handle.set_global_steps.remote(global_steps)

        if self.replica_rank == 0 and self.rollout_rank == 0:
            logger.info("delta update_weights done, global_steps=%s", global_steps)

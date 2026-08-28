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
import logging
import os
import sys
import time
from pathlib import Path

import torch
from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID, VLLM_LORA_NAME, VLLM_LORA_PATH, set_death_signal
from vllm_omni.diffusion.worker.diffusion_worker import CustomPipelineWorkerExtension

from verl_omni.utils.vllm_omni import OmniTensorLoRARequest, VLLMOmniHijack
from verl_omni.workers.rollout.vllm_rollout.zmq_utils import make_update_zmq_handle

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Linux maps file; tests may patch this.
_PROC_SELF_MAPS = "/proc/self/maps"


def torchvision_bundled_cudart_is_mapped() -> bool:
    """True when torchvision's private libcudart is mapped into this process."""
    try:
        with open(_PROC_SELF_MAPS, encoding="utf-8") as maps:
            for line in maps:
                if "libcudart" in line and "torchvision" in line:
                    return True
    except OSError:
        pass
    tv = sys.modules.get("torchvision")
    if tv is None:
        return False
    tv_file = getattr(tv, "__file__", None)
    if not tv_file:
        return False
    parent = Path(tv_file).resolve().parent
    for libs in (parent / "libs", parent.parent / "torchvision.libs"):
        if libs.is_dir() and any(libs.glob("*cudart*")):
            return True
    return False


def is_diffusion_pipeline_worker(worker: object) -> bool:
    """True for vLLM-Omni diffusion workers (not AR / standard vLLM workers)."""
    model_runner = getattr(worker, "model_runner", None)
    if model_runner is not None and getattr(model_runner, "pipeline", None) is not None:
        return True
    return getattr(worker, "od_config", None) is not None


def should_skip_unsafe_cumem_sleep(worker: object) -> bool:
    """Skip CuMem offload when torchvision's libcudart would memcpy PyTorch pages."""
    return is_diffusion_pipeline_worker(worker) and torchvision_bundled_cudart_is_mapped()


def skipped_cumem_sleep_ack(task, worker: object):
    """Return a SUCCESS ACK so AsyncOmni does not treat a skipped sleep as engine death."""
    from vllm_omni.diffusion.data import OmniACK, OmniSleepTask

    if isinstance(task, dict):
        task = OmniSleepTask(**task)
    rank = getattr(worker, "rank", 0)
    if rank != 0:
        return None
    return OmniACK(
        task_id=task.task_id,
        status="SUCCESS",
        stage_id=getattr(worker, "stage_id", 0),
        rank=rank,
        freed_bytes=0,
        metadata={"skipped": "torchvision_libcudart"},
    )


def skipped_cumem_wake_ack(task, worker: object):
    """Return a SUCCESS ACK for a wake that is a no-op because sleep was skipped."""
    from vllm_omni.diffusion.data import OmniACK, OmniWakeTask

    if isinstance(task, dict):
        task = OmniWakeTask(**task)
    rank = getattr(worker, "rank", 0)
    if rank != 0:
        return None
    return OmniACK(
        task_id=task.task_id,
        status="SUCCESS",
        stage_id=getattr(worker, "stage_id", 0),
        rank=rank,
        freed_bytes=0,
        metadata={"state": "WARM", "skipped": "cumem_sleep_was_skipped"},
    )


class vLLMOmniColocateWorkerExtension(CustomPipelineWorkerExtension):
    """
    The class for vLLM-Omni's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.

    Feature support:
    1. LoRA
    2. NPU (Ascend) memory-pool, sleep, and wake_up — via NPUColocateWorkerMixin
    """

    _pending_lora_peft_config: dict | None = None
    _cumem_sleep_skipped: bool = False

    def __new__(cls, **kwargs):
        set_death_signal()

        # 1. patch for Lora
        VLLMOmniHijack.hijack()

        return super().__new__(cls)

    def sleep(self, level: int = 1):
        """Offload CuMem weights unless torchvision's libcudart would SIGSEGV."""
        if should_skip_unsafe_cumem_sleep(self):
            self._cumem_sleep_skipped = True
            logger.warning(
                "Skipping CuMem sleep: torchvision bundled libcudart is mapped. "
                "Offload would cudaMemcpy PyTorch CuMem pages through the wrong "
                "CUDA runtime (invalid permissions for mapped object). Rely on "
                "FSDP param_offload for colocated actor VRAM."
            )
            return 0
        self._cumem_sleep_skipped = False
        parent = getattr(super(), "sleep", None)
        if callable(parent):
            return parent(level)
        return 0

    def handle_sleep_task(self, task):
        """ACK-protocol sleep. Skip CuMem memcpy when torchvision libcudart is mapped."""
        if should_skip_unsafe_cumem_sleep(self):
            self._cumem_sleep_skipped = True
            logger.warning(
                "Skipping handle_sleep_task CuMem offload: torchvision bundled "
                "libcudart is mapped in this DiffusionWorker."
            )
            return skipped_cumem_sleep_ack(task, self)
        self._cumem_sleep_skipped = False
        parent = getattr(super(), "handle_sleep_task", None)
        if not callable(parent):
            raise TypeError("handle_sleep_task is not available on the worker MRO")
        return parent(task)

    def wake_up(self, tags: list[str] | None = None):
        """No-op when the matching sleep was skipped.

        vLLM 0.26's ``CuMemAllocator.wake_up`` re-creates every tagged
        allocation unconditionally (no per-allocation asleep guard). Waking
        after a skipped sleep therefore cuMemMaps already-mapped pages, which
        fails with "CUDA Error: invalid argument" and leaks the freshly
        created physical handle (vllm-project/vllm#36651).
        """
        if getattr(self, "_cumem_sleep_skipped", False):
            logger.info("Skipping wake_up: previous CuMem sleep was skipped, nothing to restore.")
            return True
        parent = getattr(super(), "wake_up", None)
        if callable(parent):
            return parent(tags)
        return True

    def handle_wake_task(self, task):
        """ACK-protocol wake. No-op when the matching sleep was skipped."""
        if getattr(self, "_cumem_sleep_skipped", False):
            logger.info("Skipping handle_wake_task: previous CuMem sleep was skipped.")
            return skipped_cumem_wake_ack(task, self)
        parent = getattr(super(), "handle_wake_task", None)
        if not callable(parent):
            raise TypeError("handle_wake_task is not available on the worker MRO")
        return parent(task)

    def is_worker_ready(self) -> bool:
        """Readiness probe used before hybrid CuMem sleep.

        ``LLMServerManager.create`` must not return until this RPC succeeds,
        otherwise v1 ``sleep_replicas`` can memcpy unmapped DiffusionWorker pages.
        """
        return True

    def set_pending_lora_peft_config(self, peft_config: dict | None = None):
        """Stash the actor's LoRA ``peft_config`` for the next
        ``update_weights_from_ipc`` call (separate-async NCCL path only).

        Called out-of-band via ``collective_rpc`` by
        ``OmniCheckpointEngineManager`` before the NCCL weight broadcast.
        ``update_weights_from_ipc`` consumes the stash when its ``peft_config``
        kwarg is absent (the standalone rollout path), then clears it so a
        later full-weight sync is not misrouted.
        """
        self._pending_lora_peft_config = peft_config

    def _get_standard_weight_model_and_config(self):
        """Return ``(model, model_config)`` for the standard (non-LoRA) AR weight path.

        Reaches the underlying vLLM model + ``ModelConfig`` via the worker's
        ``model_runner``. Returns ``None`` for workers without this chain (e.g. the
        diffusion pipeline worker), so the caller falls back to ``self.load_weights``.
        """
        model_runner = getattr(self, "model_runner", None)
        if model_runner is None:
            return None
        model = model_runner.get_model() if hasattr(model_runner, "get_model") else getattr(model_runner, "model", None)
        model_config = getattr(model_runner, "model_config", None)
        if model is not None and model_config is not None and hasattr(model, "load_weights"):
            return model, model_config
        return None

    def update_weights_from_ipc(
        self,
        peft_config: dict = None,
        base_sync_done=False,
        use_shm: bool = False,
        zmq_update_id: str | None = None,
    ):
        """Update the weights of the rollout model.

        For LoRA updates, all LoRA tensors are accumulated across buckets and loaded
        atomically via a single ``add_lora`` call, avoiding per-bucket partial loading.
        For full-weight updates, weights are streamed bucket-by-bucket via
        ``load_weights`` to keep GPU memory usage bounded.
        """

        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

        if peft_config is None and self._pending_lora_peft_config is not None:
            peft_config = self._pending_lora_peft_config
            base_sync_done = True
            # Consume the stash so a subsequent full-weight sync isn't misrouted.
            self._pending_lora_peft_config = None

        if self.device is None:
            raise RuntimeError("Worker device is not set.")
        zmq_handle = self._get_zmq_handle()
        if zmq_update_id is not None:
            zmq_handle = make_update_zmq_handle(zmq_handle, zmq_update_id)
        receiver = BucketedWeightReceiver(
            zmq_handle=zmq_handle,
            device=self.device,
            use_shm=use_shm,
        )

        if peft_config and base_sync_done:
            # In async mode, make sure the old lora is removed before adding the new one
            t0 = time.perf_counter()
            self.remove_lora(VLLM_LORA_INT_ID)
            t1 = time.perf_counter()
            logger.debug("remove_lora took %.3f ms", (t1 - t0) * 1000)

            # Accumulate all LoRA tensors across buckets (LoRA weights are small;
            # a single atomic ``add_lora`` is both correct for multi-bucket edge
            # cases and more efficient than per-bucket loading).
            t_recv_start = time.perf_counter()
            accumulated_weights: dict[str, torch.Tensor] = {}
            receiver.receive_weights(
                on_bucket_received=lambda weights, *args, **kwargs: accumulated_weights.update(weights)
            )
            t_recv_end = time.perf_counter()
            lora_total_bytes = sum(t.element_size() * t.numel() for t in accumulated_weights.values())
            logger.debug(
                "IPC receive took %.3f ms (%d params, %.2f MB)",
                (t_recv_end - t_recv_start) * 1000,
                len(accumulated_weights),
                lora_total_bytes / (1024 * 1024),
            )

            # AR (standard vLLM) workers go through verl's base VLLMHijack, which
            # dispatches on ``isinstance(req, TensorLoRARequest)``; diffusion workers
            # go through vllm-omni's DiffusionLoRAManager, which expects the
            # OmniLoRARequest-derived ``OmniTensorLoRARequest``. Pick by worker type.
            if self._get_standard_weight_model_and_config() is not None:
                from verl.utils.vllm.utils import TensorLoRARequest

                lora_request = TensorLoRARequest(
                    lora_name=VLLM_LORA_NAME,
                    lora_int_id=VLLM_LORA_INT_ID,
                    lora_path=VLLM_LORA_PATH,
                    peft_config=peft_config,
                    lora_tensors=accumulated_weights,
                )
            else:
                lora_request = OmniTensorLoRARequest(
                    lora_name=VLLM_LORA_NAME,
                    lora_int_id=VLLM_LORA_INT_ID,
                    lora_path=VLLM_LORA_PATH,
                    peft_config=peft_config,
                    lora_tensors=accumulated_weights,
                )
            t2 = time.perf_counter()
            self.add_lora(lora_request)
            t3 = time.perf_counter()
            logger.debug("add_lora took %.3f ms", (t3 - t2) * 1000)
            logger.debug(
                "LoRA update total: %.3f ms (remove=%.3f, recv=%.3f, add=%.3f)",
                (t3 - t0) * 1000,
                (t1 - t0) * 1000,
                (t_recv_end - t_recv_start) * 1000,
                (t3 - t2) * 1000,
            )
        else:
            # Full-weight path: stream bucket-by-bucket to bound GPU memory.
            logger.info("Loading standard weights (async)")
            standard = self._get_standard_weight_model_and_config()
            if standard is not None:
                # AR (standard vLLM) model: load each bucket via the low-level
                # model.load_weights (no per-bucket finalize), then run the single
                # post-load processing pass once all buckets are received.
                model, model_config = standard
                # Re-attach weight_loader on Ascend FusedMoE params via verl's
                # built-in patch (handles ACLGraph unwrap + SUPPORTED_MOE_MODELS
                # whitelist, which Qwen3-Omni is registered into via
                # patch_register_vllm_moe_model_weight_loader).
                from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

                patch_vllm_moe_model_weight_loader(model)
                receiver.receive_weights(
                    on_bucket_received=lambda weights, *args, **kwargs: model.load_weights(weights)
                )
                from vllm.model_executor.model_loader.utils import process_weights_after_loading

                process_weights_after_loading(model, model_config, self.device)
            else:
                # Diffusion pipeline worker: load via the pipeline. vllm-omni
                # 0.26 removed DiffusionWorker/DiffusionModelRunner.load_weights;
                # each pipeline exposes load_weights via AutoWeightsLoader.
                pipeline = getattr(getattr(self, "model_runner", None), "pipeline", None)
                if pipeline is not None and hasattr(pipeline, "load_weights"):
                    load_fn = pipeline.load_weights
                elif hasattr(self, "load_weights"):
                    load_fn = self.load_weights
                else:
                    raise RuntimeError("Diffusion pipeline worker has no load_weights-capable pipeline")
                receiver.receive_weights(on_bucket_received=lambda weights, *args, **kwargs: load_fn(weights))

    def _get_zmq_handle(self) -> str:
        """Get ZMQ handle for communication.
        Uses Ray job id + replica_rank + local_rank to form the handle so it
        matches the sender side regardless of CUDA_VISIBLE_DEVICES differences,
        avoids collisions when multiple replicas share the same node, and is
        unique per Ray job to avoid cross-job collisions on shared hosts. The
        job id is forwarded by the vLLMHttpServer actor as VERL_RAY_JOB_ID and
        inherited by this vLLM worker subprocess.
        """
        replica_rank = os.environ.get("VERL_REPLICA_RANK", "0")
        job_id = os.environ.get("VERL_RAY_JOB_ID", "0")
        return f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{replica_rank}-rank-{self.local_rank}.sock"

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
"""CPU checks: Qwen-Image T2I blocks torchvision CUDA; other models do not."""

from unittest.mock import patch

from vllm_omni.diffusion.worker.diffusion_worker import CustomPipelineWorkerExtension

from verl_omni.workers.rollout.vllm_rollout.utils import (
    _is_qwen_image_t2i_pipeline,
    vLLMOmniColocateWorkerExtension,
)


def test_qwen_image_t2i_is_detected_and_edit_is_not():
    assert _is_qwen_image_t2i_pipeline(
        "verl_omni.pipelines.qwen_image_dpo.vllm_omni_rollout_adapter.QwenImageDPOPipeline"
    )
    assert _is_qwen_image_t2i_pipeline(
        "verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter.QwenImageFlowGRPOPipeline"
    )
    assert _is_qwen_image_t2i_pipeline("vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image.QwenImagePipeline")
    assert not _is_qwen_image_t2i_pipeline(
        "verl_omni.pipelines.qwen_image_edit_flow_grpo.vllm_omni_rollout_adapter.QwenImageEditPipeline"
    )
    assert not _is_qwen_image_t2i_pipeline("vllm_omni.diffusion.models.sd3.pipeline_sd3.StableDiffusion3Pipeline")


def test_re_init_pipeline_blocks_torchvision_only_for_qwen_image_t2i():
    worker = object.__new__(vLLMOmniColocateWorkerExtension)

    with (
        patch("verl_omni.workers.rollout.vllm_rollout.utils._block_torchvision_cuda_runtime") as block,
        patch.object(CustomPipelineWorkerExtension, "re_init_pipeline", return_value=None) as parent,
    ):
        vLLMOmniColocateWorkerExtension.re_init_pipeline(
            worker,
            {"pipeline_class": "verl_omni.pipelines.qwen_image_dpo.vllm_omni_rollout_adapter.QwenImageDPOPipeline"},
        )
        block.assert_called_once()
        parent.assert_called_once()

        block.reset_mock()
        parent.reset_mock()
        vLLMOmniColocateWorkerExtension.re_init_pipeline(
            worker,
            {"pipeline_class": "vllm_omni.diffusion.models.sd3.pipeline_sd3.StableDiffusion3Pipeline"},
        )
        block.assert_not_called()
        parent.assert_called_once()

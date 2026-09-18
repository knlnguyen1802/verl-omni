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
"""FSDP engine for omni models, registered as ``model_type="omni_model"``."""

import logging
import warnings

import torch
from transformers import AutoModelForMultimodalLM
from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import (
    get_init_weight_context_manager,
)
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

from verl_omni.workers.config import OmniModelConfig
from verl_omni.workers.engine.lora_export import LoRAExportMixin

logger = logging.getLogger(__name__)


@EngineRegistry.register(model_type="omni_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class OmniFSDPEngine(LoRAExportMixin, FSDPEngineWithLMHead):
    """FSDP engine for omni models"""

    def prepare_model_inputs(self, micro_batch):
        """Prepare standard LM inputs, then add model-native replay fields."""
        model_inputs, output_args = super().prepare_model_inputs(micro_batch)
        if not hasattr(self, "model_adapter_cls"):
            raise RuntimeError("Omni model inputs cannot be prepared before the model adapter is initialized.")
        model_inputs = self.model_adapter_cls.prepare_model_inputs(model_inputs, micro_batch, self.model_config)
        if not isinstance(model_inputs, dict):
            raise TypeError(
                f"OmniModelBase.prepare_model_inputs must return a dict, got {type(model_inputs).__name__}."
            )
        return model_inputs, output_args

    def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, **kwargs):
        per_tensor_param, peft_config = self.export_for_sync(
            layered_summon=layered_summon,
            base_sync_done=base_sync_done,
            adapter_name=kwargs.get("adapter_name", "default"),
            # First sync into a LoRA-enabled vLLM renames base keys to the
            # ``base_layer`` form the LoRA-wrapped AR modules expect.
            rename_base_layers=True,
        )

        if self._qat_enabled:
            from verl.utils.qat.quantizer import QATQuantizer
            from verl.utils.torch_dtypes import PrecisionType

            mixed_precision_config = self.engine_config.mixed_precision
            if mixed_precision_config is not None:
                param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            else:
                param_dtype = torch.bfloat16

            quantizer = QATQuantizer(
                mode=self._qat_config.mode,
                group_size=self._qat_config.group_size,
                ignore_patterns=list(self._qat_config.ignore_patterns),
                device=torch.device(get_device_id()),
                param_dtype=param_dtype,
            )
            per_tensor_param = quantizer.quantize_with_fusion(
                per_tensor_param,
                target_device=torch.device("cpu"),
            )

        return per_tensor_param, peft_config

    def _merged_lora_per_tensor_param(self):
        """Stream materialized merged weights before restoring the actor."""
        return self._merged_export()

    def _build_module(self):
        unsupported_options = [
            option for option in ("use_liger", "use_fused_kernels") if getattr(self.model_config, option, False)
        ]
        if unsupported_options:
            enabled_options = ", ".join(f"{option}=True" for option in unsupported_options)
            raise ValueError(
                f"Omni models do not support these enabled optimizations: {enabled_options}. "
                "Set them to false before starting the worker."
            )

        from verl.utils.torch_dtypes import PrecisionType

        from verl_omni.pipelines.model_base import OmniModelBase

        self.model_config: OmniModelConfig
        architecture = self.model_config.architecture
        adapter_cls = OmniModelBase.get_class_by_name(
            architecture,
            self.model_config.model_stage,
            self.model_config.get("external_lib"),
        )
        self.model_adapter_cls = adapter_cls

        torch_dtype = self.engine_config.model_dtype

        if torch_dtype is None:
            torch_dtype = torch.float32 if not self.engine_config.forward_only else torch.bfloat16

        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # Use the stage sub-config for the meta-tensor decision; fall back to the umbrella config.
        stage_config = getattr(
            self.model_config.hf_config, f"{self.model_config.model_stage}_config", self.model_config.hf_config
        )
        tie_word_embeddings = getattr(stage_config, "tie_word_embeddings", False)
        if not hasattr(self.model_config.hf_config, "tie_word_embeddings"):
            self.model_config.hf_config.tie_word_embeddings = tie_word_embeddings

        init_context = get_init_weight_context_manager(use_meta_tensor=not tie_word_embeddings, mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")

            auto_model_cls = getattr(adapter_cls, "auto_model_class", None) or AutoModelForMultimodalLM
            module = auto_model_cls.from_pretrained(
                pretrained_model_name_or_path=self.model_config.local_path,
                torch_dtype=torch_dtype,
                config=self.model_config.hf_config,
                trust_remote_code=self.model_config.trust_remote_code,
            )
            module = adapter_cls.configure_model(module, self.model_config)

            if self.engine_config.strategy == "fsdp" and not self.engine_config.use_orig_params:
                trainability = {parameter.requires_grad for parameter in module.parameters()}
                if len(trainability) > 1:
                    raise ValueError(
                        "FSDP1 requires use_orig_params=true when a model adapter freezes only part of the model."
                    )

            module.to(torch_dtype)

            if self.model_config.enable_gradient_checkpointing:
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return module

    def _build_lora_module(self, module):
        module = super()._build_lora_module(module)

        lora_dtype = getattr(self.model_config, "lora_dtype", None)
        if lora_dtype is not None:
            from peft.tuners.tuners_utils import BaseTunerLayer
            from verl.utils.torch_dtypes import PrecisionType

            target_dtype = PrecisionType.to_dtype(lora_dtype)
            for name, param in module.named_parameters():
                if param.requires_grad:
                    orig_dtype = param.dtype
                    param.data = param.data.to(target_dtype)
                    logger.debug("LoRA param %s: %s -> %s", name, orig_dtype, param.dtype)

            for submodule in module.modules():
                if isinstance(submodule, BaseTunerLayer):
                    submodule.cast_input_dtype_enabled = False

        return module

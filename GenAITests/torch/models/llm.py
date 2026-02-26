# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Generic Torch LLM class"""

import warnings
import torch

from aimet_torch.common.defs import QuantScheme
from aimet_torch.onnx_utils import map_torch_types_to_onnx
from aimet_torch.v2.nn.true_quant import QuantizationMixin
from aimet_torch import QuantizationSimModel

from GenAITests.shared.helpers.yaml_config_parser import YAMLConfigParser
from GenAITests.shared.models.base import SimCollection
from GenAITests.shared.models.utils.model_utils import ONNXExportableModuleWithCache
from GenAITests.shared.models.base import LLM


@YAMLConfigParser.register_default_llm
class LLM_Torch(LLM):
    """Generic LLM for AIMET-Torch quantization."""

    @classmethod
    def instantiate_quantsim(
        cls,
        model_id: str,
        context_length: int,
        sequence_length: int,
        small_model: bool = False,
        dtype: torch.dtype = torch.float32,
        kv_bits: int = 8,
        *args,
        **kwargs,
    ) -> SimCollection:
        warnings.warn(
            f"kv_bits parameter (value: {kv_bits}) is ignored in Torch GenAI framework. "
            f"KV Cache quantization is not simulated. If you would like this setting to be "
            f"simulated on your model, please enable eval_in_onnx in your config file."
        )

        model = cls.instantiate_model(model_id, small_model)
        model = model.to(dtype=dtype)

        # Wrap model to enable JIT trace
        traceable_model = ONNXExportableModuleWithCache(model)
        quantsim = QuantizationSimModel(
            model=traceable_model,
            quant_scheme=QuantScheme.post_training_tf,
            dummy_input=cls.get_sample_backbone_inputs(
                traceable_model, context_length, sequence_length
            ),
            default_output_bw=16,
            default_param_bw=4,
            in_place=True,
            config_file=cls.get_quantsim_config(),
        )

        # Configure bitwidths
        quantsim.model.model.lm_head.param_quantizers["weight"].bitwidth = 8
        for module in quantsim.model.modules():
            if cls._is_quantized_rms_norm(module):
                module.param_quantizers["weight"].bitwidth = 16

        return SimCollection(quantsim)

    @staticmethod
    def _is_quantized_rms_norm(module: torch.nn.Module) -> bool:
        """Check if the given module is a quantized RMSNormalization layer."""
        return (
            isinstance(module, QuantizationMixin)
            and map_torch_types_to_onnx.get(type(module), "") == "RMSNormalization"
        )

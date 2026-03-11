# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Generic Torch LLM class"""

from __future__ import annotations

import torch

from aimet_torch.common.defs import QuantScheme, float16, float32
from aimet_torch.onnx_utils import map_torch_types_to_onnx
from aimet_torch.v2.nn.true_quant import QuantizationMixin
from aimet_torch.v2.utils import remove_activation_quantizers
from aimet_torch import QuantizationSimModel

from GenAITests.shared.helpers.precision_config import PrecisionConfig
from GenAITests.shared.helpers.yaml_config_parser import YAMLConfigParser
from GenAITests.shared.models.base import SimCollection
from GenAITests.shared.models.utils.model_utils import ONNXExportableModuleWithCache
from GenAITests.shared.models.base import LLM

from GenAITests.torch.models.utils.quantsim_utils import (
    _apply_block_granularity_to_decoder_stack,
    _set_lm_head_precision,
)


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
        precision: PrecisionConfig | None = None,
        *args,
        **kwargs,
    ) -> SimCollection:
        if precision is None:
            precision = PrecisionConfig()

        model = cls.instantiate_model(model_id, small_model)
        model = model.to(dtype=dtype)

        default_param_bw = precision.blocks["default"].qtype.bits
        default_output_bw = (
            precision.activations.bits
            if precision.activations not in (float16, float32)
            else 16
        )

        # Wrap model to enable JIT trace
        traceable_model = ONNXExportableModuleWithCache(model)
        quantsim = QuantizationSimModel(
            model=traceable_model,
            quant_scheme=QuantScheme.post_training_tf,
            dummy_input=cls.get_sample_backbone_inputs(
                traceable_model, context_length, sequence_length
            ),
            default_output_bw=default_output_bw,
            default_param_bw=default_param_bw,
            in_place=True,
            config_file=cls.get_quantsim_config(),
        )

        if precision.activations in (float16, float32):
            remove_activation_quantizers(quantsim.model)

        # Configure RMS Norm weights to 16-bits
        for module in quantsim.model.modules():
            if cls._is_quantized_rms_norm(module):
                module.param_quantizers["weight"].bitwidth = 16

        # Set LM Head precision if specified
        _set_lm_head_precision(quantsim, precision.lm_head)
        # Apply block-level granularity (LPBQ/BQ) if configured
        _apply_block_granularity_to_decoder_stack(quantsim, precision)

        return SimCollection(quantsim)

    @staticmethod
    def _is_quantized_rms_norm(module: torch.nn.Module) -> bool:
        """Check if the given module is a quantized RMSNormalization layer."""
        return (
            isinstance(module, QuantizationMixin)
            and map_torch_types_to_onnx.get(type(module), "") == "RMSNormalization"
        )

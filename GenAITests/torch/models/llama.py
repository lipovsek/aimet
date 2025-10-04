# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Llama model class"""

import torch

from aimet_common.defs import QuantScheme
from aimet_torch import QuantizationSimModel

from aimet_torch.v2.nn.transformers.models.llama.modeling_llama import (
    QuantizedLlamaRMSNorm,
)

from GenAITests.shared.helpers.yaml_config_parser import YAMLConfigParser
from GenAITests.shared.models.generator import Generator
from GenAITests.shared.models.utils.model_utils import ONNXExportableModuleWithCache
from GenAITests.shared.models.llama import Llama_32


@YAMLConfigParser.register_model
class Llama_32_Torch(Llama_32):
    """Generic LLaMa 3.2 for AIMET-Torch"""

    @classmethod
    def instantiate_quantsim(
        cls,
        model_id: str,
        context_length: int,
        sequence_length: int,
        small_model: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> QuantizationSimModel:
        model = cls.instantiate_model(model_id, small_model)
        model = model.to(dtype=dtype)

        # Need to wrap model in this in order to enable JIT trace
        traceable_model = ONNXExportableModuleWithCache(model)

        dummy_input_ids = torch.zeros((1, sequence_length), dtype=torch.int)
        dummy_attention_mask = torch.ones((1, sequence_length), dtype=torch.int)

        assembled_dummy_inputs = Generator.prepare_inputs(
            model=traceable_model,
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            past_key_values=[],
            context_length=context_length,
            sequence_length=sequence_length,
        )

        quantsim = QuantizationSimModel(
            model=traceable_model,
            quant_scheme=QuantScheme.post_training_tf,
            dummy_input=assembled_dummy_inputs,
            default_output_bw=16,
            default_param_bw=4,
            in_place=True,
            config_file=cls.get_quantsim_config(),
        )

        quantsim.model.model.lm_head.param_quantizers["weight"].bitwidth = 8
        for _, module in quantsim.model.named_modules():
            if isinstance(module, QuantizedLlamaRMSNorm):
                module.param_quantizers["weight"].bitwidth = 16

        return quantsim

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen 3 model class"""

import warnings
import torch

from aimet_torch.common.defs import QuantScheme
from aimet_torch import QuantizationSimModel
from aimet_torch.v2.nn.transformers.models.qwen3.modeling_qwen3 import (
    QuantizedQwen3RMSNorm,
)

from GenAITests.shared.helpers.yaml_config_parser import YAMLConfigParser
from GenAITests.shared.models.base import SimCollection
from GenAITests.shared.models.qwen3 import (
    Qwen_3,
    Qwen_3_SHA_Mixin,
    Qwen_3_SHA_Conv_Mixin,
)
from GenAITests.shared.models.utils.model_utils import ONNXExportableModuleWithCache


@YAMLConfigParser.register_model
class Qwen_3_Torch(Qwen_3):
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

        # Need to wrap model in this in order to enable JIT trace
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

        quantsim.model.model.lm_head.param_quantizers["weight"].bitwidth = 8
        for _, module in quantsim.model.named_modules():
            if isinstance(module, QuantizedQwen3RMSNorm):
                module.param_quantizers["weight"].bitwidth = 16

        return SimCollection(quantsim)


@YAMLConfigParser.register_model
class Qwen_3_SHA_Torch(Qwen_3_SHA_Mixin, Qwen_3_Torch):
    pass


@YAMLConfigParser.register_model
class Qwen_3_SHA_Conv_Torch(Qwen_3_SHA_Conv_Mixin, Qwen_3_Torch):
    pass

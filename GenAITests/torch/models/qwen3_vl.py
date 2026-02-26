# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen-2.5-VL model class"""

import warnings
import torch

from aimet_torch.common.defs import QuantScheme
from aimet_torch import QuantizationSimModel
from aimet_torch.v2.nn.true_quant import QuantizationMixin
from aimet_torch.v2.nn.transformers.models.qwen3_vl.modeling_qwen3_vl import (
    QuantizedQwen3VLTextRMSNorm,
)

from GenAITests.shared.helpers.yaml_config_parser import YAMLConfigParser
from GenAITests.shared.models.base import SimCollection
from GenAITests.shared.models.qwen3_vl import (
    Qwen_3_VL,
    Qwen3VLVisualWrapper,
)
from GenAITests.shared.models.utils.model_utils import ONNXExportableModuleWithCache


@YAMLConfigParser.register_model("qwen3_vl")
class Qwen_3_VL_Torch(Qwen_3_VL):
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

        # 1) Wrap LLM model to make it traceable
        # Need to wrap model in this in order to enable JIT trace
        traceable_backbone = ONNXExportableModuleWithCache(
            model.model.language_model,
            lm_head=model.lm_head,
            use_inputs_embeds=True,
            extra_input_names=cls.get_visual_output_names()[1:],
        )
        language_sim = QuantizationSimModel(
            model=traceable_backbone,
            quant_scheme=QuantScheme.post_training_tf,
            dummy_input=cls.get_sample_backbone_inputs(
                traceable_backbone,
                context_length=context_length,
                sequence_length=sequence_length,
            ),
            default_output_bw=16,
            default_param_bw=4,
            in_place=True,
            config_file=cls.get_quantsim_config(),
        )

        language_sim.model.lm_head.param_quantizers["weight"].bitwidth = 8
        for _, module in language_sim.model.named_modules():
            if isinstance(module, QuantizedQwen3VLTextRMSNorm):
                module.param_quantizers["weight"].bitwidth = 16

        # 2) Wrap visual model to make it traceable
        traceable_visual = Qwen3VLVisualWrapper(model.model.visual)
        visual_sim = QuantizationSimModel(
            model=traceable_visual,
            quant_scheme=QuantScheme.post_training_tf,
            dummy_input=cls.get_sample_vision_inputs(
                model.config.vision_config.hidden_size
            ),
            default_output_bw=16,
            default_param_bw=4,
            in_place=True,
            config_file=cls.get_quantsim_config(),
        )

        # 3) Convert embedding table to quantized equivalent
        if not isinstance(model.model.language_model.embed_tokens, QuantizationMixin):
            model.model.language_model.embed_tokens = QuantizationMixin.from_module(
                model.model.language_model.embed_tokens
            )

        return SimCollection(
            language_sim,
            visual=visual_sim,
            embedding=model.model.language_model.embed_tokens,
            config=model.config,
            position_id_processor=cls.generate_position_ids,
        )

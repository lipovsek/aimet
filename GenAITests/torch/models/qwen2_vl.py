# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen-2.5-VL model class"""

from __future__ import annotations

import torch

from aimet_torch.common.defs import QuantScheme, float16, float32
from aimet_torch import QuantizationSimModel
from aimet_torch.v2.nn.true_quant import QuantizationMixin
from aimet_torch.v2.utils import remove_activation_quantizers
from aimet_torch.v2.nn.transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    QuantizedQwen2_5_VLRMSNorm,
)

from GenAITests.shared.helpers.precision_config import PrecisionConfig
from GenAITests.shared.helpers.yaml_config_parser import YAMLConfigParser
from GenAITests.shared.models.base import SimCollection
from GenAITests.shared.models.qwen2_vl import (
    Qwen_25_VL,
    Qwen2VLVisualWrapper,
)
from GenAITests.shared.models.utils.model_utils import ONNXExportableModuleWithCache
from GenAITests.torch.models.utils.quantsim_utils import (
    _apply_block_granularity_to_decoder_stack,
    _set_lm_head_precision,
)


@YAMLConfigParser.register_model("qwen2_5_vl")
class Qwen_25_VL_Torch(Qwen_25_VL):
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

        # 1) Wrap LLM model to make it traceable
        # Need to wrap model in this in order to enable JIT trace
        traceable_backbone = ONNXExportableModuleWithCache(
            model.model.language_model,
            lm_head=model.lm_head,
            use_inputs_embeds=True,
        )
        language_sim = QuantizationSimModel(
            model=traceable_backbone,
            quant_scheme=QuantScheme.post_training_tf,
            dummy_input=cls.get_sample_backbone_inputs(
                traceable_backbone,
                context_length=context_length,
                sequence_length=sequence_length,
            ),
            default_output_bw=default_output_bw,
            default_param_bw=default_param_bw,
            in_place=True,
            config_file=cls.get_quantsim_config(),
        )

        if precision.activations in (float16, float32):
            remove_activation_quantizers(language_sim.model)

        for _, module in language_sim.model.named_modules():
            if isinstance(module, QuantizedQwen2_5_VLRMSNorm):
                module.param_quantizers["weight"].bitwidth = 16

        # Set LM Head precision if specified
        _set_lm_head_precision(language_sim, precision.lm_head)
        # Apply block-level granularity (LPBQ/BQ) if configured
        _apply_block_granularity_to_decoder_stack(language_sim, precision)

        # 2) Wrap visual model to make it traceable
        visual_param_bw = (
            precision.visual_weight.qtype.bits
            if precision.visual_weight
            else default_param_bw
        )
        visual_output_bw = (
            precision.visual_activations.bits
            if precision.visual_activations is not None
            and precision.visual_activations not in (float16, float32)
            else default_output_bw
        )
        traceable_visual = Qwen2VLVisualWrapper(model.model.visual)
        visual_sim = QuantizationSimModel(
            model=traceable_visual,
            quant_scheme=QuantScheme.post_training_tf,
            dummy_input=cls.get_sample_vision_inputs(
                model.config.vision_config.hidden_size
            ),
            default_output_bw=visual_output_bw,
            default_param_bw=visual_param_bw,
            in_place=True,
            config_file=cls.get_quantsim_config(),
        )

        visual_activation_qtype = (
            precision.visual_activations
            if precision.visual_activations
            else precision.activations
        )
        if visual_activation_qtype in (float16, float32):
            remove_activation_quantizers(visual_sim.model)

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

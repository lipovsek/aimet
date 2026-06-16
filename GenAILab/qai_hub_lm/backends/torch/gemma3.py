# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Gemma3 Torch model class"""

from __future__ import annotations

import torch

from aimet_torch.common.defs import QuantScheme
from aimet_torch import QuantizationSimModel
from aimet_torch.v2.utils import remove_activation_quantizers
from aimet_torch.nn.transformers.models.gemma3.modeling_gemma3 import (
    QuantizedGemma3RMSNorm,
)

from GenAILab.qai_hub_lm.backends import QUANTSIM_CONFIG
from GenAILab.qai_hub_lm.precision import PrecisionConfig, float16, float32
from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.qai_hub_lm.models.base import SimCollection
from GenAILab.qai_hub_lm.models.gemma3 import (
    Gemma3_VLM,
    Gemma3VisionWrapper,
)
from GenAILab.qai_hub_lm.models.utils.exportable import ONNXExportableModuleWithCache
from GenAILab.qai_hub_lm.models.utils.layer_cache import build_layer_cache_descriptors
from GenAILab.qai_hub_lm.backends.torch.quantsim_utils import (
    _apply_block_granularity_to_decoder_stack,
    _set_lm_head_precision,
)


@YAMLConfigParser.register_model("gemma3")
class Gemma3_Torch(Gemma3_VLM):
    """Gemma3 quantization (text backbone + vision tower)."""

    @classmethod
    def instantiate_quantsim(
        cls,
        model,
        context_length: int,
        sequence_length: int,
        precision: PrecisionConfig | None = None,
        image_size: tuple[int, int] | None = None,
        *args,
        **kwargs,
    ) -> SimCollection:
        if precision is None:
            precision = PrecisionConfig()
        precision.ensure_visual_defaults()

        default_param_bw = precision.blocks["default"].qtype.bits
        default_output_bw = (
            16
            if precision.activations in (float16, float32)
            else precision.activations.bits
        )

        # Backbone: wrap language model with static cache
        layer_cache_descs = build_layer_cache_descriptors(model.config.text_config)
        traceable_backbone = ONNXExportableModuleWithCache(
            model.model.language_model,
            lm_head=model.lm_head,
            cache_type=cls.get_cache_type(),
            input_names=cls.get_backbone_input_names(layer_cache_descs),
        )
        language_sim = QuantizationSimModel(
            model=traceable_backbone,
            quant_scheme=QuantScheme.post_training_tf,
            dummy_input=cls.get_sample_backbone_inputs(
                traceable_backbone,
                context_length=context_length,
                sequence_length=sequence_length,
                layer_cache_descriptors=layer_cache_descs,
            ),
            default_output_bw=default_output_bw,
            default_param_bw=default_param_bw,
            in_place=True,
            config_file=QUANTSIM_CONFIG,
        )

        if precision.activations in (float16, float32):
            remove_activation_quantizers(language_sim.model)

        for _, module in language_sim.model.named_modules():
            if isinstance(module, QuantizedGemma3RMSNorm):
                module.param_quantizers["weight"].bitwidth = 16

        _set_lm_head_precision(
            language_sim, precision.lm_head, lm_head=language_sim.model.lm_head
        )
        _apply_block_granularity_to_decoder_stack(
            language_sim, precision, lm_head=language_sim.model.lm_head
        )

        # Vision encoder: vision_tower + multi_modal_projector
        visual_param_bw = precision.visual_weight.qtype.bits
        visual_output_bw = (
            16
            if precision.visual_activations in (float16, float32)
            else precision.visual_activations.bits
        )
        traceable_visual = Gemma3VisionWrapper(
            model.model.vision_tower, model.model.multi_modal_projector
        )
        visual_sim = QuantizationSimModel(
            model=traceable_visual,
            quant_scheme=QuantScheme.post_training_tf,
            dummy_input=cls.get_sample_vision_inputs(
                model.config, image_size=image_size
            ),
            default_output_bw=visual_output_bw,
            default_param_bw=visual_param_bw,
            in_place=True,
            config_file=QUANTSIM_CONFIG,
        )

        if precision.visual_activations in (float16, float32):
            remove_activation_quantizers(visual_sim.model)

        return SimCollection(
            backbone=language_sim,
            visual=visual_sim,
            embedding=model.model.language_model.embed_tokens,
            config=model.config,
        )

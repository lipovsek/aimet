# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen-3-VL model class"""

from __future__ import annotations

import torch

from aimet_torch.common.defs import QuantScheme
from aimet_torch import QuantizationSimModel
from aimet_torch.v2.nn.true_quant import QuantizationMixin
from aimet_torch.v2.utils import remove_activation_quantizers
from aimet_torch.v2.nn.transformers.models.qwen3_vl.modeling_qwen3_vl import (
    QuantizedQwen3VLTextRMSNorm,
)

from GenAILab.qai_hub_lm.backends import QUANTSIM_CONFIG
from GenAILab.qai_hub_lm.precision import PrecisionConfig, float16, float32
from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.qai_hub_lm.models.base import SimCollection
from GenAILab.qai_hub_lm.models.qwen3_vl import (
    Qwen_3_VL,
    Qwen3VLVisualWrapper,
)
from GenAILab.qai_hub_lm.models.utils.exportable import ONNXExportableModuleWithCache
from GenAILab.qai_hub_lm.models.utils.layer_cache import build_layer_cache_descriptors
from GenAILab.qai_hub_lm.backends.torch.quantsim_utils import (
    _apply_block_granularity_to_decoder_stack,
    _set_lm_head_precision,
)


@YAMLConfigParser.register_model("qwen3_vl")
class Qwen_3_VL_Torch(Qwen_3_VL):
    @classmethod
    def instantiate_quantsim(
        cls,
        model,
        context_length: int,
        sequence_length: int | list[int],
        precision: PrecisionConfig | None = None,
        image_size: tuple[int, int] | None = None,
        *args,
        **kwargs,
    ) -> SimCollection:
        if precision is None:
            precision = PrecisionConfig()
        precision.ensure_visual_defaults()

        max_sequence_length = (
            max(sequence_length)
            if isinstance(sequence_length, list)
            else sequence_length
        )

        default_param_bw = precision.blocks["default"].qtype.bits
        default_output_bw = (
            16
            if precision.activations in (float16, float32)
            else precision.activations.bits
        )

        # 1) Wrap LLM model to make it traceable
        layer_cache_descs = build_layer_cache_descriptors(model.config)
        traceable_backbone = ONNXExportableModuleWithCache(
            model.model.language_model,
            lm_head=model.lm_head,
            cache_type=cls.get_cache_type(),
            input_names=cls.get_backbone_input_names(
                layer_cache_descs, config=model.config
            ),
        )
        language_sim = QuantizationSimModel(
            model=traceable_backbone,
            quant_scheme=QuantScheme.post_training_tf,
            dummy_input=cls.get_sample_backbone_inputs(
                traceable_backbone,
                context_length=context_length,
                sequence_length=max_sequence_length,
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
            if isinstance(module, QuantizedQwen3VLTextRMSNorm):
                module.param_quantizers["weight"].bitwidth = 16

        # Set LM Head precision if specified
        _set_lm_head_precision(
            language_sim, precision.lm_head, lm_head=language_sim.model.lm_head
        )
        # Apply block-level granularity (LPBQ/BQ) if configured
        _apply_block_granularity_to_decoder_stack(
            language_sim, precision, lm_head=language_sim.model.lm_head
        )

        # 2) Wrap visual model to make it traceable
        visual_param_bw = precision.visual_weight.qtype.bits
        visual_output_bw = (
            16
            if precision.visual_activations in (float16, float32)
            else precision.visual_activations.bits
        )
        traceable_visual = Qwen3VLVisualWrapper(model.model.visual)
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

        visual_activation_qtype = precision.visual_activations
        if visual_activation_qtype in (float16, float32):
            remove_activation_quantizers(visual_sim.model)

        # 3) Convert embedding table to quantized equivalent
        # Note: encodings are computed after recipe application (in the test
        # runner). The weights themselves are already rotated when SpinQuant ran
        # on the float model before this sim was built.
        quantized_embedding = precision.embedding not in (float16, float32)
        if quantized_embedding and not isinstance(
            model.model.language_model.embed_tokens, QuantizationMixin
        ):
            model.model.language_model.embed_tokens = QuantizationMixin.from_module(
                model.model.language_model.embed_tokens
            )
            model.model.language_model.embed_tokens.param_quantizers[
                "weight"
            ].bitwidth = precision.embedding.bits

        return SimCollection(
            backbone=language_sim,
            visual=visual_sim,
            embedding=model.model.language_model.embed_tokens,
            config=model.config,
            position_id_processor=cls.generate_position_ids,
        )

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen-2.5-VL model class"""

import gc
import torch
import os
from transformers import AutoConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

from aimet_onnx import quantsim
from aimet_onnx.quantsim import QuantizationSimModel

from GenAITests.shared.models.base import SimCollection
from GenAITests.shared.helpers.yaml_config_parser import YAMLConfigParser
from GenAITests.shared.models.qwen2_vl import (
    Qwen_25_VL,
    VisualWrapper,
    Qwen2_5_VL_FastExportable_Mixin,
)
from GenAITests.shared.models.generator import VLM_Generator
from GenAITests.shared.models.utils.model_utils import ONNXExportableBackboneWithCache

from GenAITests.onnx.models.utils.torch_onnx_export_utils import (
    get_onnx_model,
    load_model_components_from_disk,
    get_model_checkpoint_path,
    is_huggingface_ckpt,
)
from GenAITests.onnx.models.utils.quantsim_utils import (
    _set_tensors_to_output_n_bit_symmmetric,
    _tie_quantizers_for_kv_cache,
    _set_lm_head_to_8b,
    get_ort_providers,
    AttributePatch,
)


@YAMLConfigParser.register_model
class Qwen_25_VL_ONNX(Qwen_25_VL):
    @classmethod
    def instantiate_quantsim(
        cls,
        model_id: str,
        context_length: int,
        sequence_length: int,
        small_model: bool = False,
        kv_bits: int = 8,
        *args,
        **kwargs,
    ):
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID

        if is_huggingface_ckpt(model_id):
            model = cls.instantiate_model(model_id, small_model).to(dtype=torch.float32)
            config = model.config

            traceable_backbone = ONNXExportableBackboneWithCache(
                model.model.language_model, model.lm_head
            )
            traceable_visual = VisualWrapper(model.model.visual)

            backbone_onnx_model, visual_onnx_model = get_onnx_model(
                checkpoint=get_model_checkpoint_path(model_id),
                fp_backbone_model=traceable_backbone,
                context_length=context_length,
                sequence_length=sequence_length,
                sample_input=cls.get_sample_backbone_inputs(
                    traceable_backbone, context_length, sequence_length
                ),
                input_names=VLM_Generator.get_input_names(
                    model.config.text_config.num_hidden_layers
                ),
                output_names=VLM_Generator.get_output_names(
                    model.config.text_config.num_hidden_layers
                ),
                fp_visual_model=traceable_visual,
                sample_visual_input=cls.get_sample_vision_inputs(config),
                visual_input_names=VLM_Generator.get_visual_input_names(),
                visual_output_names=VLM_Generator.get_visual_output_names(),
            )

            embedding = model.model.language_model.embed_tokens
        else:
            config = AutoConfig.from_pretrained(get_model_checkpoint_path(model_id))
            backbone_onnx_model, visual_onnx_model, embedding = (
                load_model_components_from_disk(
                    model_id,
                    context_length=context_length,
                    sequence_length=sequence_length,
                )
            )
            if visual_onnx_model is None or embedding is None:
                raise ValueError(
                    "Required model components could not be loaded from disk."
                )

        with (
            AttributePatch(quantsim, "op_types_to_tie_qtzrs", ["Concat"]),
            AttributePatch(quantsim, "_tie_qtzrs", True),
            AttributePatch(
                quantsim,
                "op_outputs_to_ignore",
                quantsim.op_outputs_to_ignore + ["Slice", "Constant"],
            ),
        ):
            backbone_quantsim = QuantizationSimModel(
                model=backbone_onnx_model,
                quant_scheme="min_max",
                default_activation_bw=16,
                default_param_bw=4,
                config_file=cls.get_quantsim_config(),
                providers=get_ort_providers(
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                ),
            )
            visual_quantsim = QuantizationSimModel(
                model=visual_onnx_model,
                quant_scheme="min_max",
                default_activation_bw=16,
                default_param_bw=4,
                config_file=cls.get_quantsim_config(),
                providers=get_ort_providers(
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                ),
            )

        # Setting kv_cache and some other layers to 8-bit
        _set_tensors_to_output_n_bit_symmmetric(backbone_quantsim, kv_bits)
        # Setting the LM head weights to 8-bit.
        _set_lm_head_to_8b(backbone_quantsim)
        # Tie kv_cache
        _tie_quantizers_for_kv_cache(backbone_quantsim)

        return SimCollection(
            backbone=backbone_quantsim,
            visual=visual_quantsim,
            embedding=embedding,
            config=config,
            position_id_processor=cls.generate_position_ids,
        )


@YAMLConfigParser.register_model
class Qwen_25_VL_FastExportable_ONNX(Qwen_25_VL_ONNX, Qwen2_5_VL_FastExportable_Mixin):
    pass

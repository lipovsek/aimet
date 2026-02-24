# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Llama ONNX model class"""

import torch
from transformers import AutoConfig

from aimet_onnx import quantsim
from aimet_onnx.quantsim import QuantizationSimModel

from GenAITests.shared.models.base import SimCollection
from GenAITests.shared.models.generator import HubCompatibleGenerator
from GenAITests.shared.helpers.yaml_config_parser import YAMLConfigParser
from GenAITests.shared.models.llama import Llama_32, Llama_32_SHA_Mixin
from GenAITests.shared.models.utils.model_utils import ONNXExportableModuleWithCache

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
class Llama_32_ONNX(Llama_32):
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
            exportable_model = ONNXExportableModuleWithCache(model)
            onnx_model, *_ = get_onnx_model(
                checkpoint=get_model_checkpoint_path(model_id),
                fp_backbone_model=exportable_model,
                context_length=context_length,
                sequence_length=sequence_length,
                sample_input=cls.get_sample_backbone_inputs(
                    exportable_model, context_length, sequence_length
                ),
                input_names=cls.get_backbone_input_names(
                    model.config.num_hidden_layers
                ),
                output_names=cls.get_backbone_output_names(
                    model.config.num_hidden_layers
                ),
            )
            config = model.config
        else:
            onnx_model, *_ = load_model_components_from_disk(
                model_id,
                context_length=context_length,
                sequence_length=sequence_length,
            )
            config = AutoConfig.from_pretrained(get_model_checkpoint_path(model_id))

        with (
            AttributePatch(quantsim, "op_types_to_tie_qtzrs", ["Concat"]),
            AttributePatch(quantsim, "_tie_qtzrs", True),
            AttributePatch(
                quantsim,
                "op_outputs_to_ignore",
                quantsim.op_outputs_to_ignore + ["Slice", "Constant"],
            ),
        ):
            quant_sim = QuantizationSimModel(
                model=onnx_model,
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
        _set_tensors_to_output_n_bit_symmmetric(quant_sim, kv_bits)
        # Setting the LM head weights to 8-bit.
        _set_lm_head_to_8b(quant_sim)
        # Tie kv_cache
        _tie_quantizers_for_kv_cache(quant_sim)

        return SimCollection(quant_sim, config=config)


@YAMLConfigParser.register_model
class Llama_32_SHA_ONNX(Llama_32_SHA_Mixin, Llama_32_ONNX):
    pass


@YAMLConfigParser.register_model
class Llama_32_AIHM_ONNX(Llama_32_ONNX):
    @classmethod
    def instantiate_model(cls, *args, **kwargs):
        raise RuntimeError("Please generate a quantized checkpoint using AIHM.")

    @staticmethod
    def get_generator_cls():
        return HubCompatibleGenerator

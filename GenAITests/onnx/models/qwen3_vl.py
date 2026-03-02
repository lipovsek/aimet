# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen-3-VL model class"""

import tempfile
import torch
from transformers import AutoConfig

from aimet_onnx import quantsim
from aimet_onnx.quantsim import QuantizationSimModel

from GenAITests.shared.helpers.model_cache import DiskBackedModelCache, ModelCacheEntry
from GenAITests.shared.models.base import SimCollection
from GenAITests.shared.helpers.yaml_config_parser import YAMLConfigParser
from GenAITests.shared.models.qwen3_vl import (
    Qwen_3_VL,
    Qwen3VLVisualWrapper,
)
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


@YAMLConfigParser.register_model("qwen3_vl")
class Qwen_3_VL_ONNX(Qwen_3_VL):
    @classmethod
    def instantiate_quantsim(
        cls,
        model_id: str,
        context_length: int,
        sequence_length: int,
        small_model: bool = False,
        kv_bits: int = 8,
        model_cache: DiskBackedModelCache | None = None,
        *args,
        **kwargs,
    ):
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID

        is_hf = is_huggingface_ckpt(model_id)

        if is_hf:
            if model_cache is not None:
                with tempfile.TemporaryDirectory() as tmpdir:
                    params = {
                        "model_id": model_id,
                        "class": cls.__name__,
                        "sequence_length": sequence_length,
                        "context_length": context_length,
                        "small_model": small_model,
                    }
                    key = DiskBackedModelCache.build_key(params)
                    entry = model_cache.get_or_export(
                        key,
                        lambda: cls._export_to_cache_entry(
                            model_id,
                            context_length,
                            sequence_length,
                            small_model,
                            tmpdir,
                        ),
                        metadata=params,
                    )
            else:
                entry = cls._export_to_cache_entry(
                    model_id,
                    context_length,
                    sequence_length,
                    small_model,
                    get_model_checkpoint_path(model_id),
                )
            backbone_onnx_model = entry.backbone
            visual_onnx_model = entry.visual
            embedding = entry.embedding
            config = entry.config
        else:
            config = AutoConfig.from_pretrained(model_id)
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

    @classmethod
    def _export_to_cache_entry(
        cls,
        model_id: str,
        context_length: int,
        sequence_length: int,
        small_model: bool,
        directory: str,
    ) -> ModelCacheEntry:
        """Export the torch model to ONNX and return a :class:`ModelCacheEntry`."""
        model = cls.instantiate_model(model_id, small_model).to(dtype=torch.float32)

        traceable_backbone = ONNXExportableModuleWithCache(
            model.model.language_model,
            lm_head=model.lm_head,
            use_inputs_embeds=True,
            extra_input_names=cls.get_visual_output_names()[1:],
        )
        traceable_visual = Qwen3VLVisualWrapper(model.model.visual)

        backbone_onnx_model, visual_onnx_model = get_onnx_model(
            checkpoint=directory,
            fp_backbone_model=traceable_backbone,
            context_length=context_length,
            sequence_length=sequence_length,
            sample_input=cls.get_sample_backbone_inputs(
                traceable_backbone, context_length, sequence_length
            ),
            input_names=cls.get_backbone_input_names(
                model.config.text_config.num_hidden_layers
            ),
            output_names=cls.get_backbone_output_names(
                model.config.text_config.num_hidden_layers
            ),
            fp_visual_model=traceable_visual,
            sample_visual_input=cls.get_sample_vision_inputs(model.config),
            visual_input_names=cls.get_visual_input_names(),
            visual_output_names=cls.get_visual_output_names(),
        )

        embedding = model.model.language_model.embed_tokens

        return ModelCacheEntry(
            backbone=backbone_onnx_model,
            visual=visual_onnx_model,
            embedding=embedding,
            config=model.config,
        )

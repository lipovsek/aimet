# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen-3-VL model class"""

from __future__ import annotations

import tempfile
import torch
from transformers import AutoConfig

from aimet_onnx import quantsim
from aimet_onnx.quantsim import QuantizationSimModel

from GenAITests.shared.helpers.model_cache import DiskBackedModelCache, ModelCacheEntry
from GenAITests.shared.helpers.precision_config import PrecisionConfig, float16, float32
from GenAITests.shared.models.base import SimCollection
from GenAITests.shared.helpers.yaml_config_parser import YAMLConfigParser
from GenAITests.shared.models.qwen3_vl import (
    Qwen_3_VL,
    Qwen3VLVisualWrapper,
)
from GenAITests.shared.models.utils.layer_cache import build_layer_cache_descriptors
from GenAITests.shared.models.utils.model_utils import ONNXExportableModuleWithCache

from GenAITests.onnx.models.utils.torch_onnx_export_utils import (
    get_onnx_model,
    load_model_components_from_disk,
    get_model_checkpoint_path,
    is_huggingface_ckpt,
)
from GenAITests.onnx.models.utils.quantsim_utils import (
    _resolve_kv_cache_quantization,
    _set_lm_head_precision,
    _apply_block_granularity_to_decoder_stack,
    _remove_activation_quantizers,
    quantize_embedding_weights,
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
        precision: PrecisionConfig | None = None,
        model_cache: DiskBackedModelCache | None = None,
        image_size: tuple[int, int] | None = None,
        *args,
        **kwargs,
    ):
        if precision is None:
            precision = PrecisionConfig()
        precision.ensure_visual_defaults()

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
                        "image_size": image_size,
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
                            image_size=image_size,
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
                    image_size=image_size,
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

        default_param_qtype = precision.blocks["default"].qtype
        default_activation_qtype = precision.activations
        visual_param_qtype = precision.visual_weight.qtype
        visual_activation_qtype = precision.visual_activations

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
                param_type=default_param_qtype,
                activation_type=default_activation_qtype,
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
                param_type=visual_param_qtype,
                activation_type=visual_activation_qtype,
                config_file=cls.get_quantsim_config(),
                providers=get_ort_providers(
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                ),
            )

        # Setting the LM head weights
        _set_lm_head_precision(backbone_quantsim, precision.lm_head)
        # Tie KV cache, and set quantization type
        _resolve_kv_cache_quantization(
            backbone_quantsim, precision.resolve_kv_cache_qtype()
        )
        # Apply block-level granularity (LPBQ/BQ) if configured
        _apply_block_granularity_to_decoder_stack(backbone_quantsim, precision)

        if default_activation_qtype in (float16, float32):
            _remove_activation_quantizers(backbone_quantsim)
        if visual_activation_qtype in (float16, float32):
            _remove_activation_quantizers(visual_quantsim)

        if precision.embedding not in (float16, float32):
            quantize_embedding_weights(embedding, precision.embedding.bits)

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
        image_size: tuple[int, int] | None = None,
    ) -> ModelCacheEntry:
        """Export the torch model to ONNX and return a :class:`ModelCacheEntry`."""
        model = cls.instantiate_model(model_id, small_model).to(dtype=torch.float32)

        traceable_backbone = ONNXExportableModuleWithCache(
            model.model.language_model,
            lm_head=model.lm_head,
            use_inputs_embeds=True,
            extra_input_names=cls.get_visual_output_names()[1:],
            cache_type=cls.get_cache_type(),
        )
        traceable_visual = Qwen3VLVisualWrapper(model.model.visual)

        backbone_onnx_model, visual_onnx_model = get_onnx_model(
            checkpoint=directory,
            fp_backbone_model=traceable_backbone,
            context_length=context_length,
            sequence_length=sequence_length,
            sample_input=cls.get_sample_backbone_inputs(
                traceable_backbone,
                context_length,
                sequence_length,
                layer_cache_descriptors=build_layer_cache_descriptors(
                    traceable_backbone.config
                ),
            ),
            input_names=cls.get_backbone_input_names(
                build_layer_cache_descriptors(traceable_backbone.config)
            ),
            output_names=cls.get_backbone_output_names(
                build_layer_cache_descriptors(traceable_backbone.config)
            ),
            fp_visual_model=traceable_visual,
            sample_visual_input=cls.get_sample_vision_inputs(
                model.config, image_size=image_size
            ),
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

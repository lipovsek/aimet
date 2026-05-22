# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Generic ONNX LLM class"""

from __future__ import annotations

import tempfile
import onnx
import torch
from transformers import AutoConfig

from aimet_onnx import quantsim
from aimet_onnx.quantsim import (
    QuantizationSimModel,
)

from GenAILab.qai_hub_lm.backends import QUANTSIM_CONFIG
from GenAILab.bench.model_cache import DiskBackedModelCache, ModelCacheEntry
from GenAILab.qai_hub_lm.precision import (
    PrecisionConfig,
    float16,
    float32,
    int8,
    int16,
)
from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.qai_hub_lm.models.base import LLM, SimCollection
from GenAILab.qai_hub_lm.models.utils.layer_cache import build_layer_cache_descriptors
from GenAILab.qai_hub_lm.models.utils.exportable import ONNXExportableModuleWithCache

from GenAILab.qai_hub_lm.backends.onnx.adaptations.hub_models import AIHMAdaptation
from GenAILab.qai_hub_lm.backends.onnx.export_utils import (
    get_onnx_model,
    load_model_components_from_disk,
    get_model_checkpoint_path,
    is_huggingface_ckpt,
)
from GenAILab.qai_hub_lm.backends.onnx.quantsim_utils import (
    _resolve_kv_cache_quantization,
    _set_lm_head_precision,
    _apply_block_granularity_to_decoder_stack,
    _remove_activation_quantizers,
    _remove_decoder_block_weight_quantizers,
    get_ort_providers,
    AttributePatch,
)


@YAMLConfigParser.register_default_llm
class LLM_ONNX(LLM):
    """Generic LLM for AIMET-ONNX quantization."""

    @classmethod
    def instantiate_quantsim(
        cls,
        model_id: str,
        context_length: int,
        sequence_length: int | list[int],
        small_model: bool = False,
        precision: PrecisionConfig | None = None,
        model_cache: DiskBackedModelCache | None = None,
        *args,
        **kwargs,
    ) -> SimCollection:
        if precision is None:
            precision = PrecisionConfig()

        max_sequence_length = (
            max(sequence_length)
            if isinstance(sequence_length, list)
            else sequence_length
        )
        cache_sl = (
            "dynamic"
            if isinstance(sequence_length, list) and len(sequence_length) > 1
            else max_sequence_length
        )

        if issubclass(cls, AIHMAdaptation):
            # If we are working with AIHM adapted models, we need to change the block detection strategy for Qwen3
            # todo: remove this when we have more robust block matching
            from aimet_onnx.graph_passes.passes.decoder_block import DecoderBlockQwen3

            DecoderBlockQwen3.NUM_RMSNORM_PER_BLK = 41

        is_hf = is_huggingface_ckpt(model_id)

        # Use model cache for HuggingFace checkpoints when a cache is provided
        if is_hf:
            if model_cache is not None:
                with tempfile.TemporaryDirectory() as tmpdir:
                    params = {
                        "model_id": model_id,
                        "class": cls.__name__,
                        "sequence_length": cache_sl,
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
            onnx_model = entry.backbone
            config = entry.config
        else:
            onnx_model, *_ = load_model_components_from_disk(
                model_id,
                context_length=context_length,
                sequence_length=cache_sl,
            )
            config = AutoConfig.from_pretrained(model_id)

        block_prec = precision.blocks["default"]
        # default_param_qtype must be int for the quantsim wiring; when block
        # weights are FP we strip those weight quantizers below, so int8 here
        # is just a placeholder.
        default_param_qtype = int8 if block_prec.is_float else block_prec.qtype
        default_activation_qtype = precision.activations

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
                param_type=default_param_qtype,
                activation_type=default_activation_qtype,
                config_file=QUANTSIM_CONFIG,
                providers=get_ort_providers(
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                ),
            )

        # Setting the LM head weights
        _set_lm_head_precision(quant_sim, precision.lm_head)
        # Tie KV cache, and set quantization type
        _resolve_kv_cache_quantization(quant_sim, precision.resolve_kv_cache_qtype())
        # If block weights are FP, drop their weight quantizers entirely.
        # Otherwise, apply block-level granularity (LPBQ/BQ) if configured.
        if block_prec.is_float:
            _remove_decoder_block_weight_quantizers(quant_sim)
        else:
            _apply_block_granularity_to_decoder_stack(quant_sim, precision)

        if default_activation_qtype in (float16, float32):
            _remove_activation_quantizers(quant_sim)

        # Plain (non-VLM) LLMs do not wire embed_tokens into SimCollection,
        # so the runner's quantize_embedding_weights() call is always
        # skipped. Reject any override to avoid silently ignoring user
        # intent. VLM subclasses honor it via their own instantiate_quantsim
        # overrides.
        if precision.embedding != int16:
            raise NotImplementedError(
                "Embedding quantization other than int16 is not currently "
                "supported for plain LLMs."
            )

        return SimCollection(quant_sim, config=config)

    @classmethod
    def _export_to_cache_entry(
        cls,
        model_id: str,
        context_length: int,
        sequence_length: int | list[int],
        small_model: bool,
        directory: str,
    ) -> ModelCacheEntry:
        """Export the torch model to ONNX in a temp dir and return a :class:`ModelCacheEntry`."""
        max_seq_len = (
            max(sequence_length)
            if isinstance(sequence_length, list)
            else sequence_length
        )

        instantiated_model = cls.instantiate_model(model_id, small_model)
        if isinstance(instantiated_model, tuple):
            if context_length != 4096 or max_seq_len != 2048:
                raise ValueError(
                    "Context length and sequence length must be 4096 and 2048 for AIHM adapted models."
                )
            assert isinstance(instantiated_model[0], onnx.ModelProto)
            return ModelCacheEntry(
                backbone=instantiated_model[0],
                config=instantiated_model[1],
            )

        assert isinstance(instantiated_model, torch.nn.Module)
        instantiated_model = instantiated_model.to(dtype=torch.float32)
        layer_cache_descs = build_layer_cache_descriptors(instantiated_model.config)
        exportable_model = ONNXExportableModuleWithCache(
            instantiated_model,
            cache_type=cls.get_cache_type(),
            input_names=cls.get_backbone_input_names(layer_cache_descs),
        )

        onnx_model, *_ = get_onnx_model(
            checkpoint=directory,
            fp_backbone_model=exportable_model,
            context_length=context_length,
            sequence_length=sequence_length,
            sample_input=cls.get_sample_backbone_inputs(
                exportable_model, context_length, max_seq_len
            ),
            input_names=cls.get_backbone_input_names(layer_cache_descs),
            output_names=cls.get_backbone_output_names(layer_cache_descs),
            dynamo=cls.use_dynamo_export(),
            dynamic_axes=cls.get_backbone_dynamic_axes(layer_cache_descs),
        )

        return ModelCacheEntry(
            backbone=onnx_model,
            config=instantiated_model.config,
        )

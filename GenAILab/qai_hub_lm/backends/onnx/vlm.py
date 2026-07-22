# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""VLM ONNX quantization — base class and model registrations."""

from __future__ import annotations

import tempfile
import warnings

import torch
from transformers import AutoConfig

from aimet_onnx import quantsim
from aimet_onnx.quantsim import QuantizationSimModel

from GenAILab.qai_hub_lm.backends import QUANTSIM_CONFIG
from GenAILab.qai_hub_lm.precision import PrecisionConfig, float16, float32
from GenAILab.bench.model_cache import DiskBackedModelCache, ModelCacheEntry
from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.qai_hub_lm.models.base import SimCollection
from GenAILab.qai_hub_lm.models.utils.exportable import ONNXExportableModuleWithCache
from GenAILab.qai_hub_lm.models.utils.layer_cache import (
    build_layer_cache_descriptors,
    _resolve_text_config,
)

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
    get_ort_providers,
    AttributePatch,
)


class VLM_ONNX:
    """Mixin providing common ONNX VLM float-export + quantsim instantiation.

    Subclasses need only declare the class (inheriting from this mixin and
    their model-specific VLM base) and register via @YAMLConfigParser.register_model.
    All model-structure knowledge comes from methods on the VLM model class:
    get_language_model, get_lm_head, get_embedding, build_vision_wrapper, get_extras.

    The float-export step (:meth:`instantiate_float_model`) is separated from
    sim construction (:meth:`instantiate_quantsim`) so the caller can transform
    the float ONNX graph(s) (e.g. apply SpinQuant) before the sims are built.
    """

    @classmethod
    def instantiate_float_model(
        cls,
        model_id: str,
        context_length: int,
        sequence_length: int | list[int],
        small_model: bool = False,
        dtype: torch.dtype = torch.float32,
        model_cache: DiskBackedModelCache | None = None,
        image_size: tuple[int, int] | None = None,
        *args,
        **kwargs,
    ) -> ModelCacheEntry:
        """Export (or load) the raw float backbone/visual ONNX models + embedding.

        Separated from :meth:`instantiate_quantsim` so the caller can transform
        the float graph(s) (e.g. apply SpinQuant) before the sims are built.
        """
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID

        cache_sl = (
            "dynamic"
            if isinstance(sequence_length, list) and len(sequence_length) > 1
            else (
                max(sequence_length)
                if isinstance(sequence_length, list)
                else sequence_length
            )
        )

        is_hf = is_huggingface_ckpt(model_id)

        if is_hf:
            if model_cache is not None:
                with tempfile.TemporaryDirectory() as tmpdir:
                    params = {
                        "model_id": model_id,
                        "class": cls.__name__,
                        "sequence_length": cache_sl,
                        "context_length": context_length,
                        "small_model": small_model,
                        "image_size": image_size,
                        "dtype": str(dtype),
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
                            dtype=dtype,
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
                    dtype=dtype,
                )
            return entry

        config = AutoConfig.from_pretrained(model_id)
        backbone_onnx_model, visual_onnx_model, embedding, extras = (
            load_model_components_from_disk(
                model_id,
                context_length=context_length,
                sequence_length=cache_sl,
            )
        )
        if visual_onnx_model is None or embedding is None:
            raise ValueError("Required model components could not be loaded from disk.")
        return ModelCacheEntry(
            backbone=backbone_onnx_model,
            visual=visual_onnx_model,
            embedding=embedding,
            config=config,
            extras=extras or None,
        )

    @classmethod
    def instantiate_quantsim(
        cls,
        entry: ModelCacheEntry,
        precision: PrecisionConfig | None = None,
        *args,
        **kwargs,
    ) -> SimCollection:
        if precision is None:
            precision = PrecisionConfig()
        precision.ensure_visual_defaults()

        backbone_onnx_model = entry.backbone
        visual_onnx_model = entry.visual
        embedding = entry.embedding
        config = entry.config
        extras = entry.extras or {}

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
                config_file=QUANTSIM_CONFIG,
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
                config_file=QUANTSIM_CONFIG,
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

        # Note: embedding quantization is deferred to after recipe application
        # (in the test runner) to allow recipes like SpinQuant to rotate weights first.

        return SimCollection(
            backbone=backbone_quantsim,
            visual=visual_quantsim,
            embedding=embedding,
            config=config,
            position_id_processor=cls.instantiate_position_processor(),
            extras=extras or None,
        )

    @classmethod
    def _export_to_cache_entry(
        cls,
        model_id: str,
        context_length: int,
        sequence_length: int | list[int],
        small_model: bool,
        directory: str,
        image_size: tuple[int, int] | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> ModelCacheEntry:
        """Export the torch model to ONNX and return a :class:`ModelCacheEntry`."""
        max_seq_len = (
            max(sequence_length)
            if isinstance(sequence_length, list)
            else sequence_length
        )

        model = cls.instantiate_model(model_id, small_model).to(dtype=dtype)
        text_config = _resolve_text_config(model.config)
        layer_cache_descs = build_layer_cache_descriptors(text_config)

        language_model = cls.get_language_model(model)
        lm_head = cls.get_lm_head(model)

        backbone_kwargs = {
            "cache_type": cls.get_cache_type(),
            "input_names": cls.get_backbone_input_names(
                layer_cache_descs, config=model.config
            ),
        }
        if lm_head is not None:
            backbone_kwargs["lm_head"] = lm_head

        traceable_backbone = ONNXExportableModuleWithCache(
            language_model, **backbone_kwargs
        )
        traceable_visual = cls.build_vision_wrapper(model)

        backbone_onnx_model, visual_onnx_model = get_onnx_model(
            checkpoint=directory,
            fp_backbone_model=traceable_backbone,
            context_length=context_length,
            sequence_length=sequence_length,
            sample_input=cls.get_sample_backbone_inputs(
                traceable_backbone,
                context_length,
                max_seq_len,
                layer_cache_descriptors=layer_cache_descs,
                image_size=image_size,
                config=model.config,
            ),
            input_names=cls.get_backbone_input_names(
                layer_cache_descs, config=model.config
            ),
            output_names=cls.get_backbone_output_names(layer_cache_descs),
            fp_visual_model=traceable_visual,
            sample_visual_input=cls.get_sample_vision_inputs(
                model.config, image_size=image_size, dtype=model.dtype
            ),
            visual_input_names=cls.get_visual_input_names(),
            visual_output_names=cls.get_visual_output_names(config=model.config),
            dynamo=cls.use_dynamo_export(),
            dynamic_axes=cls.get_backbone_dynamic_axes(
                layer_cache_descs, config=model.config
            ),
            visual_dynamic_axes=cls.get_visual_dynamic_axes(),
        )

        embedding = cls.get_embedding(model)
        extras = cls.get_extras(model) or None

        return ModelCacheEntry(
            backbone=backbone_onnx_model,
            visual=visual_onnx_model,
            embedding=embedding,
            config=model.config,
            extras=extras,
        )


# ---------------------------------------------------------------------------
# Model registrations
# ---------------------------------------------------------------------------

from GenAILab.qai_hub_lm.models.qwen2_vl import Qwen_25_VL


@YAMLConfigParser.register_model("qwen2_5_vl")
class Qwen_25_VL_ONNX(VLM_ONNX, Qwen_25_VL):
    pass


try:
    from GenAILab.qai_hub_lm.models.qwen3_vl import Qwen_3_VL

    @YAMLConfigParser.register_model("qwen3_vl")
    class Qwen_3_VL_ONNX(VLM_ONNX, Qwen_3_VL):
        pass

except ImportError:
    warnings.warn(
        "Qwen 3VL is not available. Please upgrade to a later version of transformers to use this model."
    )

try:
    from GenAILab.qai_hub_lm.models.gemma3 import Gemma3_VLM

    @YAMLConfigParser.register_model("gemma3")
    class Gemma3_ONNX(VLM_ONNX, Gemma3_VLM):
        pass

except ImportError:
    pass

try:
    from GenAILab.qai_hub_lm.models.gemma4 import Gemma4_VLM

    @YAMLConfigParser.register_model("gemma4")
    class Gemma4_ONNX(VLM_ONNX, Gemma4_VLM):
        pass

except ImportError:
    pass

try:
    from GenAILab.qai_hub_lm.models.internvl import InternVL_VLM

    @YAMLConfigParser.register_model("internvl_chat")
    class InternVL_ONNX(VLM_ONNX, InternVL_VLM):
        pass

except ImportError:
    pass

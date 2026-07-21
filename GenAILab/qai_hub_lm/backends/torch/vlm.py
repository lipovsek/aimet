# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""VLM Torch quantization — base class and model registrations."""

from __future__ import annotations

import warnings

import torch

from aimet_torch.common.defs import QuantScheme
from aimet_torch import QuantizationSimModel
from aimet_torch.onnx_utils import map_torch_types_to_onnx
from aimet_torch.v2.nn.true_quant import QuantizationMixin
from aimet_torch.v2.utils import remove_activation_quantizers

from GenAILab.qai_hub_lm.backends import QUANTSIM_CONFIG
from GenAILab.qai_hub_lm.precision import PrecisionConfig, float16, float32
from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.qai_hub_lm.models.base import SimCollection
from GenAILab.qai_hub_lm.models.utils.exportable import ONNXExportableModuleWithCache
from GenAILab.qai_hub_lm.models.utils.layer_cache import (
    build_layer_cache_descriptors,
    _resolve_text_config,
)
from GenAILab.qai_hub_lm.backends.torch.quantsim_utils import (
    _apply_block_granularity_to_decoder_stack,
    _set_lm_head_precision,
)


class VLM_Torch:
    """Mixin providing common Torch VLM quantsim instantiation.

    Subclasses need only declare the class (inheriting from this mixin and
    their model-specific VLM base) and register via @YAMLConfigParser.register_model.
    All model-structure knowledge comes from methods on the VLM model class:
    get_language_model, get_lm_head, get_embedding, build_vision_wrapper, get_extras.
    """

    @staticmethod
    def _is_quantized_rms_norm(module: torch.nn.Module) -> bool:
        return isinstance(
            module, QuantizationMixin
        ) and "RMSNormalization" in map_torch_types_to_onnx.get(type(module), [])

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

        # Backbone
        text_config = _resolve_text_config(model.config)
        layer_cache_descs = build_layer_cache_descriptors(text_config)

        language_model = cls.get_language_model(model)
        lm_head = cls.get_lm_head(model)
        traceable_backbone = ONNXExportableModuleWithCache(
            language_model,
            lm_head=lm_head,
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
                config=model.config,
            ),
            default_output_bw=default_output_bw,
            default_param_bw=default_param_bw,
            in_place=True,
            config_file=QUANTSIM_CONFIG,
        )

        if precision.activations in (float16, float32):
            remove_activation_quantizers(language_sim.model)

        for module in language_sim.model.modules():
            if (
                cls._is_quantized_rms_norm(module)
                and "weight" in module.param_quantizers
            ):
                module.param_quantizers["weight"].bitwidth = 16

        sim_lm_head = getattr(language_sim.model, "lm_head", None)
        if sim_lm_head is not None and hasattr(sim_lm_head, "linear"):
            sim_lm_head = sim_lm_head.linear
        _set_lm_head_precision(language_sim, precision.lm_head, lm_head=sim_lm_head)
        _apply_block_granularity_to_decoder_stack(
            language_sim, precision, lm_head=sim_lm_head
        )

        # Vision
        visual_param_bw = precision.visual_weight.qtype.bits
        visual_output_bw = (
            16
            if precision.visual_activations in (float16, float32)
            else precision.visual_activations.bits
        )
        traceable_visual = cls.build_vision_wrapper(model)
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

        # Embedding quantization
        embedding = cls.get_embedding(model)
        if precision.embedding not in (float16, float32):
            if not isinstance(embedding, QuantizationMixin):
                embedding = QuantizationMixin.from_module(embedding)
                embedding.param_quantizers["weight"].bitwidth = precision.embedding.bits
                # Update reference on the language model so the generator uses
                # the quantized version.
                language_model = cls.get_language_model(model)
                if hasattr(language_model, "embed_tokens"):
                    language_model.embed_tokens = embedding
                elif hasattr(language_model, "model") and hasattr(
                    language_model.model, "embed_tokens"
                ):
                    language_model.model.embed_tokens = embedding

        return SimCollection(
            backbone=language_sim,
            visual=visual_sim,
            embedding=embedding,
            config=model.config,
            position_id_processor=cls.instantiate_position_processor(),
            extras=cls.get_extras(model) or None,
        )


# ---------------------------------------------------------------------------
# Model registrations
# ---------------------------------------------------------------------------

from GenAILab.qai_hub_lm.models.qwen2_vl import Qwen_25_VL


@YAMLConfigParser.register_model("qwen2_5_vl")
class Qwen_25_VL_Torch(VLM_Torch, Qwen_25_VL):
    pass


try:
    from GenAILab.qai_hub_lm.models.qwen3_vl import Qwen_3_VL

    @YAMLConfigParser.register_model("qwen3_vl")
    class Qwen_3_VL_Torch(VLM_Torch, Qwen_3_VL):
        pass

except ImportError:
    warnings.warn(
        "Qwen 3VL is not available. Please upgrade to a later version of transformers to use this model."
    )

try:
    from GenAILab.qai_hub_lm.models.gemma3 import Gemma3_VLM

    @YAMLConfigParser.register_model("gemma3")
    class Gemma3_Torch(VLM_Torch, Gemma3_VLM):
        pass

except ImportError:
    pass

try:
    from GenAILab.qai_hub_lm.models.gemma4 import Gemma4_VLM

    @YAMLConfigParser.register_model("gemma4")
    class Gemma4_Torch(VLM_Torch, Gemma4_VLM):
        @classmethod
        def instantiate_quantsim(
            cls,
            model,
            context_length,
            sequence_length,
            precision=None,
            image_size=None,
            model_id=None,
            **kwargs,
        ):
            """Build the W4A8 sim; if the checkpoint is a packed Gemma QAT model,
            dequantize it, then load its trained QAT scales onto the sim and
            disable the quantizers the load left uninitialized.

            The returned SimCollection is what Stage 2 produces: QAT scales loaded
            (frozen) and uninitialized quantizers nulled -- no recipe needed. A
            standard float gemma4 model_id skips all QAT work (no-op).
            """
            is_qat = cls.is_qat_checkpoint(model.config)
            if is_qat:
                print(f"[Gemma4] Dequantizing packed QAT weights for {model_id}")
                cls.dequantize_packed_weights(model)

            sim_collection = super().instantiate_quantsim(
                model,
                context_length,
                sequence_length,
                precision=precision,
                image_size=image_size,
                **kwargs,
            )

            if is_qat:
                if model_id is None:
                    raise RuntimeError(
                        "Gemma4 QAT checkpoint requires model_id to locate "
                        "model.safetensors; pass model_id to instantiate_quantsim."
                    )
                counts = cls.load_qat_encodings(sim_collection, model_id)
                n = cls.disable_uninitialized_quantizers(sim_collection.backbone.model)
                print(
                    f"[Gemma4 QAT] loaded {counts}; nulled {n} uninitialized quantizers"
                )

            return sim_collection

except ImportError:
    pass

try:
    from GenAILab.qai_hub_lm.models.internvl import InternVL_VLM

    @YAMLConfigParser.register_model("internvl_chat")
    class InternVL_Torch(VLM_Torch, InternVL_VLM):
        pass

except ImportError:
    pass

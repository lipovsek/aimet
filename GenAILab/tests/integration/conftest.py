# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared fixtures for Generator parity integration tests."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from GenAILab.qai_hub_lm.models.generator import VLM_Generator
from GenAILab.qai_hub_lm.models.utils.exportable import ONNXExportableModuleWithCache
from GenAILab.qai_hub_lm.models.utils.layer_cache import build_layer_cache_descriptors
from GenAILab.qai_hub_lm.models.base import LLM


# ---------------------------------------------------------------------------
# LLM models — parametrize at module scope so each model is loaded once
# ---------------------------------------------------------------------------
LLM_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
]

SEQUENCE_LENGTHS = [32, 64]
CONTEXT_LENGTH = 256
ATTENTION_MASK_MIN = -1e9


@pytest.fixture(scope="module", params=LLM_MODELS)
def llm_bundle(request):
    """Load a full-size LLM and its tokenizer (shared across tests in module)."""
    model_id = request.param
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation="eager",
        dtype=torch.float32,
    ).cpu()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, model_id


# ---------------------------------------------------------------------------
# VLM models
# ---------------------------------------------------------------------------
VLM_MODELS = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen3-VL-2B-Instruct",
]


def _get_vlm_class(model_id: str):
    """Return the VLM descriptor class for a given model ID."""
    if "Qwen2.5-VL" in model_id:
        from GenAILab.qai_hub_lm.models.qwen2_vl import Qwen_25_VL

        return Qwen_25_VL
    if "Qwen3-VL" in model_id:
        from GenAILab.qai_hub_lm.models.qwen3_vl import Qwen_3_VL

        return Qwen_3_VL
    raise ValueError(f"No VLM class registered for {model_id}")


def _get_visual_wrapper(model_id: str):
    """Return the visual wrapper class for a given model ID."""
    if "Qwen2.5-VL" in model_id:
        from GenAILab.qai_hub_lm.models.qwen2_vl import Qwen2VLVisualWrapper

        return Qwen2VLVisualWrapper
    if "Qwen3-VL" in model_id:
        from GenAILab.qai_hub_lm.models.qwen3_vl import Qwen3VLVisualWrapper

        return Qwen3VLVisualWrapper
    raise ValueError(f"No visual wrapper registered for {model_id}")


def _torch_visual_output_names(vlm_cls, config):
    """Collapse ONNX-flattened deepstack names into a single list entry for Torch."""
    try:
        names = vlm_cls.get_visual_output_names(config=config)
    except TypeError:
        names = vlm_cls.get_visual_output_names()
    result = []
    seen_ds = False
    for n in names:
        if n.startswith("deepstack_visual_embeds_"):
            if not seen_ds:
                result.append("deepstack_visual_embeds")
                seen_ds = True
        else:
            result.append(n)
    return tuple(result)


def build_vlm_generator(model, model_id: str, sequence_length: int) -> VLM_Generator:
    """Build a VLM_Generator from a full VLM model, dispatching on model_id."""
    vlm_cls = _get_vlm_class(model_id)
    wrapper_cls = _get_visual_wrapper(model_id)
    generator_cls = vlm_cls.get_generator_cls()

    visual_output_names = _torch_visual_output_names(vlm_cls, model.config)

    try:
        backbone_input_names = vlm_cls.get_backbone_input_names(
            build_layer_cache_descriptors(model.config), config=model.config
        )
    except TypeError:
        backbone_input_names = vlm_cls.get_backbone_input_names(
            build_layer_cache_descriptors(model.config)
        )
    backbone = ONNXExportableModuleWithCache(
        model.model.language_model,
        lm_head=model.lm_head,
        cache_type=vlm_cls.get_cache_type(),
        input_names=backbone_input_names,
    )
    vision = wrapper_cls(model.model.visual)
    embedding = model.model.language_model.embed_tokens

    return generator_cls(
        backbone_model=backbone,
        vision_model=vision,
        embedding=embedding,
        tokenizer=None,
        sequence_length=sequence_length,
        context_length=CONTEXT_LENGTH,
        position_id_processor=vlm_cls.generate_position_ids,
        config=model.config,
        attention_mask_min=ATTENTION_MASK_MIN,
        visual_output_names=visual_output_names,
    )


_ort_vlm_session_cache: dict[tuple, tuple] = {}


def build_ort_vlm_generator(
    model, model_id: str, sequence_length: int, export_and_load_fn
) -> VLM_Generator:
    """Export both backbone and vision to ONNX, wrap in VLM_Generator.

    *export_and_load_fn* is ``_export_and_load_ort_session`` from the ONNX test
    module (kept there because it depends on ``onnxruntime``).
    """
    from types import SimpleNamespace
    from GenAILab.qai_hub_lm.backends.onnx.torch_onnx_interface import (
        TorchONNXInterface,
    )

    vlm_cls = _get_vlm_class(model_id)
    wrapper_cls = _get_visual_wrapper(model_id)
    generator_cls = vlm_cls.get_generator_cls()

    try:
        visual_output_names = vlm_cls.get_visual_output_names(config=model.config)
    except TypeError:
        visual_output_names = vlm_cls.get_visual_output_names()

    key = (model_id, sequence_length)
    if key not in _ort_vlm_session_cache:
        text_config = getattr(model.config, "text_config", model.config)
        layer_cache_descriptors = build_layer_cache_descriptors(text_config)

        try:
            backbone_input_names = vlm_cls.get_backbone_input_names(
                layer_cache_descriptors, config=model.config
            )
        except TypeError:
            backbone_input_names = vlm_cls.get_backbone_input_names(
                layer_cache_descriptors
            )

        backbone_wrapped = ONNXExportableModuleWithCache(
            model.model.language_model,
            lm_head=model.lm_head,
            cache_type=vlm_cls.get_cache_type(),
            input_names=backbone_input_names,
        )
        backbone_wrapped.eval()
        backbone_output_names = LLM.get_backbone_output_names(layer_cache_descriptors)

        backbone_sample = vlm_cls.get_sample_backbone_inputs(
            backbone_wrapped,
            context_length=CONTEXT_LENGTH,
            sequence_length=sequence_length,
            layer_cache_descriptors=layer_cache_descriptors,
            config=model.config,
        )

        use_dynamo = vlm_cls.use_dynamo_export()
        try:
            backbone_dynamic_axes = vlm_cls.get_backbone_dynamic_axes(
                layer_cache_descriptors, config=model.config
            )
        except TypeError:
            backbone_dynamic_axes = vlm_cls.get_backbone_dynamic_axes(
                layer_cache_descriptors
            )
        backbone_session = export_and_load_fn(
            backbone_wrapped,
            backbone_sample,
            backbone_input_names,
            backbone_output_names,
            dynamic_axes=backbone_dynamic_axes,
            dynamo=use_dynamo,
        )

        vision_wrapped = wrapper_cls(model.model.visual)
        vision_wrapped.eval()

        visual_input_names = vlm_cls.get_visual_input_names()
        sample_vision = vlm_cls.get_sample_vision_inputs(model.config)

        vision_dynamic_axes = {
            name: {0: "num_patches"} for name in visual_input_names if name != "mask"
        }
        vision_dynamic_axes[visual_output_names[0]] = {0: "num_patches"}

        vision_session = export_and_load_fn(
            vision_wrapped,
            sample_vision,
            visual_input_names,
            visual_output_names,
            dynamic_axes=vision_dynamic_axes,
        )

        _ort_vlm_session_cache[key] = (backbone_session, vision_session)

    backbone_session, vision_session = _ort_vlm_session_cache[key]
    text_config = getattr(model.config, "text_config", model.config)

    mock_backbone = SimpleNamespace(session=backbone_session)
    ort_backbone = TorchONNXInterface(mock_backbone, text_config)

    mock_visual = SimpleNamespace(session=vision_session)
    ort_vision = TorchONNXInterface(mock_visual, model.config)

    embedding = model.model.language_model.embed_tokens

    return generator_cls(
        backbone_model=ort_backbone,
        vision_model=ort_vision,
        embedding=embedding,
        tokenizer=None,
        sequence_length=sequence_length,
        context_length=CONTEXT_LENGTH,
        position_id_processor=vlm_cls.generate_position_ids,
        config=model.config,
        attention_mask_min=ATTENTION_MASK_MIN,
        visual_output_names=visual_output_names,
    )


@pytest.fixture(scope="module", params=VLM_MODELS)
def vlm_bundle(request):
    """Load a full-size VLM and its processor (shared across tests in module)."""
    import types
    from transformers import AutoProcessor

    model_id = request.param
    vlm_cls = _get_vlm_class(model_id)
    model = vlm_cls.instantiate_model(model_id)
    model = model.to(dtype=torch.float32).cpu()
    model.eval()

    # Patch _deepstack_process for Qwen3-VL so torchscript export works
    if "Qwen3-VL" in model_id:
        from GenAILab.qai_hub_lm.models.qwen3_vl import (
            _exportable_deepstack_process,
        )

        text_model = model.model.language_model
        text_model._deepstack_process = types.MethodType(
            _exportable_deepstack_process, text_model
        )

    processor = AutoProcessor.from_pretrained(
        model_id, use_fast=True, trust_remote_code=True
    )

    return model, processor, model_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def tokenize(tokenizer, text: str) -> dict[str, torch.Tensor]:
    """Tokenize text and return dict with input_ids and attention_mask."""
    return tokenizer(text, return_tensors="pt")


def make_test_image(width: int = 56, height: int = 56):
    """Create a small synthetic PIL image for VLM tests."""
    from PIL import Image
    import numpy as np

    return Image.fromarray(
        np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    )

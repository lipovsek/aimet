# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Verify that the ONNX Generator produces the same logits as a vanilla HF model.

These tests export models to ONNX, load them with ORT, wrap them in
TorchONNXInterface + Generator, and compare outputs token-for-token against
vanilla HF forward().  Designed to run on CPU in under a minute each.
"""

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch
import onnxruntime as ort

from GenAILab.shared.models.generator import Generator, VLM_Generator
from GenAILab.shared.models.utils.model_utils import ONNXExportableModuleWithCache
from GenAILab.shared.models.utils.layer_cache import build_layer_cache_descriptors
from GenAILab.shared.models.base import LLM, VLM
from GenAILab.onnx.models.utils.torch_onnx_interface import TorchONNXInterface

from .conftest import (
    SEQUENCE_LENGTHS,
    CONTEXT_LENGTH,
    ATTENTION_MASK_MIN,
    tokenize,
    make_test_image,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _export_and_load_ort_session(
    wrapped_model,
    sample_inputs: tuple[torch.Tensor, ...],
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> ort.InferenceSession:
    """Export a wrapped model to ONNX and return an ORT InferenceSession."""
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "model.onnx")
        with torch.no_grad():
            torch.onnx.export(
                wrapped_model,
                sample_inputs,
                onnx_path,
                input_names=list(input_names),
                output_names=list(output_names),
                opset_version=17,
                dynamo=False,
                dynamic_axes=dynamic_axes,
            )
        session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
    return session


def _build_ort_llm_generator(model, tokenizer, sequence_length):
    """Export an LLM to ONNX, load via ORT, and wrap in Generator."""
    wrapped = ONNXExportableModuleWithCache(model)
    wrapped.eval()

    layer_cache_descriptors = build_layer_cache_descriptors(model.config)
    input_names = LLM.get_backbone_input_names(layer_cache_descriptors)
    output_names = LLM.get_backbone_output_names(layer_cache_descriptors)

    # Build sample inputs using Generator.prepare_inputs
    dummy_ids = torch.zeros((1, 1), dtype=torch.int32)
    dummy_mask = torch.ones((1, 1), dtype=torch.int32)
    sample_inputs = Generator.prepare_inputs(
        model=wrapped,
        input_ids=dummy_ids,
        attention_mask=dummy_mask,
        past_key_values=[],
        sequence_length=sequence_length,
        context_length=CONTEXT_LENGTH,
        layer_cache_descriptors=layer_cache_descriptors,
    )

    session = _export_and_load_ort_session(
        wrapped, sample_inputs, input_names, output_names
    )

    # TorchONNXInterface expects quantsim.session — use a simple namespace
    mock_quantsim = SimpleNamespace(session=session)
    ort_model = TorchONNXInterface(mock_quantsim, model.config)

    generator = Generator(
        model=ort_model,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        context_length=CONTEXT_LENGTH,
        attention_mask_min=ATTENTION_MASK_MIN,
    )
    return generator


# ============================================================================
# LLM tests
# ============================================================================
class TestLLMOnnxGeneratorParity:
    """Compare Generator(TorchONNXInterface(ORT session)) vs vanilla HF model()."""

    TEXT = "The quick brown fox jumps over the lazy dog"

    @pytest.mark.parametrize("sequence_length", SEQUENCE_LENGTHS)
    def test_logits_match(self, llm_bundle, sequence_length):
        model, tokenizer, _ = llm_bundle
        tokens = tokenize(tokenizer, self.TEXT)

        if tokens["input_ids"].shape[1] > sequence_length:
            pytest.skip("Input longer than sequence_length")

        # --- Vanilla HF ---
        with torch.no_grad():
            hf_out = model(**tokens)

        # --- ONNX Generator ---
        generator = _build_ort_llm_generator(model, tokenizer, sequence_length)
        with torch.no_grad():
            gen_out = generator(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )

        assert hf_out.logits.shape == gen_out.logits.shape, (
            f"Shape mismatch: HF {hf_out.logits.shape} vs Gen {gen_out.logits.shape}"
        )
        torch.testing.assert_close(
            hf_out.logits.cpu(), gen_out.logits.cpu(), atol=1e-3, rtol=1e-3
        )

    @pytest.mark.parametrize("sequence_length", SEQUENCE_LENGTHS)
    def test_argmax_match(self, llm_bundle, sequence_length):
        """Stronger check: predicted tokens are identical."""
        model, tokenizer, _ = llm_bundle
        tokens = tokenize(tokenizer, self.TEXT)

        if tokens["input_ids"].shape[1] > sequence_length:
            pytest.skip("Input longer than sequence_length")

        with torch.no_grad():
            hf_out = model(**tokens)

        generator = _build_ort_llm_generator(model, tokenizer, sequence_length)
        with torch.no_grad():
            gen_out = generator(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )

        hf_preds = hf_out.logits.cpu().argmax(dim=-1)
        gen_preds = gen_out.logits.cpu().argmax(dim=-1)
        assert torch.equal(hf_preds, gen_preds), (
            f"Predicted tokens differ:\nHF:  {hf_preds}\nGen: {gen_preds}"
        )

    def test_multi_slice_logits_match(self, llm_bundle):
        """When input exceeds sequence_length, Generator processes in slices."""
        model, tokenizer, _ = llm_bundle
        long_text = " ".join(["hello world"] * 20)
        tokens = tokenize(tokenizer, long_text)
        seq_len = 32

        if tokens["input_ids"].shape[1] <= seq_len:
            pytest.skip("Input not long enough to trigger multi-slice")

        with torch.no_grad():
            hf_out = model(**tokens)

        generator = _build_ort_llm_generator(model, tokenizer, seq_len)
        with torch.no_grad():
            gen_out = generator(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )

        assert hf_out.logits.shape == gen_out.logits.shape
        # Multi-slice + ONNX may accumulate slightly more error
        torch.testing.assert_close(
            hf_out.logits.cpu(), gen_out.logits.cpu(), atol=5e-3, rtol=5e-3
        )


# ============================================================================
# VLM tests
# ============================================================================
class TestVLMOnnxGeneratorParity:
    """Compare VLM_Generator(TorchONNXInterface(ORT)) vs vanilla Qwen2.5-VL."""

    def _build_ort_vlm_generator(self, model, sequence_length):
        """Export both backbone and vision to ONNX, wrap in VLM_Generator."""
        from GenAILab.shared.models.qwen2_vl import Qwen_25_VL, Qwen2VLVisualWrapper

        # --- Backbone ---
        backbone_wrapped = ONNXExportableModuleWithCache(
            model.model.language_model,
            lm_head=model.lm_head,
            use_inputs_embeds=True,
            cache_type=Qwen_25_VL.get_cache_type(),
        )
        backbone_wrapped.eval()

        layer_cache_descriptors = build_layer_cache_descriptors(
            model.config.text_config
        )
        backbone_input_names = VLM.get_backbone_input_names(layer_cache_descriptors)
        backbone_output_names = LLM.get_backbone_output_names(layer_cache_descriptors)

        embedding = model.model.language_model.embed_tokens
        embed_dim = embedding.embedding_dim

        # Build sample backbone inputs (inputs_embeds instead of input_ids)
        # Qwen2.5-VL uses 3D position_ids: [3, batch, seq_len]
        dummy_embeds = torch.zeros((1, 1, embed_dim), dtype=torch.float32)
        dummy_mask = torch.ones((1, 1), dtype=torch.int32)
        dummy_position_ids = torch.zeros((3, 1, 1), dtype=torch.int32)
        backbone_sample = Generator.prepare_inputs(
            model=backbone_wrapped,
            input_ids=None,
            inputs_embeds=dummy_embeds,
            attention_mask=dummy_mask,
            past_key_values=[],
            sequence_length=sequence_length,
            context_length=CONTEXT_LENGTH,
            layer_cache_descriptors=layer_cache_descriptors,
            position_ids=dummy_position_ids,
        )

        backbone_session = _export_and_load_ort_session(
            backbone_wrapped,
            backbone_sample,
            backbone_input_names,
            backbone_output_names,
        )
        mock_backbone = SimpleNamespace(session=backbone_session)
        ort_backbone = TorchONNXInterface(mock_backbone, model.config.text_config)

        # --- Vision ---
        vision_wrapped = Qwen2VLVisualWrapper(model.model.visual)
        vision_wrapped.eval()

        visual_input_names = Qwen_25_VL.get_visual_input_names()
        visual_output_names = Qwen_25_VL.get_visual_output_names()

        # Sample vision inputs — use dynamic axes so pixel_values can vary
        sample_vision = Qwen_25_VL.get_sample_vision_inputs(model.config)
        vision_dynamic_axes = {
            name: {0: "num_patches"} for name in visual_input_names if name != "mask"
        }
        vision_dynamic_axes[visual_output_names[0]] = {0: "num_patches"}
        vision_session = _export_and_load_ort_session(
            vision_wrapped,
            sample_vision,
            visual_input_names,
            visual_output_names,
            dynamic_axes=vision_dynamic_axes,
        )
        mock_visual = SimpleNamespace(session=vision_session)
        ort_vision = TorchONNXInterface(mock_visual, model.config)

        generator = VLM_Generator(
            backbone_model=ort_backbone,
            vision_model=ort_vision,
            embedding=embedding,
            tokenizer=None,
            sequence_length=sequence_length,
            context_length=CONTEXT_LENGTH,
            position_id_processor=Qwen_25_VL.generate_position_ids,
            config=model.config,
            attention_mask_min=ATTENTION_MASK_MIN,
        )
        return generator

    def _prepare_text_only_inputs(self, processor, text="Describe this scene."):
        """Prepare VLM inputs without any images."""
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return processor(text=[prompt], return_tensors="pt", padding=True)

    def _prepare_single_image_inputs(self, processor):
        """Prepare VLM inputs with one small image."""
        image = make_test_image(56, 56)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "What is this?"},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return processor(text=[prompt], images=[image], return_tensors="pt")

    def _prepare_multi_image_inputs(self, processor):
        """Prepare VLM inputs with two small images."""
        img1 = make_test_image(56, 56)
        img2 = make_test_image(56, 56)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img1},
                    {"type": "image", "image": img2},
                    {"type": "text", "text": "Compare these images."},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return processor(text=[prompt], images=[img1, img2], return_tensors="pt")

    @pytest.mark.parametrize("sequence_length", [64, 128])
    def test_text_only(self, vlm_bundle, sequence_length):
        model, processor, _ = vlm_bundle
        inputs = self._prepare_text_only_inputs(processor)

        if inputs["input_ids"].shape[1] > sequence_length:
            pytest.skip("Input longer than sequence_length")

        with torch.no_grad():
            hf_out = model(**inputs)

        generator = self._build_ort_vlm_generator(model, sequence_length)
        gen_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        with torch.no_grad():
            gen_out = generator(**gen_inputs)

        assert hf_out.logits.shape == gen_out.logits.shape
        torch.testing.assert_close(
            hf_out.logits.cpu(), gen_out.logits.cpu(), atol=1e-3, rtol=1e-3
        )

    @pytest.mark.xfail(
        reason="Vision ONNX export traces data-dependent ops (Gather by grid_thw) "
        "as constants, so the exported model only works for the sample image size.",
        strict=True,
    )
    def test_single_image(self, vlm_bundle):
        model, processor, _ = vlm_bundle
        inputs = self._prepare_single_image_inputs(processor)
        seq_len = max(128, inputs["input_ids"].shape[1])

        with torch.no_grad():
            hf_out = model(**inputs)

        generator = self._build_ort_vlm_generator(model, seq_len)
        with torch.no_grad():
            gen_out = generator(**inputs)

        assert hf_out.logits.shape == gen_out.logits.shape
        torch.testing.assert_close(
            hf_out.logits.cpu(), gen_out.logits.cpu(), atol=1e-3, rtol=1e-3
        )

    @pytest.mark.xfail(
        reason="Vision ONNX export traces data-dependent ops (Gather by grid_thw) "
        "as constants, so the exported model only works for the sample image size.",
        strict=True,
    )
    def test_multi_image(self, vlm_bundle):
        model, processor, _ = vlm_bundle
        inputs = self._prepare_multi_image_inputs(processor)
        seq_len = max(128, inputs["input_ids"].shape[1])

        with torch.no_grad():
            hf_out = model(**inputs)

        generator = self._build_ort_vlm_generator(model, seq_len)
        with torch.no_grad():
            gen_out = generator(**inputs)

        assert hf_out.logits.shape == gen_out.logits.shape
        torch.testing.assert_close(
            hf_out.logits.cpu(), gen_out.logits.cpu(), atol=1e-3, rtol=1e-3
        )

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

from GenAILab.qai_hub_lm.models.generator import Generator
from GenAILab.qai_hub_lm.models.utils.exportable import ONNXExportableModuleWithCache
from GenAILab.qai_hub_lm.models.utils.layer_cache import build_layer_cache_descriptors
from GenAILab.qai_hub_lm.models.base import LLM
from GenAILab.qai_hub_lm.backends.onnx.torch_onnx_interface import TorchONNXInterface
from GenAILab.qai_hub_lm.backends.onnx.export_utils import ONNX_OPSET_VERSION

from .conftest import (
    SEQUENCE_LENGTHS,
    CONTEXT_LENGTH,
    ATTENTION_MASK_MIN,
    build_ort_vlm_generator,
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
    dynamo: bool = False,
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
                opset_version=ONNX_OPSET_VERSION,
                dynamo=dynamo,
                dynamic_axes=dynamic_axes,
            )
        session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
    return session


_ort_llm_session_cache: dict[tuple, ort.InferenceSession] = {}


def _build_ort_llm_generator(model, tokenizer, model_id, sequence_length):
    """Export an LLM to ONNX, load via ORT, and wrap in Generator."""
    key = (model_id, sequence_length)
    if key not in _ort_llm_session_cache:
        layer_cache_descriptors = build_layer_cache_descriptors(model.config)
        input_names = LLM.get_backbone_input_names(layer_cache_descriptors)

        wrapped = ONNXExportableModuleWithCache(model, input_names=input_names)
        wrapped.eval()
        output_names = LLM.get_backbone_output_names(layer_cache_descriptors)

        dummy_ids = torch.zeros((1, 1), dtype=torch.int32)
        dummy_mask = torch.ones((1, 1), dtype=torch.int32)
        sample_inputs = tuple(
            Generator.prepare_inputs(
                model=wrapped,
                input_ids=dummy_ids,
                attention_mask=dummy_mask,
                past_key_values=[],
                sequence_length=sequence_length,
                context_length=CONTEXT_LENGTH,
                layer_cache_descriptors=layer_cache_descriptors,
            ).values()
        )

        _ort_llm_session_cache[key] = _export_and_load_ort_session(
            wrapped, sample_inputs, input_names, output_names
        )

    mock_quantsim = SimpleNamespace(session=_ort_llm_session_cache[key])
    ort_model = TorchONNXInterface(mock_quantsim, model.config)

    return Generator(
        model=ort_model,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        context_length=CONTEXT_LENGTH,
        attention_mask_min=ATTENTION_MASK_MIN,
    )


# ============================================================================
# LLM tests
# ============================================================================
class TestLLMOnnxGeneratorParity:
    """Compare Generator(TorchONNXInterface(ORT session)) vs vanilla HF model()."""

    TEXT = "The quick brown fox jumps over the lazy dog"

    @pytest.mark.parametrize("sequence_length", SEQUENCE_LENGTHS)
    def test_logits_match(self, llm_bundle, sequence_length):
        model, tokenizer, model_id = llm_bundle
        tokens = tokenize(tokenizer, self.TEXT)

        if tokens["input_ids"].shape[1] > sequence_length:
            pytest.skip("Input longer than sequence_length")

        # --- Vanilla HF ---
        with torch.no_grad():
            hf_out = model(**tokens)

        # --- ONNX Generator ---
        generator = _build_ort_llm_generator(
            model, tokenizer, model_id, sequence_length
        )
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
        model, tokenizer, model_id = llm_bundle
        tokens = tokenize(tokenizer, self.TEXT)

        if tokens["input_ids"].shape[1] > sequence_length:
            pytest.skip("Input longer than sequence_length")

        with torch.no_grad():
            hf_out = model(**tokens)

        generator = _build_ort_llm_generator(
            model, tokenizer, model_id, sequence_length
        )
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
        model, tokenizer, model_id = llm_bundle
        long_text = " ".join(["hello world"] * 20)
        tokens = tokenize(tokenizer, long_text)
        seq_len = 32

        if tokens["input_ids"].shape[1] <= seq_len:
            pytest.skip("Input not long enough to trigger multi-slice")

        with torch.no_grad():
            hf_out = model(**tokens)

        generator = _build_ort_llm_generator(model, tokenizer, model_id, seq_len)
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

    def test_autoregressive_decode(self, llm_bundle):
        """Prefill followed by one decode step produces valid output."""
        model, tokenizer, model_id = llm_bundle
        tokens = tokenize(tokenizer, self.TEXT)
        seq_len = SEQUENCE_LENGTHS[0]

        if tokens["input_ids"].shape[1] > seq_len:
            pytest.skip("Input longer than sequence_length")

        generator = _build_ort_llm_generator(model, tokenizer, model_id, seq_len)

        with torch.no_grad():
            prefill_out = generator(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )

        device = generator.device
        next_token = prefill_out.logits[:, -1:].argmax(dim=-1).to(device)
        decode_mask = torch.cat(
            [
                tokens["attention_mask"].to(device),
                torch.ones(1, 1, dtype=torch.int, device=device),
            ],
            dim=1,
        )

        with torch.no_grad():
            decode_out = generator(
                input_ids=next_token,
                attention_mask=decode_mask,
                past_key_values=prefill_out.past_key_values,
            )

        assert decode_out.logits.shape == (1, 1, model.config.vocab_size)


# ============================================================================
# VLM tests
# ============================================================================
class TestVLMOnnxGeneratorParity:
    """Compare VLM_Generator(TorchONNXInterface(ORT)) vs vanilla VLM forward."""

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

    @pytest.mark.parametrize("sequence_length", [128])
    def test_text_only(self, vlm_bundle, sequence_length):
        model, processor, model_id = vlm_bundle
        inputs = self._prepare_text_only_inputs(processor)

        if inputs["input_ids"].shape[1] > sequence_length:
            pytest.skip("Input longer than sequence_length")

        with torch.no_grad():
            hf_out = model(**inputs)

        generator = build_ort_vlm_generator(
            model, model_id, sequence_length, _export_and_load_ort_session
        )
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

    @pytest.mark.parametrize("sequence_length", [128])
    def test_autoregressive_decode(self, vlm_bundle, sequence_length):
        """Prefill + one decode step over ONNX backbone."""
        model, processor, model_id = vlm_bundle
        inputs = self._prepare_text_only_inputs(processor)

        if inputs["input_ids"].shape[1] > sequence_length:
            pytest.skip("Input longer than sequence_length")

        generator = build_ort_vlm_generator(
            model, model_id, sequence_length, _export_and_load_ort_session
        )
        gen_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        with torch.no_grad():
            prefill_out = generator(**gen_inputs)

        next_token = prefill_out.logits[:, -1:].argmax(dim=-1).cpu()
        decode_mask = torch.cat(
            [inputs["attention_mask"], torch.ones(1, 1, dtype=torch.int)], dim=1
        )

        with torch.no_grad():
            decode_out = generator(
                input_ids=next_token,
                attention_mask=decode_mask,
                past_key_values=prefill_out.past_key_values,
            )

        text_config = getattr(model.config, "text_config", model.config)
        assert decode_out.logits.shape == (1, 1, text_config.vocab_size)

    @pytest.mark.xfail(
        reason="Vision ONNX export traces data-dependent ops (Gather by grid_thw) "
        "as constants, so the exported model only works for the sample image size.",
        strict=True,
    )
    def test_single_image(self, vlm_bundle):
        model, processor, model_id = vlm_bundle
        inputs = self._prepare_single_image_inputs(processor)
        seq_len = max(128, inputs["input_ids"].shape[1])

        with torch.no_grad():
            hf_out = model(**inputs)

        generator = build_ort_vlm_generator(
            model, model_id, seq_len, _export_and_load_ort_session
        )
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
        model, processor, model_id = vlm_bundle
        inputs = self._prepare_multi_image_inputs(processor)
        seq_len = max(128, inputs["input_ids"].shape[1])

        with torch.no_grad():
            hf_out = model(**inputs)

        generator = build_ort_vlm_generator(
            model, model_id, seq_len, _export_and_load_ort_session
        )
        with torch.no_grad():
            gen_out = generator(**inputs)

        assert hf_out.logits.shape == gen_out.logits.shape
        torch.testing.assert_close(
            hf_out.logits.cpu(), gen_out.logits.cpu(), atol=1e-3, rtol=1e-3
        )

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Verify that the Torch Generator produces the same logits as a vanilla HF model.

These tests load real models and compare outputs token-for-token.  They are designed
to run on CPU in under a minute each.
"""

import pytest
import torch

from GenAILab.qai_hub_lm.models.generator import Generator
from GenAILab.qai_hub_lm.models.base import LLM
from GenAILab.qai_hub_lm.utils.model_utils import ONNXExportableModuleWithCache
from GenAILab.qai_hub_lm.utils.layer_cache import build_layer_cache_descriptors

from .conftest import (
    SEQUENCE_LENGTHS,
    CONTEXT_LENGTH,
    ATTENTION_MASK_MIN,
    build_vlm_generator,
    tokenize,
    make_test_image,
)


# ============================================================================
# LLM tests
# ============================================================================
class TestLLMTorchGeneratorParity:
    """Compare Generator(ONNXExportableModuleWithCache(model)) vs model()."""

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

        # --- Generator ---
        wrapped = ONNXExportableModuleWithCache(
            model,
            input_names=LLM.get_backbone_input_names(
                build_layer_cache_descriptors(model.config)
            ),
        )
        generator = Generator(
            model=wrapped,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            context_length=CONTEXT_LENGTH,
            attention_mask_min=ATTENTION_MASK_MIN,
        )
        with torch.no_grad():
            gen_out = generator(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )

        assert hf_out.logits.shape == gen_out.logits.shape, (
            f"Shape mismatch: HF {hf_out.logits.shape} vs Gen {gen_out.logits.shape}"
        )
        torch.testing.assert_close(hf_out.logits, gen_out.logits, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("sequence_length", SEQUENCE_LENGTHS)
    def test_argmax_match(self, llm_bundle, sequence_length):
        """Stronger check: predicted tokens are identical."""
        model, tokenizer, _ = llm_bundle
        tokens = tokenize(tokenizer, self.TEXT)

        if tokens["input_ids"].shape[1] > sequence_length:
            pytest.skip("Input longer than sequence_length")

        with torch.no_grad():
            hf_out = model(**tokens)

        wrapped = ONNXExportableModuleWithCache(
            model,
            input_names=LLM.get_backbone_input_names(
                build_layer_cache_descriptors(model.config)
            ),
        )
        generator = Generator(
            model=wrapped,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            context_length=CONTEXT_LENGTH,
            attention_mask_min=ATTENTION_MASK_MIN,
        )
        with torch.no_grad():
            gen_out = generator(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )

        hf_preds = hf_out.logits.argmax(dim=-1)
        gen_preds = gen_out.logits.argmax(dim=-1)
        assert torch.equal(hf_preds, gen_preds), (
            f"Predicted tokens differ:\nHF:  {hf_preds}\nGen: {gen_preds}"
        )

    def test_multi_slice_logits_match(self, llm_bundle):
        """When input exceeds sequence_length, Generator processes in slices."""
        model, tokenizer, _ = llm_bundle
        # Use a long-ish prompt to exceed the smallest sequence_length
        long_text = " ".join(["hello world"] * 20)
        tokens = tokenize(tokenizer, long_text)
        seq_len = 32
        ctx_len = CONTEXT_LENGTH

        if tokens["input_ids"].shape[1] <= seq_len:
            pytest.skip("Input not long enough to trigger multi-slice")

        with torch.no_grad():
            hf_out = model(**tokens)

        wrapped = ONNXExportableModuleWithCache(
            model,
            input_names=LLM.get_backbone_input_names(
                build_layer_cache_descriptors(model.config)
            ),
        )
        generator = Generator(
            model=wrapped,
            tokenizer=tokenizer,
            sequence_length=seq_len,
            context_length=ctx_len,
            attention_mask_min=ATTENTION_MASK_MIN,
        )
        with torch.no_grad():
            gen_out = generator(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )

        assert hf_out.logits.shape == gen_out.logits.shape
        # Multi-slice may accumulate slightly more error
        torch.testing.assert_close(hf_out.logits, gen_out.logits, atol=5e-3, rtol=5e-3)

    def test_autoregressive_decode(self, llm_bundle):
        """Prefill followed by one decode step produces valid output."""
        model, tokenizer, _ = llm_bundle
        tokens = tokenize(tokenizer, self.TEXT)
        seq_len = SEQUENCE_LENGTHS[0]

        if tokens["input_ids"].shape[1] > seq_len:
            pytest.skip("Input longer than sequence_length")

        wrapped = ONNXExportableModuleWithCache(
            model,
            input_names=LLM.get_backbone_input_names(
                build_layer_cache_descriptors(model.config)
            ),
        )
        generator = Generator(
            model=wrapped,
            tokenizer=tokenizer,
            sequence_length=seq_len,
            context_length=CONTEXT_LENGTH,
            attention_mask_min=ATTENTION_MASK_MIN,
        )

        with torch.no_grad():
            prefill_out = generator(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )

        next_token = prefill_out.logits[:, -1:].argmax(dim=-1)
        decode_mask = torch.cat(
            [tokens["attention_mask"], torch.ones(1, 1, dtype=torch.int)], dim=1
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
class TestVLMTorchGeneratorParity:
    """Compare VLM_Generator vs vanilla VLM forward across all VLM models."""

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
        model, processor, model_id = vlm_bundle
        inputs = self._prepare_text_only_inputs(processor)

        if inputs["input_ids"].shape[1] > sequence_length:
            pytest.skip("Input longer than sequence_length")

        with torch.no_grad():
            hf_out = model(**inputs)

        generator = build_vlm_generator(model, model_id, sequence_length)
        gen_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        with torch.no_grad():
            gen_out = generator(**gen_inputs)

        assert hf_out.logits.shape == gen_out.logits.shape
        torch.testing.assert_close(hf_out.logits, gen_out.logits, atol=1e-3, rtol=1e-3)

    def test_single_image(self, vlm_bundle):
        model, processor, model_id = vlm_bundle
        if "Qwen3-VL" in model_id:
            pytest.xfail(
                "Qwen3-VL deepstack: Generator stacks visual embeds into a "
                "single tensor but HF model expects list[Tensor] per layer"
            )
        inputs = self._prepare_single_image_inputs(processor)
        seq_len = max(128, inputs["input_ids"].shape[1])

        with torch.no_grad():
            hf_out = model(**inputs)

        generator = build_vlm_generator(model, model_id, seq_len)
        with torch.no_grad():
            gen_out = generator(**inputs)

        assert hf_out.logits.shape == gen_out.logits.shape
        torch.testing.assert_close(hf_out.logits, gen_out.logits, atol=1e-3, rtol=1e-3)

    def test_autoregressive_decode(self, vlm_bundle):
        """Prefill + one decode step — catches position processor shape mismatches."""
        model, processor, model_id = vlm_bundle
        inputs = self._prepare_single_image_inputs(processor)
        seq_len = max(128, inputs["input_ids"].shape[1])

        generator = build_vlm_generator(model, model_id, seq_len)

        with torch.no_grad():
            prefill_out = generator(**inputs)

        next_token = prefill_out.logits[:, -1:].argmax(dim=-1)
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

    def test_multi_image(self, vlm_bundle):
        model, processor, model_id = vlm_bundle
        if "Qwen3-VL" in model_id:
            pytest.xfail(
                "Qwen3-VL deepstack: Generator stacks visual embeds into a "
                "single tensor but HF model expects list[Tensor] per layer"
            )
        inputs = self._prepare_multi_image_inputs(processor)
        seq_len = max(128, inputs["input_ids"].shape[1])

        with torch.no_grad():
            hf_out = model(**inputs)

        generator = build_vlm_generator(model, model_id, seq_len)
        with torch.no_grad():
            gen_out = generator(**inputs)

        assert hf_out.logits.shape == gen_out.logits.shape
        torch.testing.assert_close(hf_out.logits, gen_out.logits, atol=1e-3, rtol=1e-3)

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Generator class — static shapes, KV cache, slicing, forward."""

import contextlib
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from GenAILab.shared.models.generator import (
    Generator,
    get_past_keyval_with_shift,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _Cfg:
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    hidden_size: int = 64
    head_dim: int = 16
    vocab_size: int = 256


class _DummyModel(torch.nn.Module):
    """Model stub that returns (logits, *kv_pairs) with correct shapes."""

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or _Cfg()
        self._param = torch.nn.Parameter(torch.zeros(1))

    @property
    def config(self):
        return self.cfg

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def dtype(self):
        return torch.float32

    def forward(self, *args):
        # args: (input_tokens, attention_mask, position_ids, *kv_pairs)
        input_tokens = args[0]
        batch = input_tokens.shape[0]
        seq_len = input_tokens.shape[1]
        logits = torch.randn(batch, seq_len, self.cfg.vocab_size)
        kv_shape = (batch, self.cfg.num_key_value_heads, seq_len, self.cfg.head_dim)
        kv_tensors = [
            torch.randn(kv_shape) for _ in range(self.cfg.num_hidden_layers * 2)
        ]
        return (logits, *kv_tensors)


@pytest.fixture
def cfg():
    return _Cfg()


@pytest.fixture
def model(cfg):
    return _DummyModel(cfg)


@pytest.fixture
def tokenizer():
    tok = MagicMock()
    tok.eos_token_id = 0
    tok.pad_token_id = 0
    return tok


@pytest.fixture
def gen(model, tokenizer):
    return Generator(
        model=model,
        tokenizer=tokenizer,
        sequence_length=8,
        context_length=32,
    )


# ---------------------------------------------------------------------------
# prepare_inputs
# ---------------------------------------------------------------------------


class TestPrepareInputs:
    def test_pads_short_input(self, model):
        input_ids = torch.tensor([[1, 2, 3]])  # length 3
        attn = torch.ones(1, 3, dtype=torch.int32)
        result = Generator.prepare_inputs(
            model=model,
            input_ids=input_ids,
            attention_mask=attn,
            past_key_values=[],
            sequence_length=8,
            context_length=32,
        )
        padded_input = result[0]
        assert padded_input.shape == (1, 8)
        # Original tokens should be at the end
        assert padded_input[0, -1].item() == 3
        assert padded_input[0, -3].item() == 1

    def test_attention_mask_padding(self, model):
        input_ids = torch.tensor([[1, 2, 3]])
        attn = torch.ones(1, 3, dtype=torch.int32)
        result = Generator.prepare_inputs(
            model=model,
            input_ids=input_ids,
            attention_mask=attn,
            past_key_values=[],
            sequence_length=8,
            context_length=32,
        )
        # result[1] is the 4D attention mask
        mask_4d = result[1]
        assert mask_4d.shape == (1, 1, 8, 32)

    def test_kv_cache_shape(self, model, cfg):
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        attn = torch.ones(1, 8, dtype=torch.int32)
        result = Generator.prepare_inputs(
            model=model,
            input_ids=input_ids,
            attention_mask=attn,
            past_key_values=[],
            sequence_length=8,
            context_length=32,
        )
        # KV tensors start at index 3
        kv_tensors = result[3:]
        assert len(kv_tensors) == cfg.num_hidden_layers * 2
        for kv in kv_tensors:
            assert kv.shape == (1, cfg.num_key_value_heads, 32 - 8, cfg.head_dim)

    def test_position_ids_from_mask(self, model):
        input_ids = torch.tensor([[1, 2, 3]])
        attn = torch.ones(1, 3, dtype=torch.int32)
        result = Generator.prepare_inputs(
            model=model,
            input_ids=input_ids,
            attention_mask=attn,
            past_key_values=[],
            sequence_length=8,
            context_length=32,
        )
        position_ids = result[2]
        assert position_ids.shape[-1] == 8

    def test_position_ids_passthrough(self, model):
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        attn = torch.ones(1, 8, dtype=torch.int32)
        custom_pos = torch.arange(8).unsqueeze(0)
        result = Generator.prepare_inputs(
            model=model,
            input_ids=input_ids,
            attention_mask=attn,
            past_key_values=[],
            sequence_length=8,
            context_length=32,
            position_ids=custom_pos,
        )
        position_ids = result[2]
        assert position_ids.shape[-1] == 8

    def test_embeds_instead_of_ids(self, model, cfg):
        embeds = torch.randn(1, 4, cfg.hidden_size)
        attn = torch.ones(1, 4, dtype=torch.int32)
        result = Generator.prepare_inputs(
            model=model,
            input_ids=None,
            attention_mask=attn,
            past_key_values=[],
            sequence_length=8,
            context_length=32,
            inputs_embeds=embeds,
        )
        padded = result[0]
        assert padded.shape == (1, 8, cfg.hidden_size)

    def test_both_ids_and_embeds_raises(self, model, cfg):
        with pytest.raises(ValueError, match="exactly one"):
            Generator.prepare_inputs(
                model=model,
                input_ids=torch.tensor([[1, 2]]),
                attention_mask=torch.ones(1, 2, dtype=torch.int32),
                past_key_values=[],
                sequence_length=8,
                context_length=32,
                inputs_embeds=torch.randn(1, 2, cfg.hidden_size),
            )

    def test_neither_ids_nor_embeds_raises(self, model):
        with pytest.raises(ValueError, match="exactly one"):
            Generator.prepare_inputs(
                model=model,
                input_ids=None,
                attention_mask=torch.ones(1, 2, dtype=torch.int32),
                past_key_values=[],
                sequence_length=8,
                context_length=32,
            )

    def test_default_attention_mask(self, model):
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        result = Generator.prepare_inputs(
            model=model,
            input_ids=input_ids,
            attention_mask=None,
            past_key_values=[],
            sequence_length=8,
            context_length=32,
        )
        # Should not raise, default mask is all-ones
        assert result[0].shape == (1, 8)


# ---------------------------------------------------------------------------
# get_past_keyval_with_shift
# ---------------------------------------------------------------------------


class TestGetPastKeyvalWithShift:
    def test_empty_past(self):
        new_kvs = [
            torch.randn(1, 4, 5, 16),  # key
            torch.randn(1, 4, 5, 16),  # value
        ]
        result = get_past_keyval_with_shift([], new_kvs, length=10)
        assert len(result) == 2
        assert result[0].shape[2] == 5  # 0 + 5 = 5, clipped to min(5, 10)

    def test_empty_new(self):
        past_kvs = [
            torch.randn(1, 4, 8, 16),
            torch.randn(1, 4, 8, 16),
        ]
        result = get_past_keyval_with_shift(past_kvs, [], length=10)
        assert len(result) == 2
        assert result[0].shape[2] == 8

    def test_concatenation(self):
        past_kvs = [
            torch.randn(1, 4, 3, 16),
            torch.randn(1, 4, 3, 16),
        ]
        new_kvs = [
            torch.randn(1, 4, 2, 16),
            torch.randn(1, 4, 2, 16),
        ]
        result = get_past_keyval_with_shift(past_kvs, new_kvs, length=10)
        assert result[0].shape[2] == 5  # 3 + 2

    def test_truncation(self):
        past_kvs = [
            torch.randn(1, 4, 8, 16),
            torch.randn(1, 4, 8, 16),
        ]
        new_kvs = [
            torch.randn(1, 4, 5, 16),
            torch.randn(1, 4, 5, 16),
        ]
        result = get_past_keyval_with_shift(past_kvs, new_kvs, length=10)
        assert result[0].shape[2] == 10  # 8 + 5 = 13, truncated to 10

    def test_device(self):
        past_kvs = [torch.randn(1, 4, 3, 16), torch.randn(1, 4, 3, 16)]
        new_kvs = [torch.randn(1, 4, 2, 16), torch.randn(1, 4, 2, 16)]
        result = get_past_keyval_with_shift(
            past_kvs, new_kvs, length=10, device=torch.device("cpu")
        )
        assert result[0].device == torch.device("cpu")

    def test_dtype(self):
        past_kvs = [torch.randn(1, 4, 3, 16), torch.randn(1, 4, 3, 16)]
        new_kvs = [torch.randn(1, 4, 2, 16), torch.randn(1, 4, 2, 16)]
        result = get_past_keyval_with_shift(
            past_kvs, new_kvs, length=10, dtype=torch.float16
        )
        assert result[0].dtype == torch.float16


# ---------------------------------------------------------------------------
# slice_inputs_for_inference
# ---------------------------------------------------------------------------


class TestSliceInputs:
    def test_single_slice(self):
        inputs = torch.randn(1, 4)
        attn = torch.ones(1, 4)
        slices = list(
            Generator.slice_inputs_for_inference(inputs, attn, sequence_length=8)
        )
        assert len(slices) == 1
        assert slices[0][0].shape[1] == 4

    def test_multiple_slices(self):
        inputs = torch.randn(1, 24)
        attn = torch.ones(1, 24)
        slices = list(
            Generator.slice_inputs_for_inference(inputs, attn, sequence_length=8)
        )
        assert len(slices) == 3

    def test_reverse_order(self):
        # First yielded slice should contain the earliest tokens
        inputs = torch.arange(16).float().unsqueeze(0)
        attn = torch.ones(1, 16)
        slices = list(
            Generator.slice_inputs_for_inference(inputs, attn, sequence_length=8)
        )
        # First slice should have tokens 0-7
        assert slices[0][0][0, 0].item() == 0.0
        # Second slice should have tokens 8-15
        assert slices[1][0][0, 0].item() == 8.0

    def test_position_ids_sliced(self):
        inputs = torch.randn(1, 16)
        attn = torch.ones(1, 16)
        pos = torch.arange(16).unsqueeze(0)
        slices = list(
            Generator.slice_inputs_for_inference(
                inputs, attn, sequence_length=8, position_ids=pos
            )
        )
        assert slices[0][2].shape[-1] == 8
        assert slices[1][2].shape[-1] == 8


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------


class TestForward:
    def test_returns_causal_lm_output(self, gen):
        input_ids = torch.randint(0, 256, (1, 6))
        result = gen.forward(input_ids=input_ids)
        assert isinstance(result, CausalLMOutputWithPast)

    def test_logits_strip_padding(self, gen):
        input_ids = torch.randint(0, 256, (1, 4))
        result = gen.forward(input_ids=input_ids)
        # Logits should have seq_len matching input, not padded sequence_length
        assert result.logits.shape[1] == 4

    def test_kv_cache_populated(self, gen):
        input_ids = torch.randint(0, 256, (1, 6))
        result = gen.forward(input_ids=input_ids)
        assert isinstance(result.past_key_values, DynamicCache)
        assert result.past_key_values.get_seq_length() > 0

    def test_kv_cache_accumulates(self, gen):
        input_ids = torch.randint(0, 256, (1, 4))
        result1 = gen.forward(input_ids=input_ids)
        kv_len1 = result1.past_key_values.get_seq_length()

        input_ids2 = torch.randint(0, 256, (1, 2))
        attn = torch.ones(1, 2, dtype=torch.int32)
        result2 = gen.forward(
            input_ids=input_ids2,
            attention_mask=attn,
            past_key_values=result1.past_key_values,
        )
        kv_len2 = result2.past_key_values.get_seq_length()
        assert kv_len2 > kv_len1

    def test_long_input_multi_slice(self, gen):
        # Input longer than sequence_length (8) should be auto-sliced
        input_ids = torch.randint(0, 256, (1, 20))
        result = gen.forward(input_ids=input_ids)
        assert result.logits.shape[1] == 20

    def test_creates_default_attention_mask(self, gen):
        input_ids = torch.randint(0, 256, (1, 4))
        result = gen.forward(input_ids=input_ids, attention_mask=None)
        assert result.logits.shape[1] == 4

    def test_both_ids_and_embeds_raises(self, gen, cfg):
        with pytest.raises(ValueError, match="exactly one"):
            gen.forward(
                input_ids=torch.tensor([[1, 2]]),
                inputs_embeds=torch.randn(1, 2, cfg.hidden_size),
            )


# ---------------------------------------------------------------------------
# prefill
# ---------------------------------------------------------------------------


class TestPrefill:
    def test_yields_prepared_inputs(self, gen):
        input_ids = torch.randint(0, 256, (1, 6))
        slices = list(gen.prefill(input_ids=input_ids))
        assert len(slices) >= 1
        # Each yielded item should be a tuple of tensors
        assert isinstance(slices[0], tuple)
        assert isinstance(slices[0][0], torch.Tensor)

    def test_single_slice_input(self, gen):
        input_ids = torch.randint(0, 256, (1, 4))
        slices = list(gen.prefill(input_ids=input_ids))
        assert len(slices) == 1

    def test_multi_slice_yields_multiple(self, gen):
        input_ids = torch.randint(0, 256, (1, 20))
        slices = list(gen.prefill(input_ids=input_ids))
        assert len(slices) >= 2


# ---------------------------------------------------------------------------
# prepare_inputs_for_generation (HF generate API)
# ---------------------------------------------------------------------------


class TestPrepareInputsForGeneration:
    def test_first_call_no_stripping(self, gen):
        input_ids = torch.tensor([[1, 2, 3, 4]])
        cache = DynamicCache()
        result = gen.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=cache,
        )
        assert result["input_ids"].shape[1] == 4

    def test_strips_consumed_tokens(self, gen, cfg):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        cache = DynamicCache()
        # Simulate 3 tokens already consumed (need one entry per layer)
        for i in range(cfg.num_hidden_layers):
            cache.update(
                torch.randn(1, cfg.num_key_value_heads, 3, cfg.head_dim),
                torch.randn(1, cfg.num_key_value_heads, 3, cfg.head_dim),
                layer_idx=i,
            )
        result = gen.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=cache,
        )
        assert result["input_ids"].shape[1] == 2  # 5 - 3


# ---------------------------------------------------------------------------
# combine_local_and_global_outputs
# ---------------------------------------------------------------------------


class TestCombineOutputs:
    def test_logits_concatenated(self, gen, cfg):
        global_outputs = {
            "past_key_values": [],
            "logits": torch.randn(1, 4, cfg.vocab_size),
        }
        local_logits = torch.randn(1, 8, cfg.vocab_size)  # padded to seq_len=8
        kv_shape = (1, cfg.num_key_value_heads, 8, cfg.head_dim)
        local_kvs = [torch.randn(kv_shape) for _ in range(cfg.num_hidden_layers * 2)]
        local_outputs = (local_logits, *local_kvs)
        gen.combine_local_and_global_outputs(3, local_outputs, global_outputs)
        # Should be 4 existing + 3 new valid tokens
        assert global_outputs["logits"].shape[1] == 7

    def test_logits_strip_padding(self, gen, cfg):
        global_outputs = {"past_key_values": []}
        local_logits = torch.randn(1, 8, cfg.vocab_size)
        kv_shape = (1, cfg.num_key_value_heads, 8, cfg.head_dim)
        local_kvs = [torch.randn(kv_shape) for _ in range(cfg.num_hidden_layers * 2)]
        local_outputs = (local_logits, *local_kvs)
        gen.combine_local_and_global_outputs(3, local_outputs, global_outputs)
        # Only 3 valid tokens from an 8-wide padded output
        assert global_outputs["logits"].shape[1] == 3

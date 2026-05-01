# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for HubCompatibleGenerator — RoPE injection and KV permutations."""

from collections import OrderedDict
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers.models.llama import LlamaConfig

from GenAILab.qai_hub_lm.models.generator import HubCompatibleGenerator


@pytest.fixture
def llama_config():
    return LlamaConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        intermediate_size=128,
        vocab_size=256,
    )


class _DummyModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._param = torch.nn.Parameter(torch.zeros(1))

    @property
    def config(self):
        return self._config

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def dtype(self):
        return torch.float32

    def forward(self, *args):
        input_tokens = args[0]
        b, s = input_tokens.shape[:2]
        logits = torch.randn(b, s, self._config.vocab_size)
        # Hub format: keys (H, B, D, S), values (H, B, S, D)
        H = self._config.num_key_value_heads
        D = self._config.hidden_size // self._config.num_attention_heads
        kvs = []
        for _ in range(self._config.num_hidden_layers):
            kvs.append(torch.randn(H, b, D, s))  # key: hub format
            kvs.append(torch.randn(H, b, s, D))  # value: hub format
        return (logits, *kvs)


@pytest.fixture
def hub_gen(llama_config):
    model = _DummyModel(llama_config)
    tok = MagicMock()
    tok.eos_token_id = 0
    return HubCompatibleGenerator(
        model=model,
        tokenizer=tok,
        sequence_length=8,
        context_length=32,
    )


class TestHubCompatibleGenerator:
    def test_prepare_inputs_adds_rope(self, hub_gen, llama_config):
        input_ids = torch.randint(0, 256, (1, 8))
        attn = torch.ones(1, 8, dtype=torch.int32)
        result = hub_gen.prepare_inputs(
            model=hub_gen.model,
            input_ids=input_ids,
            attention_mask=attn,
            past_key_values=[],
            sequence_length=8,
            context_length=32,
        )
        assert isinstance(result, OrderedDict)
        assert "position_ids_cos" in result
        assert "position_ids_sin" in result
        assert "position_ids" not in result
        assert result["position_ids_cos"].shape[-1] > 0
        assert result["position_ids_sin"].shape[-1] > 0

    def test_prepare_inputs_kv_permutation(self, hub_gen, llama_config):
        input_ids = torch.randint(0, 256, (1, 8))
        attn = torch.ones(1, 8, dtype=torch.int32)
        B, H = 1, llama_config.num_key_value_heads
        D = llama_config.hidden_size // llama_config.num_attention_heads
        S = 24  # context_length - sequence_length
        past_kvs = []
        for _ in range(llama_config.num_hidden_layers):
            past_kvs.append(torch.randn(B, H, S, D))  # key
            past_kvs.append(torch.randn(B, H, S, D))  # value

        result = hub_gen.prepare_inputs(
            model=hub_gen.model,
            input_ids=input_ids,
            attention_mask=attn,
            past_key_values=past_kvs,
            sequence_length=8,
            context_length=32,
        )
        kv_keys = [
            k
            for k in result
            if k.startswith("past_key_") or k.startswith("past_value_")
        ]
        assert len(kv_keys) == llama_config.num_hidden_layers * 2
        # Keys should be in hub format: (H, B, D, S)
        key = result["past_key_0_in"]
        assert key.shape[0] == H
        # Values should be in hub format: (H, B, S, D)
        value = result["past_value_0_in"]
        assert value.shape[0] == H

    def test_extra_kwargs_accepted_as_tensors(self, hub_gen, llama_config):
        result = hub_gen.prepare_inputs(
            model=hub_gen.model,
            input_ids=torch.randint(0, 256, (1, 8)),
            attention_mask=torch.ones(1, 8, dtype=torch.int32),
            past_key_values=[],
            sequence_length=8,
            context_length=32,
            extra_kwarg=torch.zeros(1),
        )
        assert "extra_kwarg" in result

    def test_combine_outputs_kv_back_to_hf(self, hub_gen, llama_config):
        B, H = 1, llama_config.num_key_value_heads
        D = llama_config.hidden_size // llama_config.num_attention_heads
        S = 8

        global_outputs = {"past_key_values": [], "logits": torch.randn(1, 4, 256)}
        logits = torch.randn(B, S, 256)
        raw_kvs = []
        for _ in range(llama_config.num_hidden_layers):
            raw_kvs.append(torch.randn(H, B, D, S))  # hub key format
            raw_kvs.append(torch.randn(H, B, S, D))  # hub value format
        raw_outputs = (logits, *raw_kvs)

        local_outputs = hub_gen.parse_model_outputs(raw_outputs)
        hub_gen.combine_local_and_global_outputs(4, local_outputs, global_outputs)
        # After combine, KV should be in HF format: (B, H, S, D)
        kv_cache = global_outputs["past_key_values"]
        assert len(kv_cache) > 0
        # Keys should have B first
        assert kv_cache[0].shape[0] == B

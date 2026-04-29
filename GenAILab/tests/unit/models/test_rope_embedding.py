# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for RoPE embedding utility."""

import pytest
import torch
from unittest.mock import MagicMock
from transformers.models.llama import LlamaConfig
from transformers.models.qwen2 import Qwen2Config

from GenAILab.qai_hub_lm.utils.rope_embedding import RopeEmbedding


def _make_model(config):
    model = MagicMock()
    model.config = config
    return model


class TestRopeEmbedding:
    def test_llama_config(self):
        config = LlamaConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_hidden_layers=2,
            intermediate_size=128,
        )
        model = _make_model(config)
        rope = RopeEmbedding(model, context_length=32)
        assert rope.cos is not None
        assert rope.sin is not None

    def test_qwen2_config(self):
        config = Qwen2Config(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_hidden_layers=2,
            intermediate_size=128,
        )
        model = _make_model(config)
        rope = RopeEmbedding(model, context_length=32)
        assert rope.cos is not None

    def test_unknown_config_raises(self):
        class UnknownConfig:
            pass

        model = _make_model(UnknownConfig())
        with pytest.raises(RuntimeError, match="Unknown rotary"):
            RopeEmbedding(model, context_length=32)

    def test_get_embedding_shape(self):
        config = LlamaConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_hidden_layers=2,
            intermediate_size=128,
        )
        model = _make_model(config)
        rope = RopeEmbedding(model, context_length=64)
        position_ids = torch.arange(8).unsqueeze(0)  # (1, 8)
        cos, sin = rope.get_embedding(position_ids)
        assert cos.shape[0] == 1  # batch
        assert cos.shape[2] == 8  # sequence_length

    def test_position_aware(self):
        config = LlamaConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_hidden_layers=2,
            intermediate_size=128,
        )
        model = _make_model(config)
        rope = RopeEmbedding(model, context_length=64)
        pos_a = torch.tensor([[0, 1, 2, 3]])
        pos_b = torch.tensor([[4, 5, 6, 7]])
        cos_a, _ = rope.get_embedding(pos_a)
        cos_b, _ = rope.get_embedding(pos_b)
        assert not torch.equal(cos_a, cos_b)

    def test_dtype(self):
        config = LlamaConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_hidden_layers=2,
            intermediate_size=128,
        )
        model = _make_model(config)
        rope = RopeEmbedding(model, context_length=32)
        pos = torch.tensor([[0, 1, 2, 3]])
        cos, sin = rope.get_embedding(pos, dtype=torch.float16)
        assert cos.dtype == torch.float16
        assert sin.dtype == torch.float16

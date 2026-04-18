# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared test fixtures for GenAILab unit and integration tests."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch


# ---------------------------------------------------------------------------
# Mock model config
# ---------------------------------------------------------------------------


@dataclass
class _MinimalConfig:
    """Minimal model config that mirrors the fields Generator and LLM base use."""

    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    hidden_size: int = 64
    head_dim: int = 16  # hidden_size // num_attention_heads
    model_type: str = "llama"
    vocab_size: int = 256
    image_token_id: int = 128
    layer_types: list = None


@pytest.fixture
def mock_model_config():
    return _MinimalConfig()


# ---------------------------------------------------------------------------
# Mock torch model
# ---------------------------------------------------------------------------


class _TinyModel(torch.nn.Module):
    """Minimal nn.Module with .config and .dtype for tests that need a model object."""

    def __init__(self, config=None):
        super().__init__()
        self.cfg = config or _MinimalConfig()
        self.linear = torch.nn.Linear(
            self.cfg.hidden_size, self.cfg.vocab_size, bias=False
        )

    @property
    def config(self):
        return self.cfg

    @property
    def device(self):
        return self.linear.weight.device

    @property
    def dtype(self):
        return self.linear.weight.dtype

    def forward(self, *args):
        # Return (logits, *kv_pairs) mimicking an LLM forward signature
        batch = args[0].shape[0] if len(args) > 0 else 1
        seq_len = args[0].shape[1] if len(args) > 0 and args[0].dim() >= 2 else 1
        logits = torch.randn(batch, seq_len, self.cfg.vocab_size)
        num_kv = self.cfg.num_hidden_layers * 2
        kv_shape = (batch, self.cfg.num_key_value_heads, seq_len, self.cfg.head_dim)
        kv_tensors = [torch.randn(kv_shape) for _ in range(num_kv)]
        return (logits, *kv_tensors)


@pytest.fixture
def mock_model(mock_model_config):
    return _TinyModel(mock_model_config)


# ---------------------------------------------------------------------------
# Mock tokenizer
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.eos_token_id = 0
    tok.pad_token_id = 0
    tok.bos_token = "<s>"
    tok.__call__ = MagicMock(
        return_value={"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
    )
    return tok


# ---------------------------------------------------------------------------
# Sample YAML config dict
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config():
    return {
        "model": {
            "model_id": "meta-llama/Llama-3.2-1B-Instruct",
            "sequence_length": 32,
            "context_length": 64,
        },
        "recipe": {
            "backbone": {
                "name": "RemoveQuantization",
            }
        },
        "metrics": [{"name": "TinyMMLU"}],
    }


# ---------------------------------------------------------------------------
# Sample precision config
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_precision():
    from GenAILab.shared.helpers.precision_config import PrecisionConfig

    return PrecisionConfig()


# ---------------------------------------------------------------------------
# Temp cache directory
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_cache_dir(tmp_path):
    return tmp_path / "cache"


# ---------------------------------------------------------------------------
# Sample profiler data structures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_metric_results():
    from GenAILab.shared.helpers.profiler import MetricResult

    return [
        MetricResult(metric_name="PPL", result=12.5, profiler=None),
        MetricResult(metric_name="TinyMMLU", result=65.0, profiler=None),
    ]


@pytest.fixture
def sample_components():
    from GenAILab.shared.helpers.profiler import ComponentRecipeStats

    return {
        "backbone": ComponentRecipeStats(
            recipe_name="Calibration",
            recipe_kwargs={"num_batches": 32},
            dataset_name="Wikitext",
            dataset_kwargs={"split": "train"},
            profiler=None,
        )
    }

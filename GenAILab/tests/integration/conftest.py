# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared fixtures for Generator parity integration tests."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
]


@pytest.fixture(scope="module", params=VLM_MODELS)
def vlm_bundle(request):
    """Load a full-size VLM and its processor (shared across tests in module)."""
    from transformers import AutoProcessor

    model_id = request.param
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation="eager",
        dtype=torch.float32,
    ).cpu()
    model.eval()

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

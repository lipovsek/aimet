# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for VLM_Generator — vision fusion, position processing."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from GenAILab.shared.models.generator import VLM_Generator


@dataclass
class _VLMCfg:
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    hidden_size: int = 64
    head_dim: int = 16
    vocab_size: int = 256
    image_token_id: int = 128
    video_token_id: int = 129


class _DummyBackbone(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
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
        embeds = args[0]
        b, s = embeds.shape[:2]
        logits = torch.randn(b, s, self.cfg.vocab_size)
        kv_shape = (b, self.cfg.num_key_value_heads, s, self.cfg.head_dim)
        kvs = [torch.randn(kv_shape) for _ in range(self.cfg.num_hidden_layers * 2)]
        return (logits, *kvs)


class _DummyVision(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, pixel_values=None, image_grid_thw=None, mask=None, **kwargs):
        if pixel_values is None:
            return None
        # Return embeddings matching the image token count.
        # mask is a 2D boolean tensor (batch, padded_seq_len).
        num_image_tokens = mask.sum().item() if mask is not None else 4
        return torch.randn(int(num_image_tokens), self.hidden_size)


@pytest.fixture
def vlm_cfg():
    return _VLMCfg()


@pytest.fixture
def vlm_gen(vlm_cfg):
    backbone = _DummyBackbone(vlm_cfg)
    vision = _DummyVision(vlm_cfg.hidden_size)
    embedding = torch.nn.Embedding(vlm_cfg.vocab_size, vlm_cfg.hidden_size)
    tok = MagicMock()
    tok.eos_token_id = 0
    return VLM_Generator(
        backbone_model=backbone,
        vision_model=vision,
        embedding=embedding,
        tokenizer=tok,
        sequence_length=8,
        context_length=32,
        config=vlm_cfg,
    )


class TestVLMFusion:
    def test_fuse_text_image_embeddings(self, vlm_gen, vlm_cfg):
        # Create input_ids with some image tokens
        input_ids = torch.randint(0, 200, (1, 6))
        input_ids[0, 2] = vlm_cfg.image_token_id
        input_ids[0, 3] = vlm_cfg.image_token_id
        pixel_values = torch.randn(2, 3, 224, 224)

        embeds, mm_token_type_ids, extra = vlm_gen.fuse_text_image_video(
            input_ids=input_ids,
            pixel_values=pixel_values,
        )
        assert embeds.shape == (1, 6, vlm_cfg.hidden_size)

    def test_fuse_text_no_images(self, vlm_gen, vlm_cfg):
        input_ids = torch.randint(0, 100, (1, 6))  # no image tokens
        embeds, mm_token_type_ids, extra = vlm_gen.fuse_text_image_video(
            input_ids=input_ids,
            pixel_values=None,
        )
        assert embeds.shape == (1, 6, vlm_cfg.hidden_size)

    def test_video_raises(self, vlm_gen):
        with pytest.raises(RuntimeError, match="No support for video"):
            vlm_gen.fuse_text_image_video(
                input_ids=torch.tensor([[1, 2, 3]]),
                pixel_values_videos=torch.randn(1, 3, 224, 224),
            )


class TestVLMForward:
    def test_forward_returns_output(self, vlm_gen, vlm_cfg):
        input_ids = torch.randint(0, 100, (1, 4))
        result = vlm_gen.forward(input_ids=input_ids)
        assert isinstance(result, CausalLMOutputWithPast)
        assert result.logits.shape[1] == 4

    def test_position_processor_called(self, vlm_cfg):
        backbone = _DummyBackbone(vlm_cfg)
        vision = _DummyVision(vlm_cfg.hidden_size)
        embedding = torch.nn.Embedding(vlm_cfg.vocab_size, vlm_cfg.hidden_size)
        tok = MagicMock()
        tok.eos_token_id = 0

        processor_called = False

        def position_processor(
            self_gen,
            input_ids=None,
            image_grid_thw=None,
            video_grid_thw=None,
            attention_mask=None,
            mm_token_type_ids=None,
        ):
            nonlocal processor_called
            processor_called = True
            return None

        gen = VLM_Generator(
            backbone_model=backbone,
            vision_model=vision,
            embedding=embedding,
            tokenizer=tok,
            sequence_length=8,
            context_length=32,
            config=vlm_cfg,
            position_id_processor=position_processor,
        )
        gen.forward(input_ids=torch.randint(0, 100, (1, 4)))
        assert processor_called

    def test_no_position_processor(self, vlm_gen):
        # Should work without a position_id_processor
        assert vlm_gen.position_id_processor is None
        result = vlm_gen.forward(input_ids=torch.randint(0, 100, (1, 4)))
        assert result.logits is not None


class TestVLMPrefill:
    def test_prefill_yields(self, vlm_gen):
        input_ids = torch.randint(0, 100, (1, 4))
        slices = list(vlm_gen.prefill(input_ids=input_ids))
        assert len(slices) >= 1
        assert isinstance(slices[0], tuple)

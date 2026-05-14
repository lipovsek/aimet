# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for LLM/VLM/SimCollection base classes."""

from unittest.mock import MagicMock

import pytest

from GenAILab.qai_hub_lm.models.base import LLM, VLM, SimCollection
from GenAILab.qai_hub_lm.models.utils.layer_cache import (
    LayerCacheDescriptor,
    AttentionType,
)


def _make_descriptors(num_layers):
    """Create simple FULL-attention descriptors for testing."""
    return [
        LayerCacheDescriptor(
            layer_idx=i,
            attention_type=AttentionType.FULL,
            num_kv_heads=4,
            head_dim=16,
        )
        for i in range(num_layers)
    ]


class TestSimCollection:
    def test_is_vlm_false(self):
        sc = SimCollection(backbone=MagicMock(), visual=None)
        assert sc.is_vlm() is False

    def test_is_vlm_true(self):
        sc = SimCollection(backbone=MagicMock(), visual=MagicMock())
        assert sc.is_vlm() is True

    def test_fields(self):
        backbone = MagicMock()
        visual = MagicMock()
        embedding = MagicMock()
        config = MagicMock()
        sc = SimCollection(backbone, visual, embedding, config)
        assert sc.backbone is backbone
        assert sc.visual is visual
        assert sc.embedding is embedding
        assert sc.config is config


class TestLLMBase:
    def test_backbone_input_names(self):
        descs = _make_descriptors(2)
        names = LLM.get_backbone_input_names(descs)
        assert names[0] == "input_ids"
        assert names[1] == "attention_mask"
        assert names[2] == "position_ids"
        # 2 layers * 2 (key + value) = 4 KV entries
        assert "past_key_0_in" in names
        assert "past_value_0_in" in names
        assert "past_key_1_in" in names
        assert "past_value_1_in" in names
        assert len(names) == 3 + 2 * 2  # 3 base + 4 KV

    def test_backbone_output_names(self):
        descs = _make_descriptors(2)
        names = LLM.get_backbone_output_names(descs)
        assert names[0] == "logits"
        assert "past_key_0_out" in names
        assert "past_value_1_out" in names
        assert len(names) == 1 + 2 * 2  # 1 logits + 4 KV


class TestVLMBase:
    def test_backbone_input_names_uses_inputs_embeds(self):
        descs = _make_descriptors(2)
        names = VLM.get_backbone_input_names(descs)
        assert names[0] == "inputs_embeds"
        assert "input_ids" not in names

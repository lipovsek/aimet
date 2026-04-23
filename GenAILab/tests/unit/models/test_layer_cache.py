# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for per-layer cache descriptors."""

from dataclasses import dataclass

import pytest

from GenAILab.shared.models.utils.layer_cache import (
    AttentionType,
    LayerCacheDescriptor,
    build_layer_cache_descriptors,
)


class TestLayerCacheDescriptor:
    def test_full_attention_shapes(self):
        desc = LayerCacheDescriptor(
            layer_idx=0,
            attention_type=AttentionType.FULL,
            num_kv_heads=4,
            head_dim=16,
        )
        shape_a, shape_b = desc.dummy_state_shapes(
            batch_size=1, context_length=32, sequence_length=8
        )
        # Full attention: (batch, kv_heads, context - seq, head_dim)
        assert shape_a == (1, 4, 24, 16)
        assert shape_b == (1, 4, 24, 16)

    def test_linear_attention_shapes(self):
        desc = LayerCacheDescriptor(
            layer_idx=0,
            attention_type=AttentionType.LINEAR,
            num_kv_heads=4,
            head_dim=16,
            conv_dim=640,
            conv_kernel_size=4,
            linear_num_v_heads=8,
            linear_head_k_dim=64,
            linear_head_v_dim=128,
        )
        conv_shape, recurrent_shape = desc.dummy_state_shapes(
            batch_size=1, context_length=32, sequence_length=8
        )
        assert conv_shape == (1, 640, 4)
        assert recurrent_shape == (1, 8, 64, 128)

    def test_clip_length_full(self):
        desc = LayerCacheDescriptor(
            layer_idx=0,
            attention_type=AttentionType.FULL,
            num_kv_heads=4,
            head_dim=16,
        )
        assert desc.clip_length(100) == 100

    def test_clip_length_sliding_window(self):
        desc = LayerCacheDescriptor(
            layer_idx=0,
            attention_type=AttentionType.SLIDING_WINDOW,
            num_kv_heads=4,
            head_dim=16,
            sliding_window_size=32,
        )
        assert desc.clip_length(100) == 32
        assert desc.clip_length(16) == 16

    def test_clip_length_linear(self):
        desc = LayerCacheDescriptor(
            layer_idx=0,
            attention_type=AttentionType.LINEAR,
            num_kv_heads=4,
            head_dim=16,
        )
        assert desc.clip_length(100) is None


class TestBuildLayerCacheDescriptors:
    def test_simple_full_attention(self):
        @dataclass
        class Cfg:
            num_hidden_layers: int = 4
            num_attention_heads: int = 8
            num_key_value_heads: int = 4
            hidden_size: int = 128
            head_dim: int = 16

        descs = build_layer_cache_descriptors(Cfg())
        assert len(descs) == 4
        assert all(d.attention_type == AttentionType.FULL for d in descs)
        assert all(d.num_kv_heads == 4 for d in descs)
        assert all(d.head_dim == 16 for d in descs)

    def test_mixed_attention_from_layer_types(self):
        @dataclass
        class Cfg:
            num_hidden_layers: int = 4
            num_attention_heads: int = 8
            num_key_value_heads: int = 4
            hidden_size: int = 128
            head_dim: int = 16
            layer_types: list = None
            linear_num_key_heads: int = 8
            linear_num_value_heads: int = 8
            linear_key_head_dim: int = 64
            linear_value_head_dim: int = 128
            linear_conv_kernel_dim: int = 4

        cfg = Cfg(
            layer_types=[
                "linear_attention",
                "full_attention",
                "linear_attention",
                "full_attention",
            ]
        )
        descs = build_layer_cache_descriptors(cfg)
        assert descs[0].attention_type == AttentionType.LINEAR
        assert descs[1].attention_type == AttentionType.FULL
        assert descs[2].attention_type == AttentionType.LINEAR
        assert descs[3].attention_type == AttentionType.FULL

    def test_linear_dimensions_populated(self):
        @dataclass
        class Cfg:
            num_hidden_layers: int = 2
            num_attention_heads: int = 8
            num_key_value_heads: int = 4
            hidden_size: int = 128
            head_dim: int = 16
            layer_types: list = None
            linear_num_key_heads: int = 8
            linear_num_value_heads: int = 8
            linear_key_head_dim: int = 64
            linear_value_head_dim: int = 128
            linear_conv_kernel_dim: int = 4

        cfg = Cfg(layer_types=["linear_attention", "full_attention"])
        descs = build_layer_cache_descriptors(cfg)

        linear_desc = descs[0]
        # conv_dim = k_dim * k_heads * 2 + v_dim * v_heads
        assert linear_desc.conv_dim == 64 * 8 * 2 + 128 * 8
        assert linear_desc.conv_kernel_size == 4
        assert linear_desc.linear_num_v_heads == 8
        assert linear_desc.linear_head_k_dim == 64
        assert linear_desc.linear_head_v_dim == 128

        full_desc = descs[1]
        assert full_desc.conv_dim is None
        assert full_desc.conv_kernel_size is None

    def test_head_dim_derived_from_hidden_size(self):
        @dataclass
        class Cfg:
            num_hidden_layers: int = 1
            num_attention_heads: int = 8
            num_key_value_heads: int = 4
            hidden_size: int = 128

        descs = build_layer_cache_descriptors(Cfg())
        assert descs[0].head_dim == 16  # 128 // 8

    def test_sliding_window_pattern(self):
        @dataclass
        class Cfg:
            num_hidden_layers: int = 4
            num_attention_heads: int = 8
            num_key_value_heads: int = 4
            hidden_size: int = 128
            head_dim: int = 16
            sliding_window: int = 32
            sliding_window_pattern: int = 2

        descs = build_layer_cache_descriptors(Cfg())
        # Pattern=2: layers 0, 2 are sliding window
        assert descs[0].attention_type == AttentionType.SLIDING_WINDOW
        assert descs[1].attention_type == AttentionType.FULL
        assert descs[2].attention_type == AttentionType.SLIDING_WINDOW
        assert descs[3].attention_type == AttentionType.FULL
        assert descs[0].sliding_window_size == 32

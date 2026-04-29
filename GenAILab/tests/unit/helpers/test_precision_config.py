# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for PrecisionConfig and WeightPrecision."""

import pytest

from GenAILab.qai_hub_lm.precision import (
    Granularity,
    PrecisionConfig,
    WeightPrecision,
    resolve_qtype,
    int2,
    int4,
    int8,
    int16,
    float16,
    float32,
)


# ---------------------------------------------------------------------------
# resolve_qtype
# ---------------------------------------------------------------------------


class TestResolveQtype:
    def test_int(self):
        assert resolve_qtype(4) == int4
        assert resolve_qtype(8) == int8

    def test_string(self):
        assert resolve_qtype("int4") == int4
        assert resolve_qtype("int8") == int8

    def test_passthrough(self):
        assert resolve_qtype(int4) is int4

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            resolve_qtype("garbage")


# ---------------------------------------------------------------------------
# WeightPrecision
# ---------------------------------------------------------------------------


class TestWeightPrecision:
    def test_defaults(self):
        wp = WeightPrecision()
        assert wp.qtype == int4
        assert wp.granularity == Granularity.PCQ
        assert wp.block_size is None

    def test_invalid_granularity(self):
        with pytest.raises(ValueError, match="Invalid granularity"):
            WeightPrecision(granularity="XYZ")

    def test_from_dict_none(self):
        wp = WeightPrecision.from_dict(None, qtype=int8)
        assert wp.qtype == int8

    def test_from_dict_int_shorthand(self):
        wp = WeightPrecision.from_dict(4)
        assert wp.qtype == int4

    def test_from_dict_full(self):
        wp = WeightPrecision.from_dict(
            {"qtype": 8, "granularity": "BQ", "block_size": 128}
        )
        assert wp.qtype == int8
        assert wp.granularity == Granularity.BQ
        assert wp.block_size == 128

    def test_float_qtype_raises(self):
        with pytest.raises(ValueError, match="Floating-point"):
            WeightPrecision.from_dict({"qtype": "float16"})

    def test_to_dict(self):
        wp = WeightPrecision(qtype=int4, granularity=Granularity.LPBQ, block_size=32)
        d = wp.to_dict()
        assert d["granularity"] == "LPBQ"
        assert d["block_size"] == 32
        assert "int4" in d["qtype"]


# ---------------------------------------------------------------------------
# PrecisionConfig
# ---------------------------------------------------------------------------


class TestPrecisionConfig:
    def test_defaults(self):
        pc = PrecisionConfig()
        assert pc.activations == int16
        assert pc.kv_cache == int8
        assert pc.embedding == int16
        assert pc.lm_head.qtype == int8
        assert pc.blocks["default"].qtype == int4
        assert pc.visual_weight is None

    def test_from_dict_none(self):
        pc = PrecisionConfig.from_dict(None)
        assert pc.activations == int16

    def test_from_dict_activations(self):
        pc = PrecisionConfig.from_dict({"activations": 8})
        assert pc.activations == int8

    def test_from_dict_kv_cache(self):
        pc = PrecisionConfig.from_dict({"kv_cache": 4})
        assert pc.kv_cache == int4

    def test_from_dict_lm_head(self):
        pc = PrecisionConfig.from_dict(
            {"lm_head": {"qtype": 4, "granularity": "BQ", "block_size": 64}}
        )
        assert pc.lm_head.qtype == int4
        assert pc.lm_head.granularity == Granularity.BQ
        assert pc.lm_head.block_size == 64

    def test_from_dict_blocks_shorthand_int(self):
        pc = PrecisionConfig.from_dict({"blocks": 8})
        assert pc.blocks["default"].qtype == int8

    def test_from_dict_blocks_dict(self):
        pc = PrecisionConfig.from_dict(
            {
                "blocks": {
                    "default": {"qtype": 4, "granularity": "BQ", "block_size": 128}
                }
            }
        )
        assert pc.blocks["default"].qtype == int4
        assert pc.blocks["default"].granularity == Granularity.BQ
        assert pc.blocks["default"].block_size == 128

    def test_from_dict_blocks_flat_wp_dict(self):
        pc = PrecisionConfig.from_dict(
            {"blocks": {"qtype": 4, "granularity": "LPBQ", "block_size": 128}}
        )
        assert pc.blocks["default"].granularity == Granularity.LPBQ
        assert pc.blocks["default"].block_size == 128

    def test_from_dict_blocks_non_default_raises(self):
        with pytest.raises(ValueError, match="not yet supported"):
            PrecisionConfig.from_dict({"blocks": {"layer_0": {"qtype": 4}}})

    def test_from_dict_visual(self):
        pc = PrecisionConfig.from_dict(
            {
                "visual": {
                    "weight": {"qtype": 8, "granularity": "PCQ"},
                    "activations": 16,
                }
            }
        )
        assert pc.visual_weight.qtype == int8
        assert pc.visual_activations == int16

    def test_from_dict_visual_no_weight(self):
        pc = PrecisionConfig.from_dict({"visual": {}})
        assert pc.visual_weight.qtype == int8  # default

    def test_from_dict_embedding(self):
        pc = PrecisionConfig.from_dict({"embedding": 8})
        assert pc.embedding == int8

    def test_to_dict_roundtrip(self):
        original = PrecisionConfig.from_dict(
            {
                "activations": 8,
                "kv_cache": 4,
                "lm_head": {"qtype": 4, "granularity": "BQ", "block_size": 64},
                "blocks": {"qtype": 4, "granularity": "LPBQ", "block_size": 32},
            }
        )
        d = original.to_dict()
        assert "activations" in d
        assert "kv_cache" in d
        assert "lm_head" in d
        assert "blocks" in d

    def test_resolve_kv_cache_float_activation(self):
        pc = PrecisionConfig(activations=float16)
        resolved = pc.resolve_kv_cache_qtype()
        assert resolved == float16

    def test_resolve_kv_cache_default(self):
        pc = PrecisionConfig()
        resolved = pc.resolve_kv_cache_qtype()
        assert resolved == int8

    def test_resolve_kv_cache_override(self):
        pc = PrecisionConfig()
        resolved = pc.resolve_kv_cache_qtype(int4)
        assert resolved == int4

    def test_resolve_embedding_float_activation(self):
        pc = PrecisionConfig(activations=float32)
        resolved = pc.resolve_embedding_qtype()
        assert resolved == float32

    def test_resolve_embedding_default(self):
        pc = PrecisionConfig()
        resolved = pc.resolve_embedding_qtype()
        assert resolved == int16

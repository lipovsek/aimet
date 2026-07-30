# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the clean (aimet-free) precision schema.

These run anywhere -- no aimet. Cluster-only round-trip parity against the old
PrecisionConfig.from_dict lives in test_precision_parity.py (skipped here).
"""

import pytest
from pydantic import ValidationError

from GenAILab.qai_hub_lm.schema import (
    Granularity,
    PrecisionSchema,
    QType,
    WeightPrecisionSchema,
)


class TestDefaultsAreTheContract:
    """Omitted section / fields must reproduce the documented defaults:
    W4A16, lm_head W8 PCQ, KV cache int8, embedding int16."""

    def test_empty_precision_defaults(self):
        p = PrecisionSchema()
        assert p.activations == QType.int16
        assert p.kv_cache == QType.int8
        assert p.embedding == QType.int16
        assert p.lm_head.qtype == QType.int8
        assert p.lm_head.granularity == Granularity.PCQ
        assert p.blocks["default"].qtype == QType.int4
        assert p.visual is None

    def test_partial_override_keeps_other_defaults(self):
        p = PrecisionSchema.model_validate({"activations": "int8"})
        assert p.activations == QType.int8
        assert p.kv_cache == QType.int8  # default retained
        assert p.blocks["default"].qtype == QType.int4  # default retained


class TestRealConfigShapes:
    """The precision blocks that appear in configs/*.yaml."""

    def test_torch_nightly_blocks_lpbq(self):
        # precision: {blocks: {qtype: int4, granularity: LPBQ, block_size: 64}}
        p = PrecisionSchema.model_validate(
            {"blocks": {"qtype": "int4", "granularity": "LPBQ", "block_size": 64}}
        )
        d = p.blocks["default"]
        assert d.qtype == QType.int4
        assert d.granularity == Granularity.LPBQ
        assert d.block_size == 64

    def test_onnx_nightly_blocks_bq(self):
        p = PrecisionSchema.model_validate(
            {"blocks": {"qtype": "int4", "granularity": "BQ", "block_size": 32}}
        )
        assert p.blocks["default"].granularity == Granularity.BQ


class TestBlocksShorthand:
    """from_dict accepted: bare int/str, flat dict, or {default: {...}}."""

    def test_bare_int_shorthand(self):
        p = PrecisionSchema.model_validate({"blocks": 8})
        assert p.blocks["default"].qtype == 8  # bare int bitwidth preserved

    def test_flat_dict_wrapped_as_default(self):
        p = PrecisionSchema.model_validate(
            {"blocks": {"qtype": "int4", "granularity": "PCQ"}}
        )
        assert p.blocks["default"].qtype == QType.int4

    def test_explicit_default_mapping(self):
        p = PrecisionSchema.model_validate({"blocks": {"default": {"qtype": "int8"}}})
        assert p.blocks["default"].qtype == QType.int8

    def test_per_block_range_rejected(self):
        with pytest.raises(ValidationError):
            PrecisionSchema.model_validate(
                {"blocks": {"default": {"qtype": "int4"}, "0-10": {"qtype": "int8"}}}
            )


class TestWeightPrecisionRules:
    def test_block_size_required_for_lpbq_int(self):
        with pytest.raises(ValidationError):
            WeightPrecisionSchema.model_validate(
                {"qtype": "int4", "granularity": "LPBQ"}
            )

    def test_float_weight_ignores_block_size_requirement(self):
        # float weights ignore granularity -> no block_size needed
        wp = WeightPrecisionSchema.model_validate(
            {"qtype": "float16", "granularity": "LPBQ"}
        )
        assert wp.is_float

    def test_qtype_shorthand_bare_value(self):
        wp = WeightPrecisionSchema.model_validate("int8")
        assert wp.qtype == QType.int8


class TestVisual:
    def test_visual_weight_and_activations(self):
        p = PrecisionSchema.model_validate(
            {"visual": {"weight": {"qtype": "int8"}, "activations": "int16"}}
        )
        assert p.visual.weight.qtype == QType.int8
        assert p.visual.activations == QType.int16

    def test_float_visual_weight_rejected(self):
        with pytest.raises(ValidationError):
            PrecisionSchema.model_validate({"visual": {"weight": {"qtype": "float16"}}})


class TestForbidExtra:
    def test_typo_key_rejected(self):
        # the silent-drop bug the conversion fixes: 'granularty' typo
        with pytest.raises(ValidationError):
            WeightPrecisionSchema.model_validate({"qtype": "int4", "granularty": "PCQ"})

    def test_unknown_top_level_key_rejected(self):
        with pytest.raises(ValidationError):
            PrecisionSchema.model_validate({"activation": "int16"})  # missing 's'


class TestJsonSchema:
    def test_emits_constrained_spec(self):
        s = PrecisionSchema.model_json_schema()
        assert "properties" in s
        # QType enum should appear as a constrained set, not an opaque object
        assert "$defs" in s and "QType" in s["$defs"]

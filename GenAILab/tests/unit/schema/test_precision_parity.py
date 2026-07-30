# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Round-trip parity: the new schema path must match the old PrecisionConfig.

This is the de-risking net for the precision conversion. For every input dict,
the OLD path ``PrecisionConfig.from_dict(d).to_dict()`` must equal the NEW path
``PrecisionConfig.from_schema(PrecisionSchema.model_validate(d)).to_dict()``.

Requires aimet (PrecisionConfig resolves qtype objects), so it SKIPS where aimet
is unavailable and runs on the cluster / in CI. The corpus is the real
configs/*.yaml precision blocks PLUS synthesized cases covering every from_dict
branch -- the real configs alone are too few to exercise the polymorphic
``blocks`` / ``visual`` / shorthand surface.
"""

import pytest

aimet_precision = pytest.importorskip(
    "GenAILab.bench.precision",
    reason="precision parity requires aimet (qtype resolution)",
)
PrecisionConfig = aimet_precision.PrecisionConfig

from GenAILab.qai_hub_lm.schema import PrecisionSchema  # noqa: E402


# Synthesized corpus: each entry exercises a distinct from_dict branch.
CASES = [
    pytest.param(None, id="omitted-section"),
    pytest.param({}, id="empty-dict"),
    pytest.param({"activations": "int8"}, id="activations-override"),
    pytest.param({"activations": 16}, id="activations-bare-int"),
    pytest.param({"kv_cache": "int16"}, id="kv-cache-override"),
    pytest.param({"embedding": "int8"}, id="embedding-override"),
    pytest.param({"lm_head": {"qtype": "int4"}}, id="lm-head-dict"),
    pytest.param({"lm_head": 8}, id="lm-head-bare-int"),
    pytest.param({"blocks": 8}, id="blocks-bare-int"),
    pytest.param({"blocks": {"qtype": "int4"}}, id="blocks-flat-dict"),
    pytest.param(
        {"blocks": {"qtype": "int4", "granularity": "LPBQ", "block_size": 64}},
        id="blocks-lpbq",
    ),
    pytest.param(
        {"blocks": {"qtype": "int4", "granularity": "BQ", "block_size": 32}},
        id="blocks-bq",
    ),
    pytest.param(
        {"blocks": {"default": {"qtype": "int8"}}}, id="blocks-explicit-default"
    ),
    pytest.param({"blocks": {"qtype": "float16"}}, id="blocks-float"),
    pytest.param(
        {"visual": {"weight": {"qtype": "int8"}, "activations": "int16"}},
        id="visual-full",
    ),
    pytest.param({"visual": {"weight": {"qtype": "int4"}}}, id="visual-weight-only"),
    pytest.param(
        {
            "activations": "int16",
            "kv_cache": "int8",
            "embedding": "int16",
            "lm_head": {"qtype": "int8", "granularity": "PCQ"},
            "blocks": {"qtype": "int4", "granularity": "LPBQ", "block_size": 64},
        },
        id="full-w4a16-recipe",
    ),
]


@pytest.mark.parametrize("raw", CASES)
def test_from_schema_matches_from_dict(raw):
    old = PrecisionConfig.from_dict(raw).to_dict()
    schema = PrecisionSchema.model_validate(raw if raw is not None else {})
    new = PrecisionConfig.from_schema(schema).to_dict()
    assert new == old, f"parity mismatch for {raw!r}\n old={old}\n new={new}"

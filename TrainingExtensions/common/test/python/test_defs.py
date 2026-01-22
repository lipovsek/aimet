# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import pytest

try:
    from aimet_onnx.common.defs import qtype, QTYPE_ALIASES
except ImportError:
    from aimet_torch.common.defs import qtype, QTYPE_ALIASES


def test_qtypes():
    assert str(qtype.float(5, 10, False, False)) == "float16"
    assert str(qtype.float(2, 1, False, False)) == "float4e2m1"
    assert str(qtype.float(4, 3, True, True)) == "float8e4m3fnuz"
    assert str(qtype.float(5, 2, False, False)) == "float8e5m2"

    assert str(qtype.int(3)) == "int3"
    assert qtype.int(16) == QTYPE_ALIASES["int16"]
    assert qtype.int(8).bits == 8

    assert qtype.float(5, 10, False, False) == QTYPE_ALIASES["float16"]
    assert QTYPE_ALIASES["float16"].mantissa_bits == 10
    assert QTYPE_ALIASES["float16"].exponent_bits == 5


def test_invalid_qtypes():
    with pytest.raises(ValueError):
        qtype.int(0)

    with pytest.raises(ValueError):
        qtype.float(-1, 4)

    with pytest.raises(ValueError):
        qtype.float(1, -1)

# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

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

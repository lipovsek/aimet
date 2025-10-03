# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

import random
import tempfile
import torch
import numpy as np
import warnings
from aimet_torch.v2.quantization.encoding_analyzer import MinMaxEncodingAnalyzer
from aimet_torch.v2.quantization import DequantizedTensor
from aimet_torch.v2.quantization.float import FloatQuantizeDequantize, FloatEncoding
from aimet_torch.v2.quantization.float._finfo import _finfo
from aimet_torch.fp_quantization import fake_cast_to_ieee_float


@pytest.fixture(autouse=True)
def set_seed():
    random.seed(999)
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture()
def x():
    """
    Returns [
        [-2., -1.99, -1.98, ..., -1.01],
        [-1., -0.99, -0.98, ..., -0.01],
        [ 0.,  0.01,  0.02, ...,  0.99],
        [ 1.,  1.01,  1.02, ...,  1.99],
    ]
    """
    return torch.arange(-200, 200).view(4, 100) / 100


@torch.no_grad()
@pytest.mark.parametrize(
    "dtype,               exponent_bits, mantissa_bits, finite, unsigned_zero",
    [
        (torch.float16, 5, 10, False, False),
        (torch.bfloat16, 8, 7, False, False),
        (torch.float8_e5m2, 5, 2, False, False),
        (torch.float8_e4m3fn, 4, 3, True, False),
        # NOTE: Not supported in torch 2.1
        # (torch.float8_e5m2fnuz, 5,             2,             False,  False),
        # (torch.float8_e4m3fnuz, 4,             3,             True,   False),
    ],
)
def test_qdq_output_standard_dtypes(
    x, dtype, exponent_bits, mantissa_bits, finite, unsigned_zero
):
    """
    Given: Instantiated FloatQuantizeDequantize with a well-known dtype of pytorch
    When: Run forward
    Then: Output should be equal to downcasting and upcasting the input
    """
    float_qdq = FloatQuantizeDequantize(dtype=dtype)
    expected_output = x.to(dtype).float()
    assert torch.equal(float_qdq(x), expected_output)

    """
    Given: Instantiated two quantizers:
        - FloatQuantizeDequantize(dtype=dtype)
        - FloatQuantizeDequantize(exponent_bits, mantissa_bits)

        where exponent_bits and mantissa_bits corresponds to dtype
    When: Run forward
    Then: The two quantizers should produce same output
    """
    float_qdq_1 = FloatQuantizeDequantize(dtype=dtype)
    float_qdq_2 = FloatQuantizeDequantize(
        exponent_bits, mantissa_bits, finite, unsigned_zero
    )
    float_qdq_out_1 = float_qdq_1(x)
    float_qdq_out_2 = float_qdq_2(x)
    assert torch.equal(float_qdq_out_1, float_qdq_out_2)
    assert isinstance(float_qdq_out_1, DequantizedTensor)
    assert isinstance(float_qdq_out_2, DequantizedTensor)
    assert (
        float_qdq_out_1.encoding.exponent_bits
        == float_qdq_out_2.encoding.exponent_bits
        == exponent_bits
    )
    assert (
        float_qdq_out_1.encoding.mantissa_bits
        == float_qdq_out_2.encoding.mantissa_bits
        == mantissa_bits
    )
    assert float_qdq_out_1.dequantize() is float_qdq_out_1
    assert float_qdq_out_2.dequantize() is float_qdq_out_2


@pytest.mark.parametrize(
    "finite, unsigned_zero",
    [
        (True, True),
        (True, False),
        (False, True),
    ],
)
def test_special_floats_sanity(finite, unsigned_zero):
    ...
    """
    When: Instantiate non-builtin finite/unsigned_zero float qdq
    Then: Should throw runtime error
    """
    with pytest.raises(RuntimeError):
        _ = FloatQuantizeDequantize(3, 3, finite=finite, unsigned_zero=unsigned_zero)

    """
    Given: Start from a non-fininte, non-unsigned_zero float qdq
    When: Forcefully set fininte/unsigned_zero to True
    Then: Should throw runtime error
    """
    qdq = FloatQuantizeDequantize(3, 3, finite=False, unsigned_zero=False)
    qdq._finfo = _finfo(qdq.exponent_bits, qdq.mantissa_bits, finite, unsigned_zero)
    with pytest.raises(RuntimeError):
        _ = qdq(torch.randn(10))


@torch.no_grad()
@pytest.mark.parametrize("exponent_bits", [3, 4])
@pytest.mark.parametrize("mantissa_bits", [3, 4])
def test_qdq_output_non_standard_dtypes(x, exponent_bits, mantissa_bits):
    """
    Given: Instantiated FloatQuantizeDequantize with a non-standard float dtype
    When: Run forward
    Then: Output should be equal to fake-casting the input to the non-standard float
    """
    float_qdq = FloatQuantizeDequantize(exponent_bits, mantissa_bits)
    max_representable_value = _finfo(exponent_bits, mantissa_bits, False, False).max
    expected_output = fake_cast_to_ieee_float(
        x, max_representable_value, exponent_bits, mantissa_bits
    )
    assert torch.equal(float_qdq(x), expected_output)


@torch.no_grad()
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float8_e5m2,
        torch.float8_e4m3fn,
        # NOTE: Not supported in torch 2.1
        # torch.float8_e5m2fnuz,
        # torch.float8_e4m3fnuz,
    ],
)
def test_qdq_compute_encodings(dtype):
    """
    Given: Instantiated FloatQuantizeDequantize with a min-max encoding analyzer
    When: compute_encodings() and run forwad
    Then: Output should be equal to fake-casting the input
          with maximum representable value = observed maximum input
    """
    float8_tiny = torch.finfo(dtype).tiny
    float8_max = torch.finfo(dtype).max
    for x in [
        torch.arange(-2, 2, 0.004) * float8_max,
        torch.arange(-0.5, 0.5, 0.001) * float8_max,
    ]:
        x = x.view(10, 100)

        encoding_analyzer = MinMaxEncodingAnalyzer((100,))
        float8_qdq = FloatQuantizeDequantize(
            dtype=dtype, encoding_analyzer=encoding_analyzer
        )
        with float8_qdq.compute_encodings():
            _ = float8_qdq(x)

        scale = float8_qdq.get_scale()
        expected_scale = x.abs().max(dim=0).values / float8_max
        assert torch.allclose(scale, expected_scale)

        expected_output = (x / scale).clamp(-float8_max, float8_max).to(dtype).to(
            x.dtype
        ) * scale
        assert torch.allclose(float8_qdq(x), expected_output, atol=float8_tiny)


def test_allow_overwrite(x):
    exponent_bits, mantissa_bits = 3, 4
    q = FloatQuantizeDequantize(
        exponent_bits, mantissa_bits, encoding_analyzer=MinMaxEncodingAnalyzer((1, 100))
    )
    with q.compute_encodings():
        q(x)
    q_max = q.maxval.detach().clone()

    """
    Given: allow_overwrite set to False
    When: Try to recompute encodings
    Then: Encoding does NOT get overwritten by compute_encodings
    """
    q.allow_overwrite(False)
    assert not q.is_overwrite_allowed("maxval")
    # Check deprecated _allow_overwrite flag for backwards compatibility
    assert not q._allow_overwrite

    with q.compute_encodings():
        q(x * 2)

    assert torch.equal(q_max, q.maxval)

    """
    Given: allow_overwrite set to True
    When: Try to recompute encodings
    Then: Encoding does NOT get overwritten by compute_encodings
    """
    q.allow_overwrite(True)
    assert q.is_overwrite_allowed("maxval")
    # Check deprecated _allow_overwrite flag for backwards compatibility
    assert q._allow_overwrite

    with q.compute_encodings():
        q(x * 2)

    assert torch.equal(q.maxval, q_max * 2)


@pytest.mark.parametrize(
    "exponent_1, mantissa_1, encoding_analyzer_1",
    [(1, 2, MinMaxEncodingAnalyzer((1, 3))), (3, 4, None)],
)
@pytest.mark.parametrize(
    "exponent_2, mantissa_2, encoding_analyzer_2",
    [(5, 6, MinMaxEncodingAnalyzer((1, 3))), (7, 8, None)],
)
def test_save_and_load_state_dict(
    exponent_1,
    mantissa_1,
    encoding_analyzer_1,
    exponent_2,
    mantissa_2,
    encoding_analyzer_2,
):
    qtzr_1 = FloatQuantizeDequantize(
        exponent_1, mantissa_1, encoding_analyzer=encoding_analyzer_1
    )
    dummy_input = torch.randn(1, 3)
    with qtzr_1.compute_encodings():
        qtzr_1(dummy_input)

    qtzr_2 = FloatQuantizeDequantize(
        exponent_2, mantissa_2, encoding_analyzer=encoding_analyzer_2
    )
    with qtzr_2.compute_encodings():
        qtzr_2(dummy_input)
    assert not torch.allclose(
        qtzr_1(dummy_input), qtzr_2(dummy_input), atol=1e-7, rtol=1e-7
    )

    qtzr_1_state_dict = qtzr_1.state_dict()
    qtzr_2.load_state_dict(qtzr_1_state_dict)
    assert torch.equal(qtzr_1(dummy_input), qtzr_2(dummy_input))


def test_extreme_values_warning():
    extreme_val = torch.finfo(torch.float16).max
    dummy_input = torch.arange(start=0, end=extreme_val, dtype=torch.float16)
    encoding_shape = (1,)
    qdq = FloatQuantizeDequantize(
        dtype=torch.float16, encoding_analyzer=MinMaxEncodingAnalyzer(encoding_shape)
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with qdq.compute_encodings():
            qdq(dummy_input)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "Extreme values" in str(w[-1].message)


def test_onnx_export():
    """
    When: torch.onnx.export a quantizer
    Then: export shouldn't throw error
    """
    qdq = FloatQuantizeDequantize(dtype=torch.float16)
    with tempfile.TemporaryFile() as f:
        torch.onnx.export(qdq, torch.randn(10, 10), f)


def test_float_encoding_to():
    """
    Given: FloatEncoding with maxval=None
    When: Call .to()
    Then: Should return identical object
    """
    encoding = FloatEncoding(
        exponent_bits=5,
        mantissa_bits=10,
        finite=False,
        unsigned_zero=False,
        maxval=None,
    )
    new_encoding = encoding.to(device="cpu", dtype=torch.float16)
    assert new_encoding is encoding

    """
    Given: FloatEncoding with maxval=None
    """
    encoding = FloatEncoding(
        exponent_bits=5,
        mantissa_bits=10,
        finite=False,
        unsigned_zero=False,
        maxval=torch.tensor(124.0),
    )
    """
    When: Call .to() with same dtype and device
    Then: Should return identical object
    """
    new_encoding = encoding.to(device="cpu", dtype=torch.float32)
    assert new_encoding is encoding

    """
    When: Call .to() with new dtype and device
    Then: 1. New encoding object should be in proper dtype and device
          2. Old encoding object should not be affected
    """
    new_encoding = encoding.to(device="cpu", dtype=torch.float16)
    assert new_encoding.maxval.device == torch.device("cpu")
    assert new_encoding.maxval.dtype == torch.float16

    assert encoding.maxval.device == torch.device("cpu")
    assert encoding.maxval.dtype == torch.float32


def test_default_args():
    float16_qdq = FloatQuantizeDequantize(exponent_bits=5, mantissa_bits=10)
    assert float16_qdq.is_float16()

    bfloat16_qdq = FloatQuantizeDequantize(exponent_bits=8, mantissa_bits=7)
    assert bfloat16_qdq.is_bfloat16()


@pytest.mark.parametrize("dtype", [torch.bool, torch.int32])
def test_qdq_ignore_boolean_and_integers(dtype):
    """
    When: Pass boolean or integer tensor as input to FloatQuantizeDequantize
    Then: Output should be same as input, with dtype preserved.
    """
    x = (torch.arange(10) % 2).to(dtype)
    float_qdq = FloatQuantizeDequantize(dtype=torch.float8_e5m2)
    out = float_qdq(x)
    assert torch.equal(out, x)
    assert out.dtype == x.dtype

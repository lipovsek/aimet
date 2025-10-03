# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
import numpy as np

from aimet_common.quantsim import (
    calculate_delta_offset,
    compute_min_max_given_delta_offset,
    _is_bias_out_of_int32_range,
    _get_adjusted_weight_scale,
)
from aimet_common import libpymo


class TestCommonQuantSim:
    def test_offset_delta_compute(self):
        """test computation of delta and offset for export"""

        # Test asymmetric quantization with scalar inputs
        max_val = 1.700559472933134
        min_val = -2.1006477158567995
        bitwidth = 8

        expected_delta = (max_val - min_val) / (2**bitwidth - 1)
        expected_offset = np.round(min_val / expected_delta)
        delta, offset = calculate_delta_offset(
            min_val,
            max_val,
            bitwidth,
            use_strict_symmetric=False,
            use_symmetric_encodings=False,
        )
        assert np.isclose(delta, expected_delta)
        assert expected_offset == offset

        # Test symmetric quantization with scalar inputs
        max_val = 2.0
        min_val = -2.0
        bitwidth = 8

        num_steps = 2**bitwidth - 1
        num_positive_steps = np.floor(num_steps / 2)
        expected_delta = max_val / num_positive_steps
        expected_offset = -num_positive_steps - 1

        delta, offset = calculate_delta_offset(
            min_val,
            max_val,
            bitwidth,
            use_symmetric_encodings=True,
            use_strict_symmetric=False,
        )
        assert np.isclose(delta, expected_delta)
        assert offset == expected_offset

        # Test symmetric quantization with array inputs
        min_val = np.array([-2.0, 0.0, -1.0], dtype=np.float32)
        max_val = np.array([2.0, 1.0, 1.0], dtype=np.float32)
        bitwidth = 8

        num_steps = 2**bitwidth - 1
        num_positive_steps = np.floor(num_steps / 2)
        expected_delta = np.array(
            [
                max_val[0] / num_positive_steps,  # symmetric
                max_val[1] / num_positive_steps,  # symmetric
                max_val[2] / num_positive_steps,  # symmetric
            ]
        )
        expected_offset = np.array(
            [
                -num_positive_steps - 1,
                -num_positive_steps - 1,
                -num_positive_steps - 1,
            ]
        )

        delta, offset = calculate_delta_offset(
            min_val,
            max_val,
            bitwidth,
            use_symmetric_encodings=True,
            use_strict_symmetric=False,
        )
        assert np.allclose(delta, expected_delta)
        assert np.array_equal(offset, expected_offset)

        # Test asymmetric quantization with array inputs
        min_val = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        max_val = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        bitwidth = 8

        delta, offset = calculate_delta_offset(
            min_val,
            max_val,
            bitwidth,
            use_symmetric_encodings=True,
            use_strict_symmetric=False,
        )
        symmetric_mask = min_val < 0
        assert not np.any(symmetric_mask)

        expected_delta = (max_val - min_val) / (2**bitwidth - 1)
        expected_offset = np.round(min_val / expected_delta).astype(np.int32)

        assert np.allclose(delta, expected_delta)
        assert np.array_equal(offset, expected_offset)

    @pytest.mark.parametrize(
        "enc_min, enc_max, is_symmetric, is_strict",
        [
            (-5.0, 8.0, False, False),  # Expected new min/max = [-5.5714283  7.428571]
            (-5.0, 10.0, False, False),  # Expected new min/max = [-4.285714 10.714285]
            (-4.0, 3.0, True, False),  # Expected new min/max = [-4.0 3.0]
            (-3.0, 3.0, True, True),  # Expected new min/max = [-3.0 3.0]
            (0.0, 10.0, True, False),
        ],
    )  # Expected new min/max = [0.0 10.0]
    def test_encoding_param_calculation_python_vs_cpp(
        self, enc_min, enc_max, is_symmetric, is_strict
    ):
        """
        Test that the recomputed encoding within libpymo TensorQuantizer matches with the way encodings are recomputed
        in calculate_delta_offset and compute_min_max_given_delta_offset.
        """
        tensor_quantizer = libpymo.TensorQuantizer(
            libpymo.QuantizationMode.QUANTIZATION_TF, libpymo.RoundingMode.ROUND_NEAREST
        )
        tensor_quantizer.isEncodingValid = True
        in_tensor = np.array([-100.0, 100.0])
        out_tensor = np.zeros(in_tensor.shape).astype(np.float32)
        tensor_quantizer.quantizeDequantize(
            in_tensor, out_tensor, enc_min, enc_max, 3, False
        )

        delta, offset = calculate_delta_offset(
            enc_min, enc_max, 3, is_symmetric, is_strict
        )
        new_enc_min, new_enc_max = compute_min_max_given_delta_offset(
            delta, offset, 3, is_symmetric, is_strict
        )
        assert np.allclose(out_tensor[0], new_enc_min, atol=1e-5)
        assert np.allclose(out_tensor[1], new_enc_max, atol=1e-5)

    def test_is_bias_out_of_int32_range(self):
        bias = np.array([1.0])
        scale = np.array([1e-9])
        result = _is_bias_out_of_int32_range(bias, scale, num_steps=2**31)
        assert not result[0]  # within signed int32 range

        bias = np.array([10])
        scale = np.array([1e-9])
        result = _is_bias_out_of_int32_range(bias, scale, num_steps=2**31)
        assert result[0]  # not within signed int32 range

        bias = np.array([10.0, -10.0, 1.0])
        scale = np.array([1e-9, 1e-9, 1e-9])
        result = _is_bias_out_of_int32_range(bias, scale, num_steps=2**31)
        expected = np.array([True, True, False])
        assert np.all(result == expected)  # mix

        bias = np.array([1.0, -1.0, 1.0])
        scale = np.array([1e-9, 1e-9, 1e-9])
        result = _is_bias_out_of_int32_range(bias, scale, num_steps=2**31)
        assert np.all(result == False)  # all-within int32 range

        bias = np.array([10.0, -10.0, 10.0])
        scale = np.array([1e-9, 1e-9, 1e-9])
        result = _is_bias_out_of_int32_range(bias, scale, num_steps=2**31)
        assert np.all(result == True)  # all-exceeding int32 range

    def test_adjust_weight_scale_for_bias_overflow(self):
        # Weight adjustment not adjustment needed
        bias = np.array([1.0], dtype=np.float32)
        input_scale = np.array([0.1], dtype=np.float32)
        weight_scale = np.array([0.1])
        result = _get_adjusted_weight_scale(bias, input_scale, weight_scale)
        assert result == np.asarray(weight_scale, dtype=np.float32)

        # Weight adjustment needed
        bias = np.array([1e10])
        input_scale = np.array([0.1])
        weight_scale = np.array([0.1])
        expected = np.array(
            [np.abs(bias[0]) / (2**31 * input_scale[0])], dtype=np.float32
        )
        result = _get_adjusted_weight_scale(
            bias, input_scale, weight_scale, num_steps=2**31
        )
        assert np.allclose(result, expected)

        # Mix case (per-channel)
        bias = np.array([1.0, 1e10])
        input_scale = np.array([0.1])
        weight_scale = np.array([0.1, 0.1])
        expected = np.array(
            [weight_scale[0], np.abs(bias[1]) / (2**31 * input_scale[0])],
            dtype=np.float32,
        )
        result = _get_adjusted_weight_scale(
            bias, input_scale, weight_scale, num_steps=2**31
        )
        assert np.allclose(result, expected)  # mix

        # vector bias, 1D weight_scale (per-tensor)
        bias = np.array([1.0, 1e10, -1.0])
        input_scale = np.array([0.1])
        weight_scale = np.array([0.1])
        expected = np.array(
            [np.abs(bias[1]) / (2**31 * input_scale[0])], dtype=np.float32
        )
        result = _get_adjusted_weight_scale(
            bias, input_scale, weight_scale, num_steps=2**31
        )
        assert np.allclose(result, expected)

        # vector bias, float weight_scale (per-tensor)
        bias = np.array([1.0, 1e10, -1.0])
        input_scale = 0.1
        weight_scale = 0.1
        expected = np.array([np.abs(bias[1]) / (2**31 * input_scale)], dtype=np.float32)
        result = _get_adjusted_weight_scale(
            bias, input_scale, weight_scale, num_steps=2**31
        )
        assert np.allclose(result, expected)

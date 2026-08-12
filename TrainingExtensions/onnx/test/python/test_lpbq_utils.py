# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np

from aimet_onnx import lpbq_utils
from aimet_onnx.qc_quantize_op import LPBQScaleQuantizer
from aimet_onnx.utils import numpy_from_TfEncoding, numpy_to_TfEncoding
from aimet_onnx import qtype


class TestLPBQUtils:
    def test_get_per_group_scale_factor(self):
        bw = 4
        scale = np.asarray(
            [[100.2, 32.1, 0.001, 0.4], [23.1, 22.1, 10.0, 9.0]], np.float32
        )
        per_group_scale = lpbq_utils._get_per_group_scale_factor(scale, (1, 4), bw)
        expected_scale = np.asarray([100.2, 23.1]) / 2**bw
        assert np.allclose(
            per_group_scale, expected_scale.reshape(per_group_scale.shape)
        )

        per_group_scale = lpbq_utils._get_per_group_scale_factor(scale, (2, 1), bw)
        expected_scale = np.asarray([100.2, 32.1, 10.0, 9.0]) / 2**bw
        assert np.allclose(
            per_group_scale, expected_scale.reshape(per_group_scale.shape)
        )

    def test_per_group_int_scales(self):
        scale = np.asarray([[16, 1.6, 0.16], [0.12, 1.111, 0.033]])
        grouping = (2, 1)
        expected_int_scale = [16, 16, 16, 1, 11, 3]
        int_scale, scale_factor = lpbq_utils.grouped_dynamic_quantize(
            scale, grouping, 4
        )
        assert scale_factor.flatten().tolist() == [1, 0.1, 0.01]
        assert int_scale.flatten().tolist() == expected_int_scale

    def test_compress_encoding_scales(self):
        scale_bw = 8
        scale = np.asarray([[25.6, 11.111], [256.0, 25.555]], np.float32)
        offset = np.asarray([[-128, -128], [-128, -128]])

        encodings = numpy_to_TfEncoding(scale, offset, qtype.int(4))
        lpbq_encodings = lpbq_utils.compress_encoding_scales(
            encodings, scale.shape, (1, 2), scale_bw
        )
        lpbq_scale, lpbq_offset = numpy_from_TfEncoding(lpbq_encodings, scale.shape)

        expected_lpbq_scale = np.asarray(
            [
                [25.6, 11.1],
                [256.0, 26.0],
            ],
            np.float32,
        )

        assert np.allclose(lpbq_scale, expected_lpbq_scale)
        assert np.allclose(lpbq_offset, offset)


class TestLPBQScaleQuantizer:
    def test_as_encoding_dict(self):
        scale = np.abs(np.random.randn(4, 8)) + 0.1
        enc = LPBQScaleQuantizer(scale_bits=4).as_encoding_dict(
            scale, block_axis=1, channel_axis=0
        )
        assert set(enc) == {"x", "x_scale", "input_dtype", "axis"}
        # x_scale is squeezed to one value per channel; x keeps the per-block shape
        assert enc["x_scale"].shape == (4, 1)
        assert enc["x"].shape == scale.shape
        assert enc["input_dtype"] == qtype.int(4)
        assert enc["axis"] == 0

    def test_quantize_dequantize_without_block_axis_is_identity(self):
        scale = np.abs(np.random.randn(4, 8)) + 0.1
        quantizer = LPBQScaleQuantizer(scale_bits=4)
        qdq = quantizer.quantize_dequantize(scale, block_axis=None)
        assert np.array_equal(qdq, scale)

    def test_equality(self):
        # _merge_constraints relies on scale_bits equality
        assert LPBQScaleQuantizer(scale_bits=4) == LPBQScaleQuantizer(scale_bits=4)
        assert LPBQScaleQuantizer(scale_bits=4) != LPBQScaleQuantizer(scale_bits=8)

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np

from aimet_onnx import lpbq_utils


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

        encodings = lpbq_utils.scale_offset_arrays_to_encodings(scale, offset, 4)
        lpbq_encodings = lpbq_utils.compress_encoding_scales(
            encodings, scale.shape, (1, 2), scale_bw
        )
        lpbq_scale, lpbq_offset = lpbq_utils.encodings_to_scale_offset_arrays(
            lpbq_encodings, scale.shape
        )

        expected_lpbq_scale = np.asarray(
            [
                [25.6, 11.1],
                [256.0, 26.0],
            ],
            np.float32,
        )

        assert np.allclose(lpbq_scale, expected_lpbq_scale)
        assert np.allclose(lpbq_offset, offset)

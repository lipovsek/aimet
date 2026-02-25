# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Hadamard matrix utilities"""

import pytest
import numpy as np

try:
    from aimet_onnx.common.hadamard import get_hadamard_matrix
except ImportError:
    from aimet_torch.common.hadamard import get_hadamard_matrix


class TestGetHadamardMatrix:
    """Tests for get_hadamard_matrix function"""

    def test_power_of_two_sizes(self):
        """Power of 2 sizes"""
        for size in [2, 4, 8, 16, 32, 64]:
            H = get_hadamard_matrix(size)
            assert H.shape == (size, size)
            assert H.dtype == np.float32
            product = H @ H.T
            expected = size * np.eye(size)
            assert np.allclose(product, expected)  # H @ H.T = size * I

            # Orthogonality property of normalized Hadamard matrices
            R = H / np.sqrt(size)
            product = R @ R.T
            expected = np.eye(size)
            assert np.allclose(product, expected)  # R @ R.T = I

    @pytest.mark.parametrize("size", [12, 20, 28])
    def test_base_factors(self, size):
        """Base factor sizes (12, 20, 28)"""
        H = get_hadamard_matrix(size)
        assert H.shape == (size, size)
        assert H.dtype == np.float32
        product = H @ H.T
        expected = size * np.eye(size)
        assert np.allclose(product, expected)  # H @ H.T = size * I

        # Orthogonality property of normalized Hadamard matrices
        R = H / np.sqrt(size)
        product = R @ R.T
        expected = np.eye(size)
        assert np.allclose(product, expected)  # R @ R.T = I

    @pytest.mark.parametrize(
        "size",
        [
            24,
            48,
            96,  # 12 * 2^n for n=1,2,3
            40,
            80,  # 20 * 2^n for n=1,2
            56,
            112,  # 28 * 2^n for n=1,2
        ],
    )
    def test_composite_sizes_factor_times_power_of_2(self, size):
        """Composite sizes (factor * 2^n) using H2 doubling"""
        H = get_hadamard_matrix(size)
        assert H.shape == (size, size), f"Wrong shape for size {size}"
        assert H.dtype == np.float32
        product = H @ H.T
        expected = size * np.eye(size)
        assert np.allclose(product, expected)  # H @ H.T = size * I

        # Orthogonality property of normalized Hadamard matrices
        R = H / np.sqrt(size)
        product = R @ R.T
        expected = np.eye(size)
        assert np.allclose(product, expected)  # R @ R.T = I

    def test_common_model_sizes(self):
        """Common LLM hidden sizes"""
        test_sizes = [
            1280,
            1536,
            2048,
            2560,
            3072,
            3584,
        ]
        for size in test_sizes:
            H = get_hadamard_matrix(size)
            assert H.shape == (size, size)
            assert H.dtype == np.float32
            product = H @ H.T
            expected = size * np.eye(size)
            assert np.allclose(product, expected)  # H @ H.T = size * I

            # Orthogonality property of normalized Hadamard matrices
            R = H / np.sqrt(size)
            product = R @ R.T
            expected = np.eye(size)
            assert np.allclose(product, expected)  # R @ R.T = I

    def test_unsupported_sizes_raise_error(self):
        """Test that unsupported sizes raise AssertionError"""
        # Supported sizes of the current implementation are:
        # 1. Powers of 2: 2^n where n >= 1
        # 2. factor * 2^n where factor in {12, 20, 28} and n >= 0
        unsupported_sizes = [
            3,
            7,
            36,
        ]
        for size in unsupported_sizes:
            with pytest.raises(AssertionError):
                get_hadamard_matrix(size)

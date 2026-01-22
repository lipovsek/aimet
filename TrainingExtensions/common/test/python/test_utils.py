# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Module for testing aimet common utils"""

import os
import tempfile
import numpy as np
import pytest

try:
    from aimet_onnx.common.utils import profile, AimetLogger, compute_psnr
    from aimet_onnx.common import utils
except ImportError:
    from aimet_torch.common.utils import profile, AimetLogger, compute_psnr
    from aimet_torch.common import utils

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


def test_save_json_yaml():
    test_dict = {"1": 1, "2": 2, "3": 3}
    with tempfile.TemporaryDirectory() as tmpdir:
        utils.save_json_yaml(os.path.join(tmpdir, "saved_dict"), test_dict)
        assert os.path.isfile(os.path.join(tmpdir, "saved_dict"))


def test_profile():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path_and_name = os.path.join(tmpdir, "temp_profile.txt")
        with profile("profile 1", file_path_and_name, new_file=True, logger=logger):
            _ = 1 + 1
        with profile("profile 2", file_path_and_name, logger=logger):
            _ = 1 + 1
        with open(file_path_and_name, "r") as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert lines[0].startswith("profile 1: ")
        assert lines[1].startswith("profile 2: ")

        with profile("profile 3", file_path_and_name, new_file=True, logger=logger):
            _ = 1 + 1
        with open(file_path_and_name, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1
        assert lines[0].startswith("profile 3: ")

        with profile("profile 4"):
            _ = 1 + 1


def test_compute_psnr():
    # Identical arrays
    expected = np.array([[1, 2], [3, 4]])
    actual = np.array([[1, 2], [3, 4]])
    psnr = compute_psnr(expected, actual)
    assert psnr == 100

    # Valid PSNR
    expected = np.array([[1.62434536, -0.61175641], [-0.52817175, -1.07296862]])
    actual = np.array([[0.86540763, -2.3015387], [1.74481176, -0.7612069]])
    psnr = compute_psnr(expected, actual)
    assert np.isfinite(psnr)
    assert psnr > 0

    # Empty arrays
    expected = np.array([])
    actual = np.array([])
    with pytest.raises(ValueError):
        compute_psnr(expected, actual)

    # Shape mismatch
    expected = np.array([1, 2, 3, 4])
    actual = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        compute_psnr(expected, actual)

    # data_range contains all zero
    expected = np.array([[0, 0], [0, 0]])
    actual = np.array([[1, 2], [3, 4]])
    psnr = compute_psnr(expected, actual)
    assert psnr == -100

    # Expected should contain finite values only
    expected = np.array([[1, 2], [np.nan, 4]])
    actual = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        compute_psnr(expected, actual)

    # Clip the PSNR to -100 dB for non-finite values
    expected = np.array([[1, 2], [3, 4]])
    actual = np.array([[1, np.inf], [np.nan, 4]])
    psnr = compute_psnr(expected, actual)
    assert psnr == -100

    # When data_range = 0 and noise_pw = 0, treat it as perfect match
    expected = np.array([[0, 0], [0, 0]])
    actual = np.array([[0, 0], [0, 0]])
    psnr = compute_psnr(expected, actual)
    assert psnr <= 100

    # Handle scalars
    expected = 1.0
    actual = 1.000001
    psnr = compute_psnr(expected, actual)
    assert psnr <= 100

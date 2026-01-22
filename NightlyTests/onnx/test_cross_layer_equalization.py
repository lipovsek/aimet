# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
import pytest

from aimet_onnx.cross_layer_equalization import equalize_model
import test_models


class TestCLEAcceptance:
    """Acceptance test for AIMET ONNX"""

    @pytest.mark.skip(reason="Find better test criteria.")
    @pytest.mark.parametrize(
        "model", [test_models.mobilenetv2(), test_models.mobilenetv3_large_model()]
    )
    def test_cle_mv2(self, model):
        """Test for E2E quantization"""
        np.random.seed(0)
        equalize_model(model)

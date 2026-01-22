# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import unittest
from matplotlib import pyplot as plt

try:
    from aimet_onnx.common.curve_fit import MonotonicIncreasingCurveFit
except ImportError:
    from aimet_torch.common.curve_fit import MonotonicIncreasingCurveFit


class TestCommonCurveFit(unittest.TestCase):
    def test_curve_fit(self):
        x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        y = [0.1, 0.12, 0.18, 0.22, 0.4, 0.8, 0.7, 0.87, 0.7, 0.92]

        new_y, _ = MonotonicIncreasingCurveFit.fit(x, y)

        # Check is results are truly monotonically increasing
        for index in range(1, len(new_y) - 1):
            self.assertTrue(new_y[index] >= new_y[index - 1])

        # plt.plot(x, y, label='original')
        # plt.plot(x, new_y, label='curve_fit')
        #
        # plt.legend()
        # plt.show()

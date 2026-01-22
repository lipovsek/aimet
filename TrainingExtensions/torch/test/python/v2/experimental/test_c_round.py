# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import torch
from aimet_torch.v2.experimental.c_round import c_round


def test_c_round():
    x = torch.tensor([-1.5, -1.25, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.25, 1.5])
    expected = torch.tensor([-2.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0])
    assert torch.equal(c_round(x), expected)

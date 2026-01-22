# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import pytest
import torch
from aimet_torch.model_preparer import prepare_model


@pytest.mark.parametrize("mode", ["bilinear", "nearest", "bicubic"])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align_corners", [True, False])
def test_grid_sample(mode, padding_mode, align_corners):
    torch.manual_seed(0)

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, input, grid):
            return torch.nn.functional.grid_sample(
                input,
                grid,
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
            )

    model = Model()
    dummy_input = (torch.randn(1, 3, 8, 8), torch.randn(1, 5, 5, 2))

    original_out = model(*dummy_input)
    print(original_out)
    prepared_model = prepare_model(model)
    prepared_out = prepared_model(*dummy_input)

    assert torch.equal(original_out, prepared_out)
    assert len([module for module in prepared_model.modules()]) == 2

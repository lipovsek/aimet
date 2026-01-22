# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from torch.nn import Conv2d, Linear

from aimet_torch.layer_database import Layer
from aimet_torch.layer_selector import ConvFcLayerSelector, ConvNoDepthwiseLayerSelector


class TestLayerSelector(unittest.TestCase):
    def test_select_all_conv_layers(self):
        mock_output_shape = (1, 1, 1, 1)

        # Two regular conv layers
        layer1 = Layer(Conv2d(10, 20, 5), "", mock_output_shape)
        layer2 = Layer(Conv2d(10, 20, 5), "", mock_output_shape)
        layer3 = Layer(Conv2d(10, 10, 5, groups=10), "", mock_output_shape)

        layer_db = MagicMock()
        layer_db.__iter__.return_value = [layer1, layer2, layer3]

        layer_selector = ConvNoDepthwiseLayerSelector()
        layer_selector.select(layer_db, [])
        layer_db.mark_picked_layers.assert_called_once_with([layer1, layer2])

        # One conv and one linear layer
        layer1 = Layer(Conv2d(10, 20, 5), "", mock_output_shape)
        layer2 = Layer(Linear(10, 20), "", mock_output_shape)

        layer_db = MagicMock()
        layer_db.__iter__.return_value = [layer1, layer2]

        layer_selector.select(layer_db, [])
        layer_db.mark_picked_layers.assert_called_once_with([layer1])

        # Two regular conv layers - one in ignore list
        layer1 = Layer(Conv2d(10, 20, 5), "", mock_output_shape)
        layer2 = Layer(Conv2d(10, 20, 5), "", mock_output_shape)

        layer_db = MagicMock()
        layer_db.__iter__.return_value = [layer1, layer2]

        layer_selector.select(layer_db, [layer2.module])
        layer_db.mark_picked_layers.assert_called_once_with([layer1])

    def test_select_all_conv_and_fc_layers(self):
        mock_output_shape = (1, 1, 1, 1)

        # one regular conv layer, one depth wise conv layer and one FC layer
        layer1 = Layer(Conv2d(10, 10, 5, groups=10), "", mock_output_shape)
        layer2 = Layer(Linear(10, 20), "", mock_output_shape)
        layer3 = Layer(Conv2d(20, 40, 5), "", mock_output_shape)

        layer_db = MagicMock()
        layer_db.__iter__.return_value = [layer1, layer2, layer3]

        layer_selector = ConvFcLayerSelector()
        layer_selector.select(layer_db, [])
        layer_db.mark_picked_layers.assert_called_once_with([layer2, layer3])

        # Two regular conv layers and one FC layer - one in ignore list
        layer1 = Layer(Conv2d(10, 20, 5), "", mock_output_shape)
        layer2 = Layer(Linear(10, 20), "", mock_output_shape)
        layer3 = Layer(Conv2d(20, 40, 5), "", mock_output_shape)

        layer_db = MagicMock()
        layer_db.__iter__.return_value = [layer1, layer2, layer3]

        layer_selector.select(layer_db, [layer2.module])
        layer_db.mark_picked_layers.assert_called_once_with([layer1, layer3])

    def test_grouped_convolution_support(self):
        dummy_input = torch.randn(1, 4, 8, 8)
        grouped_convolution = nn.Conv2d(4, 16, kernel_size=3, groups=2)

        output_shape = grouped_convolution(dummy_input)
        layer = Layer(grouped_convolution, "grouped_conv", output_shape)
        assert layer.type_specific_params.groups == 2

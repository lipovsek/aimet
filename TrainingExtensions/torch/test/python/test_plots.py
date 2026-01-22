# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""
Contains unit tests to test the visualizations created given a model."""

import unittest
import aimet_torch.plots as plots
import numpy as np
import torch
import torchvision.models as models


class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # input channel, output channels, 5x5 square convolution
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 150, 5)
        self.conv3 = torch.nn.Conv2d(1, 60, 5)
        self.conv4 = torch.nn.Conv2d(6, 80, 5)


model = CNNModel()
resnet18 = models.resnet18(False)


class VisualizeNetwork(unittest.TestCase):
    def test_tensor_input(self):
        # 2 input channels, 6 output channels, 5*5 square convolution
        conv1 = torch.nn.Conv2d(2, 4, 5)

        # each column contains all the weights from one output channel
        conv1_weights = plots.get_weights(conv1)
        # print("conv1 weights after numpy reshaping", conv1_weights)

        # number of output channels, or columns in the reshaped matrix
        num_weights_out_channel1 = len(conv1_weights[0])
        total_weights_expected = np.prod(list(conv1.weight.shape))
        total_weights_actual = np.prod(conv1_weights.shape)

        # the length of any row should equal the number of output channels
        self.assertEqual(conv1.weight.shape[0], num_weights_out_channel1)

        # ensure the number of weights is the same before and after
        self.assertEqual(total_weights_expected, total_weights_actual)

    def test_clear_event_files(self):
        plots.clear_event_files("./data")

    def test_multivisualization_functions(self):
        conv1 = torch.nn.Conv2d(6, 15, 5)
        conv2 = torch.nn.Conv2d(6, 15, 5)
        before = plots.get_weights(conv1)
        after = plots.get_weights(conv2)
        # plots.compare_boxplots_before_after_quantization(before, after, "testing", subplot_name="boxplot")
        # plots.compare_key_stats_scatter_plot(before, after, "other", subplot_name="scatter")

        #
        # plots.compare_overall_changes_line_plot(before, after, "other", subplot_name="line")
        # plots.compare_overall_model_changes_violinplot(before, after, "other", subplot_name="violin")

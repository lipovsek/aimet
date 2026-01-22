# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-docstring
""" These are code examples to be used when generating AIMET documentation via Sphinx """

import torch

from aimet_torch.arch_checker.arch_checker import ArchChecker

class ModelWithNotEnoughChannels(torch.nn.Module):
    """ Model that prelu module. Expects input of shape (1, 3, 32, 32) """

    def __init__(self):
        super(ModelWithNotEnoughChannels, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 31, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(31)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.bn1(x)
        return x

class ModelWithPrelu(torch.nn.Module):
    """ Model that prelu module. Expects input of shape (1, 3, 32, 32) """

    def __init__(self):
        super(ModelWithPrelu, self).__init__()
        self.conv1 = torch.nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.prelu1 = torch.nn.PReLU()

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.bn1(x)
        x = self.prelu1(x)
        return x


class ModelWithNonfoldableBN(torch.nn.Module):
    """ Model that has non-foldable batch norm. """

    def __init__(self):
        super(ModelWithNonfoldableBN, self).__init__()
        self.conv1 = torch.nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.avg_pool1 = torch.nn.AvgPool2d(3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.avg_pool1(x)
        x = self.bn1(x)
        return x



def example_check_for_number_of_conv_channels():

    model = ModelWithNotEnoughChannels()
    ArchChecker.check_model_arch(model, dummy_input=torch.rand(1, 3, 32, 32))

def example_check_for_non_performant_activations():

    model = ModelWithPrelu()
    ArchChecker.check_model_arch(model, dummy_input=torch.rand(1, 32, 32, 32))

def example_check_for_standalone_bn():

    model = ModelWithNonfoldableBN()
    ArchChecker.check_model_arch(model, dummy_input=torch.rand(1, 32, 32, 32))

if __name__ == '__main__':
    example_check_for_number_of_conv_channels()
    example_check_for_non_performant_activations()
    example_check_for_standalone_bn()

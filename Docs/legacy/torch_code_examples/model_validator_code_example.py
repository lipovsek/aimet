# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-docstring
""" These are code examples to be used when generating AIMET documentation via Sphinx """

import torch
import torch.nn.functional as F

# Model Validator related imports
from aimet_torch.model_validator.model_validator import ModelValidator


class ModelWithReusedNodes(torch.nn.Module):
    """ Model that reuses a relu module. Expects input of shape (1, 3, 32, 32) """

    def __init__(self):
        super(ModelWithReusedNodes, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.linear = torch.nn.Linear(2592, 10)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class ModelWithoutReusedNodes(torch.nn.Module):
    """ Model that is fixed to not reuse modules. Expects input of shape (1, 3, 32, 32) """

    def __init__(self):
        super(ModelWithoutReusedNodes, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.linear = torch.nn.Linear(2592, 10)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class ModelWithFunctionalLinear(torch.nn.Module):
    """ Model that uses a torch functional linear layer. Expects input of shape (1, 3, 32, 32) """

    def __init__(self):
        super(ModelWithFunctionalLinear, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.relu2 = torch.nn.ReLU(inplace=True)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1)
        x = F.linear(x, torch.randn(10, 2592))
        return x


class ModelWithoutFunctionalLinear(torch.nn.Module):
    """ Model that is fixed to use a linear module instead of functional. Expects input of shape (1, 3, 32, 32) """

    def __init__(self):
        super(ModelWithoutFunctionalLinear, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.linear = torch.nn.Linear(2592, 10)
        with torch.no_grad():
            self.linear.weight = torch.nn.Parameter(torch.randn(10, 2592))

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def validate_example_model():

    # Load the model to validate
    model = ModelWithReusedNodes()

    # Output of ModelValidator.validate_model will be True if model is valid, False otherwise
    ModelValidator.validate_model(model, model_input=torch.rand(1, 3, 32, 32))


if __name__ == '__main__':
    validate_example_model()

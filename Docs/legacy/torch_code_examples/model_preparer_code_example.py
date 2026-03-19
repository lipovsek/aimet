# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: skip-file

# ModelPreparer imports

import torch
import torch.nn.functional as F
from aimet_torch.model_preparer import prepare_model

# End of import statements


class ModelWithFunctionalReLU(torch.nn.Module):
    """ Model that uses functional ReLU instead of nn.Modules. Expects input of shape (1, 3, 32, 32) """
    def __init__(self):
        super(ModelWithFunctionalReLU, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).relu()
        return x


class ModelWithReusedReLU(torch.nn.Module):
    """ Model that uses single ReLU instances multiple times in the forward. Expects input of shape (1, 3, 32, 32) """
    def __init__(self):
        super(ModelWithReusedReLU, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x


class ModelWithElementwiseAddOp(torch.nn.Module):
    def __init__(self):
        super(ModelWithElementwiseAddOp, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5, bias=False)
        self.conv2 = torch.nn.Conv2d(3, 6, 5)

    def forward(self, *inputs):
        x1 = self.conv1(inputs[0])
        x2 = self.conv2(inputs[1])
        x = x1 + x2
        return x


def model_preparer_functional_example():

    # Load the model and keep in eval() mode
    model = ModelWithFunctionalReLU().eval()
    input_shape = (1, 3, 32, 32)
    input_tensor = torch.randn(*input_shape)

    # Call to prepare_model API
    prepared_model = prepare_model(model)
    print(prepared_model)

    # Compare the outputs of original and transformed model
    assert torch.allclose(model(input_tensor), prepared_model(input_tensor))


def model_preparer_reused_example():

    # Load the model and keep in eval() mode
    model = ModelWithReusedReLU().eval()
    input_shape = (1, 3, 32, 32)
    input_tensor = torch.randn(*input_shape)

    # Call to prepare_model API
    prepared_model = prepare_model(model)
    print(prepared_model)

    # Compare the outputs of original and transformed model
    assert torch.allclose(model(input_tensor), prepared_model(input_tensor))


def model_preparer_elementwise_add_example():

    # Load the model and keep in eval() mode
    model = ModelWithElementwiseAddOp().eval()
    input_shape = (1, 3, 32, 32)
    input_tensor = [torch.randn(*input_shape), torch.randn(*input_shape)]

    # Call to prepare_model API
    prepared_model = prepare_model(model)
    print(prepared_model)

    # Compare the outputs of original and transformed model
    assert torch.allclose(model(*input_tensor), prepared_model(*input_tensor))


if __name__ == '__main__':
    model_preparer_functional_example()
    model_preparer_reused_example()
    model_preparer_elementwise_add_example()
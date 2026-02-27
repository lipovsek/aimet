# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=missing-docstring
import torch
from torch.utils.data import Dataset, DataLoader
from aimet_torch.experimental.adascale import adascale_optimizer
from aimet_torch.experimental.adascale.adascale_optimizer import AdaScaleModelConfig


class ModelWithLinears(torch.nn.Module):
    def __init__(self):
        super(ModelWithLinears, self).__init__()

        self.fc1 = torch.nn.Linear(64, 32)
        self.relu1 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout()
        self.fc2 = torch.nn.Linear(32, 64)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class ModelWithConsecutiveLinearBlocks(torch.nn.Module):
    def __init__(self):
        super(ModelWithConsecutiveLinearBlocks, self).__init__()
        self.linear_blocks = torch.nn.ModuleList(ModelWithLinears() for _ in range(5))
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        for linear_block in self.linear_blocks:
            x = linear_block(x)
        x = self.softmax(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# [setup]
# Load the model
# General setup that can be changed as needed
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = ModelWithConsecutiveLinearBlocks().eval().to(device)

# Register ModelWithLinears as the block type to AdaScale
adascale_optimizer.adascale_model_config_dict[ModelWithConsecutiveLinearBlocks] = (
    AdaScaleModelConfig(ModelWithLinears)
)
# End of [setup]

# [prepare-dataloader]
num_batches = 32
num_samples = 96
dummy_input = torch.rand(num_samples, 3, 32, 64).to(device)
data_set = CustomDataset(dummy_input)
data_loader = DataLoader(
    data_set, batch_size=int(num_samples / num_batches), shuffle=True
)
# End of [prepare-dataloader]

# [create-sim]
from aimet_torch.common.defs import QuantScheme
from aimet_torch import QuantizationSimModel

sim = QuantizationSimModel(
    model,
    dummy_input=dummy_input,
    quant_scheme=QuantScheme.training_range_learning_with_tf_init,
    default_param_bw=4,
    default_output_bw=16,
)
# End of [create-sim]

# [apply-adascale]
# Find and freeze optimal encodings candidate for weight parameters of supported layers
from aimet_torch.experimental.adascale import apply_adascale
from aimet_torch.v2.utils import default_forward_fn

apply_adascale(
    qsim=sim,
    data_loader=data_loader,
    forward_fn=default_forward_fn,
    num_iterations=1500,
)

# End of [apply-adascale]


# [compute_encodings]
def forward_pass(model: torch.nn.Module, _):
    with torch.no_grad():
        for data in data_loader:
            model(data)


# Compute the Quantization Encodings
# compute encodings for all activations and parameters of uninitialized layer(s)/operations(s)
sim.compute_encodings(forward_pass, None)
# End of [compute_encodings]

# [evaluation]
# Determine simulated quantized accuracy
...
# End of [evaluation]

# [export]
# Export the model for on-target inference
import aimet_torch
aimet_torch.onnx.export(
    sim.model,
    dummy_input,
    "quantized_model.onnx",
    dynamo=False,
    opset_version=21,
)
# End of [export]

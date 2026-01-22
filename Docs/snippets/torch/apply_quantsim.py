# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: skip-file

""" QuantSim and QAT code example to be used for documentation generation. """

# PyTorch imports
import torch
import torch.cuda
from torchvision.datasets import ImageNet
from tqdm import tqdm
# End of PyTorch imports

# Dataloaders
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

BATCH_SIZE = 32
NUM_CALIBRATION_SAMPLES = 1024

def get_calibration_and_eval_data_loaders(dataset_path: str, batch_size: int):
    """
    Returns calibration and evaluation data-loader for ImageNet dataset from provided path
    """
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = ImageNet(dataset_path, split='val', transform=transform)
    calibration_dataset, eval_dataset = random_split(
        dataset, [.9, 0.1]
    )

    calibration_data_loader = torch.utils.data.DataLoader(calibration_dataset, shuffle=True, batch_size=batch_size)
    eval_data_loader = torch.utils.data.DataLoader(eval_dataset, shuffle=True, batch_size=batch_size)
    return calibration_data_loader, eval_data_loader

PATH_TO_IMAGENET = './imagenet_dataset'
calibration_data_loader, eval_data_loader = get_calibration_and_eval_data_loaders(PATH_TO_IMAGENET, BATCH_SIZE)
# End of dataloaders

# Calibration callback
from typing import Any, Optional
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def pass_calibration_data(model: torch.nn.Module, forward_pass_args: Optional[Any]=None):
    """
    The User of the QuantizationSimModel API is expected to write this callback based on their dataset.
    """
    data_loader = forward_pass_args
    num_batches = NUM_CALIBRATION_SAMPLES // BATCH_SIZE

    model.eval()
    with torch.no_grad():
        for batch, (input_data, _) in enumerate(data_loader):
            inputs_batch = input_data.to(device)  # labels are ignored
            model(inputs_batch)
            if batch >= num_batches:
                break
# End of calibration callback

# Load the model
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).to(device)
# End of load the model

# Create Quantization Simulation Model
from aimet_torch.common.defs import QuantScheme
from aimet_torch.common.quantsim_config.utils import get_path_for_per_channel_config
from aimet_torch import QuantizationSimModel

input_shape = (1, 3, 224, 224)
dummy_input = torch.randn(input_shape).to(device)
sim = QuantizationSimModel(model,
                           dummy_input=dummy_input,
                           quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                           default_param_bw=8,
                           default_output_bw=16,
                           config_file=get_path_for_per_channel_config())
# End of QuantizationSimModel

# Compute the Quantization Encodings
sim.compute_encodings(pass_calibration_data, forward_pass_callback_args=calibration_data_loader)
# End of compute_encodings

# Evaluation
# Determine simulated quantized accuracy
sim.model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in tqdm(eval_data_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = sim.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy: {correct / total:.4f}')
# End of evaluation

# Export
# Export the model for on-target inference.
# Export the model which saves pytorch model without any simulation nodes and saves encodings file for both
# activations and parameters in JSON format at provided path.
sim.export(path='./', filename_prefix='quantized_mobilenet_v2', dummy_input=dummy_input.cpu())
# End of export

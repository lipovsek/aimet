# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=missing-docstring
# [setup]
import os
import onnxruntime as ort
import onnx
import torch
from torchvision import transforms
import torchvision

# Load the model
pt_model = torchvision.models.mobilenet_v2(pretrained=True)
input_shape = (1, 3, 224, 224)
dummy_input = torch.randn(input_shape)

# Modify file_path as you wish
file_path = os.path.join(".", f"mobilenet_v2.onnx")
torch.onnx.export(
    pt_model,
    (dummy_input,),
    file_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
    dynamo=False,
)
# Load exported ONNX model
model = onnx.load_model(file_path)

# Choose providers
if "CUDAExecutionProvider" in ort.get_available_providers():
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
else:
    providers = ["CPUExecutionProvider"]

# End of load the model

# Prepare the dataloader
DATASET_ROOT = ... # Set your path to imagenet dataset root directory
BATCH_SIZE = 32
NUM_CALIBRATION_SAMPLES = 128

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

imagenet_data = torchvision.datasets.ImageNet(
    DATASET_ROOT,
    split="val",
    transform=preprocess    
)

dataloader = torch.utils.data.DataLoader(
    imagenet_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)
# End of dataloader


# Step 1
# Create the QuantizationSimModel
import aimet_onnx

sim = aimet_onnx.QuantizationSimModel(
    model,
    param_type=aimet_onnx.int4,
    activation_type=aimet_onnx.int8,
    providers=providers
)
# End of step 1

# Step 2
import itertools

# Get unlabeled onnx data
input_name = model.graph.input[0].name
num_batches = NUM_CALIBRATION_SAMPLES // BATCH_SIZE
unlabeled_data = [{input_name: data.numpy()} for data, _ in itertools.islice(dataloader, num_batches)]

# Apply SeqMSE to the sim
aimet_onnx.apply_seq_mse(sim, unlabeled_data)
# End of step 2

# Step 3
sim.compute_encodings(unlabeled_data)
# End of step 3

# Step 4
from tqdm import tqdm
import numpy as np

correct_predictions = 0
total_samples = 0
for inputs, labels in tqdm(dataloader):
    inputs, labels = inputs.numpy(), labels.numpy()
    output, = sim.session.run(None, {input_name: inputs})
    pred_labels = np.argmax(output, axis=1)
    correct_predictions += np.sum(pred_labels == labels)
    total_samples += labels.shape[0]

accuracy = correct_predictions / total_samples
print(f"Quantized accuracy: {accuracy}")
# End of step 4

# Step 5
sim.export(path=".", filename_prefix="quantized_mobilenet_v2")
# End of step 5
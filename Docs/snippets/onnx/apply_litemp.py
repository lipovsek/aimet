# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
# pylint: skip-file
# imports start
import os
import copy
import numpy as np
import onnx
import torch
import onnxruntime
from tqdm import tqdm
# imports end

# Load the model
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
pt_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
input_shape = (1, 3, 224, 224)
dummy_input = torch.randn(input_shape)

# Modify file_path to save model at a different location
file_path = os.path.join('./', 'mobilenet_v2.onnx')
torch.onnx.export(pt_model,
                  (dummy_input,),
                  file_path,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={
                      'input': {0: 'batch_size'},
                      'output': {0: 'batch_size'},
                  },
                  dynamo=False,
                  )

# Load exported ONNX model
onnx_model = onnx.load_model(file_path)
# End of loading the model

# Prepare model with onnx-simplifier
import onnxsim
onnx_model, _ = onnxsim.simplify(onnx_model)
# End of prepare model

# Set up dataloader
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

BATCH_SIZE = 32
NUM_CALIBRATION_SAMPLES = 100

def get_calibration_and_eval_data_loaders(path: str, batch_size: int = BATCH_SIZE):
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

    dataset = datasets.ImageNet(path, split='val', transform=transform)
    calibration_dataset, eval_dataset = random_split(
        dataset, [NUM_CALIBRATION_SAMPLES, len(dataset) - NUM_CALIBRATION_SAMPLES]
    )

    calibration_data_loader = DataLoader(calibration_dataset, shuffle=False, batch_size=batch_size)
    eval_data_loader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size)
    return calibration_data_loader, eval_data_loader

# Change path here to point to different dataset
PATH_TO_IMAGENET = './imagenet_dataset'
calibration_data_loader, eval_data_loader = get_calibration_and_eval_data_loaders(PATH_TO_IMAGENET, BATCH_SIZE)
# End of setting up dataloader

# Evaluate FP32 model accuracy
import math
import itertools

def evaluate(session, data_loader, num_samples=1000):
    """
    Evaluate an ONNX model on a subset of ImageNet data
    """
    correct_predictions = 0
    total_samples = 0
    input_name = session.get_inputs()[0].name
    max_batches = math.ceil(num_samples / BATCH_SIZE)

    for i, (inputs, labels) in enumerate(tqdm(itertools.islice(data_loader, max_batches))):
        pred_probs, *_ = session.run(None, {input_name: inputs.numpy()})
        pred_labels = np.argmax(pred_probs, axis=1)
        correct_predictions += np.sum(pred_labels == labels.numpy())
        total_samples += labels.shape[0]

    accuracy = correct_predictions / total_samples
    return accuracy

fp_session = onnxruntime.InferenceSession(
    onnx_model.SerializeToString(),
    providers=["CUDAExecutionProvider"]
)
fp_accuracy = evaluate(fp_session, eval_data_loader)
print(f'fp32 accuracy: {fp_accuracy:.4f}')
# End of FP32 evaluation

# Create QuantSim object
from aimet_onnx.common.defs import QuantScheme
from aimet_onnx import int8, int16
from aimet_onnx import QuantizationSimModel, compute_encodings

# If CUDA acceleration is not available, simply use providers = ["CPUExecutionProvider"]
providers = ["CUDAExecutionProvider"]
sim = QuantizationSimModel(
    copy.deepcopy(onnx_model),
    param_type=int8,
    activation_type=int8,
    quant_scheme=QuantScheme.min_max,
    config_file="default",
    providers=providers
)
# End of creating QuantSim object

# Compute quantization encodings
# Compute quantization parameters using representative data for all the quantizers in the model
input_name = sim.session.get_inputs()[0].name
with compute_encodings(sim):
    for i, (inputs, _) in enumerate(calibration_data_loader):
        _ = sim.session.run(None, {input_name: inputs.numpy()})
# End of computing quantization encodings

# Evaluate quantized accuracy
w8a8_accuracy = evaluate(sim.session, eval_data_loader)
print(f'quantized accuracy (w8a8): {w8a8_accuracy:.4f}')
# End of quantized accuracy

# Perform sensitivity analysis
from aimet_onnx.utils import make_psnr_eval_fn
from aimet_onnx import analyze_per_layer_sensitivity

# Only few samples are required.
fp_inputs = [{input_name: x.numpy()} for x, _ in itertools.islice(calibration_data_loader, 1)]
psnr_eval_fn = make_psnr_eval_fn(fp_session, fp_inputs)
layer_sensitivity_dict = analyze_per_layer_sensitivity(
    sim, eval_fn=psnr_eval_fn
)
# End of sensitivity analysis

# Apply precision adjustment
from aimet_onnx.lite_mp import flip_layers_to_higher_precision

percentage = 10 # Percentage of layers to flip
override_precision = int16 # Precision to sets layers to
flip_layers_to_higher_precision(
    sim, layer_sensitivity_dict, percentage, override_precision
)
# End of precision adjustment

# Recompute quantization encodings
with compute_encodings(sim):
    for i, (inputs, _) in enumerate(calibration_data_loader):
        _ = sim.session.run(None, {input_name: inputs.numpy()})
# End of recompute quantization encodings

# Reevaluate model's accuracy after quantization
w8a8_mp_accuracy = evaluate(sim.session, eval_data_loader)
print(f'quantized accuracy (w8a8_mixed): {w8a8_mp_accuracy:.4f}')
# End of re-evaluation
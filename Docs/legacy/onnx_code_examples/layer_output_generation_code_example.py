# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

""" Code example to generate intermediate layer outputs of a model """

# Step 0. Import statements
import onnx
from onnxruntime import InferenceSession
from onnxsim import simplify

from aimet_onnx.quantsim import QuantizationSimModel, load_encodings_to_sim
from aimet_onnx.layer_output_utils import LayerOutputUtil
# End step 0

# Step 1. Obtain original or quantsim model
# Load the model.
model = onnx.load('path/to/aimet_export_artifacts/model.onnx')
# Simplify the model
model, _ = simplify(model)

# Use same arguments as that were used for the exported QuantSim model. For sake of simplicity only mandatory arguments are passed below.
quantsim = QuantizationSimModel(model=model, dummy_input=dummy_input_dict, use_cuda=False)

# Load exported encodings into quantsim object
load_encodings_to_sim(quantsim, 'path/to/aimet_export_artifacts/model.encodings')

# Check whether constructed original and quantsim model are running properly before using Layer Output Generation API.
_ = InferenceSession(model.SerializeToString()).run(None, dummy_input_dict)
_ = quantsim.session.run(None, dummy_input_dict)
# End step 1

# Step 2. Obtain pre-processed inputs
# Use same input pre-processing pipeline as was used for computing the quantization encodings.
input_batches = get_pre_processed_inputs()
# End step 2

# Step 3. Generate outputs
# Use original model to get fp32 layer-outputs
fp32_layer_output_util = LayerOutputUtil(model=model, dir_path='./fp32_layer_outputs')

# Use quantsim model to get quantsim layer-outputs
quantsim_layer_output_util = LayerOutputUtil(model=quantsim.model.model, dir_path='./quantsim_layer_outputs')

for input_batch in input_batches:
    fp32_layer_output_util.generate_layer_outputs(input_batch)
    quantsim_layer_output_util.generate_layer_outputs(input_batch)

# Note: Generate layer-outputs for fp32 model before creating quantsim model becuase the fp32 model itself is modified to get quantsim version.
# End step 3

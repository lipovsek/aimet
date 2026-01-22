# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: skip-file
""" Code example to generate intermediate layer outputs of a model """

# Step 0. Import statements
import tensorflow as tf

from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_tensorflow.keras.layer_output_utils import LayerOutputUtil
# End step 0

# Step 1. Obtain original or quantsim model
# Load the model.
model = tf.keras.models.load_model('path/to/aimet_export_artifacts/model.h5')

# Use same arguments as that were used for the exported QuantSim model. For sake of simplicity only mandatory arguments are passed below.
quantsim = QuantizationSimModel(model)

# Load exported encodings into quantsim object.
quantsim.load_encodings_to_sim('path/to/aimet_export_artifacts/model.encodings')

# Check whether constructed original and quantsim model are running properly before using Layer Output Generation API.
_ = model.predict(dummy_input)
_ = quantsim.predict(dummy_input)
# End step 1

# Step 2. Obtain pre-processed inputs
# Use same input pre-processing pipeline as was used for computing the quantization encodings.
input_batches = get_pre_processed_inputs()
# End step 2

# Step 3. Generate outputs
# Use original model to get fp32 layer-outputs
fp32_layer_output_util = LayerOutputUtil(model=model, save_dir='fp32_layer_outputs')

# Use quantsim model to get quantsim layer-outputs
quantsim_layer_output_util = LayerOutputUtil(model=quantsim.model, save_dir='quantsim_layer_outputs')

for input_batch in input_batches:
    fp32_layer_output_util.generate_layer_outputs(input_batch=input_batch)
    quantsim_layer_output_util.generate_layer_outputs(input_batch=input_batch)
# End step 3

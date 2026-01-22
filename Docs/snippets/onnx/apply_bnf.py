# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=missing-docstring
import os

import onnx
import onnxsim
import torch
from aimet_onnx.batch_norm_fold import fold_all_batch_norms_to_weight
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

pt_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
print(pt_model)

# Shape for each ImageNet sample is (3 channels) x (224 height) x (224 width)
input_shape = (1, 3, 224, 224)
dummy_input = torch.randn(input_shape)

# Modify file_path as you wish, we are using temporary directory for now
file_path = os.path.join('/tmp', f'mobilenet_v2.onnx')
torch.onnx.export(
    pt_model,
    (dummy_input,),
    file_path,
    do_constant_folding=False,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'},
    },
    dynamo=False,
)
# Load exported ONNX model
model = onnx.load_model(file_path)
# End of step 1

# Step 2
# Unlike AIMET, which supports both forward/backward folding, ONNX simplifier only performs backward folding.
# Therefore, we disable the corresponding optimization in `skipped_optimizers` and proceed with the example
try:
    model, _ = onnxsim.simplify(model, skipped_optimizers=['fuse_bn_into_conv'])
except:
    print('ONNX Simplifier failed. Proceeding with unsimplified model')

initializers = {init.name: init for init in model.graph.initializer}
conv_weight_name = model.graph.node[0].input[1]
conv_weight = initializers[conv_weight_name]
conv_weight_array = onnx.numpy_helper.to_array(conv_weight)

print("*** Before batch norm folding ***")

print("\nmodel.graph.node[0]:")
print(model.graph.node[0].name)

print("\nmodel.graph.node[1]:")
print(model.graph.node[1].name)

print("\nConv weight")
print(conv_weight_array)
# End of step 2

# Step 3
_ = fold_all_batch_norms_to_weight(model=model)

conv_weight = initializers[conv_weight_name]
conv_weight_array = onnx.numpy_helper.to_array(conv_weight)

print("*** After batch norm folding ***")

print("\nmodel.graph.node[0]:")
print(model.graph.node[0].name)

print("\nmodel.graph.node[1]:")
print(model.graph.node[1].name)

print("\nConv weight")
print(conv_weight_array)
# End of step 3

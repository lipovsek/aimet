# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=missing-docstring
# [step_1]
import os
import onnx
import torch
from torchvision.models import mobilenet_v2

model = mobilenet_v2(weights='DEFAULT').eval()

dummy_input = torch.randn((10, 3, 224, 224))
file_path = os.path.join('/tmp', f'mobilenet_v2.onnx')
torch.onnx.export(model, dummy_input, file_path, dynamo=False)
onnx_model = onnx.load_model(file_path)
# End of [step_1]

# [step_2]
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx import int8, int16
from aimet_onnx.utils import make_dummy_input

sim = QuantizationSimModel(onnx_model,
                           param_type=int8,
                           activation_type=int16)
# End of [step_2]

# [step_3]
calibration_data = make_dummy_input(onnx_model)
sim.compute_encodings(inputs=[calibration_data])
# End of [step_3]

# [step_4]
input_name = tuple(calibration_data.keys())[0]
output = sim.session.run(None, { input_name : dummy_input.numpy() })
print(output)
# End of [step_4]

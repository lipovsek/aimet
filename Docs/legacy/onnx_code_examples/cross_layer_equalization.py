# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: skip-file

from onnxsim import simplify
from aimet_onnx.cross_layer_equalization import equalize_model

def cross_layer_equalization():
    onnx_model = Model()
    # Simplify the model
    onnx_model, _ = simplify(onnx_model)
    equalize_model(onnx_model)
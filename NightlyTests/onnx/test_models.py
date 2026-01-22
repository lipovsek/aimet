# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import torch
from onnxruntime.quantization.onnx_quantizer import ONNXModel
from onnx import load_model

import io
from torchvision import models


def mobilenetv2():
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    model = models.MobileNetV2().eval()

    buffer = io.BytesIO()
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        buffer,
        training=torch.onnx.TrainingMode.PRESERVE,
        export_params=True,
        do_constant_folding=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,
    )
    buffer.seek(0)
    model = ONNXModel(load_model(buffer))
    return model


def mobilenetv3_large_model():
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    model = models.mobilenet_v3_large().eval()

    buffer = io.BytesIO()
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        buffer,
        training=torch.onnx.TrainingMode.PRESERVE,
        export_params=True,
        do_constant_folding=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,
    )
    buffer.seek(0)
    model = ONNXModel(load_model(buffer))
    return model


def resnet18():
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    model = models.resnet18().eval()

    buffer = io.BytesIO()
    # Export the model
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        buffer,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=12,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],
        dynamo=False,
    )
    buffer.seek(0)
    model = ONNXModel(load_model(buffer))
    return model

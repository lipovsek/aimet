# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
import os
import tempfile
from typing import Sequence
import onnx
import torch


def _qdq_op1_op2_qdq(
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    op_type_1: str,
    op_type_2: str,
    op1_kwargs=None,
    op2_kwargs=None,
) -> onnx.ModelProto:
    """
    (input) -> QDQ -> op1 -> op2 -> QDQ -> (output)
    """
    output_scale = 0.2
    model = onnx.helper.make_model(
        ir_version=10,
        opset_imports=[onnx.helper.make_operatorsetid("", 13)],
        graph=onnx.helper.make_graph(
            name="qdq_graph",
            inputs=[
                onnx.helper.make_tensor_value_info(
                    "input", onnx.TensorProto.FLOAT, shape=input_shape
                )
            ],
            outputs=[
                onnx.helper.make_tensor_value_info(
                    "output", onnx.TensorProto.FLOAT, shape=output_shape
                )
            ],
            nodes=[
                onnx.helper.make_node(
                    "QuantizeLinear",
                    inputs=["input", "input_scale", "input_zero_point"],
                    outputs=["input_q"],
                    name="input_q",
                ),
                onnx.helper.make_node(
                    "DequantizeLinear",
                    inputs=["input_q", "input_scale", "input_zero_point"],
                    outputs=["input_dq"],
                    name="input_dq",
                ),
                onnx.helper.make_node(
                    op_type_1,
                    inputs=["input_dq"],
                    outputs=[f"{op_type_1}_output_0"],
                    name=op_type_1.lower(),
                    **(op1_kwargs or {}),
                ),
                onnx.helper.make_node(
                    op_type_2,
                    inputs=[f"{op_type_1}_output_0"],
                    outputs=[f"{op_type_2}_output_0"],
                    name=op_type_2.lower(),
                    **(op2_kwargs or {}),
                ),
                onnx.helper.make_node(
                    "QuantizeLinear",
                    inputs=[
                        f"{op_type_2}_output_0",
                        "output_scale",
                        "output_zero_point",
                    ],
                    outputs=["output_q"],
                    name="output_q",
                ),
                onnx.helper.make_node(
                    "DequantizeLinear",
                    inputs=["output_q", "output_scale", "output_zero_point"],
                    outputs=["output"],
                    name="output_dq",
                ),
            ],
            initializer=[
                onnx.helper.make_tensor(
                    name="input_scale",
                    data_type=onnx.TensorProto.FLOAT,
                    dims=[],
                    vals=[0.1],
                ),
                onnx.helper.make_tensor(
                    name="input_zero_point",
                    data_type=onnx.TensorProto.UINT8,
                    dims=[],
                    vals=[128],
                ),
                onnx.helper.make_tensor(
                    name="output_scale",
                    data_type=onnx.TensorProto.FLOAT,
                    dims=[],
                    vals=[output_scale],
                ),
                onnx.helper.make_tensor(
                    name="output_zero_point",
                    data_type=onnx.TensorProto.UINT8,
                    dims=[],
                    vals=[128],
                ),
            ],
        ),
    )

    return model, (output_scale,)


def qdq_relu_cast_qdq():
    """
    (input) -> QDQ -> Relu -> Cast -> QDQ -> (output)
    """
    return _qdq_op1_op2_qdq(
        input_shape=[1, 3, 224, 224],
        output_shape=[1, 3, 224, 224],
        op_type_1="Relu",
        op_type_2="Cast",
        op2_kwargs={"to": onnx.TensorProto.FLOAT},
    )


def qdq_relu_identity_qdq():
    """
    (input) -> QDQ -> Relu -> Identity -> QDQ -> (output)
    """
    return _qdq_op1_op2_qdq(
        input_shape=[1, 3, 224, 224],
        output_shape=[1, 3, 224, 224],
        op_type_1="Relu",
        op_type_2="Identity",
    )


def qdq_relu_transpose_qdq():
    """
    (input) -> QDQ -> Relu -> Transpose -> QDQ -> (output)
    """
    return _qdq_op1_op2_qdq(
        input_shape=[1, 3, 224, 224],
        output_shape=[1, 3, 224, 224],
        op_type_1="Relu",
        op_type_2="Transpose",
        op2_kwargs={"perm": [0, 1, 3, 2]},
    )


def split_qdq(
    split_input_quantized: bool,
    mul_input_quantized: bool,
    mul_output_quantized: bool,
    reshape_input_quantized: bool,
    reshape_output_quantized: bool,
):
    """
                      +--> Mul ------> (output1)
    (input) -> Split -+
                      +--> Reshape --> (output2)
    """
    input_scale = mul_input_scale = reshape_input_scale = reshape_output_scale = 0.01
    mul_output_scale = 0.02

    class Model(torch.nn.Module):
        def forward(self, x):
            if split_input_quantized:
                x = torch.fake_quantize_per_tensor_affine(
                    x, scale=input_scale, zero_point=128, quant_min=0, quant_max=255
                )

            x1, x2 = torch.split(x, 3, dim=1)

            if mul_input_quantized:
                x1 = torch.fake_quantize_per_tensor_affine(
                    x1,
                    scale=mul_input_scale,
                    zero_point=128,
                    quant_min=0,
                    quant_max=255,
                )

            if reshape_input_quantized:
                x2 = torch.fake_quantize_per_tensor_affine(
                    x2,
                    scale=reshape_input_scale,
                    zero_point=128,
                    quant_min=0,
                    quant_max=255,
                )

            y1 = x1 * 2
            y2 = x2.reshape(-1)

            if mul_output_quantized:
                y1 = torch.fake_quantize_per_tensor_affine(
                    y1,
                    scale=mul_output_scale,
                    zero_point=128,
                    quant_min=0,
                    quant_max=255,
                )

            if reshape_output_quantized:
                y2 = torch.fake_quantize_per_tensor_affine(
                    y2,
                    scale=reshape_output_scale,
                    zero_point=128,
                    quant_min=0,
                    quant_max=255,
                )

            return y1, y2

    model = Model()
    dummy_input = torch.randn(1, 6, 224, 224)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = "."
        path = os.path.join(tmp_dir, "split_qdq.onnx")
        torch.onnx.export(
            model,
            (dummy_input,),
            path,
            input_names=["input"],
            output_names=["output1", "output2"],
            opset_version=13,
        )
        return onnx.load(path), (mul_output_scale, reshape_output_scale)


def concat_qdq(
    mul_input_quantized: bool,
    mul_output_quantized: bool,
    reshape_input_quantized: bool,
    reshape_output_quantized: bool,
    concat_output_quantized: bool,
):
    """
    (input1) --> Mul -------+
                            +-> Concat -> (output)
    (input2) --> Reshape ---+
    """
    reshape_input_scale = reshape_output_scale = concat_output_scale = 0.01
    mul_input_scale = 0.011
    mul_output_scale = 0.022

    class Model(torch.nn.Module):
        def forward(self, x1, x2):
            if mul_input_quantized:
                x1 = torch.fake_quantize_per_tensor_affine(
                    x1,
                    scale=mul_input_scale,
                    zero_point=128,
                    quant_min=0,
                    quant_max=255,
                )

            if reshape_input_quantized:
                x2 = torch.fake_quantize_per_tensor_affine(
                    x2,
                    scale=reshape_input_scale,
                    zero_point=128,
                    quant_min=0,
                    quant_max=255,
                )

            y1 = x1 * 2
            y2 = x2.reshape(y1.shape)

            if mul_output_quantized:
                y1 = torch.fake_quantize_per_tensor_affine(
                    y1,
                    scale=mul_output_scale,
                    zero_point=128,
                    quant_min=0,
                    quant_max=255,
                )

            if reshape_output_quantized:
                y2 = torch.fake_quantize_per_tensor_affine(
                    y2,
                    scale=reshape_output_scale,
                    zero_point=128,
                    quant_min=0,
                    quant_max=255,
                )

            out = torch.cat((y1, y2), dim=1)

            if concat_output_quantized:
                out = torch.fake_quantize_per_tensor_affine(
                    out,
                    scale=concat_output_scale,
                    zero_point=128,
                    quant_min=0,
                    quant_max=255,
                )

            return out

    model = Model()
    dummy_input = (
        torch.randn(1, 3, 224, 224),
        torch.randn(1, 3 * 224 * 224),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = "."
        path = os.path.join(tmp_dir, "concat_qdq.onnx")
        torch.onnx.export(
            model,
            dummy_input,
            path,
            input_names=["input1", "input2"],
            output_names=["output"],
            opset_version=13,
        )
        return onnx.load(path), (concat_output_scale,)

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Collection of onnx-related util functions that can be shared across aimet-onnx and aimet-torch"""

# pylint: disable=no-member, import-error

from collections import deque, defaultdict
import functools
import itertools
from typing import Iterable, Optional, Sequence, Dict, List, Union, Mapping
import math

import os
import tempfile

import numpy as np
import onnx
from onnx import ModelProto, NodeProto, TensorProto
from onnx.numpy_helper import from_array, to_array
from onnx.external_data_helper import (
    load_external_data_for_tensor,
    uses_external_data,
    _get_all_tensors,
)
from onnx.defs import OpSchema

from . import opset10, opset13, opset19, opset21, opset23, opset25
from ..utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


def _add_onnx_qdq_node(
    model: ModelProto,
    input_name: str,
    output_name: str,
    node_name_prefix: str,
    encodings: dict,
    float_type: np.dtype,
    onnx_opset: int,
    prequantize_constants: bool,
    base_dir: str = "",
):
    """
    Add onnx::QuantizeLinear and/or onnx::DequantizeLinear as below

     -------> onnx::QuantizeLinear -------------> onnx::DequantizeLinear ----->
    (input)                        (input_q)                             (output)


    except for int32 bias encoding, for which we take alternative representation as below
    since onnx::QuantizeLinear doesn't allow int32 outputs.

    -------------> onnx::DequantizeLinear ----->
    (bias_q)                             (bias_qdq)

    """
    _add_onnx_qdq_nodes(
        model,
        [input_name],
        [output_name],
        [node_name_prefix],
        [encodings],
        [float_type],
        onnx_opset,
        prequantize_constants,
        base_dir=base_dir,
    )


def _add_onnx_qdq_nodes(
    model: ModelProto,
    input_names: Iterable[str],
    output_names: Iterable[str],
    node_name_prefixes: Iterable[str],
    encodings: Iterable[dict],
    float_types: Iterable[np.dtype],
    onnx_opset: int,
    prequantize_constants: bool,
    base_dir: str = "",
):
    """
    Add onnx::QuantizeLinear and/or onnx::DequantizeLinear as below

     -------> onnx::QuantizeLinear -------------> onnx::DequantizeLinear ----->
    (input)                        (input_q)                             (output)


    except for int32 bias encodings, for which we take alternative representation as below
    since onnx::QuantizeLinear doesn't allow int32 outputs.

    -------------> onnx::DequantizeLinear ----->
    (bias_q)                             (bias_qdq)

    """
    if onnx_opset < 10:
        raise RuntimeError(
            "ONNX opset {} cannot represent QuantizeLinear and DequantizeLinear nodes."
            "So not able to export model as ONNX QDQ graph"
        )

    if onnx_opset < 13:
        opset = opset10
    elif onnx_opset < 19:
        opset = opset13
    elif onnx_opset < 21:
        opset = opset19
    elif onnx_opset < 23:
        opset = opset21
    elif onnx_opset < 25:
        opset = opset23
    else:
        opset = opset25

    constants = _get_all_constants(model)
    nodes_to_add = []
    tensors_to_add = []
    tensors_to_remove = {}
    inputs_to_rename = {}

    for input_name, output_name, node_name_prefix, encoding, float_type in zip(
        input_names, output_names, node_name_prefixes, encodings, float_types
    ):
        inputs_to_rename[input_name] = output_name
        output_dtype = encoding["output_dtype"]
        axis = encoding.get("axis", None)

        if input_name in constants:
            input_shape = constants[input_name].dims

            # Convert to positive index. Not strictly necessary; just for convenience
            if axis is not None:
                axis = (axis + len(input_shape)) % len(input_shape)
        else:
            input_shape = None

        block_size = encoding.get("block_size", None)
        y_zero_point = encoding.get("y_zero_point", None)

        y_scale = np.array(
            encoding.get("y_scale") or encoding.get("per_channel_float_scale")
        ).astype(float_type)
        per_block_int_scale = (
            np.array(encoding["per_block_int_scale"])
            if "per_block_int_scale" in encoding
            else None
        )

        if y_zero_point is not None:
            y_zero_point = np.array(encoding["y_zero_point"], dtype=np.int64)
        elif per_block_int_scale is not None:
            y_zero_point = np.zeros(per_block_int_scale.shape, dtype=np.int64)
        else:
            y_zero_point = np.zeros(y_scale.shape, dtype=np.int64)

        tensors_to_add.append(
            opset.DequantizeLinear.make_zero_point(
                y_zero_point, dtype=output_dtype, name=f"{input_name}_zero_point"
            )
        )

        if per_block_int_scale is None:
            tensors_to_add.append(from_array(y_scale, name=f"{input_name}_scale"))
        else:
            # Export LPBQ.
            #
            # Strategy: Derive y_scale from per_channel_float_scale and per_block_uint_scale
            #
            #           (FLOAT)
            # per_channel_float_scale -----+
            #                              +--> DequantizeLinear -----+ (blockwise scale)
            #    per_block_uint_scale -----+                          |
            #           (UINT8)                               +-------+---------+
            #                                                 V                 V
            #                              weight ---> QuantizeLinear -> DequantizeLinear -> ...
            if output_dtype != "int4":
                raise RuntimeError(
                    f"LPBQ can be only exported with int4; got {output_dtype}"
                )

            channel_axis = axis - 1  # Assume channel_axis = block_axis - 1 by default

            if input_shape is not None:
                # Convert to positive index
                channel_axis = (channel_axis + len(input_shape)) % len(input_shape)

                non_singleton_axes = tuple(
                    i for i, dim in enumerate(input_shape) if dim != 1
                )

                if len(non_singleton_axes) > 2 or (
                    len(non_singleton_axes) == 2 and axis not in non_singleton_axes
                ):
                    raise RuntimeError(
                        "When exported to onnx QDQ, LPBQ can be only applied to tensors with "
                        "at most two non-singleton dimensions, "
                        "each representing channel and block axes. "
                        f'Got "{input_name}" with shape {input_shape} and block axis {axis}'
                    )

                try:
                    # The non-singleton axis which isn't block axis (if any) is channel axis
                    channel_axis = next(i for i in non_singleton_axes if i != axis)
                except StopIteration:
                    pass

            tensors_to_add.extend(
                [
                    from_array(
                        y_scale.flatten(), name=f"{input_name}_per_channel_float_scale"
                    ),
                    from_array(
                        per_block_int_scale.astype(np.uint8),
                        name=f"{input_name}_per_block_uint_scale",
                    ),
                ]
            )
            nodes_to_add.extend(
                [
                    opset.DequantizeLinear.make_node(
                        name=f"{node_name_prefix}_scale_dq",
                        inputs=[
                            f"{input_name}_per_block_uint_scale",
                            f"{input_name}_per_channel_float_scale",
                        ],
                        output=f"{input_name}_scale",
                        dtype="uint8",
                        axis=channel_axis,
                    )
                ]
            )

        input_q = None
        if prequantize_constants or output_dtype in ("int32", "uint32"):
            const = constants.get(input_name)
            if const:
                input_q = _quantize_const(
                    const,
                    f"{input_name}_q",
                    y_scale,
                    y_zero_point,
                    axis,
                    block_size,
                    output_dtype,
                    per_block_int_scale=per_block_int_scale,
                    opset=opset,
                    base_dir=base_dir,
                )

        if input_q:
            nodes_to_add.append(
                opset.DequantizeLinear.make_node(
                    name=f"{node_name_prefix}_dq",
                    inputs=[
                        input_q.name,
                        f"{input_name}_scale",
                        f"{input_name}_zero_point",
                    ],
                    output=output_name,
                    dtype=output_dtype,
                    axis=axis,
                    block_size=block_size,
                )
            )
            tensors_to_remove[input_name] = True
            tensors_to_add.append(input_q)

        else:
            nodes_to_add.extend(
                [
                    opset.QuantizeLinear.make_node(
                        name=f"{node_name_prefix}_q",
                        inputs=[
                            input_name,
                            f"{input_name}_scale",
                            f"{input_name}_zero_point",
                        ],
                        output=f"{input_name}_q",
                        dtype=output_dtype,
                        axis=axis,
                        block_size=block_size,
                    ),
                    opset.DequantizeLinear.make_node(
                        name=f"{node_name_prefix}_dq",
                        inputs=[
                            f"{input_name}_q",
                            f"{input_name}_scale",
                            f"{input_name}_zero_point",
                        ],
                        output=output_name,
                        dtype=output_dtype,
                        axis=axis,
                        block_size=block_size,
                    ),
                ]
            )

    _finalize_graph_changes(
        model, nodes_to_add, inputs_to_rename, tensors_to_add, tensors_to_remove
    )


def _quantize_const(
    const: TensorProto,
    name: str,
    y_scale: np.ndarray,
    y_zero_point: np.ndarray,
    axis: Optional[int],
    block_size: Optional[int],
    output_dtype: str,
    per_block_int_scale: Optional[np.ndarray],
    opset,
    base_dir: str = "",
) -> TensorProto:
    const = to_array(const, base_dir=base_dir).astype(np.float32)
    # Always quantize in float32
    y_scale = y_scale.astype(np.float32)

    if per_block_int_scale is not None:
        block_axis = axis
        channel_axis = 0 if block_axis in (1, -1) else 1
        y_scale = y_scale.reshape(
            *(-1 if axis == channel_axis else 1 for axis in range(const.ndim))
        )
        y_scale = (y_scale * per_block_int_scale).astype(np.float32)

    y_scale = _broadcast(y_scale, const.ndim, axis=axis, block_size=block_size)
    y_zero_point = (
        _broadcast(y_zero_point, const.ndim, axis=axis, block_size=block_size)
        if y_zero_point is not None
        else np.zeros(y_scale.shape, dtype=np.int32)
    )

    y_scale = y_scale.astype(np.float32)
    if np.any(y_scale == 0.0):
        raise RuntimeError(
            f"y_scale for constant {name} contains zero entries "
            f"(count={int(np.sum(y_scale == 0.0))}/{y_scale.size}); "
            "cannot divide `const` by zero. This usually means an upstream "
            "encoding (weight/input) collapsed to zero. "
            "Please check provided input for calibration or increase precision of the model."
        )
    const_q = const / y_scale + y_zero_point

    if "int" in output_dtype:
        unsigned, bitwidth = output_dtype.split("int")
        bitwidth = int(bitwidth)

        if unsigned:
            clip_min = 0
            clip_max = 2**bitwidth - 1
        else:
            clip_min = -(2 ** (bitwidth - 1))
            clip_max = -clip_min - 1

        const_q = const_q.round().clip(clip_min, clip_max)

    return opset.DequantizeLinear.make_arr(const_q, dtype=output_dtype, name=name)


def _dequantize_const(
    const_q: TensorProto,
    name: str,
    y_scale: np.ndarray,
    y_zero_point: np.ndarray,
    axis: Optional[int],
    block_size: Optional[int],
    output_dtype: str,
    per_block_int_scale: Optional[np.ndarray],
) -> TensorProto:
    if output_dtype == "bfloat16":
        raise RuntimeError("Unsupported data type: {}")

    const_q = to_array(const_q)
    # Always dequantize in float32
    y_scale = y_scale.astype(np.float32)

    if per_block_int_scale is not None:
        block_axis = axis
        channel_axis = 0 if block_axis in (1, -1) else 1
        y_scale = y_scale.reshape(
            *(-1 if axis == channel_axis else 1 for axis in range(const_q.ndim))
        )
        y_scale = (y_scale * per_block_int_scale).astype(np.float32)

    y_scale = _broadcast(y_scale, const_q.ndim, axis=axis, block_size=block_size)
    y_zero_point = (
        _broadcast(y_zero_point, const_q.ndim, axis=axis, block_size=block_size)
        if y_zero_point is not None
        else np.zeros(y_scale.shape, dtype=np.int32)
    )

    const_q = const_q.astype(np.int64)
    y_scale = y_scale.astype(np.float32)
    y_zero_point = y_zero_point.astype(np.int64)

    const_dq = (const_q - y_zero_point) * y_scale
    return from_array(const_dq.astype(output_dtype), name=name)


def _broadcast(
    x: np.ndarray, ndim: int, axis: Optional[int], block_size: Optional[int]
) -> np.ndarray:
    if axis is None:
        return x

    axis = (ndim + axis) % ndim  # Make positive
    if block_size is None:
        channel_axis = axis
        broadcast_shape = tuple(
            -1 if axis == channel_axis else 1 for axis in range(ndim)
        )
        x = x.reshape(broadcast_shape)
    else:
        block_axis = axis
        x = x.repeat(block_size, axis=block_axis)

    return x


def _finalize_graph_changes(
    model: ModelProto,
    nodes_to_add: Iterable,
    inputs_to_rename: Dict,
    tensors_to_add: List[TensorProto],
    tensors_to_remove: Dict,
):
    # Remove dangling tensors/nodes
    initializers = [
        init
        for init in model.graph.initializer
        if not tensors_to_remove.pop(init.name, None)
    ]
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(initializers)

    nodes = [
        node
        for node in model.graph.node
        if not (
            node.op_type == "Constant" and tensors_to_remove.pop(node.output[0], None)
        )
    ]
    model.graph.ClearField("node")
    model.graph.node.extend(nodes)

    # Redirect consumers that took the removed biases to take qdq bias instead
    # before:
    #     bias --------------------> consumer
    # after:
    #     bias_int32 --> DQ -------> consumer
    for node in model.graph.node:
        for i, old_name in enumerate(node.input):
            new_name = inputs_to_rename.get(old_name, None)
            if new_name is not None:
                node.input[i] = new_name

    # Add new tensors
    for t in tensors_to_add:
        model.graph.initializer.append(t)

    # Insert new nodes in a topologically order
    original_nodes = deque(list(model.graph.node))
    new_nodes = {node.input[0]: node for node in nodes_to_add if node.input}
    queue = deque([node for node in nodes_to_add if not node.input])

    queue.extend(
        [new_nodes.pop(inp.name) for inp in model.graph.input if inp.name in new_nodes]
    )
    queue.extend(
        [
            new_nodes.pop(init.name)
            for init in model.graph.initializer
            if init.name in new_nodes
        ]
    )

    if not queue and original_nodes:
        queue.append(original_nodes.popleft())

    model.graph.ClearField("node")

    while queue:
        node = queue.popleft()
        model.graph.node.append(node)

        qdq_nodes = [
            new_nodes.pop(output_name)
            for output_name in node.output
            if output_name in new_nodes
        ]
        if qdq_nodes:
            queue.extend(qdq_nodes)

        if not queue and original_nodes:
            queue.append(original_nodes.popleft())

    model.graph.node.extend(new_nodes.values())


class _ParamUtils:
    """Param utilities"""

    @staticmethod
    def get_shape(
        model: ModelProto, node: NodeProto, param_index: int
    ) -> Optional[Sequence[int]]:
        """
        Returns a list of shape for the param specifies
        :param model: ONNX model
        :param node: ONNX node to which the param feeds to
        :param param_index: Index at which param feeds to the ONNX node
        """
        param = _ParamUtils.get_param(model, node, param_index)
        if param:
            return param.dims
        return None

    @staticmethod
    def get_param(
        model: ModelProto, node: NodeProto, param_index: int
    ) -> Optional[TensorProto]:
        """
        Returns the param tensor
        :param model: ONNX model
        :param node: ONNX node to which the param feeds to
        :param param_index: Index at which param feeds to the ONNX node
        """
        if len(node.input) >= param_index + 1:
            param_name = node.input[param_index]
            return _ParamUtils.get_param_by_name(model, param_name)
        return None

    @staticmethod
    def get_param_by_name(model: ModelProto, param_name: str) -> Optional[TensorProto]:
        """
        Returns the param tensor

        :param model: ONNX model
        :param param_name: Name of parameter to retrieve
        """

        def find_param_in_model_initializers(param_name: str, model: ModelProto):
            for param in model.graph.initializer:
                if param.name == param_name:
                    return param
            return None

        def find_param_in_model_constants(param_name: str, model: ModelProto):
            for node in model.graph.node:
                if node.op_type == "Constant" and param_name in node.output:
                    for attribute in node.attribute:
                        if attribute.name == "value":
                            param = attribute.t
                            param.name = param_name
                            return param
                if node.op_type == "Identity" and param_name == node.output[0]:
                    return _ParamUtils.get_param(model, node, 0)
            return None

        param = find_param_in_model_initializers(param_name, model)
        if param is None:
            param = find_param_in_model_constants(param_name, model)
        return param


_all_op_schemas = {schema.name: schema for schema in onnx.defs.get_all_schemas()}


def _is_float_input(op_type: str, i: int) -> bool:
    schema = _all_op_schemas[op_type]

    if (
        len(schema.inputs) > 0
        and schema.inputs[-1].option == OpSchema.FormalParameterOption.Variadic
    ):
        i = min(i, len(schema.inputs) - 1)

    if i >= len(schema.inputs):
        raise ValueError(
            f"Input index {i} is out of range for operator {op_type} with {len(schema.inputs)} inputs"
        )

    input = schema.inputs[i]
    return _is_float(schema, input)


def _is_float_output(op_type: str, i: int) -> bool:
    """
    Returns True if op_type can return float output
    """
    schema = _all_op_schemas[op_type]

    if schema.outputs[-1].option == OpSchema.FormalParameterOption.Variadic:
        i = min(i, len(schema.outputs) - 1)

    if i >= len(schema.outputs):
        raise ValueError(
            f"Output index {i} is out of range for operator {op_type} with {len(schema.outputs)} outputs"
        )

    output = schema.outputs[i]
    return _is_float(schema, output)


def _is_float(schema: OpSchema, tensor_spec: OpSchema.FormalParameter) -> bool:
    type_str = tensor_spec.type_str
    try:
        type_constraint = next(
            type_constraint
            for type_constraint in schema.type_constraints
            if type_constraint.type_param_str == type_str
        )
    except StopIteration:
        type_constraint = None

    if type_constraint:
        allowed_type_strs = type_constraint.allowed_type_strs
    else:
        allowed_type_strs = [type_str]

    return any(
        t in ("tensor(float)", "tensor(double)", "tensor(float16)")
        for t in allowed_type_strs
    )


def _is_grid_preserving_op(op_type: str, domain: str = "") -> bool:
    """
    Returns True if op_type is a grid-preserving op.

    Unary function `f(x1, x2, ..., xn)` is grid-preserving
    if and only if `q(f(x)) == f(q(x))` for arbitrary quantization function `q`.

    Note that grid-preserving ops is a subset of grid-equivariant ops
    defined in `_is_grid_equivariant_op`
    """
    return (domain, op_type) in (
        ("", "Col2Im"),
        ("", "Compress"),
        ("", "DepthToSpace"),
        ("", "Dropout"),
        ("", "Expand"),
        ("", "Flatten"),
        ("", "Gather"),
        ("", "GatherElements"),
        ("", "GatherND"),
        ("", "Identity"),
        ("", "MaxPool"),
        ("", "MaxRoiPool"),
        ("", "NonZero"),
        ("", "Pad"),
        ("", "ReduceMax"),
        ("", "ReduceMin"),
        ("", "Reshape"),
        ("", "Slice"),
        ("", "SpaceToDepth"),
        ("", "Split"),
        ("", "SplitToSequence"),
        ("", "Squeeze"),
        ("", "Tile"),
        ("", "TopK"),
        ("", "Transpose"),
        ("", "Unsqueeze"),
        ("qti_aisw", "BatchToSpace"),
        ("qti_aisw", "SpaceToBatch"),
    )


def _is_grid_equivariant_op(op_type: str, domain: str = "") -> bool:
    """
    Returns True if op_type is a grid-equivariant op.

    N-ary function `f(x1, x2, ..., xn)` is grid-equivariant
    if and only if `q(f(x1, x2, ..., xn)) == f(q(x1), q(x2), ..., q(xn))`
    for arbitrary quantization function `q`.
    """
    return _is_grid_preserving_op(op_type, domain) or (domain, op_type) in (
        ("", "Concat"),
        ("", "Scatter"),
        ("", "Where"),
    )


def _is_htp_interpolation_op(op_type: str, domain: str = "") -> bool:
    """
    Returns True if op_type can be considered an interpolation op in HTP.
    Although these operators aren't strictly data movement ops,
    HTP reuses the same quantization encoding for both input and output of
    the interpolation ops
    """
    # TODO: Absorb this function into redesigned config file
    return (domain, op_type) in (
        ("", "Resize"),
        ("", "ScatterElements"),
        ("", "ScatterND"),
        ("", "Upsample"),
        ("qti_aisw", "CropAndResize"),
    )


def _is_metadata_op(op_type: str, domain: str = "") -> bool:
    return (domain, op_type) in (
        ("", "Shape"),
        ("", "Size"),
    )


def _parse(markdown: str) -> dict[str, list[str]]:
    onnx_to_qnn_ir = {}
    rows = markdown.strip().split("\n")[2:]  # Skip the header and separator lines
    for row in rows:
        _, onnx_op, qnn_ir_op, *_ = (entry.strip() for entry in row.split("|"))
        assert onnx_op in _all_op_schemas
        onnx_to_qnn_ir.setdefault(onnx_op, []).append(qnn_ir_op)

    return onnx_to_qnn_ir


_onnx_to_qnn_ir: dict[str, list[str]] = _parse(
    """
| ONNX Operator        | QNN IR Operation         |
|----------------------|--------------------------|
| Abs                  | ElementWiseAbs           |
| Add                  | ElementWiseAdd           |
| And                  | ElementWiseAnd           |
| ArgMax               | Argmax                   |
| ArgMin               | Argmin                   |
| Asin                 | ElementWiseAsin          |
| Atan                 | ElementWiseAtan          |
| AveragePool          | PoolAvg2d                |
| AveragePool          | PoolAvg3d                |
| BatchNormalization   | Batchnorm                |
| Cast                 | Cast                     |
| Ceil                 | ElementWiseCeil          |
| Clip                 | ReluMinMax               |
| Col2Im               | Col2Im                   |
| Concat               | Concat                   |
| ConstantOfShape      | ConstantOfShape          |
| Conv                 | Conv2d                   |
| Conv                 | Conv3d                   |
| Conv                 | DepthWiseConv2d          |
| ConvTranspose        | TransposeConv2d          |
| ConvTranspose        | TransposeConv3d          |
| Cos                  | ElementWiseCos           |
| CumSum               | CumulativeSum            |
| DepthToSpace         | DepthToSpace             |
| DequantizeLinear     | Dequantize               |
| Div                  | ElementWiseDivide        |
| Elu                  | Elu                      |
| Equal                | ElementWiseEqual         |
| Exp                  | ElementWiseExp           |
| Flatten              | Reshape                  |
| Floor                | ElementWiseFloor         |
| Gather               | Gather                   |
| GatherElements       | GatherElements           |
| GatherND             | GatherNd                 |
| Gelu                 | Gelu                     |
| Gemm                 | FullyConnected           |
| GlobalAveragePool    | PoolAvg2d                |
| GlobalAveragePool    | PoolAvg3d                |
| GlobalMaxPool        | PoolMax2d                |
| GlobalMaxPool        | PoolMax3d                |
| Greater              | ElementWiseGreater       |
| GreaterOrEqual       | ElementWiseGreaterEqual  |
| GridSample           | GridSample               |
| GroupNormalization   | GroupNorm                |
| GRU                  | Gru                      |
| HardSwish            | HardSwish                |
| InstanceNormalization| InstanceNorm             |
| IsInf                | IsInf                    |
| IsNaN                | ElementWiseNotEqual      |
| IsNaN                | IsNan                    |
| LayerNormalization   | LayerNorm                |
| LeakyRelu            | Prelu                    |
| Less                 | ElementWiseLess          |
| LessOrEqual          | ElementWiseLessEqual     |
| Log                  | ElementWiseLog           |
| LogSoftmax           | LogSoftmax               |
| LpNormalization      | L2Norm                   |
| LpPool               | L2Pool2d                 |
| LRN                  | Lrn                      |
| LSTM                 | Lstm                     |
| MatMul               | MatMul                   |
| MatMul               | FullyConnected           |
| Max                  | ElementWiseMaximum       |
| MaxPool              | PoolMax2d                |
| MaxPool              | PoolMax3d                |
| MaxRoiPool           | RoiPooling               |
| Min                  | ElementWiseMinimum       |
| Mod                  | ElementWiseFmod          |
| Mod                  | ElementWiseMod           |
| Mul                  | ElementWiseMultiply      |
| Neg                  | ElementWiseNeg           |
| NonMaxSuppression    | NonMaxSuppression        |
| NonZero              | NonZero                  |
| Not                  | ElementWiseNot           |
| OneHot               | OneHot                   |
| Or                   | ElementWiseOr            |
| Pad                  | Pad                      |
| Pow                  | ElementWisePower         |
| PRelu                | Prelu                    |
| QLinearConv          | Conv2d                   |
| QLinearConv          | Conv3d                   |
| QLinearConv          | DepthWiseConv2d          |
| QLinearMatMul        | MatMul                   |
| QuantizeLinear       | Quantize                 |
| RandomUniformLike    | RandomUniformLike        |
| Reciprocal           | ElementWiseDivide        |
| ReduceMax            | ReduceMax                |
| ReduceMean           | ReduceMean               |
| ReduceMin            | ReduceMin                |
| ReduceProd           | ReduceProd               |
| ReduceSum            | ReduceSum                |
| ReduceSumSquare      | ReduceSumSquare          |
| Relu                 | Relu                     |
| Reshape              | Reshape                  |
| Resize               | Resize                   |
| RMSNormalization     | RmsNorm                  |
| RoiAlign             | RoiAlign                 |
| RotaryEmbedding      | RotaryEmbedding          |
| Round                | ElementWiseRound         |
| Scatter              | ScatterElements          |
| ScatterElements      | ScatterElements          |
| ScatterND            | ScatterNd                |
| Shape                | Shape                    |
| Sigmoid              | Sigmoid                  |
| Sign                 | ElementWiseSign          |
| Sin                  | ElementWiseSin           |
| Slice                | StridedSlice             |
| Softmax              | Softmax                  |
| Softplus             | ElementWiseSoftplus      |
| SpaceToDepth         | SpaceToDepth             |
| Split                | Split                    |
| Sqrt                 | ElementWiseSquareRoot    |
| Squeeze              | Reshape                  |
| STFT                 | Stft                     |
| Sub                  | ElementWiseSubtract      |
| Sum                  | ElementWiseAdd           |
| Tanh                 | Tanh                     |
| Tile                 | Tile                     |
| TopK                 | TopK                     |
| Transpose            | Transpose                |
| Unsqueeze            | Reshape                  |
| Where                | ElementWiseSelect        |
| Xor                  | ElementWiseXor           |
"""
)

del _parse


def _convert_version_with_external_weights(model, target_opset_version):
    """
    Upgrade opset version with weights flushed to disk temporarily
    """
    regular_tensors = {
        tensor.name
        for tensor in _get_all_tensors(model)
        if not uses_external_data(tensor)
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_file = os.path.join(tmp_dir, "model.onnx")

        # Temporarily switch regular (internal) tensors to external
        onnx.save_model(
            model,
            onnx_file,
            save_as_external_data=True,
            location="model.data",
        )
        external_tensors = {
            tensor.name: tensor.external_data[:]
            for tensor in _get_all_tensors(model)
            if uses_external_data(tensor)
        }

        model = onnx.version_converter.convert_version(model, target_opset_version)

        # Restore original state of the model
        for tensor in _get_all_tensors(model):
            if tensor.name in external_tensors:
                # Step 1. Restore external_data of all tensors
                # NOTE: This step is only necessary in onnx < 1.19,
                # where version converter strips away external_data information
                external_data = external_tensors[tensor.name]
                tensor.data_location = TensorProto.EXTERNAL
                del tensor.external_data[:]
                tensor.external_data.extend(external_data)

                # Step 2. Load raw_data of tensors that were original non-external
                if tensor.name in regular_tensors:
                    load_external_data_for_tensor(tensor, tmp_dir)
                    tensor.data_location = TensorProto.DEFAULT
                    del tensor.external_data[:]

    return model


def _convert_version(
    model: onnx.ModelProto, target_opset_version: int
) -> onnx.ModelProto:
    if any(uses_external_data(tensor) for tensor in _get_all_tensors(model)):
        model = _convert_version_with_external_weights(model, target_opset_version)
    else:
        try:
            model = onnx.version_converter.convert_version(model, target_opset_version)
        except Exception as e:  # pylint: disable=broad-exception-caught
            # convert_version throws an exception on model > 2GB the observed exception was a
            # RuntimeError exception about ir_version, but possible other exceptions could be
            # triggered. leaving this very generic for now.
            logger.warning(
                "onnx.version_converter.convert_version failed with exception: %s. Retrying with external data",
                str(e),
            )
            model = _convert_version_with_external_weights(model, target_opset_version)

    logger.info("The opset of the onnx model is updated to %s.", target_opset_version)
    return model


def _remove_onnx_qdq_nodes(
    model: onnx.ModelProto,
) -> List[Dict[str, Union[str, int, np.ndarray]]]:
    initializers: Dict[str, TensorProto] = {
        init.name: init for init in model.graph.initializer
    }
    constants: Dict[str, TensorProto] = _get_all_constants(model)
    q_nodes: Dict[str, NodeProto] = {}
    dq_nodes: Dict[str, NodeProto] = {}
    producers: Dict[str, NodeProto] = {}
    consumers: Dict[str, Dict[str, NodeProto]] = defaultdict(dict)
    graph_outputs = set(output.name for output in model.graph.output)

    _validate_model(model, constants, consumers, producers)

    to_encoding = functools.partial(
        _to_encoding,
        model=model,
        constants=constants,
        consumers=consumers,
        producers=producers,
    )
    get_lpbq_nodes = functools.partial(
        _get_lpbq_nodes, producers=producers, consumers=consumers, constants=constants
    )
    get_qdq_nodes = functools.partial(
        _get_qdq_nodes, producers=producers, consumers=consumers, constants=constants
    )

    for node in model.graph.node:
        if node.op_type == "QuantizeLinear":
            q_nodes[node.name] = node
        elif node.op_type == "DequantizeLinear":
            dq_nodes[node.name] = node

        for inp in node.input:
            consumers[inp].update({node.name: node})
        for out in node.output:
            producers[out] = node

    to_be_removed = {}
    for dq in dq_nodes.values():
        lpbq_nodes = get_lpbq_nodes(dq)
        if lpbq_nodes:
            to_be_removed.update({node.name: node for node in lpbq_nodes})
            continue

        qdq_nodes = get_qdq_nodes(dq)
        if qdq_nodes:
            to_be_removed.update({node.name: node for node in qdq_nodes})

    encodings = {
        dq.name: to_encoding(dq)
        for dq in to_be_removed.values()
        if dq.op_type == "DequantizeLinear"
    }
    encodings = {
        node_name: encoding
        for node_name, encoding in encodings.items()
        if encoding is not None
    }

    # Reconnect nodes
    for dq in model.graph.node:
        if dq.op_type != "DequantizeLinear":
            continue

        producer = producers.get(dq.input[0])
        if producer and producer.op_type == "QuantizeLinear":
            q = producer
            producer = producers.get(q.input[0])
        else:
            q = None

        if dq.output[0] in graph_outputs and not producer:
            # Edge case: This means the model was in form of:
            #   (model_input) --> Q -> DQ -> (model_output)
            # or:
            #   (constant) -----> Q -> DQ -> (model_output)
            #
            # We can't preserve the I/O names in this case
            raise RuntimeError(
                f"Node {q.name} (op_type: {q.op_type}) can't be removed because "
                "it's the only connection between the model's input and output."
            )

        if not q:
            # Standalone DQs can be removed if it only takes static inputs.
            #
            # Before:
            #   (constant) -----> Q -> DQ ----> (consumers or model_output)
            #                               ↑
            #                           dq.output[0]
            #                           (=new_name)
            # After:
            #   (constant) -------------------> (consumers or model_output)
            #                               ↑
            #                           dq.output[0]
            #                           (=new_name)
            const = constants.get(dq.input[0])

            if const and const.name not in initializers:
                const_node = producers[const.name]
                to_be_removed[const_node.name] = const_node

            if const and dq.name in encodings:
                new_name = dq.output[0]
                e = encodings[dq.name]
                initializers[dq.output[0]] = _dequantize_const(
                    const,
                    name=new_name,
                    y_scale=e.get("y_scale", e.get("per_channel_float_scale")),
                    y_zero_point=e.get("y_zero_point"),
                    axis=e.get("axis"),
                    block_size=e.get("block_size"),
                    output_dtype="float32",
                    per_block_int_scale=e.get("per_block_int_scale"),
                )

            continue

        if dq.output[0] in graph_outputs:
            # DQ output is part of graph outputs.
            # We should preserve DQ's output name to preserve the graph output name
            #
            # Before:
            #                             +--> consumers
            #   producer -----> Q -> DQ --+--> (model_output)
            #                             ↑
            #                         dq.output[0]
            #                         (=new_name)
            # After:
            #                             +--> consumers
            #   producer -----------------+--> (model_output)
            #                             ↑
            #                         dq.output[0]
            #                         (=new_name)
            new_name = dq.output[0]
        else:
            # Before:
            #   producer -----> Q -> DQ -----> consumers
            #              ↑
            #           q.input[0]
            #          (=new_name)
            # After:
            #   producer --------------------> consumers
            #              ↑
            #           q.input[0]
            #          (=new_name)
            new_name = q.input[0]

        for consumer in consumers[dq.output[0]].values():
            for i, inp in enumerate(consumer.input):
                if inp == dq.output[0]:
                    consumer.input[i] = new_name
        if producer:
            for i, out in enumerate(producer.output):
                if out == q.input[0]:
                    producer.output[i] = new_name

    included_nodes = set()
    node = []
    # Nodes may appear in producer.values() multiple times, only use the first appearance
    for producer in producers.values():
        if producer.name not in to_be_removed and producer.name not in included_nodes:
            included_nodes.add(producer.name)
            node.append(producer)

    model.graph.ClearField("node")
    model.graph.node.extend(node)

    # model.graph.ClearField("initializer")
    # model.graph.initializer.extend(list(initializers.values()))
    # `initializers` is purely additive: it starts from every existing graph
    # initializer and only gains new dequantized-const entries (with DQ-output
    # names that never collide with existing ones). Re-serializing the whole set
    # via ClearField + extend would needlessly re-encode large tensors (e.g. a
    # ~2 GiB tied lm_head weight) and overflow protobuf's 2 GiB message ceiling.
    # Instead, leave existing initializers in place and append only the new ones.
    existing_init_names = {init.name for init in model.graph.initializer}
    model.graph.initializer.extend(
        init for name, init in initializers.items() if name not in existing_init_names
    )
    from onnxruntime.quantization.onnx_quantizer import ONNXModel

    ONNXModel(model).remove_unused_constant()

    # Convert removed Q/DQ nodes to encoding
    return list(encodings.values())


def _validate_model(
    model: ModelProto,
    constants: Dict[str, TensorProto],
    consumers: Dict[str, Dict[str, NodeProto]],
    producers: Dict[str, NodeProto],
) -> None:
    invalid_nodes = []

    for node in model.graph.node:
        if node.op_type == "QuantizeLinear":
            is_qdq = _is_q_dq_sequence(node, consumers)
        elif node.op_type == "DequantizeLinear":
            is_qdq = _get_qdq_nodes(
                node, producers, consumers, constants
            ) or _is_lpbq_subgraph(node, producers, consumers, constants)
        else:
            continue

        if not is_qdq:
            invalid_nodes.append(node)

    if not invalid_nodes:
        return

    invalid_node_names = ", ".join([node.name for node in invalid_nodes])
    raise RuntimeError(
        f"Invalid QuantizeLinear/DequantizeLinear detected: {invalid_node_names}.\n\n"
        "To import onnx QDQ model, please ensure the following requirements:"
        "  - All QuantizeLinear (if any) must be followed by DequantizeLinear.\n"
        "  - All DequantizeLinaer must be 1) preceded by QuantizeLinear or 2) take static constant as input"
    )


def _to_encoding(
    dq: NodeProto,
    model: ModelProto,
    constants: Dict[str, TensorProto],
    consumers: Dict[str, Dict[str, NodeProto]],
    producers: Dict[str, NodeProto],
) -> Optional[Dict[str, Union[str, int, np.ndarray]]]:
    q = producers.get(dq.input[0])

    if q and q.op_type != "QuantizeLinear":
        raise RuntimeError(
            f"DequantizeLinear can be only preceded by QuantizeLinear. "
            f"Got {q.op_type} (name: {q.name})"
        )

    for consumer in consumers[dq.output[0]].values():
        if consumer.op_type == "DequantizeLinear":
            lpbq = _get_lpbq_nodes(consumer, producers, consumers, constants)
            if not lpbq:
                raise RuntimeError(
                    f"Back-to-back DequantizeLinear detected at {dq.name}. "
                    "Back-to-back DequantizeLinear is only supported in LPBQ"
                )
            return None

    if any(dq.output[0] == graph_out.name for graph_out in model.graph.output):
        input_name = dq.output[0]
    else:
        input_name = q.input[0] if q else dq.output[0]

    lpbq = _get_lpbq_nodes(dq, producers, consumers, constants)
    if lpbq:
        *_, scale_dq = lpbq
        scale = {
            "per_block_int_scale": to_array(constants[scale_dq.input[0]]),
            "per_channel_float_scale": to_array(constants[scale_dq.input[1]]),
        }
        per_tensor = False
    else:
        scale_arr = to_array(constants[dq.input[1]])
        per_tensor = scale_arr.ndim == 0
        scale = {"y_scale": scale_arr}

    if len(dq.input) > 2:
        zp_name = dq.input[2]
        zp_tensor_proto = constants[zp_name]

        if zp_tensor_proto.data_type not in (
            TensorProto.INT4,
            TensorProto.INT8,
            TensorProto.INT16,
            TensorProto.INT32,
            TensorProto.UINT4,
            TensorProto.UINT8,
            TensorProto.UINT16,
        ):
            raise RuntimeError(
                f'Found zero_point "{zp_name}" with unsupported dtype '
                f"{onnx.helper.tensor_dtype_to_string(zp_tensor_proto.data_type)}. "
                "Only [u]int4, [u]int8, [u]int16, and int32 are supported."
            )

        zp = to_array(zp_tensor_proto)
        output_dtype = zp_tensor_proto.data_type
    else:
        zp = None
        try:
            output_dtype = (
                next(attr.i for attr in q.attribute if attr.name == "output_dtype")
                if q
                else TensorProto.UINT8
            )
        except StopIteration:
            # ONNX assumes uint8 if neither zero_point nor output_dtype is specified
            output_dtype = TensorProto.UINT8

    *_, output_dtype = (
        onnx.helper.tensor_dtype_to_string(output_dtype).lower().split(".")
    )

    encoding = {
        "name": input_name,
        "output_dtype": output_dtype,
        **scale,
    }

    if zp is not None:
        encoding["y_zero_point"] = zp

    for attr in dq.attribute:
        if attr.name == "axis":
            encoding["axis"] = attr.i
        elif attr.name == "block_size":
            encoding["block_size"] = attr.i
        elif attr.name == "output_dtype":
            output_dtype = onnx.helper.tensor_dtype_to_np_dtype(attr.i)
            if output_dtype != encoding["output_dtype"]:
                raise RuntimeError(
                    f"Attribute output_dtype={output_dtype} of node {dq.name} "
                    "is inconsistent with "
                    f"the dtype of zero_point {encoding['output_dtype']} "
                )

    # ONNX QuantizeLinear's axis and block_size attributes are set to 1 and 0
    # respectively, sometimes even for per-tensor QDQ. Pop them to avoid confusion.
    if per_tensor:
        encoding.pop("axis", None)
        encoding.pop("block_size", None)

    return encoding


def _iterate_graph_nodes_recursive(graph: onnx.GraphProto) -> Iterable[onnx.NodeProto]:
    for node in graph.node:
        yield node

        if node.op_type == "If":
            for attr in node.attribute:
                # then/else branch subgraphs
                yield from _iterate_graph_nodes_recursive(attr.g)

        elif node.op_type in ("Loop", "Scan"):
            body = next(attr for attr in node.attribute if attr.name == "body")
            yield from _iterate_graph_nodes_recursive(body.g)


def _get_all_constants(
    model: onnx.ModelProto, consumers: dict[str, list[onnx.NodeProto]] | None = None
) -> dict[str, onnx.TensorProto]:
    """
    Get all constants in the ONNX model, including
      * Initializers
      * Output of Constant nodes.
      * Output of Identity nodes that takes initializers or Constant nodes as input (recursively).
    """
    if consumers is None:
        consumers = {}
        for node in _iterate_graph_nodes_recursive(model.graph):
            for input_name in node.input:
                consumers.setdefault(input_name, []).append(node)

    constants: dict[str, onnx.TensorProto] = {
        const.name: const for const in _get_all_tensors(model)
    }

    constants |= {
        const_node.output[0]: attr.t
        for const_node in _iterate_graph_nodes_recursive(model.graph)
        if const_node.op_type == "Constant"
        for attr in const_node.attribute
        if attr.HasField("t")
    }

    for const in constants.copy().values():
        queue = consumers.get(const.name, []).copy()
        while queue:
            consumer = queue.pop()
            if consumer.op_type == "Identity":
                constants[consumer.output[0]] = const
                queue += consumers.get(consumer.output[0], [])

    return constants


def _get_qdq_nodes(
    dq: NodeProto,
    producers: Dict[str, NodeProto],
    consumers: Dict[str, Dict[str, NodeProto]],
    constants: Dict[str, TensorProto],
) -> List[NodeProto]:
    if dq.op_type != "DequantizeLinear":
        raise ValueError(
            f"_get_qdq_nodes can only take DequantizeLinear node as input; got {dq.op_type}"
        )

    qdq_nodes = []
    q = producers.get(dq.input[0])

    if not q:
        if set(dq.input) <= constants.keys():
            # Standalone DQ with static inputs
            qdq_nodes.append(dq)
    elif (
        _is_q_dq_sequence(q, consumers)
        # Scale and zp must be constants.
        and set(q.input[1:]) <= constants.keys()
    ):
        qdq_nodes.extend([q, *consumers[q.output[0]].values()])

    return qdq_nodes


def _get_lpbq_nodes(
    dq: NodeProto,
    producers: Dict[str, NodeProto],
    consumers: Dict[str, Dict[str, NodeProto]],
    constants: Dict[str, TensorProto],
) -> List[NodeProto]:
    if dq.op_type != "DequantizeLinear":
        raise ValueError(
            f"_get_lpbq_nodes can only take DequantizeLinear node as input; got {dq.op_type}"
        )

    scale_dq = producers.get(dq.input[1])

    if not scale_dq:
        return []

    is_lpbq = (
        set(scale_dq.input) <= constants.keys()
        and set(dq.input[2:]) <= constants.keys()
    )

    q = producers.get(dq.input[0])
    if q:
        is_lpbq &= (
            _is_q_dq_sequence(q, consumers)
            # Input of Q must be constant
            and q.input[0] in constants
        )

    if is_lpbq:
        return [q, dq, scale_dq] if q else [dq, scale_dq]

    return []


def _is_q_dq_sequence(q: NodeProto, consumers: Dict[str, Dict[str, NodeProto]]):
    return (
        q.op_type == "QuantizeLinear"
        and all(
            # All Q must be followed by DQ
            consumer.op_type == "DequantizeLinear"
            # Q-DQ must share same scale and zp
            and consumer.input[1:] == q.input[1:]
            # This rules out LPBQ which takes runtime-computed scale as input
            for consumer in consumers[q.output[0]].values()
        )
    )


def _is_lpbq_subgraph(
    dq: NodeProto,
    producers: Dict[str, NodeProto],
    consumers: Dict[str, Dict[str, NodeProto]],
    constants: Dict[str, TensorProto],
) -> bool:
    """
    per_channel_float_scale -----+
                                 +--> DequantizeLinear -----+ (blockwise scale)
       per_block_uint_scale -----+        (1st DQ)          |
                                                    +-------+---------+
                                                    V                 V
                                 weight ---> QuantizeLinear -> DequantizeLinear -> ...
                                                                   (2nd DQ)
    """
    if dq.op_type != "DequantizeLinear":
        raise ValueError(
            f"_is_lpbq_subgraph can only take DequantizeLinear node as input; got {dq.op_type}"
        )

    is_2nd_dq = bool(_get_lpbq_nodes(dq, producers, consumers, constants))
    is_1st_dq = all(
        consumer.op_type == "DequantizeLinear"
        and _get_lpbq_nodes(consumer, producers, consumers, constants)
        for consumer in consumers[dq.output[0]].values()
    )
    return is_1st_dq or is_2nd_dq


def _get_node_attribute(node: NodeProto, name: str):
    """
    Return the value of a node's attribute specified by its name

    :param node: NodeProto object to retrieve the attribute from
    :param name: string containing the name of the attribute to retrieve
    :return: value of the attribute
    """
    for item in node.attribute:
        if item.name == name:
            return onnx.helper.get_attribute_value(item)
    return None


def contains_tensor_type(model: ModelProto, tensor_type: int | List[int]):
    """
    Returns True if the model contains the specified tensor type(s).
    """
    if isinstance(tensor_type, int):
        tensor_type = [tensor_type]
    if any(
        tensor.type.tensor_type.elem_type in tensor_type
        for tensor in itertools.chain(model.graph.input, model.graph.output)
    ):
        return True

    if any(tensor.data_type in tensor_type for tensor in model.graph.initializer):
        return True

    for node in model.graph.node:
        if node.op_type == "Cast":
            cast_type = _get_node_attribute(node, "to")
            if cast_type in tensor_type:
                return True

        if node.op_type in ("Constant", "ConstantOfShape"):
            value = _get_node_attribute(node, "value")
            if value is None:
                continue

            if value.data_type in tensor_type:
                return True

    return False


def _encoding_equal(enc1: Mapping | None, enc2: Mapping | None) -> bool:
    return bool(
        enc1 is not None
        and enc2 is not None
        and enc1["output_dtype"] == enc2["output_dtype"]
        and np.array_equal(enc1["y_scale"], enc2["y_scale"])
        and np.array_equal(enc1.get("y_zero_point", 0), enc2.get("y_zero_point", 0))
        and np.array_equal(enc1.get("axis"), enc2.get("axis"))
        and np.array_equal(enc1.get("block_size"), enc2.get("block_size"))
        and np.array_equal(
            enc1.get("per_channel_float_scale"), enc2.get("per_channel_float_scale")
        )
        and np.array_equal(
            enc1.get("per_block_int_scale"), enc2.get("per_block_int_scale")
        )
    )


def _is_htp_masked_softmax_reducemin(
    node: onnx.NodeProto,
    consumers: Mapping[str, List[onnx.NodeProto]],
    constants: Mapping[str, onnx.TensorProto],
) -> bool:
    """
    Returns True if node is a ReduceMin node within HTP MaskedSoftmax.

    HTP MaskedSoftmax is represented in onnx as

    Softmax(
        Where(mask, x, ReduceMin(x, axis=-1) + B),
        axis=-1,
    )

    where
      - x: 4D
      - B: scalar constant <=-20
    """
    if node.op_type != "ReduceMin":
        return False

    reducemin_axes = node.input[1]

    if reducemin_axes not in constants:
        return False

    reducemin_axes = to_array(constants[reducemin_axes])

    if not (reducemin_axes.size == 1 and reducemin_axes.item() in (-1, 3)):
        return False

    if node.output[0] not in consumers or len(consumers[node.output[0]]) != 1:
        return False

    (add,) = consumers[node.output[0]]

    if add.op_type != "Add":
        return False

    if add.output[0] not in consumers or len(consumers[add.output[0]]) != 1:
        return False

    B = next(inp for inp in add.input if inp != node.output[0])

    if B not in constants:
        return False

    B = to_array(constants[B])

    if B.size != 1 or B > -20:
        return False

    (where,) = consumers[add.output[0]]

    if where.op_type != "Where":
        return False

    if where.output[0] not in consumers or len(consumers[where.output[0]]) != 1:
        return False

    (softmax,) = consumers[where.output[0]]

    if softmax.op_type != "Softmax":
        return False

    softmax_axis = next(
        (attr.i for attr in softmax.attribute if attr.name == "axis"),
        None,
    )

    if softmax_axis not in (-1, 3):
        return False

    return True


def _derive_data_movement_op_encodings(
    model: onnx.ModelProto,
    encodings: Mapping[str, Mapping],
) -> Dict[str, Dict]:
    data_movement_ops = [
        node
        for node in model.graph.node
        if _is_grid_equivariant_op(node.op_type, domain=node.domain)
    ]

    new_encodings = {}
    consumers: Mapping[str, List[onnx.NodeProto]] = defaultdict(list)
    constants: Mapping[str, onnx.TensorProto] = _get_all_constants(model)

    for node in model.graph.node:
        for inp in node.input:
            consumers[inp].append(node)

    def derive_encoding(node: onnx.NodeProto):
        derived_encodings = {}

        # Skip deriving encodings if node is a ReduceMin node within HTP MaskedSoftmax
        # subgraph. This is a temporary workaround for GenAI Notebooks relying
        # on aimet-torch which doesn't support MaskedSoftmax as supergroup yet.
        # In aimet-torch, even if the user manually removed all intermediate quantizers
        # within MaskedSoftmax subgraph, ReduceMin's output encoding will be
        # re-populated during export since it is a grid-preserving op. As a result,
        # this intermediate encoding prevents QAIRT Quantizer V2 (IRQV2) from
        # pattern-matching MaskedSoftmax subgraph as supergroup.
        # TODO(#7434): Remove this workaround once aimet-torch supports MaskedSoftmax as supergroup
        if _is_htp_masked_softmax_reducemin(node, consumers, constants):
            return derived_encodings

        input_names = [
            name
            for i, name in enumerate(node.input)
            if _is_float_input(node.op_type, i)
        ]
        output_names = [
            name
            for i, name in enumerate(node.output)
            if _is_float_output(node.op_type, i)
        ]

        can_propagate_forward = all(
            _encoding_equal(encodings.get(inp), encodings.get(input_names[0]))
            for inp in input_names[1:]
        )
        can_propagate_backward = all(
            _encoding_equal(encodings.get(out), encodings.get(output_names[0]))
            for out in output_names[1:]
        )

        for input_name, output_name in itertools.product(input_names, output_names):
            inp_encoding = encodings.get(input_name)
            out_encoding = encodings.get(output_name)

            if inp_encoding and out_encoding:
                # Both input and output encoding already exists; skip
                continue

            # Only per-tensor encodings can be safely propagated through data movement ops
            # because some data movement ops such as Reshape and Transpose can't reuse
            # the same channel/block axes across inputs and outputs
            if (
                out_encoding
                and out_encoding.get("axis") is None
                and can_propagate_backward
            ):
                if len(consumers[input_name]) > 1:
                    # If input has more than one consumer or if there are more than one output,
                    # it is NOT safe to reuse output encoding for input quantization
                    continue
                else:
                    # Reuse output encoding for input quantization
                    derived_encodings.update({input_name: out_encoding.copy()})
                    continue

            # Only per-tensor encodings can be safely propagated through data movement ops
            # because some data movement ops such as Reshape and Transpose can't reuse
            # the same channel/block axes across inputs and outputs
            if (
                inp_encoding
                and inp_encoding.get("axis") is None
                and can_propagate_forward
            ):
                # Reuse input encoding for output quantization
                derived_encodings.update({output_name: inp_encoding.copy()})
                continue

        return derived_encodings

    for node in data_movement_ops:
        enc = derive_encoding(node)
        new_encodings |= enc
        encodings |= enc

    # Repeat in reverse-DFS order
    for node in reversed(data_movement_ops):
        enc = derive_encoding(node)
        new_encodings |= enc
        encodings |= enc

    return {key: enc for key, enc in new_encodings.items()}


def _get_effective_encoding(
    tensor: str,
    producers: Mapping[str, onnx.NodeProto],
    encodings: Mapping[str, Mapping],
) -> Optional[Mapping]:
    """
    Returns encoding for tensor, propagating upwards through grid preserving ops if necessary
    """
    if tensor in encodings:
        return encodings[tensor]

    producer = producers.get(tensor)
    if not producer:
        return None

    if (
        _is_grid_preserving_op(producer.op_type, domain=producer.domain)
        and producer.input
    ):
        return _get_effective_encoding(producer.input[0], producers, encodings)

    return None


def _is_constant_scalar(
    tensor_name: str,
    constants: Mapping[str, TensorProto],
    base_dir: str = "",
) -> bool:
    tensor = constants.get(tensor_name)
    if tensor is None:
        return False
    array = to_array(tensor, base_dir=base_dir)
    return array.size == 1


def _derive_const_rescale_op_output_encodings(
    model: onnx.ModelProto,
    encodings: Mapping[str, Mapping],
    base_dir: str = "",
) -> Dict[str, Dict]:
    updated_encodings = encodings.copy()
    constants = _get_all_constants(model)
    producers = {output: node for node in model.graph.node for output in node.output}
    for node in model.graph.node:
        if node.op_type not in ("Mul", "Div"):
            continue
        if node.output[0] in updated_encodings:
            continue
        inp_idx, scale_idx = (0, 1)
        if node.op_type == "Mul" and not _is_constant_scalar(
            node.input[1], constants, base_dir
        ):
            inp_idx, scale_idx = scale_idx, inp_idx
        if node.input[scale_idx] in encodings:
            continue  # Skip if rescaling factor is quantized
        if not _is_constant_scalar(node.input[scale_idx], constants, base_dir):
            continue
        const_factor = to_array(constants[node.input[scale_idx]], base_dir=base_dir)
        if const_factor.item() <= 0:
            continue
        input_encoding = _get_effective_encoding(
            node.input[inp_idx], producers, updated_encodings
        )
        if input_encoding is None:
            continue
        input_scale = input_encoding.get("y_scale")
        if not isinstance(input_scale, float):
            continue

        scale_factor = (
            1 / const_factor.item() if node.op_type == "Div" else const_factor.item()
        )
        if scale_factor == 0 or math.isnan(scale_factor) or math.isinf(scale_factor):
            continue
        output_encoding = input_encoding.copy()
        output_encoding["y_scale"] = input_scale * scale_factor
        updated_encodings[node.output[0]] = output_encoding

        # Insert trivial encoding for constant scale factor
        scale_encoding = input_encoding.copy()
        scale_encoding["y_scale"] = const_factor.item()
        scale_encoding["y_zero_point"] = 0
        updated_encodings[node.input[scale_idx]] = scale_encoding

    new_encodings = {k: v for k, v in updated_encodings.items() if k not in encodings}
    return new_encodings

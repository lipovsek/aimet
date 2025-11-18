# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Tuple, Dict
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto, TensorShapeProto

from aimet_onnx.common.utils import AimetLogger

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SeqMse)


def modify_graph_with_grouped_conv(
    model: onnx.ModelProto, block_size: int, block_axis: int
) -> onnx.ModelProto:
    """
    Transforms Conv nodes in the given ONNX model into grouped convolutions based on the specified block_size and block_axis.

    :param model: The input ONNX model to be modified.
    :param block_size: The size of the block along the block axis in the weight tensor.
    :param block_axis: Block axis.
    """
    graph = model.graph
    new_nodes = []

    conv_nodes = [n for n in graph.node if n.op_type == "Conv"]
    if not conv_nodes:
        raise ValueError("No Conv nodes found in the graph.")

    for node in conv_nodes:
        input_name, weight_name, output_name = (
            node.input[0],
            node.input[1],
            node.output[0],
        )

        # Get weight tensor
        weight_tensor = _get_weight_tensor(graph, weight_name)

        num_blocks = weight_tensor.shape[block_axis] // block_size

        # Transform the weight tensor to support grouped convolution
        weight_info = _transform_conv_weight(
            graph, weight_name, weight_tensor.shape, num_blocks
        )
        transformed_weight_name = weight_info["final_output"]
        new_nodes.extend(weight_info["nodes"])

        # Extract the existing attributes and update the `group` attribute
        conv_attrs = {
            attr.name: helper.get_attribute_value(attr) for attr in node.attribute
        }
        if conv_attrs.get("group", 1) != 1:
            raise NotImplementedError(
                f"Regular Conv (group=1) only supported for now, found Conv with group: {conv_attrs.get('group')}"
            )

        conv_attrs["group"] = num_blocks * conv_attrs.get("group", 1)

        grouped_conv_output_name = _make_name(node.output[0], "grouped_conv_output")
        grouped_conv_node = helper.make_node(
            "Conv",
            inputs=[input_name, transformed_weight_name],
            outputs=[grouped_conv_output_name],
            name=node.name,
            **conv_attrs,
        )
        new_nodes.append(grouped_conv_node)

        # Update output tensor's name and shape after replacing original Conv node
        # with a grouped conv node.
        for output_vi in graph.output:
            if output_vi.name == output_name:
                output_vi.name = grouped_conv_output_name

                # Create symbolic shape
                symbolic_dims = ["batch", "c_out", "h_out", "w_out"]

                shape_proto = TensorShapeProto()
                for dim_name in symbolic_dims:
                    dim = shape_proto.dim.add()
                    dim.dim_param = dim_name

                    # Apply new shape
                    output_vi.type.tensor_type.shape.CopyFrom(shape_proto)

        # Remove original conv node
        graph.node.remove(node)

    # Add all the newly created nodes
    graph.node.extend(new_nodes)


def modify_graph_with_grouped_linear(
    model: onnx.ModelProto, block_size: int, block_axis: int
) -> onnx.ModelProto:
    """
    Transforms linear nodes (MatMul/Gemm) in the given ONNX model into block-wise grouped linear operations.

    Assumptions:

    - Inputs to the ONNX graph are preprocessed using `prepare_linear_inputs()` to shape:
        (num_blocks, batch, block_size)
    - All the Gemm/MatMul nodes in the given ONNX should have same block size and block_axis.
    - Original model input shapes are ignored and replaced with symbolic grouped shapes.

    :param model: The input ONNX model to be modified.
    :param block_size: The size of the block along the block axis in the weight tensor.
    :param block_axis: Block axis.
    """
    graph = model.graph
    new_nodes = []

    # Clear all value_info entries since they are not required - since they are mainly used
    # for shape inference.
    graph.ClearField("value_info")

    # NOTE: All inputs will now have symbolic shape (num_blocks, batch, block_size)
    symbolic_shape = ["num_blocks", "batch", "block_size"]

    # Update model inputs
    # These are external inputs to the model, we update their shape to use symbolic
    # dimensions, but preserve the original data types (e.g. FLOAT, FLOAT16)
    for i, input_tensor in enumerate(graph.input):
        input_name = input_tensor.name
        elem_type = input_tensor.type.tensor_type.elem_type

        # Create a new input tensor with symbolic dimensions and same name
        new_input = helper.make_tensor_value_info(input_name, elem_type, symbolic_shape)
        graph.input[i].CopyFrom(new_input)

    linear_nodes = [n for n in graph.node if n.op_type in ["MatMul", "Gemm"]]
    if not linear_nodes:
        raise ValueError("No Linear nodes found in the graph.")

    for node in linear_nodes:
        input_name, weight_name, output_name = (
            node.input[0],
            node.input[1],
            node.output[0],
        )

        # Default: no transposition of weights
        trans_b = 0

        # Handle Gemm specific attributes
        if node.op_type == "Gemm":
            trans_a = next(
                (attr.i for attr in node.attribute if attr.name == "transA"), 0
            )
            trans_b = next(
                (attr.i for attr in node.attribute if attr.name == "transB"), 0
            )

            # Only support transA==0 (no transposition of input)
            if trans_a != 0:
                raise RuntimeError(f"transposition of {node.name} is not supported")

        # Get weight tensor
        weight_tensor = _get_weight_tensor(graph, weight_name)

        # Transform the weight tensor to support block-wise linear operation
        num_blocks = weight_tensor.shape[block_axis] // block_size
        weight_info = _transform_linear_weight(
            graph, weight_name, weight_tensor.shape, num_blocks, trans_b
        )
        transformed_weight_name = weight_info["final_output"]
        new_nodes.extend(weight_info["nodes"])

        # Block-wise batched MatMul
        matmul_output_name = _make_name(output_name, "batched_linear_output")
        matmul_node = helper.make_node(
            "MatMul",
            inputs=[input_name, transformed_weight_name],
            outputs=[matmul_output_name],
            name=node.name,
        )
        new_nodes.append(matmul_node)

        # Update output tensor's name and shape after replacing original linear node
        # with a batched linear node.
        for output_vi in graph.output:
            if output_vi.name == output_name:
                output_vi.name = matmul_output_name
                output_vi.type.tensor_type.shape.ClearField("dim")

                # Create new symbolic output shape and apply the new shape to output value info
                symbolic_dims = ["num_blocks", "batch", "c_out"]
                shape_proto = TensorShapeProto()
                for dim_name in symbolic_dims:
                    dim = shape_proto.dim.add()
                    dim.dim_param = dim_name
                output_vi.type.tensor_type.shape.CopyFrom(shape_proto)

        # Remove original conv node
        graph.node.remove(node)

    # Add all newly created nodes to the graph
    graph.node.extend(new_nodes)


def _make_name(base: str, suffix: str) -> str:
    """
    Generates a new unique name for the given base and suffix.

    :param base: The base name.
    :param suffix: The suffix string.
    :return: New name.
    """
    return f"{base}_{suffix}"


def _get_weight_tensor(graph: onnx.GraphProto, weight_name: str) -> np.ndarray:
    """

    Retrieves the weight tensor from the graph's initializers and converts it
    to a numpy array.

    NOTE: If quantize op has been inserted, the weight name
    may include a "_qdq" suffix. This function strips that suffix to
    match the original initializer

    :param graph: The ONNX model graph.
    :param weight_name: The name of the weight tensor.
    :return: The weight tensor as numpy array.
    """
    # Find the initializer matching the weight name (strip "_qdq" if present)
    initializer = next(
        (
            init
            for init in graph.initializer
            if init.name == weight_name.replace("_qdq", "")
        ),
        None,
    )

    if initializer is None:
        raise ValueError(f"Weight initializer not found for {weight_name}")

    return numpy_helper.to_array(initializer)


def _transform_conv_weight(
    graph: onnx.GraphProto, weight_name: str, weight_shape: Tuple, num_blocks: int
) -> Dict:
    """
    Transforms a weight tensor for grouped convolution.

    The transformation involves:
     - Reshaping the weight from (c_out, c_in, kh, kw) to (c_out, num_blocks, block_size, kh, kw)
     - Transposing to (num_blocks, c_out, block_size, kh, kw)
     - Reshaping to (num_blocks * c_out, block_size, kh, kw)

    :param graph: The ONNX model graph.
    :param weight_name: The name of original weight tensor.
    :param weight_shape: The shape of original weight tensor.
    :param num_blocks: Number of blocks to split the input channel into.
    :return: A dictionary containing the list of transformation nodes and
    the name of the final transformed weight tensor.
    """
    nodes = []

    if len(weight_shape) != 4:
        raise ValueError(f"Only Conv2D weights are supported. Got shape {weight_shape}")

    c_out, c_in, k_h, k_w = weight_shape
    block_size = c_in // num_blocks

    # Reshape weight from (c_out, c_in, kh, kw) to (c_out, num_blocks, block_size, kh, kw)
    reshape1_shape_array = np.array(
        [c_out, num_blocks, block_size, k_h, k_w], dtype=np.int64
    )
    reshape1_shape_name = _make_name(weight_name, "reshape1_shape")
    graph.initializer.append(
        numpy_helper.from_array(reshape1_shape_array, name=reshape1_shape_name)
    )

    reshape1_output_name = _make_name(weight_name, "reshape1")
    reshape1_node_name = _make_name(weight_name, "reshape1_node")
    reshape1_node = helper.make_node(
        "Reshape",
        inputs=[weight_name, reshape1_shape_name],
        outputs=[reshape1_output_name],
        name=reshape1_node_name,
    )
    nodes.append(reshape1_node)

    # Transpose weights from (c_out, num_blocks, block_size, kh, kw) to (num_blocks, c_out, block_size, kh, kw)
    perm = [1, 0, 2, 3, 4]
    transpose_output_name = _make_name(weight_name, "transpose")
    transpose_node_name = _make_name(weight_name, "transpose_node")
    transpose_node = helper.make_node(
        "Transpose",
        inputs=[reshape1_output_name],
        outputs=[transpose_output_name],
        perm=perm,
        name=transpose_node_name,
    )
    nodes.append(transpose_node)

    # Reshape weight from (num_blocks, c_out, block_size, kh, kw) to (num_blocks * c_out, block_size, kh, kw)
    reshape2_shape_array = np.array(
        [num_blocks * c_out, block_size, k_h, k_w], dtype=np.int64
    )
    reshape2_shape_name = _make_name(weight_name, "reshape2_shape")
    graph.initializer.append(
        numpy_helper.from_array(reshape2_shape_array, name=reshape2_shape_name)
    )

    reshape2_output_name = _make_name(weight_name, "reshape2")
    reshape2_node_name = _make_name(weight_name, "reshape2_node")
    reshape2_node = helper.make_node(
        "Reshape",
        inputs=[transpose_output_name, reshape2_shape_name],
        outputs=[reshape2_output_name],
        name=reshape2_node_name,
    )
    nodes.append(reshape2_node)

    return {"nodes": nodes, "final_output": reshape2_output_name}


def _transform_linear_weight(
    graph: onnx.GraphProto,
    weight_name: str,
    weight_shape: Tuple,
    num_blocks: int,
    trans_b: int,
) -> Dict:
    """
    Transforms a linear weight for block-wise grouped linear operation.

    The transformation involves:
     - Optional transpose if transB = 1 (from (c_out, c_in) to (c_in, c_out))
     - Reshape to (num_blocks, block_size, c_out)

    :param graph: The ONNX model graph.
    :param weight_name: The name of the weight tensor.
    :param weight_shape: The shape of the weight tensor.
    :param num_blocks: Number of blocks.
    :param trans_b: Whether the transpose should be applied. (1 = yes, 0 = no)
    :return: A dictionary containing the list of transformation nodes and
    the name of the final transformed weight tensor.
    """

    nodes = []

    if len(weight_shape) != 2:
        raise ValueError(
            f"Expected weight tensor for node {weight_name} to be 2D,"
            f" but found shape {weight_shape}"
        )

    c_in, c_out = weight_shape

    input_name = weight_name

    if trans_b == 1:
        # Transpose weights from (c_out, c_in) to (c_in, c_out)
        c_out, c_in = weight_shape
        perm = [1, 0]
        transpose_output_name = _make_name(weight_name, "transpose")
        transpose_node_name = _make_name(weight_name, "transpose_node")
        transpose_node = helper.make_node(
            "Transpose",
            inputs=[weight_name],
            outputs=[transpose_output_name],
            perm=perm,
            name=transpose_node_name,
        )
        input_name = transpose_output_name
        nodes.append(transpose_node)

    block_size = c_in // num_blocks

    # Reshape weights from (c_in, c_out) to (num_blocks, block_size, c_out)
    reshape_shape_array = np.array([num_blocks, block_size, c_out], dtype=np.int64)
    reshape_shape_name = _make_name(weight_name, "reshape_shape")
    graph.initializer.append(
        numpy_helper.from_array(reshape_shape_array, name=reshape_shape_name)
    )

    reshape_output_name = _make_name(weight_name, "reshape_output")
    reshape_node_name = _make_name(weight_name, "reshape_node")
    reshape_node = helper.make_node(
        "Reshape",
        inputs=[input_name, reshape_shape_name],
        outputs=[reshape_output_name],
        name=reshape_node_name,
    )
    nodes.append(reshape_node)

    return {"nodes": nodes, "final_output": reshape_output_name}


def prepare_linear_inputs(
    data: Dict[str, List[np.ndarray]], block_size: int
) -> Dict[str, List[np.ndarray]]:
    """
    Transforms each tensor in the dictionary by reshaping the last dimension into blocks
    of size `block_size`, and transposing the result to shape (num_blocks, batch, block_size).

    :param data: A dictionary with NumPy arrays.
    :param block_size: Block size.
    :return: Prepared inputs for linear op.
    """

    def transform(tensor: np.ndarray) -> np.ndarray:
        last_dim = tensor.shape[-1]

        if last_dim % block_size != 0:
            raise ValueError(
                f"block_size {block_size} must divide the last dimension of the tensor {tensor.shape}."
            )

        num_blocks = last_dim // block_size
        flattened = tensor.reshape(-1, last_dim)
        reshaped = flattened.reshape(flattened.shape[0], num_blocks, block_size)
        return reshaped.transpose(1, 0, 2).copy()

    return {key: [transform(arr) for arr in arr_list] for key, arr_list in data.items()}

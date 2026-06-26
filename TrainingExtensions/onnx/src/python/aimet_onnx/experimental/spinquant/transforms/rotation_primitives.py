# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Generic Hadamard-rotation primitives shared across SpinQuant rotation passes.

These helpers operate on an arbitrary normalized rotation matrix ``R`` and an
op's storage metadata. They contain no knowledge of R1 / R2 / R3 specifics;
each rotation pass selects the matrix and decides which ops to call them on.

Storage / role conventions:

    Role          | Storage         | Formula
    --------------|-----------------|-------------------
    reading layer | [out, in]       | W @ R   (axis 1)
    reading layer | [in,  out]      | R^T @ W (axis 0)
    reading Conv  | [out, in, *k]   | W @ R   (axis 1)
    writing layer | [out, in]       | R^T @ W (axis 0)
    writing layer | [in,  out]      | W @ R   (axis -1)
    writing Conv  | [out, in, *k]   | R^T @ W (axis 0)
    Gather        | [vocab, hidden] | W @ R   (axis -1)
"""

from typing import Tuple

import numpy as np
import onnx
from onnx import numpy_helper

from aimet_onnx.common.hadamard import get_hadamard_matrix
from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.meta.operations import Op
from aimet_onnx.utils import ModelProto, ParamUtils

from aimet_onnx.experimental.spinquant.model_analysis.weight_utils import (
    get_bias_product,
    get_weight_product,
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SpinQuant)


def hadamard_rotation_matrix(hidden_size: int) -> np.ndarray:
    """Return ``H / sqrt(hidden_size)`` as a float64 normalized Hadamard rotation."""
    return (get_hadamard_matrix(hidden_size) / np.sqrt(hidden_size)).astype(np.float64)


def block_diag_repeat(R: np.ndarray, k: int) -> np.ndarray:
    """Return a ``[k*d, k*d]`` block-diagonal matrix with ``R`` repeated ``k`` times.

    Lets the existing whole-axis rotation helpers express per-head rotations:
    rotating an op's output channels per-head with ``R`` is identical to
    rotating the full axis with ``block_diag(R, R, ..., R)``.

    :param R: A ``[d, d]`` square matrix.
    :param k: Number of diagonal copies (e.g. number of attention heads).
    :return: A ``[k*d, k*d]`` block-diagonal matrix.
    """
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"block_diag_repeat: R must be square, got shape {R.shape}.")
    d = R.shape[0]
    out = np.zeros((k * d, k * d), dtype=R.dtype)
    for i in range(k):
        out[i * d : (i + 1) * d, i * d : (i + 1) * d] = R
    return out


def rotate_gather_weight(model: ModelProto, op: Op, R: np.ndarray):
    """Apply a right-side rotation ``W @ R`` to the weight of a Gather op (embed_tokens).

    :param model: ONNX ModelProto.
    :param op: The Gather op.
    :param R: Normalized rotation matrix [hidden, hidden].
    """
    for inp in op.inputs:
        if inp.is_parm or inp.is_const:
            tensor = ParamUtils.get_param_by_name(model, inp.name)
            if tensor is None:
                raise RuntimeError(
                    f"embed_tokens op '{op.name}': weight '{inp.name}' not found in "
                    f"initializers."
                )
            W = numpy_helper.to_array(tensor)
            W_new = right_multiply(W, R).astype(W.dtype)
            tensor.CopyFrom(numpy_helper.from_array(W_new, name=tensor.name))
            _logger.debug("Rotated embed_tokens '%s' shape %s.", inp.name, W.shape)
            return
    raise RuntimeError(f"embed_tokens op '{op.name}': no static weight input found.")


def rotate_linear_weight(model: ModelProto, op: Op, R: np.ndarray, is_writing: bool):
    """Apply a rotation to the weight (and bias if writing) of a MatMul/Gemm/Conv op.

    :param model: ONNX ModelProto.
    :param op: The MatMul, Gemm, or Conv op.
    :param R: Normalized rotation matrix [hidden, hidden].
    :param is_writing: True for layers that write to the residual stream
        (e.g. o_proj, down_proj); False for layers that read from it
        (e.g. qkv, gate_up, lm_head).
    """
    weight_inp, is_transposed = get_weight_product(op)
    if weight_inp is None:
        raise RuntimeError(f"Op '{op.name}': no static weight found.")

    tensor = ParamUtils.get_param_by_name(model, weight_inp.name)
    if tensor is None:
        raise RuntimeError(
            f"Op '{op.name}': weight '{weight_inp.name}' not found in initializers."
        )

    W = numpy_helper.to_array(tensor)
    W_new = apply_transform(W, R, op.type, is_transposed, is_writing).astype(W.dtype)
    tensor.CopyFrom(numpy_helper.from_array(W_new, name=tensor.name))
    _logger.debug(
        "Rotated op '%s' (%s, transposed=%s, is_writing=%s) shape %s.",
        op.name,
        op.type,
        is_transposed,
        is_writing,
        W.shape,
    )

    if is_writing:
        bias_inp = get_bias_product(op)
        if bias_inp is not None:
            bias_tensor = ParamUtils.get_param_by_name(model, bias_inp.name)
            if bias_tensor is not None:
                b = numpy_helper.to_array(bias_tensor)
                b_rot = right_multiply(b, R, axis=-1).astype(b.dtype)
                bias_tensor.CopyFrom(
                    numpy_helper.from_array(b_rot, name=bias_tensor.name)
                )
                _logger.debug(
                    "Rotated bias for writing op '%s' shape %s.", op.name, b.shape
                )


def apply_transform(
    W: np.ndarray, R: np.ndarray, op_type: str, is_transposed: bool, is_writing: bool
) -> np.ndarray:
    """Dispatch to ``right_multiply`` or ``left_multiply`` based on storage and role.

    :param W: Weight tensor of any shape.
    :param R: Normalized rotation matrix [hidden, hidden].
    :param op_type: ONNX op type.
    :param is_transposed: True if W is stored as [out, in]; False if [in, out].
    :param is_writing: True for writing layers; False for reading layers.
    :return: Rotated weight.
    """
    if op_type == "Conv":
        # [out, in, *k]
        return (
            left_multiply(W, R, axis=0) if is_writing else right_multiply(W, R, axis=1)
        )
    if is_transposed:
        # [out, in]
        return (
            left_multiply(W, R, axis=0) if is_writing else right_multiply(W, R, axis=1)
        )
    # [in, out]
    return right_multiply(W, R, axis=-1) if is_writing else left_multiply(W, R, axis=0)


def right_multiply(W: np.ndarray, R: np.ndarray, axis: int = -1) -> np.ndarray:
    """``W_new = W @ R`` along ``axis``.

    Corresponds to ``left_hand_transform(R^T)`` baked into reading layers
    ([out, in] storage), and ``right_hand_transform(R)`` baked into writing
    layers ([in, out] storage).

    :param W: Weight tensor of any shape.
    :param R: Normalized rotation matrix [hidden, hidden].
    :param axis: Axis of W corresponding to hidden_size.
    :return: Rotated weight.
    """
    W_moved = np.moveaxis(W.astype(np.float64), axis, -1)  # [*rest, hidden]
    moved_shape = W_moved.shape
    W_new = W_moved.reshape(-1, W_moved.shape[-1]) @ R  # [N, hidden]
    return np.moveaxis(W_new.reshape(moved_shape), -1, axis)


def left_multiply(W: np.ndarray, R: np.ndarray, axis: int = 0) -> np.ndarray:
    """``W_new = R^T @ W`` along ``axis``.

    Corresponds to ``right_hand_transform(R)`` baked into writing layers
    ([out, in] storage), and ``left_hand_transform(R^T)`` baked into reading
    layers ([in, out] storage).

    :param W: Weight tensor of any shape.
    :param R: Normalized rotation matrix [hidden, hidden].
    :param axis: Axis of W corresponding to hidden_size.
    :return: Rotated weight.
    """
    W_moved = np.moveaxis(W.astype(np.float64), axis, 0)  # [hidden, *rest]
    moved_shape = W_moved.shape
    W_new = R.T @ W_moved.reshape(W_moved.shape[0], -1)  # [hidden, N]
    return np.moveaxis(W_new.reshape(moved_shape), 0, axis)


def insert_online_hadamard_node(
    model: ModelProto,
    target_tensor_name: str,
    consumer_nodes: list[onnx.NodeProto],
    head_dim: int,
    name_prefix: str,
) -> Tuple[str, onnx.NodeProto]:
    """Insert ``MatMul(target_tensor, H)`` between a producer and one consumer input.

    Used by R3 to add an online Hadamard rotation immediately upstream of a
    chosen consumer (the QK^T MatMul, or the past-key Concat). The new MatMul
    reads ``target_tensor_name`` and writes a rotated tensor; only the
    requested consumer input is rewired, so other consumers of the original
    tensor (if any) keep seeing the unrotated value.

    :param model: ONNX ModelProto to mutate.
    :param target_tensor_name: Name of the tensor to rotate.
    :param consumer_nodes: The NodeProtos whose input index we will rewire.
    :param head_dim: Size of the (square) Hadamard matrix; rotates the last
        axis of ``target_tensor_name``.
    :param name_prefix: Prefix used to name the inserted initializer / node /
        output tensor (e.g. ``"block0_q"``).
    :return: ``(output_name, new_node)`` — the name of the new (rotated) output
        tensor and the inserted ``MatMul`` NodeProto (so callers can wire a
        quantizer relative to it).
    """
    for consumer_node in consumer_nodes:
        if target_tensor_name not in consumer_node.input:
            raise ValueError(
                f"insert_online_hadamard_node: consumer_node '{consumer_node.name}' "
                f"target tensor `{target_tensor_name}` does not appear in consumer node inputs: "
                f"{consumer_node.input}"
            )

    elem_type = _infer_tensor_elem_type(model, target_tensor_name)
    np_dtype = onnx.helper.tensor_dtype_to_np_dtype(elem_type)

    H = hadamard_rotation_matrix(head_dim).astype(np_dtype)
    initializer_name = f"{name_prefix}_R3_hadamard"
    output_name = f"{name_prefix}_R3_out"
    node_name = f"{name_prefix}_R3"

    initializer = numpy_helper.from_array(H, name=initializer_name)
    model.graph.initializer.append(initializer)

    new_node = onnx.helper.make_node(
        op_type="MatMul",
        inputs=[target_tensor_name, initializer_name],
        outputs=[output_name],
        name=node_name,
    )
    # protobuf ``insert`` copies the message, so use the in-graph reference for
    # the return value — callers wire quantizers onto this node's edges.
    new_node = _insert_node_after_producer(model, target_tensor_name, new_node)

    for consumer_node in consumer_nodes:
        consumer_input_idx = list(consumer_node.input).index(target_tensor_name)
        consumer_node.input[consumer_input_idx] = output_name

    _logger.debug(
        "Inserted online Hadamard MatMul '%s' on tensor '%s' (head_dim=%d, dtype=%s); "
        "rewired '%s'.input[%d].",
        node_name,
        target_tensor_name,
        head_dim,
        np_dtype,
        consumer_node.name,
        consumer_input_idx,
    )
    return output_name, new_node


def _infer_tensor_elem_type(model: ModelProto, tensor_name: str) -> int:
    """Return the ONNX TensorProto.DataType for ``tensor_name``.

    Looks first at graph value_info / inputs / outputs, then at initializers.
    Defaults to ``FLOAT`` if nothing matches — ORT will raise a clear dtype
    error at session-build time if that's wrong, which is preferable to
    silently casting weights.
    """
    for vi in (
        list(model.graph.value_info)
        + list(model.graph.input)
        + list(model.graph.output)
    ):
        if vi.name == tensor_name:
            elem_type = vi.type.tensor_type.elem_type
            if elem_type != onnx.TensorProto.UNDEFINED:
                return elem_type
    for init in model.graph.initializer:
        if init.name == tensor_name:
            return init.data_type
    return onnx.TensorProto.FLOAT


def _insert_node_after_producer(
    model: ModelProto, producer_output_name: str, new_node: onnx.NodeProto
) -> onnx.NodeProto:
    """Insert ``new_node`` immediately after the node producing ``producer_output_name``.

    ONNX requires nodes appear in a topologically valid order. ORT is
    tolerant of out-of-order nodes for many graphs, but other tools (and
    serializers that re-validate) are not. Always insert just after the
    producer. If no producer is found (the tensor is a graph input or
    initializer), append at the front of the node list.

    :return: The in-graph ``NodeProto`` (a protobuf ``insert`` copies its
        argument, so callers must use this reference to further mutate the node).
    """
    nodes = model.graph.node
    insert_at = 0
    for idx, node in enumerate(nodes):
        if producer_output_name in node.output:
            insert_at = idx + 1
            break
    nodes.insert(insert_at, new_node)
    return nodes[insert_at]

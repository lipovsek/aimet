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

import numpy as np
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

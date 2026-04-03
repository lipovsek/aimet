# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""R1 Hadamard rotation for SpinQuant: absorbs orthogonal rotation into ONNX weight initializers.

R1 = H / sqrt(hidden_size) rotates the residual stream (h_rot = h @ R1). Each weight
absorbs the rotation so the model output is unchanged.

Terminology:
  - reading layer: takes input from the residual stream (qkv, gate_up, lm_head)
  - writing layer: adds output back to the residual stream (o_proj, down_proj, embed_tokens)
  - right_hand_transform(R): PyTorch activation-space op that inserts ``y @ R`` after
    writing layer
  - left_hand_transform(R^T): PyTorch activation-space op that inserts ``x @ R^T`` before
    reading layer

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
from aimet_onnx.experimental.spinquant.block_identifier import DecoderModelRoleMap
from aimet_onnx.experimental.spinquant.fuse_norm import _get_weight_product

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SpinQuant)


def apply_r1_rotation(
    model: ModelProto,
    role_map: DecoderModelRoleMap,
    hidden_size: int,
):
    """Apply R1 Hadamard rotation to linear weights in role_map in-place.

    Modifies ONNX initializer tensors directly. After this call the model is
    mathematically equivalent to the original with tiny difference (due to precision).

    Weight existence and shape validation must be done by the caller before
    invoking this function (see _infer_hidden_size and _validate_all_weights).

    :param model: ONNX ModelProto.
    :param role_map: Role map produced by get_decoder_role_map.
    :param hidden_size: Dimension of the Hadamard rotation matrix.
    """
    R1 = (get_hadamard_matrix(hidden_size) / np.sqrt(hidden_size)).astype(np.float64)
    _logger.info("Applying R1 Hadamard rotation with hidden_size=%d.", hidden_size)

    # embed_tokens: right_hand_transform(R1)
    for op in role_map.embed_tokens:
        _rotate_gather_weight(model, op, R1)

    # lm_head: left_hand_transform(R1^T)
    for op in role_map.lm_head:
        _rotate_linear_weight(model, op, R1, is_writing=False)

    for block_idx, block in enumerate(role_map.blocks):
        _logger.debug("Applying R1 to block %d.", block_idx)

        for op in block.qkv_linears:  # left_hand_transform(R1^T)
            _rotate_linear_weight(model, op, R1, is_writing=False)

        for op in block.o_proj:  # right_hand_transform(R1)
            _rotate_linear_weight(model, op, R1, is_writing=True)

        for op in block.gate_up_linears:  # left_hand_transform(R1^T)
            _rotate_linear_weight(model, op, R1, is_writing=False)

        for op in block.down_proj:  # right_hand_transform(R1)
            _rotate_linear_weight(model, op, R1, is_writing=True)


def _infer_hidden_size(model: ModelProto, role_map: DecoderModelRoleMap) -> int:
    """Infer the model hidden size from the first embed_tokens weight in ``role_map``.

    :param model: ONNX ModelProto.
    :param role_map: Role map produced by get_decoder_role_map
    :return: The hidden dimension size.
    """
    for op in role_map.embed_tokens:
        for inp in op.inputs:
            if inp.is_parm or inp.is_const:
                tensor = ParamUtils.get_param_by_name(model, inp.name)
                if tensor is not None:
                    return numpy_helper.to_array(tensor).shape[-1]
    raise ValueError(
        "Cannot infer hidden_size: no embed_tokens initializer weight found in role_map."
    )


def _validate_all_weights(
    model: ModelProto, role_map: DecoderModelRoleMap, hidden_size: int
):
    """Verify that every weight in role_map exists and has the correct shape.

    :param model: ONNX ModelProto to validate against.
    :param role_map: Role map whose ops are checked.
    :param hidden_size: The hidden dimension size.
    """
    for op in role_map.embed_tokens:
        found = False
        for inp in op.inputs:
            if inp.is_parm or inp.is_const:
                found = True
                tensor = ParamUtils.get_param_by_name(model, inp.name)
                if tensor is None:
                    raise RuntimeError(
                        f"embed_tokens op '{op.name}': weight '{inp.name}' not found in "
                        f"initializers. Cannot apply R1 rotation."
                    )
                shape = numpy_helper.to_array(tensor).shape
                if shape[-1] != hidden_size:
                    raise RuntimeError(
                        f"embed_tokens op '{op.name}': weight '{inp.name}' has shape {shape}, "
                        f"but shape[-1]={shape[-1]} != hidden_size={hidden_size}."
                    )
        if not found:
            raise RuntimeError(
                f"embed_tokens op '{op.name}': no static weight input found. "
                f"Can't apply R1 rotation."
            )

    linear_ops_with_role = [(op, False) for op in role_map.lm_head]
    for block in role_map.blocks:
        linear_ops_with_role += [(op, False) for op in block.qkv_linears]
        linear_ops_with_role += [(op, True) for op in block.o_proj]
        linear_ops_with_role += [(op, False) for op in block.gate_up_linears]
        linear_ops_with_role += [(op, True) for op in block.down_proj]

    for op, is_writing in linear_ops_with_role:
        weight_inp, is_transposed = _get_weight_product(op)
        if weight_inp is None:
            raise RuntimeError(
                f"Op '{op.name}': no static weight found. Can't apply R1 rotation."
            )
        tensor = ParamUtils.get_param_by_name(model, weight_inp.name)
        if tensor is None:
            raise RuntimeError(
                f"Op '{op.name}': weight '{weight_inp.name}' not found in initilizers. "
                f"Can't apply R1 rotation."
            )
        shape = numpy_helper.to_array(tensor).shape

        if op.type == "Conv" or is_transposed:  # [out, in]
            rotated_axis = 0 if is_writing else 1
        else:  # [in, out]
            rotated_axis = -1 if is_writing else 0
        if shape[rotated_axis] != hidden_size:
            raise RuntimeError(
                f"Op '{op.name}': weight shape {shape}, axis {rotated_axis} "
                f"= {shape[rotated_axis]}, expected hidden_size={hidden_size}."
            )


def _rotate_gather_weight(model: ModelProto, op: Op, R: np.ndarray):
    """Apply the R1 rotation to the weight of embed_tokens op (Gather).

    :param model: ONNX ModelProto.
    :param op: The Gather op.
    :param R: Normalized Hadamard rotation matrix [hidden, hidden].
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
            W_new = _right_multiply(W, R).astype(W.dtype)
            tensor.CopyFrom(numpy_helper.from_array(W_new, name=tensor.name))
            _logger.debug("Rotated embed_tokens '%s' shape %s.", inp.name, W.shape)
            return
    raise RuntimeError(f"embed_tokens op '{op.name}': no static weight input found.")


def _rotate_linear_weight(model: ModelProto, op: Op, R: np.ndarray, is_writing: bool):
    """Apply the R1 rotation to the weight of MatMul/Gemm/Conv op.

    :param model: ONNX ModelProto.
    :param op: The MatMul, Gemm, or Conv op.
    :param R: Normalized Hadamard rotation matrix [hidden, hidden].
    :param is_writing: True for layers that write to the residual stream (o_proj,
        down_proj); False for layers that read from the residual stream
        (qkv, gate_up, lm_head).
    """
    weight_inp, is_transposed = _get_weight_product(op)
    if weight_inp is None:
        raise RuntimeError(f"Op '{op.name}': no static weight found.")

    tensor = ParamUtils.get_param_by_name(model, weight_inp.name)
    if tensor is None:
        raise RuntimeError(
            f"Op '{op.name}': weight '{weight_inp.name}' not found in initializers."
        )

    W = numpy_helper.to_array(tensor)
    W_new = _apply_transform(W, R, op.type, is_transposed, is_writing).astype(W.dtype)
    tensor.CopyFrom(numpy_helper.from_array(W_new, name=tensor.name))
    _logger.debug(
        "Rotated op '%s' (%s, transposed=%s, is_writing=%s) shape %s.",
        op.name,
        op.type,
        is_transposed,
        is_writing,
        W.shape,
    )


def _apply_transform(
    W: np.ndarray, R: np.ndarray, op_type: str, is_transposed: bool, is_writing: bool
) -> np.ndarray:
    """Dispatch to _right_multiply or _left_multiply based on storage layout and role.

    :param W: Weight tensor of any shape.
    :param R: Normalized Hadamard rotation matrix [hidden, hidden].
    :param op_type: ONNX op type.
    :param is_transposed: True if W is stored as [out, in]; False if stored as [in, out].
    :param is_writing: True for writing layers; False for reading layers.
    :return: Rotated weight.
    """
    if op_type == "Conv":
        # [out, in, *k]
        return (
            _left_multiply(W, R, axis=0)
            if is_writing
            else _right_multiply(W, R, axis=1)
        )
    if is_transposed:
        # [out, in]
        return (
            _left_multiply(W, R, axis=0)
            if is_writing
            else _right_multiply(W, R, axis=1)
        )
    # [in, out]
    return (
        _right_multiply(W, R, axis=-1) if is_writing else _left_multiply(W, R, axis=0)
    )


def _right_multiply(W: np.ndarray, R: np.ndarray, axis: int = -1) -> np.ndarray:
    """W_new = W @ R along axis.

    Corresponds to left_hand_transform(R^T) baked into reading layers ([out, in] storage),
    and right_hand_transform(R) baked into writing layers ([in, out] storage).

    :param W: Weight tensor of any shape.
    :param R: Normalized Hadamard rotation matrix [hidden, hidden].
    :param axis: Axis of W corresponding to hidden_size.
    :return: Rotated weight.
    """
    W_moved = np.moveaxis(W.astype(np.float64), axis, -1)  # [*rest, hidden]
    moved_shape = W_moved.shape
    W_new = W_moved.reshape(-1, W_moved.shape[-1]) @ R  # [N, hidden]
    return np.moveaxis(W_new.reshape(moved_shape), -1, axis)


def _left_multiply(W: np.ndarray, R: np.ndarray, axis: int = 0) -> np.ndarray:
    """W_new = R^T @ W along axis.

    Corresponds to right_hand_transform(R) baked into writing layers ([out, in] storage),
    and left_hand_transform(R^T) baked into reading layers ([in, out] storage).

    :param W: Weight tensor of any shape.
    :param R: Normalized Hadamard rotation matrix [hidden, hidden].
    :param axis: Axis of W corresponding to hidden_size.
    :return: Rotated weight.
    """
    W_moved = np.moveaxis(W.astype(np.float64), axis, 0)  # [hidden, *rest]
    moved_shape = W_moved.shape
    W_new = R.T @ W_moved.reshape(W_moved.shape[0], -1)  # [hidden, N]
    return np.moveaxis(W_new.reshape(moved_shape), 0, axis)

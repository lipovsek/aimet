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

from typing import List
import numpy as np
from onnx import numpy_helper

from aimet_onnx.common.hadamard import get_hadamard_matrix
from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.meta.operations import Op
from aimet_onnx.utils import ModelProto, ParamUtils
from aimet_onnx.experimental.spinquant.block_identifier import DecoderModelRoleMap
from aimet_onnx.experimental.spinquant.fuse_norm import (
    _get_weight_product,
    _find_post_writing_norms,
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SpinQuant)


def apply_r1_rotation(
    model: ModelProto,
    role_map: DecoderModelRoleMap,
    backbone_hidden_size: int,
):
    """Apply R1 Hadamard rotation to linear weights in role_map in-place.

    Modifies ONNX initializer tensors directly. After this call the model is
    mathematically equivalent to the original with tiny difference (due to precision).

    Weight existence and shape validation must be done by the caller before
    invoking this function (see _infer_hidden_size and _validate_all_weights).

    :param model: ONNX ModelProto.
    :param role_map: Role map produced by get_decoder_role_map.
    :param backbone_hidden_size: Language backbone hidden size.
    """
    R1 = (
        get_hadamard_matrix(backbone_hidden_size) / np.sqrt(backbone_hidden_size)
    ).astype(np.float64)
    _logger.info(
        "Backbone: Applying R1 Hadamard rotation with backbone_hidden_size=%d.",
        backbone_hidden_size,
    )

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


def apply_r1_rotation_merger(
    model: ModelProto,
    merger_linear2: List[Op],
    backbone_hidden_size: int,
):
    """Apply R_L Hadamard rotation to PatchMerger linear_fc2 weights in-place.

    Rotates merger_linear2 (linear_fc2) with R_L — the language backbone rotation
    matrix. This is non-negotiable: whenever the backbone is SpinQuant-rotated,
    merger_linear2 must also be rotated so its outputs land in the rotated backbone
    residual stream space.

    :param model: ONNX ModelProto (modified in-place).
    :param merger_linear2: List of merger linear_fc2 ops from find_merger_linear2.
    :param backbone_hidden_size: Language backbone hidden size.
    """
    R_L = (
        get_hadamard_matrix(backbone_hidden_size) / np.sqrt(backbone_hidden_size)
    ).astype(np.float64)
    _logger.info(
        "Visual: Applying R1 Hadamard rotation to merger_linear2 with backbone_hidden_size=%d.",
        backbone_hidden_size,
    )

    for op in merger_linear2:
        _rotate_linear_weight(model, op, R_L, is_writing=True)


def _infer_hidden_size(model: ModelProto, role_map: DecoderModelRoleMap) -> int:
    """Infer the model hidden size from embed_tokens or lm_head weights in ``role_map``.

    Tries embed_tokens first (Gather weight [vocab, hidden], last dim = hidden).
    Falls back to lm_head for VLM backbones exported with use_inputs_embeds=True
    that have no Gather op: reading-layer weight has hidden on the input axis.

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

    # Fallback: lm_head reading layer.
    # Gemm transB=1 stores W [vocab, hidden] → hidden = shape[-1].
    # MatMul stores W [hidden, vocab]        → hidden = shape[0].
    # Conv 1x1 stores W [vocab, hidden, 1, 1] → hidden = shape[1].
    for op in role_map.lm_head:
        weight_inp, is_transposed = _get_weight_product(op)
        if weight_inp is not None:
            tensor = ParamUtils.get_param_by_name(model, weight_inp.name)
            if tensor is not None:
                W = numpy_helper.to_array(tensor)
                if op.type == "Conv":
                    return W.shape[1]  # [out_ch, in_ch, *k]: in_ch = hidden
                return W.shape[-1] if is_transposed else W.shape[0]

    raise ValueError(
        "Cannot infer hidden_size: no embed_tokens or lm_head initializer weight found in role_map."
    )


def _get_bias_product(op: Op):
    """Return the bias Product for Gemm/Conv (third static input) or MatMul followed by Add.

    Writing layers (o_proj, down_proj, patch_embed) whose output is added to the residual
    stream must have their bias rotated alongside the weight. This function locates that
    bias initializer so the caller can apply ``b_rot = b @ R``.

    Handles three patterns:
    - Gemm (transB=1): inputs are [X, W, B]; B is the second static input.
    - Conv:            inputs are [X, W, B]; B is the second static input.
    - MatMul + Add:    a downstream Add whose second input is a static initializer.

    :param op: A MatMul, Gemm, or Conv Op.
    :return: The bias Product, or None if no static bias is found.
    """
    if op.type in ("Gemm", "Conv"):
        static_inputs = [inp for inp in op.inputs if inp.is_parm or inp.is_const]
        if len(static_inputs) >= 2:
            return static_inputs[1]

    if op.type == "MatMul":
        for out_op in op.output_ops:
            if out_op.type == "Add":
                for inp in out_op.inputs:
                    if inp.is_parm or inp.is_const:
                        return inp

    return None


def _validate_backbone_weights(
    model: ModelProto, role_map: DecoderModelRoleMap, hidden_size: int
):
    """Verify architecture compatibility and that every weight in role_map exists and has the correct shape.

    :param model: ONNX ModelProto to validate against.
    :param role_map: Role map whose ops are checked.
    :param hidden_size: The hidden dimension size.
    """
    # Architecture compatibility validation
    post_writing_norms = _find_post_writing_norms(model, role_map)
    if post_writing_norms:
        raise ValueError(
            f"R1 rotation absorption requires writing layers (o_proj, down_proj) to feed "
            f"directly into the residual add. Detected {len(post_writing_norms)} affine "
            f"RMSNorm(s) between writing layers and the residual add "
            f"- R1 absorption is not feasible for this architecture."
        )

    # Weight shape validation
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


def _validate_merger_linear2(
    model: ModelProto,
    merger_linear2: List[Op],
    backbone_hidden_size: int,
):
    """Verify that every merger_linear2 weight exists and writes into backbone_hidden_size.

    :param model: ONNX ModelProto to validate against.
    :param merger_linear2: List of merger_linear2 ops from find_merger_linear2.
    :param backbone_hidden_size: Language backbone hidden dimension d_L.
    """
    for op in merger_linear2:
        weight_inp, is_transposed = _get_weight_product(op)
        if weight_inp is None:
            raise RuntimeError(
                f"merger_linear2 op '{op.name}': no static weight found. Cannot apply R_L rotation."
            )
        tensor = ParamUtils.get_param_by_name(model, weight_inp.name)
        if tensor is None:
            raise RuntimeError(
                f"merger_linear2 op '{op.name}': weight '{weight_inp.name}' not found in initializers."
            )
        shape = numpy_helper.to_array(tensor).shape
        # writing layer: output axis = backbone_hidden_size
        # Conv/transposed [out, in]: axis 0; MatMul [in, out]: axis -1
        rotated_axis = 0 if (op.type == "Conv" or is_transposed) else -1
        if shape[rotated_axis] != backbone_hidden_size:
            raise RuntimeError(
                f"merger_linear2 op '{op.name}': weight shape {shape}, axis {rotated_axis} "
                f"= {shape[rotated_axis]}, expected backbone_hidden_size={backbone_hidden_size}."
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

    if is_writing:
        bias_inp = _get_bias_product(op)
        if bias_inp is not None:
            bias_tensor = ParamUtils.get_param_by_name(model, bias_inp.name)
            if bias_tensor is not None:
                b = numpy_helper.to_array(bias_tensor)
                b_rot = _right_multiply(b, R, axis=-1).astype(b.dtype)
                bias_tensor.CopyFrom(
                    numpy_helper.from_array(b_rot, name=bias_tensor.name)
                )
                _logger.debug(
                    "Rotated bias for writing op '%s' shape %s.", op.name, b.shape
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

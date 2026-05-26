# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Static-weight / bias / hidden-size lookup primitives shared across rotation passes."""

from typing import Optional, Tuple
from onnx import numpy_helper

from aimet_onnx.meta.connectedgraph import Product
from aimet_onnx.meta.operations import Op
from aimet_onnx.utils import ModelProto, ParamUtils


def get_weight_product(op: Op) -> Tuple[Optional[Product], bool]:
    """Return ``(weight_product, is_transposed)`` for a MatMul/Gemm/Conv op.

    Handles two patterns:

    * Direct:   W (initializer) -> MatMul/Gemm/Conv
    * Indirect: W (initializer) -> Transpose -> MatMul

    For the indirect case ``is_transposed=True`` signals the stored weight has
    shape ``[out, in]``, so any per-input-axis transform must be applied on
    axis 1 instead of axis 0.

    :param op: A MatMul, Gemm, or Conv Op whose static weight product is to be located.
    :return: Tuple of (weight_product, is_transposed). weight_product is the Product
             holding the static weight initializer, or None if no static weight is found.
             is_transposed is True when the stored weight tensor has shape [out, in] —
             either because the op is Gemm with transB=1 (detected via transposed_params),
             or because the weight passes through a Transpose node before reaching a MatMul.
    """
    for inp in op.inputs:
        if inp.is_parm or inp.is_const:
            return inp, getattr(op, "transposed_params", False)

    # W -> Transpose -> MatMul pattern
    if op.type in ("MatMul", "Gemm"):
        for inp in op.inputs:
            if inp.producer and inp.producer.type == "Transpose":
                for t_inp in inp.producer.inputs:
                    if t_inp.is_parm or t_inp.is_const:
                        return t_inp, True
    return None, False


def get_bias_product(op: Op) -> Optional[Product]:
    """Return the bias Product for Gemm/Conv (third static input) or MatMul followed by Add.

    Writing layers (o_proj, down_proj, patch_embed) whose output is added to the
    residual stream must have their bias rotated alongside the weight. This
    function locates that bias initializer so the caller can apply the per-axis
    transform to the bias as well.

    Handles three patterns:

    * Gemm (transB=1): inputs are [X, W, B]; B is the second static input.
    * Conv:            inputs are [X, W, B]; B is the second static input.
    * MatMul + Add:    a downstream Add whose second input is a static initializer.

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


def infer_hidden_size(model: ModelProto, role_map) -> int:
    """Infer the model hidden size from embed_tokens or lm_head weights in ``role_map``.

    Tries ``embed_tokens`` first (Gather weight ``[vocab, hidden]``, last dim = hidden).
    Falls back to ``lm_head`` for VLM backbones exported with ``use_inputs_embeds=True``
    that have no Gather op: reading-layer weight has hidden on the input axis.

    :param model: ONNX ModelProto.
    :param role_map: DecoderModelRoleMap produced by ``get_decoder_role_map``.
    :return: The hidden dimension size.
    """
    for op in role_map.embed_tokens:
        for inp in op.inputs:
            if inp.is_parm or inp.is_const:
                tensor = ParamUtils.get_param_by_name(model, inp.name)
                if tensor is not None:
                    return numpy_helper.to_array(tensor).shape[-1]

    # Fallback: lm_head reading layer.
    # Gemm transB=1 stores W [vocab, hidden] -> hidden = shape[-1].
    # MatMul stores W [hidden, vocab]        -> hidden = shape[0].
    # Conv 1x1 stores W [vocab, hidden, 1, 1] -> hidden = shape[1].
    for op in role_map.lm_head:
        weight_inp, is_transposed = get_weight_product(op)
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

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Static-weight / bias / hidden-size lookup primitives shared across rotation passes."""

from typing import Optional, Tuple
from onnx import numpy_helper

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.meta.connectedgraph import Product
from aimet_onnx.meta.operations import Op
from aimet_onnx.utils import ModelProto, ParamUtils

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SpinQuant)


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
        # Only the data input (a [vocab, hidden] table) yields hidden_size;
        # other static inputs (e.g. axis attributes, indices) are not embedding tables.
        for inp in op.inputs:
            if not (inp.is_parm or inp.is_const):
                continue
            tensor = ParamUtils.get_param_by_name(model, inp.name)
            if tensor is None:
                continue
            shape = numpy_helper.to_array(tensor).shape
            if len(shape) >= 2:
                return shape[-1]

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


def infer_head_dim(model: ModelProto) -> int:
    """Infer per-head dimension from a ``past_value`` graph input's last axis.

    HF/optimum LLM exports include ``past_value_*`` (or ``past_key_values.*.value``)
    inputs whose final dimension is ``head_dim`` regardless of the surrounding
    layout (``[B, num_kv_heads, past_seq, head_dim]`` or
    ``[B, past_seq, num_kv_heads, head_dim]``). This avoids having to derive
    ``head_dim`` from ``hidden_size / num_heads``, which is wrong for models
    that decouple the two (e.g. Gemma3 fixes ``head_dim=256`` independent of
    hidden size).

    :param model: ONNX ModelProto whose graph inputs are scanned.
    :return: ``head_dim`` read from the last dim of the first matching input.
    :raises ValueError: If no ``past_value`` input exists, or if its last dim
        is not a static positive integer.
    """
    for inp in model.graph.input:
        if "past_value" not in inp.name:
            continue
        dims = inp.type.tensor_type.shape.dim
        if len(dims) == 0:
            continue
        last = dims[-1]
        # Must be a statically-known positive int. Symbolic dims (dim_param set,
        # or dim_value == 0) cannot be used to derive head_dim.
        if last.HasField("dim_value") and last.dim_value > 0:
            head_dim = last.dim_value
            _logger.info(
                "Derived head_dim=%d from graph input '%s' (last dim of shape %s).",
                head_dim,
                inp.name,
                [d.dim_value if d.HasField("dim_value") else d.dim_param for d in dims],
            )
            return head_dim

    raise ValueError(
        "Cannot infer head_dim: no graph input matching 'past_value' with a "
        "static positive last dimension was found."
    )

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""RMSNorm fusion for SpinQuant: absorb norm scale weights into downstream linear layers."""

from typing import List, Optional, Tuple
import numpy as np
from onnx import numpy_helper

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.common.connected_graph.operation import Op
from aimet_onnx.graph_passes.passes.common_patterns import match_rms_norm_pattern
from aimet_onnx.meta.connectedgraph import ConnectedGraph, Product
from aimet_onnx.utils import ModelProto, ParamUtils

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SpinQuant)

# Op types that only reshape/reformat activations without changing the
# mathematical relationship between gamma and the downstream linear weight.
_OP_OUTPUTS_TO_IGNORE = [
    "Unsqueeze",
    "Squeeze",
    "Transpose",
    "Reshape",
    "Flatten",
    "Cast",
]


def fuse_norm_layers_into_linears(model: ModelProto, connected_graph: ConnectedGraph):
    """
    For every affine RMSNorm in the model, absorb the scale weight (gamma) into
    the weights of its downstream linear (MatMul/Gemm/Conv) layers, then reset
    gamma to ones.

    The transformation is numerically equivalent:
        gamma * RMSNorm(x) @ W  ==  RMSNorm(x) @ (diag(gamma) @ W)
                                ==  RMSNorm(x) @ W_fused

    where in ONNX convention W is [in_features, out_features], so:
        W_fused = gamma[:, None] * W

    After fusion, gamma is set to ones, making the norm a pure normalization
    with no learnable scale effect.

    :param model: ONNX ModelProto whose initializers are modified in-place.
    :param connected_graph: ConnectedGraph built from the model.
    """
    for op in connected_graph.ordered_ops:
        result = _find_norm_scale_and_consumers(op, model)
        if result is None:
            continue

        scale_name, downstream_linears = result
        if not downstream_linears:
            _logger.debug(
                "RMSNorm scale '%s': no downstream linear ops found, skipping.",
                scale_name,
            )
            continue

        scale_tensor = ParamUtils.get_param_by_name(model, scale_name)
        scale = numpy_helper.to_array(scale_tensor)
        scale_dtype = scale.dtype
        scale_f64 = scale.astype(np.float64)  # promote for numerical precision

        for linear_op in downstream_linears:
            weight_inp, is_transposed = _get_weight_product(linear_op)
            weight_tensor = ParamUtils.get_param_by_name(model, weight_inp.name)
            if weight_tensor is None:
                _logger.warning(
                    "RMSNorm scale '%s': weight '%s' not found in initializers, skipping.",
                    scale_name,
                    weight_inp.name,
                )
                continue
            W = numpy_helper.to_array(weight_tensor)
            orig_dtype = W.dtype

            if linear_op.type == "Conv":
                # W shape: [out_channels, in_channels, *kernel]
                # gamma [in_channels] is absorbed along axis 1
                scale_broadcast = scale_f64.reshape(1, -1, *([1] * (W.ndim - 2)))
            elif is_transposed:
                # Gemm transB=1 or W → Transpose → MatMul: stored W is [out, in]
                # gamma [in_features] is absorbed along axis 1
                scale_broadcast = scale_f64[None, :]
            else:
                # MatMul or Gemm transB=0: W shape [in_features, out_features]
                # gamma [in_features] is absorbed along axis 0
                scale_broadcast = scale_f64[:, None]

            W_fused = (scale_broadcast * W.astype(np.float64)).astype(orig_dtype)
            weight_tensor.CopyFrom(
                numpy_helper.from_array(W_fused, name=weight_tensor.name)
            )

        # Reset gamma to ones so the norm no longer applies any scaling
        ones = np.ones(scale.shape, dtype=scale_dtype)
        scale_tensor.CopyFrom(numpy_helper.from_array(ones, name=scale_tensor.name))


def _get_weight_product(op: Op) -> Tuple[Optional[Product], bool]:
    """
    Return (weight_product, is_transposed) for a MatMul/Gemm/Conv op.

    Handles two patterns:
    - Direct:   W (initializer) → MatMul/Gemm/Conv
    - Indirect: W (initializer) → Transpose → MatMul

    For the indirect case ``is_transposed=True`` signals that the stored weight
    has shape [out, in], so gamma must be absorbed along axis 1 instead of axis 0.
    Returns (None, False) if no static weight is found.

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

    # W → Transpose → MatMul pattern
    if op.type in ("MatMul", "Gemm"):
        for inp in op.inputs:
            if inp.producer and inp.producer.type == "Transpose":
                for t_inp in inp.producer.inputs:
                    if t_inp.is_parm or t_inp.is_const:
                        return t_inp, True
    return None, False


def _iter_linear_consumers(scale_mul_op: Op) -> List[Op]:
    """
    Return all downstream weight MatMul/Gemm/Conv ops reachable from scale_mul_op.

    MatMul/Gemm/Conv Op can be a direct consumer or the op may be
    reached through a chain of reshape/reformat ops (Unsqueeze, Transpose, Cast,
    Reshape, …) that adjust the activation layout into Conv-compatible format.

    :param scale_mul_op: The Mul op that applies the RMSNorm scale (gamma) to the
                         normalized activations. Its outputs are traversed to collect
                         downstream weight ops.
    :return: List of MatMul, Gemm, or Conv ops that consume the scaled activations
             and have a static weight initializer available for fusion.
    """
    result = []
    visited = set()
    queue = list(scale_mul_op.output_ops)
    while queue:
        consumer = queue.pop()
        if id(consumer) in visited:
            continue
        visited.add(id(consumer))
        if (
            consumer.type in ("MatMul", "Gemm")
            and _get_weight_product(consumer)[0] is not None
        ):
            result.append(consumer)
        elif consumer.type in _OP_OUTPUTS_TO_IGNORE:
            queue.extend(consumer.output_ops)
        elif consumer.type == "Conv" and _get_weight_product(consumer)[0] is not None:
            result.append(consumer)
    return result


def _find_norm_scale_and_consumers(
    op: Op, model: ModelProto
) -> Optional[Tuple[str, List[Op]]]:
    """
    If op starts an affine RMSNorm pattern, return (scale_initializer_name,
    list of downstream weight MatMul/Gemm/Conv ops).

    Returns None if op does not start an affine RMSNorm (non-affine norms have
    no scale to fuse).

    :param op: The candidate starting op to check for an affine RMSNorm pattern.
    :param model: ONNX ModelProto used to look up initializer names when matching
                  the RMSNorm pattern.
    :return: A tuple (scale_initializer_name, downstream_linear_ops) if op starts
             an affine RMSNorm, where scale_initializer_name is the name of the gamma
             initializer and downstream_linear_ops is the list of weight MatMul/Gemm/Conv
             ops that follow the norm's scale multiply. Returns None if the op does not
             match an affine RMSNorm pattern.
    """
    norm_ops = match_rms_norm_pattern(op, model)
    if not norm_ops:
        return None

    # match_rms_norm_pattern stops at the last Mul op. Check for the trailing
    # scale Mul: RMSNorm(x) * gamma, with an optional Cast in between.
    last_op = norm_ops[-1]
    if len(last_op.output_ops) != 1:
        return None  # non-affine norm, nothing to fuse
    next_op = last_op.output_ops[0]
    if next_op.type == "Cast" and len(next_op.output_ops) == 1:
        next_op = next_op.output_ops[0]
    if next_op.type != "Mul":
        return None  # non-affine norm, nothing to fuse

    scale_mul_op = next_op
    scale_inp = next((inp for inp in scale_mul_op.inputs if inp.is_const), None)
    if scale_inp is None:
        return None

    # Downstream weight linears: direct MatMul/Gemm consumers and Conv ops.
    downstream_linears = _iter_linear_consumers(scale_mul_op)

    return scale_inp.name, downstream_linears

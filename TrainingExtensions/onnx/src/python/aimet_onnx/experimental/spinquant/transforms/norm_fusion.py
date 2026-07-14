# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""RMSNorm scale absorption: fuse gamma into downstream linear weights."""

from typing import List
import numpy as np
from onnx import numpy_helper

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.utils import ModelProto, ParamUtils

from aimet_onnx.experimental.block_topology.norm_detection import ActiveNorm
from aimet_onnx.experimental.block_topology.weight_utils import get_weight_product

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SpinQuant)


def fuse_norm_layers_into_linears(model: ModelProto, active_norms: List[ActiveNorm]):
    """Absorb RMSNorm gamma into downstream linear weights, then reset gamma to ones.

    For every affine RMSNorm in ``active_norms``, this function multiplies the
    scale weight (gamma) into the weights of its downstream linear
    (MatMul/Gemm/Conv) layers in-place, then resets gamma to ones.

    The transformation is numerically equivalent::

        gamma * RMSNorm(x) @ W  ==  RMSNorm(x) @ (diag(gamma) @ W)
                                ==  RMSNorm(x) @ W_fused

    where in ONNX convention W is [in_features, out_features], so::

        W_fused = gamma[:, None] * W

    After fusion, gamma is set to ones, making the norm a pure normalization
    with no learnable scale effect.

    :param model: ONNX ModelProto whose initializers are modified in-place.
    :param active_norms: Active norms to fuse, as returned by ``find_active_norms``.
        Each entry carries the gamma initializer name and the downstream linear ops.
    """
    for active_norm in active_norms:
        scale_name = active_norm.scale_name
        downstream_linears = active_norm.downstream_linears

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
            weight_inp, is_transposed = get_weight_product(linear_op)
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

            # Determine in_features based on storage layout (needed for tiling check below).
            if linear_op.type == "Conv":
                in_features = W.shape[1]  # [out, in, *k]
            elif is_transposed:
                in_features = W.shape[1]  # [out, in]
            else:
                in_features = W.shape[0]  # [in, out]

            # Repeat gamma when its length is smaller than in_features.
            scale_f64_effective = scale_f64
            if len(scale_f64) < in_features:
                if in_features % len(scale_f64) != 0:
                    raise ValueError(
                        f"RMSNorm scale '{scale_name}' length {len(scale_f64)} does not "
                        f"divide in_features={in_features} of op '{linear_op.name}'."
                    )
                tile_factor = in_features // len(scale_f64)
                scale_f64_effective = np.tile(scale_f64, tile_factor)
                _logger.debug(
                    "Repeating RMSNorm scale '%s' by %d for op '%s' "
                    "(gamma dim %d < in_features %d).",
                    scale_name,
                    tile_factor,
                    linear_op.name,
                    len(scale_f64),
                    in_features,
                )

            if linear_op.type == "Conv":
                # W shape: [out_channels, in_channels, *kernel]
                # gamma [in_channels] is absorbed along axis 1
                scale_broadcast = scale_f64_effective.reshape(
                    1, -1, *([1] * (W.ndim - 2))
                )
            elif is_transposed:
                # Gemm transB=1 or W -> Transpose -> MatMul: stored W is [out, in]
                # gamma [in_features] is absorbed along axis 1
                scale_broadcast = scale_f64_effective[None, :]
            else:
                # MatMul or Gemm transB=0: W shape [in_features, out_features]
                # gamma [in_features] is absorbed along axis 0
                scale_broadcast = scale_f64_effective[:, None]

            W_fused = (scale_broadcast * W.astype(np.float64)).astype(orig_dtype)
            weight_tensor.CopyFrom(
                numpy_helper.from_array(W_fused, name=weight_tensor.name)
            )
            _logger.debug(
                "Fused RMSNorm scale '%s' into weight '%s' of op '%s'.",
                scale_name,
                weight_inp.name,
                linear_op.name,
            )

        # Reset gamma to ones so the norm no longer applies any scaling
        ones = np.ones(scale.shape, dtype=scale_dtype)
        scale_tensor.CopyFrom(numpy_helper.from_array(ones, name=scale_tensor.name))

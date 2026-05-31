# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""R2 per-head Hadamard rotation pass.

R2 = H / sqrt(head_dim) rotates the V/O path on a per-head basis. Per block:

* V projection: ``Wv <- Wv @ block_diag(R2, ..., R2)`` on the output axis,
  rotating each head's slice with R2.
* O projection: ``Wo <- block_diag(R2, ..., R2)^T @ Wo`` on the input axis,
  rotating each head's slice with R2^T.

In float, R2 inserted at V's output and R2^T inserted at O's input cancel
because the attention output ``softmax(QK^T) @ V`` is linear in V along the
``head_dim`` axis. Under quantization, R2 reduces V/O outliers.

R2 is independent of R1: R1 acts on the residual-stream (``hidden``) axis,
R2 acts on the per-head ``head_dim`` axis. They compose without interaction.
"""

from onnx import numpy_helper

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.meta.operations import Op
from aimet_onnx.utils import ModelProto, ParamUtils

from aimet_onnx.experimental.spinquant.model_analysis import (
    find_attention_topology,
)
from aimet_onnx.experimental.spinquant.model_analysis.weight_utils import (
    get_weight_product,
)
from aimet_onnx.experimental.spinquant.passes.base import (
    RotationPass,
    SpinquantContext,
)
from aimet_onnx.experimental.spinquant.transforms import (
    block_diag_repeat,
    hadamard_rotation_matrix,
    rotate_linear_weight,
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SpinQuant)


class R2RotationPass(RotationPass):
    """R2 per-head V/O Hadamard rotation.

    ``head_dim`` is read from :attr:`SpinquantContext.backbone_head_dim`, which
    the orchestrator derives from a ``past_value`` graph input. Callers do not
    pass it explicitly; an export without KV-cache inputs cannot run R2.
    """

    @property
    def name(self) -> str:
        return "R2"

    def validate(self, ctx: SpinquantContext) -> None:
        """Verify each block has unfused QKV with a clean V/O attention path."""
        head_dim = _require_head_dim(ctx)
        topologies = _get_or_build_topology_cache(ctx)
        for topology, block in zip(topologies, ctx.backbone_role_map.blocks):
            for v_op in topology.v_ops:
                _validate_v_op(ctx.backbone_sim.model.model, v_op, head_dim)
            for o_op in block.o_proj:
                _validate_o_op(ctx.backbone_sim.model.model, o_op, head_dim)

    def apply(self, ctx: SpinquantContext) -> None:
        """Rotate each block's V output channels and O input channels per head."""
        head_dim = _require_head_dim(ctx)
        topologies = _get_or_build_topology_cache(ctx)
        R2 = hadamard_rotation_matrix(head_dim)
        _logger.info(
            "Backbone: Applying R2 Hadamard rotation per head (head_dim=%d).", head_dim
        )

        for block_idx, (topology, block) in enumerate(
            zip(topologies, ctx.backbone_role_map.blocks)
        ):
            for v_op in topology.v_ops:
                v_axis_size = _get_rotated_axis_size(
                    ctx.backbone_sim.model.model, v_op, is_writing=True
                )
                R2_v = block_diag_repeat(R2, v_axis_size // head_dim)
                rotate_linear_weight(
                    ctx.backbone_sim.model.model, v_op, R2_v, is_writing=True
                )
                _logger.debug(
                    "R2 block %d: rotated v='%s' (axis=%d).",
                    block_idx,
                    v_op.name,
                    v_axis_size,
                )

            for o_op in block.o_proj:
                o_axis_size = _get_rotated_axis_size(
                    ctx.backbone_sim.model.model, o_op, is_writing=False
                )
                R2_o = block_diag_repeat(R2, o_axis_size // head_dim)
                rotate_linear_weight(
                    ctx.backbone_sim.model.model, o_op, R2_o, is_writing=False
                )
                _logger.debug(
                    "R2 block %d: rotated o='%s' (axis=%d).",
                    block_idx,
                    o_op.name,
                    o_axis_size,
                )


def _require_head_dim(ctx: SpinquantContext) -> int:
    """Return ``ctx.backbone_head_dim`` or raise if it could not be derived."""
    head_dim = ctx.backbone_head_dim
    if head_dim is None:
        raise ValueError(
            "R2 rotation: head_dim could not be derived from the backbone "
            "model. The export must expose a 'past_value' graph input whose "
            "last dimension is a static positive integer (HF/optimum LLM "
            "exports satisfy this by default)."
        )
    return head_dim


_TOPOLOGY_CACHE_KEY = "_r2_topology_cache"


def _get_or_build_topology_cache(ctx: SpinquantContext):
    """Compute attention topology once per ctx and cache it on the ctx instance."""
    cached = getattr(ctx, _TOPOLOGY_CACHE_KEY, None)
    if cached is not None:
        return cached
    topologies = find_attention_topology(ctx.backbone_role_map)
    object.__setattr__(ctx, _TOPOLOGY_CACHE_KEY, topologies)
    return topologies


def _get_rotated_axis_size(model: ModelProto, op: Op, is_writing: bool) -> int:
    """Return the size of the axis R2 rotates for ``op``.

    For writing layers (V output): output dim — ``shape[0]`` for [out, in] /
    Conv storage, ``shape[-1]`` for [in, out] storage.
    For reading layers (O input): input dim — the complementary axis.
    """
    weight_inp, is_transposed = get_weight_product(op)
    tensor = ParamUtils.get_param_by_name(model, weight_inp.name)
    shape = numpy_helper.to_array(tensor).shape

    if op.type == "Conv":
        return shape[0] if is_writing else shape[1]
    if is_transposed:  # [out, in]
        return shape[0] if is_writing else shape[1]
    # [in, out]
    return shape[-1] if is_writing else shape[0]


def _validate_v_op(model: ModelProto, v_op: Op, head_dim: int) -> None:
    out_size = _get_rotated_axis_size(model, v_op, is_writing=True)
    if out_size % head_dim != 0:
        raise ValueError(
            f"R2 rotation: V op '{v_op.name}' output size {out_size} not divisible "
            f"by head_dim={head_dim}."
        )


def _validate_o_op(model: ModelProto, o_op: Op, head_dim: int) -> None:
    in_size = _get_rotated_axis_size(model, o_op, is_writing=False)
    if in_size % head_dim != 0:
        raise ValueError(
            f"R2 rotation: O op '{o_op.name}' input size {in_size} not divisible "
            f"by head_dim={head_dim}."
        )

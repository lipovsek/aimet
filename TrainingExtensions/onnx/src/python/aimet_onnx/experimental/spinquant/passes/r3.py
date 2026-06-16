# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""R3 online Hadamard rotation pass.

R3 = H / sqrt(head_dim) is an online Hadamard inserted as a ``MatMul`` node
on each decoder block's Q-side and K-side, immediately upstream of the
attention QK^T MatMul. Because ``H @ H^T = I``,
``(Q @ H) @ (K @ H)^T = Q @ K^T`` — the two inserted rotations cancel
algebraically in float, so attention output is unchanged.

K-side placement: when the export has KV-cache inputs, R3 is inserted on the
current-K branch *before* the past-key ``Concat``. This is the paper-correct
placement: K values entering the cache (and so quantized for cache storage) are
already rotated, which is what motivates R3 in the first place. The model's
``present_key`` output now carries rotated K, and the runtime feeds it back
as ``past_key`` next step — the cache convention is self-consistent across
autoregressive steps.

Unlike R1 / R2, R3 introduces new ops and new initializers in the graph
rather than mutating existing weights. R3 is independent of R1 / R2 — composes
freely.

R3 inserts plain float ``MatMul`` Hadamards. It runs on the float ONNX model
before any ``QuantizationSimModel`` is built, so the sim — created afterward on
the rotated graph — wraps the inserted MatMuls in activation quantizers like
any other op, and ``compute_encodings`` calibrates them on the rotated values.

Limitations (this iteration):

* Requires a KV-cache-style export: the model must expose one ``past_key_*``
  graph input per decoder block. Each is consumed by exactly one ``Concat``
  whose other input is the post-RoPE current K. The Concat output reaches
  the QK^T MatMul through pass-through ops only (Transpose / Reshape /
  Cast / Identity).
* Prefill-only exports without KV-cache inputs are not supported.

Note on graph staleness: inserting nodes invalidates ``ctx.backbone_role_map``.
R3 is the last pass in the pipeline. Do not run another role-map-dependent pass
after R3 in the same context.
"""

import re
from typing import List

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.meta.operations import Op

from aimet_onnx.experimental.spinquant.model_analysis import (
    BlockR3Anchors,
    find_r3_anchors,
)
from aimet_onnx.experimental.spinquant.passes.base import (
    RotationPass,
    SpinquantContext,
)
from aimet_onnx.experimental.spinquant.transforms import (
    insert_online_hadamard_node,
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SpinQuant)

# R3 inserts its online Hadamard rotations as ``MatMul`` nodes whose names are
# the ``spinquant_block{idx}_{q|k}`` prefix from ``_rotate_q_side`` /
# ``_rotate_k_side`` below plus the ``_R3`` suffix appended by
# ``insert_online_hadamard_node``. This regex is the single source of truth for
# recognizing those ops downstream; keep it in sync with the names produced here.
_ONLINE_ROTATION_OP_RE = re.compile(r"spinquant_block\d+_[qk]_R3$")


def is_online_rotation_op(cg_op: Op) -> bool:
    """Return True if ``cg_op`` is a SpinQuant R3 online Hadamard rotation MatMul.

    R3 online rotations carry a fixed orthonormal Hadamard as their "weight"
    rather than a learnable parameter, so downstream optimizers (e.g.
    sequential MSE) should skip them.
    """
    return cg_op.type == "MatMul" and bool(_ONLINE_ROTATION_OP_RE.match(cg_op.name))


class R3RotationPass(RotationPass):
    """R3 online Hadamard rotation on Q and K paths into the QK^T MatMul.

    ``head_dim`` is read from :attr:`SpinquantContext.backbone_head_dim`,
    which the orchestrator derives from a ``past_value`` graph input. An
    export without KV-cache inputs cannot run R3.
    """

    @property
    def name(self) -> str:
        return "R3"

    def validate(self, ctx: SpinquantContext) -> None:
        """Verify the model exposes one past_key_* graph input per decoder block,
        each consumed by exactly one Concat reaching exactly one MatMul forward."""
        _require_head_dim(ctx)
        # find_r3_anchors performs all the structural validation and raises
        # with a precise per-block error if anything looks wrong.
        _get_or_build_anchor_cache(ctx)

    def apply(self, ctx: SpinquantContext) -> None:
        """Insert one online Hadamard MatMul on the Q path and one on the K path per block.

        * Q-side: the Hadamard is spliced on the edge feeding QK^T.
        * K-side: the Hadamard is spliced on the current-K edge feeding the
          past-key Concat, so the cache stores rotated K.

        The inserted MatMuls are plain float ops; the ``QuantizationSimModel``
        built afterward on the rotated graph wraps them in quantizers.
        """
        head_dim = _require_head_dim(ctx)
        anchors = _get_or_build_anchor_cache(ctx)
        model = ctx.backbone_model
        _logger.info(
            "Backbone: Applying R3 online Hadamard rotation per attention block "
            "(head_dim=%d, blocks=%d).",
            head_dim,
            len(anchors),
        )

        for block_idx, anchor in enumerate(anchors):
            self._rotate_q_side(model, anchor, head_dim, block_idx)
            self._rotate_k_side(model, anchor, head_dim, block_idx)
            _logger.debug(
                "R3 block %d: inserted Q-side rotation before '%s'[%d] "
                "and K-side rotation before '%s'[%d].",
                block_idx,
                anchor.qk_matmul_node.name,
                anchor.q_consumer_input_idx,
                anchor.k_concat_node.name,
                anchor.k_consumer_input_idx,
            )

    @staticmethod
    def _rotate_q_side(model, anchor, head_dim, block_idx) -> None:
        """Insert ``... -> R3 -> QK^T`` on the Q path."""
        insert_online_hadamard_node(
            model,
            target_tensor_name=anchor.q_input_tensor,
            consumer_node=anchor.qk_matmul_node,
            consumer_input_idx=anchor.q_consumer_input_idx,
            head_dim=head_dim,
            name_prefix=f"spinquant_block{block_idx}_q",
        )

    @staticmethod
    def _rotate_k_side(model, anchor, head_dim, block_idx) -> None:
        """Insert ``... -> R3 -> Concat`` on the current-K path.

        R3 is spliced on the current-K edge feeding the past-key Concat so K
        values entering the cache are already rotated.
        """
        insert_online_hadamard_node(
            model,
            target_tensor_name=anchor.k_input_tensor,
            consumer_node=anchor.k_concat_node,
            consumer_input_idx=anchor.k_consumer_input_idx,
            head_dim=head_dim,
            name_prefix=f"spinquant_block{block_idx}_k",
        )


def _require_head_dim(ctx: SpinquantContext) -> int:
    """Return ``ctx.backbone_head_dim`` or raise if it could not be derived."""
    head_dim = ctx.backbone_head_dim
    if head_dim is None:
        raise ValueError(
            "R3 rotation: head_dim could not be derived from the backbone "
            "model. The export must expose a 'past_value' graph input whose "
            "last dimension is a static positive integer (HF/optimum LLM "
            "exports satisfy this by default)."
        )
    return head_dim


_ANCHOR_CACHE_KEY = "_r3_anchor_cache"


def _get_or_build_anchor_cache(ctx: SpinquantContext) -> List[BlockR3Anchors]:
    """Compute R3 anchors once per ctx and cache them on the ctx instance."""
    cached = getattr(ctx, _ANCHOR_CACHE_KEY, None)
    if cached is not None:
        return cached
    anchors = find_r3_anchors(ctx.backbone_role_map, ctx.backbone_model)
    object.__setattr__(ctx, _ANCHOR_CACHE_KEY, anchors)
    return anchors

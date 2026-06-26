# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""V projection identification for SpinQuant R2 rotation.

R2 rotates the V/O path of each attention block on a per-head basis. The V
projection is identified by matching the module name in ``Op.name`` against a
small allow-list of conventional names (``v_proj``, ``v``, ``value``).

This is a deliberate, temporary heuristic. Topology-based identification (walk
back from O through Reshape/Transpose to V) is fragile when an exporter emits
dynamic ``Reshape`` shape inputs (common for HF→ONNX exports with dynamic
batch/seq), so we trust the decoder role map instead. Fused QKV is rejected
naturally because the fused linear's module name (e.g. ``qkv_proj``) does not
match any of the V allow-list names.
"""

import re
from dataclasses import dataclass
from typing import List

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.meta.operations import Op

from aimet_onnx.experimental.spinquant.model_analysis.block_identifier import (
    DecoderModelRoleMap,
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SpinQuant)

# Match a V module name (``v_proj``, ``value``, ``v``), optionally suffixed with
# ``_sha`` and/or a per-head index (``v_proj_sha``, ``v_proj.0``, ``v_proj_sha.1``).
# Longest alternatives first so ``v_proj`` wins over the ``v`` prefix.
_V_MODULE_PATTERN = re.compile(r"^(v_proj|value|v)(_sha)?(\.\d+)?$")


@dataclass
class BlockAttentionTopology:
    """V projection(s) for a single decoder block.

    :param v_ops: V projection ops (MatMul/Gemm/Conv) within the block's
        ``qkv_linears``. Multiple ops appear when an exporter splits multi-head
        attention into per-head single-head attention (one v_proj per head).
    """

    v_ops: List[Op]


def find_attention_topology(
    role_map: DecoderModelRoleMap,
) -> List[BlockAttentionTopology]:
    """Return per-block V projection ops for every decoder block in ``role_map``.

    V is identified by matching the second-to-last component of ``Op.name``
    (the originating ``nn.Module`` attribute name) against ``_V_MODULE_PATTERN``.

    :param role_map: Decoder role map produced by ``get_decoder_role_map``.
    :return: One ``BlockAttentionTopology`` per block, in topological order.
    :raises ValueError: If any block has zero V candidates among its
        ``qkv_linears`` (e.g. fused QKV).
    """
    result = []
    for block_idx, block in enumerate(role_map.blocks):
        v_candidates = [op for op in block.qkv_linears if _is_v_projection(op)]
        if not v_candidates:
            qkv_names = [op.name for op in block.qkv_linears]
            raise ValueError(
                f"R2 rotation: block {block_idx}: expected at least one V projection "
                f"in qkv_linears matching pattern '{_V_MODULE_PATTERN.pattern}', "
                f"found 0 (qkv_linears={qkv_names}). Likely fused QKV or non-standard naming."
            )
        result.append(BlockAttentionTopology(v_ops=v_candidates))
    _logger.debug(
        "Attention topology: %d block(s), v_ops=%s.",
        len(result),
        [[op.name for op in t.v_ops] for t in result],
    )
    return result


def _is_v_projection(op: Op) -> bool:
    """Return True iff ``op``'s containing module name is in the V allow-list.

    ``Op.name`` from a HF→ONNX export looks like
    ``/model/.../self_attn/v_proj/MatMul``; the second-to-last ``/``-segment is
    the originating ``nn.Module`` attribute name (``v_proj``, ``v``, ``value``).
    """
    parts = op.name.rsplit("/", 2)
    if len(parts) < 2:
        return False
    return bool(_V_MODULE_PATTERN.match(parts[-2]))

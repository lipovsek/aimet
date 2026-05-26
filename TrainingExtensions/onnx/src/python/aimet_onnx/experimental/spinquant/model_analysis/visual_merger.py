# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""PatchMerger output projection (linear_fc2) detection for VLM visual encoders."""

from typing import List

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.meta.operations import Op

from aimet_onnx.experimental.spinquant.model_analysis.weight_utils import (
    get_weight_product,
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SpinQuant)

_LINEAR_TYPES = frozenset(("MatMul", "Gemm", "Conv"))


def find_merger_linear2(connected_graph: ConnectedGraph) -> List[Op]:
    """Find PatchMerger linear_fc2 ops in a visual encoder ONNX graph.

    Identifies all weighted linear ops that are leaf nodes in the weighted-linear
    subgraph — i.e. have no downstream weighted linear consumers. These are the
    PatchMerger linear_fc2 layers that write into the language backbone residual
    stream and must always be rotated with R_L when the backbone is SpinQuant-rotated.

    NOTE:
        Assumes the PatchMerger linear_fc2 is the topological leaf of the weighted-linear
        subgraph — i.e. no downstream weighted linear follows it. This holds for Qwen2.5-VL
        and Qwen3-VL. Unknown architectures will be misdetected; an
        explicit override will be added as part of the general block-detection fallback.

    :param connected_graph: ConnectedGraph built from visual.onnx.
    :return: List of merger_linear2 ops in topological order.
    :raises ValueError: If no merger_linear2 ops are found.
    """
    weighted_linears_topo = [
        op
        for op in connected_graph.ordered_ops
        if op.type in _LINEAR_TYPES and get_weight_product(op)[0] is not None
    ]
    weighted_linear_ids = {id(op) for op in weighted_linears_topo}

    def _has_downstream_weighted_linear(op: Op) -> bool:
        visited = set()
        stack = list(op.output_ops)
        while stack:
            cur = stack.pop()
            if id(cur) in visited:
                continue
            visited.add(id(cur))
            if id(cur) in weighted_linear_ids:
                return True
            stack.extend(cur.output_ops)
        return False

    result = [
        op for op in weighted_linears_topo if not _has_downstream_weighted_linear(op)
    ]

    if not result:
        raise ValueError(
            "merger_linear2 not detected: no leaf weighted linear op found in the ViT graph."
        )

    _logger.info(
        "Visual: merger_linear2=%s will be rotated with R_L.",
        [op.name for op in result],
    )
    return result

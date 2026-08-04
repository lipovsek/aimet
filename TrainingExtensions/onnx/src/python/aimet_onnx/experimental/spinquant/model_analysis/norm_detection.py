# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""R1-specific affine RMSNorm checks for SpinQuant.

Generic active-RMSNorm detection lives in ``block_topology.norm_detection``.
This module holds only the SpinQuant R1 precondition check that reuses that
detection to find affine RMSNorms sitting between a writing layer and the
residual add.
"""

from typing import List

from aimet_onnx.utils import ModelProto

from aimet_onnx.experimental.llm_topology.norm_detection import (
    _find_norm_scale_and_consumers,
)


def find_post_writing_norms(model: ModelProto, role_map) -> List[str]:
    """Return op names of affine RMSNorms immediately after writing layers.

    Used by R1 architecture compatibility checks: R1 absorption requires writing
    layers (o_proj, down_proj) to feed directly into the residual add, with no
    affine RMSNorm in between.

    :param model: ONNX ModelProto.
    :param role_map: LlmTopology.
    :return: List of norm op names for detected post-writing norms.
    """
    found = []
    for block in role_map.blocks:
        for writing_op in block.o_proj + block.down_proj:
            for out_op in writing_op.output_ops:
                candidates = out_op.output_ops if out_op.type == "Cast" else [out_op]
                for candidate in candidates:
                    match = _find_norm_scale_and_consumers(candidate, model)
                    if match is not None:
                        found.append(candidate.name)
                        break
    return found

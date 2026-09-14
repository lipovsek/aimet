# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""R1-specific affine RMSNorm checks for SpinQuant.

Generic active-RMSNorm detection lives in ``llm_topology.norm_detection``.
This module holds only the SpinQuant R1 precondition check that reuses that
detection to find affine RMSNorms sitting between a writing layer and the
residual add.
"""

from typing import Iterable, List

from aimet_onnx.utils import ModelProto

from aimet_onnx.experimental.llm_topology import ir_analysis
from aimet_onnx.experimental.llm_topology.norm_detection import is_affine_rms_norm


def find_post_writing_norms(
    model: ModelProto, writing_output_tensors: Iterable[str]
) -> List[str]:
    """Return names of affine RMSNorms immediately after writing layers.

    Used by R1 architecture compatibility checks: R1 absorption requires writing
    layers (o_proj, down_proj) to feed directly into the residual add, with no
    affine RMSNorm in between.

    :param model: ONNX ModelProto.
    :param writing_output_tensors: Output tensor names of the writing layers to
        check (a block's o_proj and down_proj).
    :return: List of norm names for detected post-writing norms.
    """
    ir_model = ir_analysis.build_analysis_ir(model)
    node_by_output = ir_analysis.node_by_output_tensor(ir_model)

    found = []
    for tensor_name in writing_output_tensors:
        writing_node = node_by_output.get(tensor_name)
        if writing_node is None:
            continue
        for consumer in writing_node.successors():
            # A dtype hop between the writing layer and the norm is transparent.
            candidates = (
                consumer.successors() if consumer.op_type == "Cast" else [consumer]
            )
            for candidate in candidates:
                if is_affine_rms_norm(candidate):
                    found.append(ir_analysis.node_name(candidate))
                    break
    return found

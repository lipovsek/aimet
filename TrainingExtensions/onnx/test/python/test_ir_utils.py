# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import onnx
import onnx_ir
import onnxruntime as ort
import copy
import numpy as np
import aimet_onnx

from aimet_onnx import ir_utils
from .models import models_for_tests


@pytest.mark.parametrize(
    "model",
    (
        models_for_tests.build_dummy_model(),
        models_for_tests.single_residual_model().model,
        models_for_tests.multi_input_model().model,
        models_for_tests.transposed_conv_model().model,
        models_for_tests.concat_model().model,
        models_for_tests.hierarchical_model().model,
        models_for_tests.elementwise_op_model().model,
        models_for_tests.instance_norm_model().model,
        models_for_tests.layernorm_model(),
        models_for_tests.matmul_with_constant_first_input(),
        models_for_tests.model_with_split_matmul(),
    ),
)
def test_remove_aimet_quantizers(model: onnx.ModelProto):
    sim = aimet_onnx.QuantizationSimModel(copy.deepcopy(model))
    ir_sim_model = onnx_ir.from_proto(sim.model.model)
    ir_utils.remove_aimet_quantizers(ir_sim_model)
    for node in ir_sim_model.graph.all_nodes():
        assert node.op_type != "QcQuantizeOp"

    proto: onnx.ModelProto = onnx_ir.to_proto(ir_sim_model)
    onnx.checker.check_model(proto)
    assert len(proto.graph.node) == len(model.graph.node)
    for orig_inp, new_inp in zip(model.graph.input, proto.graph.input):
        assert orig_inp.name == new_inp.name
    for orig_out, new_out in zip(model.graph.output, proto.graph.output):
        assert orig_out.name == new_out.name

    node_name_to_inputs_orig = {node.name: node.input for node in model.graph.node}
    node_name_to_inputs_new = {node.name: node.input for node in proto.graph.node}
    for node_name in node_name_to_inputs_orig:
        assert node_name in node_name_to_inputs_new
        assert node_name_to_inputs_orig[node_name] == node_name_to_inputs_new[node_name]

    dummy_input = aimet_onnx.utils.make_dummy_input(sim.model.model)
    sess_orig = ort.InferenceSession(model.SerializeToString())
    sess_new = ort.InferenceSession(proto.SerializeToString())
    orig_outs = sess_orig.run(None, dummy_input)
    new_outs = sess_new.run(None, dummy_input)
    for orig_out, new_out in zip(orig_outs, new_outs):
        assert np.allclose(orig_out, new_out, atol=1e-6)

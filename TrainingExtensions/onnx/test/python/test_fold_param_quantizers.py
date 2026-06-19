# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for QuantizationSimModel.fold_param_quantizers (aimet-onnx)"""

import json
import os
import tempfile

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper

import aimet_onnx
from aimet_onnx.common.defs import QuantizationDataType
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.utils import make_dummy_input

from .models.models_for_tests import build_dummy_model


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _make_calibrated_sim(**kwargs):
    model = build_dummy_model()
    dummy_input = make_dummy_input(model)
    sim = QuantizationSimModel(model, **kwargs)
    sim.compute_encodings([dummy_input])
    return sim, model, dummy_input


def _constant_weight_matmul_model(in_features=8, out_features=6):
    """MatMul whose weight is produced by a ``Constant`` node (not an initializer)."""
    weight = np.random.randn(in_features, out_features).astype(np.float32)
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["weight"],
        name="weight_const",
        value=numpy_helper.from_array(weight, name="weight_value"),
    )
    matmul = helper.make_node(
        "MatMul", inputs=["input", "weight"], outputs=["output"], name="matmul"
    )
    inp = helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [1, in_features]
    )
    out = helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, out_features]
    )
    graph = helper.make_graph([const_node, matmul], "const_matmul", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


class TestFoldParamQuantizers:
    def test_output_is_numerically_equivalent(self):
        """Folding should not change the quantized inference output."""
        sim, _, dummy_input = _make_calibrated_sim()

        out_before = sim.session.run(None, dummy_input)[0]

        sim.fold_param_quantizers()

        out_after = sim.session.run(None, dummy_input)[0]
        assert np.allclose(out_before, out_after, atol=1e-6)

    def test_param_quantize_nodes_removed(self):
        """All folded param QcQuantizeOp nodes should be removed from the graph."""
        sim, _, _ = _make_calibrated_sim()

        sim.fold_param_quantizers()

        folded_names = set(sim._folded_param_quantizers)
        assert folded_names, "expected some param quantizers to fold"

        # No QcQuantizeOp node should remain for the folded params
        remaining = [
            node
            for node in sim.model.model.graph.node
            if node.op_type == "QcQuantizeOp" and node.input[0] in folded_names
        ]
        assert not remaining

        # The quantizer objects are retained (still enabled, in the dict) so encodings
        # can still be exported; they simply no longer have a node in the graph.
        for name in folded_names:
            assert name in sim.qc_quantize_op_dict
            assert sim.qc_quantize_op_dict[name].enabled

    def test_initializers_overwritten_with_qdq_values(self):
        """The model initializers should hold quantize-dequantize'd weights after fold."""
        sim, _, _ = _make_calibrated_sim()

        weight_name = "fc_w"
        init_before = {
            init.name: np.frombuffer(init.raw_data, dtype=np.float32).copy()
            for init in sim.model.model.graph.initializer
            if init.name == weight_name
        }[weight_name]

        # The expected QDQ value is what the partial-model QDQ computation produces.
        expected = sim._get_qdq_parameters()[weight_name].flatten()

        sim.fold_param_quantizers()

        init_after = {
            init.name: np.frombuffer(init.raw_data, dtype=np.float32).copy()
            for init in sim.model.model.graph.initializer
            if init.name == weight_name
        }[weight_name]

        # The folded weights should differ from the original float weights...
        assert not np.allclose(init_before, init_after)
        # ...and should match the simulated quantize-dequantize values.
        assert np.allclose(init_after, expected, atol=1e-6)

    def test_activation_quantizers_untouched(self):
        """Folding params should not remove activation quantizers."""
        sim, _, _ = _make_calibrated_sim()

        activation_nodes_before = {
            node.name
            for node in sim.model.model.graph.node
            if node.op_type == "QcQuantizeOp" and node.input[0] in sim.activation_names
        }

        sim.fold_param_quantizers()

        activation_nodes_after = {
            node.name
            for node in sim.model.model.graph.node
            if node.op_type == "QcQuantizeOp" and node.input[0] in sim.activation_names
        }
        assert activation_nodes_before == activation_nodes_after
        for name in sim.activation_names:
            assert name in sim.qc_quantize_op_dict

    def test_int32_params_are_skipped(self):
        """int32 (>= 32 bit) param quantizers must not be folded."""
        sim, _, _ = _make_calibrated_sim()

        # Force one param quantizer to 32 bits and recompute its encoding.
        weight_name = "fc_w"
        sim.qc_quantize_op_dict[weight_name].set_bitwidth(32)
        sim.qc_quantize_op_dict[weight_name].data_type = QuantizationDataType.int
        sim._rebuild_session()

        sim.fold_param_quantizers()

        # The int32 param should remain an active, in-graph quantizer.
        assert weight_name in sim.qc_quantize_op_dict
        assert weight_name not in sim._folded_param_quantizers
        assert any(
            node.op_type == "QcQuantizeOp" and node.input[0] == weight_name
            for node in sim.model.model.graph.node
        )

    def test_fold_is_idempotent(self):
        """A second fold call should be a safe no-op."""
        sim, _, dummy_input = _make_calibrated_sim()

        sim.fold_param_quantizers()
        folded = dict(sim._folded_param_quantizers)
        out_after_first = sim.session.run(None, dummy_input)[0]

        sim.fold_param_quantizers()
        assert dict(sim._folded_param_quantizers) == folded
        out_after_second = sim.session.run(None, dummy_input)[0]
        assert np.allclose(out_after_first, out_after_second)

    @pytest.mark.parametrize("encoding_version", ["1.0.0", "2.0.0"])
    def test_export_emits_folded_param_encodings(self, encoding_version, tmp_dir):
        """Export after folding should still emit encodings for the folded params."""
        sim, _, _ = _make_calibrated_sim()

        sim.fold_param_quantizers()
        folded_names = set(sim._folded_param_quantizers)

        sim.export(tmp_dir, "folded_sim", encoding_version=encoding_version)

        with open(os.path.join(tmp_dir, "folded_sim.encodings")) as f:
            encodings = json.load(f)

        if encoding_version == "1.0.0":
            exported_param_names = {enc["name"] for enc in encodings["param_encodings"]}
        else:
            exported_param_names = {enc["name"] for enc in encodings["encodings"]}

        assert folded_names <= exported_param_names

        # The exported model should not contain QcQuantizeOps for the folded params.
        exported = onnx.load(os.path.join(tmp_dir, "folded_sim.onnx"))
        assert not any(
            node.op_type == "QcQuantizeOp" and node.input[0] in folded_names
            for node in exported.graph.node
        )

    def test_fold_constant_node_param(self):
        """Params produced by a Constant node (not an initializer) should fold correctly."""
        model = _constant_weight_matmul_model()
        dummy_input = make_dummy_input(model)
        sim = QuantizationSimModel(
            model, param_type=aimet_onnx.int8, activation_type=aimet_onnx.int16
        )
        sim.compute_encodings([dummy_input])

        out_before = sim.session.run(None, dummy_input)[0]

        sim.fold_param_quantizers()

        # The Constant-node weight should have been folded...
        assert "weight" in sim._folded_param_quantizers
        # ...its QcQuantizeOp removed...
        assert not any(
            node.op_type == "QcQuantizeOp" and node.input[0] == "weight"
            for node in sim.model.model.graph.node
        )
        # ...and inference output should be unchanged.
        out_after = sim.session.run(None, dummy_input)[0]
        assert np.allclose(out_before, out_after, atol=1e-6)

    def test_fold_float16_params(self):
        """Float (fp16) param quantizers fold via an fp16 round-trip, not int QDQ."""
        sim, _, dummy_input = _make_calibrated_sim(
            param_type=aimet_onnx.float16, activation_type=aimet_onnx.float16
        )

        out_before = sim.session.run(None, dummy_input)[0]

        # The folded value should be the float16 round-trip of the raw weights.
        weight_name = "fc_w"
        raw = {
            init.name: numpy_helper.to_array(init)
            for init in sim.model.model.graph.initializer
        }[weight_name]
        expected = raw.astype(np.float16).astype(raw.dtype)

        sim.fold_param_quantizers()

        assert weight_name in sim._folded_param_quantizers
        init_after = {
            init.name: np.frombuffer(init.raw_data, dtype=np.float32).copy()
            for init in sim.model.model.graph.initializer
            if init.name == weight_name
        }[weight_name]
        assert np.allclose(init_after, expected.flatten(), atol=0)

        out_after = sim.session.run(None, dummy_input)[0]
        assert np.allclose(out_before, out_after, atol=1e-6)

    def test_fold_does_not_build_partial_onnx_graph(self, monkeypatch):
        """
        Regression guard for the protobuf 2GB limit.

        ``_get_qdq_parameters`` must compute QDQ params directly from each quantizer,
        without materializing every parameter into a throwaway ONNX graph (which would
        overflow protobuf's 2GB message cap for large models). This asserts the method
        never calls ``onnx.helper.make_graph``.
        """
        sim, _, _ = _make_calibrated_sim()

        def _fail(*args, **kwargs):
            raise AssertionError(
                "_get_qdq_parameters must not build an ONNX graph (protobuf 2GB limit)"
            )

        monkeypatch.setattr(onnx.helper, "make_graph", _fail)

        sim.fold_param_quantizers()
        assert sim._folded_param_quantizers

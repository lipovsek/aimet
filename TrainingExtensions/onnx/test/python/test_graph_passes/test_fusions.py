# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ONNX graph fusions."""

import os
import pytest
import numpy as np
import torch
import onnx
import onnx_ir
import onnxruntime
from aimet_onnx.utils import make_dummy_input
from aimet_onnx import QuantizationSimModel

from aimet_onnx.graph_passes.fusions import fuse_supergroups
from ..models import models_for_tests


def create_layernorm_model(
    tmpdir, elementwise_affine=True, bias=True, epsilon=1e-5, opset=16
):
    # Input shape: [batch_size, seq_len, hidden_size]
    input_shape = [1, 32, 64]
    hidden_size = input_shape[-1]

    class LayerNormModel(torch.nn.Module):
        def __init__(self):
            super(LayerNormModel, self).__init__()
            self.linear = torch.nn.Linear(64, 64)
            self.layernorm = torch.nn.LayerNorm(
                hidden_size,
                eps=epsilon,
                bias=bias,
                elementwise_affine=elementwise_affine,
            )
            self.linear2 = torch.nn.Linear(64, 64)

        def forward(self, x):
            x = self.linear(x)
            x = self.layernorm(x)
            return self.linear2(x)

    model = LayerNormModel()
    dummy_input = torch.randn(*input_shape)
    model_path = os.path.join(tmpdir, "layernorm.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )
    model = onnx.load(model_path)

    return model


def layernorm_with_pow_2_as_multiply(tmpdir):
    model = create_layernorm_model(tmpdir)
    ir_model = onnx_ir.from_proto(model)
    for node in ir_model.graph.all_nodes():
        if node.op_type == "Pow":
            node.op_type = "Mul"
            node.replace_input_with(1, node.inputs[0])
    onnx_ir.passes.common.RemoveUnusedNodesPass().call(ir_model)
    return onnx_ir.to_proto(ir_model)


def layernorm_with_pow_3(tmpdir):
    model = create_layernorm_model(tmpdir)
    ir_model = onnx_ir.from_proto(model)
    new_const = onnx_ir.val(
        "new_pow_const", const_value=onnx_ir.tensor(np.array(3.0, dtype=np.float32))
    )
    ir_model.graph.register_initializer(new_const)
    for node in ir_model.graph.all_nodes():
        if node.op_type == "Pow":
            node.replace_input_with(1, new_const)
    onnx_ir.passes.common.RemoveUnusedNodesPass().call(ir_model)
    return onnx_ir.to_proto(ir_model)


def layernorm_with_negative_epsilon(tempdir):
    model = create_layernorm_model(tempdir)
    ir_model = onnx_ir.from_proto(model)
    new_const = onnx_ir.val(
        "new_epsilon", const_value=onnx_ir.tensor(np.array(-1e-5, dtype=np.float32))
    )
    ir_model.graph.register_initializer(new_const)
    ir_model.graph.node("/layernorm/Add").replace_input_with(1, new_const)
    return onnx_ir.to_proto(ir_model)


def layernorm_with_no_reducemean_axis(tmpdir):
    model = create_layernorm_model(tmpdir)
    ir_model = onnx_ir.from_proto(model)
    reduce_means = [
        node for node in ir_model.graph.all_nodes() if node.op_type == "ReduceMean"
    ]
    for rm in reduce_means:
        rm.attributes.clear()
    return onnx_ir.to_proto(ir_model)


class TestLayerNormFusion:
    """Tests for LayerNormalization pattern fusion."""

    # TODO: Match layernorm without bias/affine transform
    @pytest.mark.parametrize("bias", [True])
    @pytest.mark.parametrize("affine", [True])
    @pytest.mark.parametrize("opset_version", range(13, 17))
    @pytest.mark.parametrize("epsilon", [1e-1, 1e-3, 1e-5])
    def test_fuses_single_layernorm(
        self, tmp_path, opset_version, bias, affine, epsilon
    ):
        layernorm_model = create_layernorm_model(
            tmp_path,
            elementwise_affine=affine,
            bias=bias,
            opset=opset_version,
            epsilon=epsilon,
        )
        model = onnx_ir.from_proto(layernorm_model)
        inputs = make_dummy_input(layernorm_model)

        session = onnxruntime.InferenceSession(layernorm_model.SerializeToString())
        output_pre_fusion = session.run(None, inputs)

        fused_model = fuse_supergroups(model, patterns=["LayerNormalization"])

        model_proto = onnx_ir.to_proto(fused_model)
        layernorms = [
            node
            for node in model_proto.graph.node
            if node.op_type == "LayerNormalization"
        ]
        assert len(layernorms) == 1

        session = onnxruntime.InferenceSession(model_proto.SerializeToString())
        output_post_fusion = session.run(None, inputs)

        assert np.allclose(output_pre_fusion[0], output_post_fusion[0], atol=1e-5)

    def test_fuses_layernorm_with_pow_as_multiply(self, tmp_path):
        layernorm_model = layernorm_with_pow_2_as_multiply(tmp_path)
        model = onnx_ir.from_proto(layernorm_model)
        inputs = make_dummy_input(layernorm_model)

        session = onnxruntime.InferenceSession(layernorm_model.SerializeToString())
        output_pre_fusion = session.run(None, inputs)

        fused_model = fuse_supergroups(model, patterns=["LayerNormalization"])

        model_proto = onnx_ir.to_proto(fused_model)
        layernorms = [
            node
            for node in model_proto.graph.node
            if node.op_type == "LayerNormalization"
        ]
        assert len(layernorms) == 1

        session = onnxruntime.InferenceSession(model_proto.SerializeToString())
        output_post_fusion = session.run(None, inputs)

        assert np.allclose(output_pre_fusion[0], output_post_fusion[0], atol=1e-5)

    @pytest.mark.parametrize(
        "invalid_graph_factory",
        [
            layernorm_with_pow_3,
            layernorm_with_negative_epsilon,
            layernorm_with_no_reducemean_axis,
            lambda tmpdir: create_layernorm_model(tmpdir, elementwise_affine=False),
            lambda tmpdir: create_layernorm_model(tmpdir, bias=False),
            lambda tmpdir: create_layernorm_model(
                tmpdir, bias=False, elementwise_affine=False
            ),
        ],
    )
    def test_rejects_invalid_layernorm_patterns(self, tmp_path, invalid_graph_factory):
        """Test that invalid LayerNorm patterns are rejected during fusion."""
        model_proto = invalid_graph_factory(tmp_path)
        num_nodes_before = len(model_proto.graph.node)
        inputs = make_dummy_input(model_proto)
        session = onnxruntime.InferenceSession(model_proto.SerializeToString())
        output_pre_fusion = session.run(None, inputs)
        model = onnx_ir.from_proto(model_proto)

        # Attempt fusion
        fused_model = fuse_supergroups(model, patterns=["LayerNormalization"])
        model_proto = onnx_ir.to_proto(fused_model)
        assert len(model_proto.graph.node) == num_nodes_before

        # Ensure no LayerNormalization nodes were created
        layernorm_nodes = [
            node
            for node in model_proto.graph.node
            if node.op_type == "LayerNormalization"
        ]
        assert len(layernorm_nodes) == 0

        session = onnxruntime.InferenceSession(model_proto.SerializeToString())
        output_post_fusion = session.run(None, inputs)
        assert np.allclose(output_pre_fusion[0], output_post_fusion[0], atol=1e-5)

    def test_multiple_layernorm_instances(self, tmp_path):
        """Test fusion with multiple LayerNorm instances in the model."""

        class DoubleLayerNormModel(torch.nn.Module):
            def __init__(self):
                super(DoubleLayerNormModel, self).__init__()
                self.ln1 = torch.nn.LayerNorm(128)
                self.ln2 = torch.nn.LayerNorm(128)

            def forward(self, x):
                x = self.ln1(x)
                x = self.ln2(x)
                return x

        model = DoubleLayerNormModel()
        dummy_input = torch.randn(1, 32, 128)
        model_path = os.path.join(tmp_path, "double_layernorm.onnx")
        torch.onnx.export(
            model, dummy_input, model_path, opset_version=13, dynamo=False
        )

        model_proto = onnx.load(model_path)
        inputs = make_dummy_input(model_proto)

        session = onnxruntime.InferenceSession(model_proto.SerializeToString())
        output = session.run(None, inputs)

        layernorm_nodes = [
            node
            for node in model_proto.graph.node
            if node.op_type == "LayerNormalization"
        ]
        assert len(layernorm_nodes) == 0

        model = onnx_ir.from_proto(model_proto)

        # Apply fusion
        fused_model = fuse_supergroups(model, patterns=["LayerNormalization"])
        model_proto = onnx_ir.to_proto(fused_model)

        # Both instances should potentially be fused
        layernorm_nodes = [
            node
            for node in model_proto.graph.node
            if node.op_type == "LayerNormalization"
        ]
        assert len(layernorm_nodes) == 2
        assert len(model_proto.functions) == 2
        session = onnxruntime.InferenceSession(model_proto.SerializeToString())
        output_post_fusion = session.run(None, inputs)
        assert np.allclose(output[0], output_post_fusion[0], atol=1e-5)

    def test_quantsim_with_fused_layernorm(self, tmp_path):
        """Test that QuantSim works with fused LayerNorm nodes."""
        # Get decomposed LayerNorm model
        layernorm_model = create_layernorm_model(tmp_path, opset=13)
        model = onnx_ir.from_proto(layernorm_model)

        # Apply fusion
        fused_model = fuse_supergroups(model, patterns=["LayerNormalization"])
        model_proto = onnx_ir.to_proto(fused_model)

        """
        When: Creating a QuantizationSimModel with the fused LayerNorm model
        Then: 1) LayerNorm weight is detected as a parameter
              2) LayerNorm weight is quantized with parameter type
              3) sim.compute_encodings runs without error
        """
        sim = QuantizationSimModel(
            model_proto,
            param_type="int8",
            activation_type="int16",
            config_file="htp_v73",
        )

        assert "layernorm.weight" in sim.param_names
        assert sim.qc_quantize_op_dict["layernorm.weight"].bitwidth == 8
        sim.compute_encodings([make_dummy_input(model_proto)])


class TestFusion:
    def test_unknown_pattern_raises_error(self, tmp_path):
        """Test that unknown pattern names raise ValueError."""
        model_proto = create_layernorm_model(tmp_path)
        model = onnx_ir.from_proto(model_proto)

        with pytest.raises(ValueError, match="Unknown pattern names"):
            fuse_supergroups(model, patterns=["UnknownPattern"])

    @pytest.mark.parametrize(
        "providers",
        [["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]],
    )
    @pytest.mark.parametrize(
        "model_factory",
        [
            lambda path: create_layernorm_model(path),
            lambda path: models_for_tests.single_residual_model().model,
            lambda path: models_for_tests.squeezenet1_0(path).model,
            lambda path: models_for_tests.simple_relu_model().model,
            lambda path: models_for_tests.standalone_layernorm([1, 32, 64]),
        ],
    )
    def test_fusion_does_not_impact_accuracy(self, tmp_path, model_factory, providers):
        """Test that fusion runs without errors on a variety of models."""
        model_proto = model_factory(tmp_path)

        dummy_input = make_dummy_input(model_proto)
        session = onnxruntime.InferenceSession(
            model_proto.SerializeToString(), providers=providers
        )
        output_pre_fusion = session.run(None, dummy_input)

        # Apply fusion
        model = onnx_ir.from_proto(model_proto)
        fused_model = fuse_supergroups(model, patterns=["LayerNormalization"])

        # Ensure the fused model can be converted back to proto
        model_proto = onnx_ir.to_proto(fused_model)
        session = onnxruntime.InferenceSession(
            model_proto.SerializeToString(), providers=providers
        )
        output_post_fusion = session.run(None, dummy_input)

        assert np.allclose(output_pre_fusion[0], output_post_fusion[0], atol=1e-5)

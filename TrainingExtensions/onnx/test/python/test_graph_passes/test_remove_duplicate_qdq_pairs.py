# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for RemoveDuplicateQDQPairsPass.
"""

import numpy as np
import onnx
import onnx_ir
import onnx_ir.passes.common
import pytest

from aimet_onnx.graph_passes.cleanup.remove_duplicate_qdq_pairs import (
    RemoveDuplicateQDQPairsPass,
    _qdq_nodes_equal,
)


def make_qdq_node(op_type, name, scale, zero_point, attrs=None):
    """Helper to create a Q or DQ node in onnx-ir."""
    attrs = attrs or {}
    return onnx_ir.node(
        op_type,
        inputs=[
            onnx_ir.Value(name=f"{name}_input"),
            onnx_ir.Value(
                name=f"{name}_scale",
                const_value=onnx_ir.Tensor(np.array(scale, dtype=np.float32)),
            ),
            onnx_ir.Value(
                name=f"{name}_zp",
                const_value=onnx_ir.Tensor(np.array(zero_point, dtype=np.uint8)),
            ),
        ],
        num_outputs=1,
        attributes=attrs,
    )


class TestQdqNodesEqual:
    """Tests for _qdq_nodes_equal function."""

    def test_same_params_no_attrs(self):
        """Nodes with same scale/zp and no attributes should be equal."""
        node1 = make_qdq_node("QuantizeLinear", "q1", scale=0.1, zero_point=128)
        node2 = make_qdq_node("QuantizeLinear", "q2", scale=0.1, zero_point=128)
        assert _qdq_nodes_equal(node1, node2)

    def test_different_scale(self):
        """Nodes with different scale should not be equal."""
        node1 = make_qdq_node("QuantizeLinear", "q1", scale=0.1, zero_point=128)
        node2 = make_qdq_node("QuantizeLinear", "q2", scale=0.2, zero_point=128)
        assert not _qdq_nodes_equal(node1, node2)

    def test_different_zero_point(self):
        """Nodes with different zero_point should not be equal."""
        node1 = make_qdq_node("QuantizeLinear", "q1", scale=0.1, zero_point=128)
        node2 = make_qdq_node("QuantizeLinear", "q2", scale=0.1, zero_point=0)
        assert not _qdq_nodes_equal(node1, node2)

    def test_same_axis_attr(self):
        """Nodes with same axis attribute should be equal."""
        node1 = make_qdq_node(
            "QuantizeLinear", "q1", scale=0.1, zero_point=128, attrs={"axis": 1}
        )
        node2 = make_qdq_node(
            "QuantizeLinear", "q2", scale=0.1, zero_point=128, attrs={"axis": 1}
        )
        assert _qdq_nodes_equal(node1, node2)

    def test_different_axis_attr(self):
        """Nodes with different axis attribute should not be equal."""
        node1 = make_qdq_node(
            "QuantizeLinear", "q1", scale=0.1, zero_point=128, attrs={"axis": 0}
        )
        node2 = make_qdq_node(
            "QuantizeLinear", "q2", scale=0.1, zero_point=128, attrs={"axis": 1}
        )
        assert not _qdq_nodes_equal(node1, node2)

    def test_one_has_axis_other_not(self):
        """Node with axis vs node without axis should not be equal."""
        node1 = make_qdq_node(
            "QuantizeLinear", "q1", scale=0.1, zero_point=128, attrs={"axis": 1}
        )
        node2 = make_qdq_node("QuantizeLinear", "q2", scale=0.1, zero_point=128)
        assert not _qdq_nodes_equal(node1, node2)

    def test_same_saturate_attr(self):
        """Nodes with same saturate attribute should be equal."""
        node1 = make_qdq_node(
            "QuantizeLinear", "q1", scale=0.1, zero_point=128, attrs={"saturate": 1}
        )
        node2 = make_qdq_node(
            "QuantizeLinear", "q2", scale=0.1, zero_point=128, attrs={"saturate": 1}
        )
        assert _qdq_nodes_equal(node1, node2)

    def test_different_saturate_attr(self):
        """Nodes with different saturate attribute should not be equal."""
        node1 = make_qdq_node(
            "QuantizeLinear", "q1", scale=0.1, zero_point=128, attrs={"saturate": 0}
        )
        node2 = make_qdq_node(
            "QuantizeLinear", "q2", scale=0.1, zero_point=128, attrs={"saturate": 1}
        )
        assert not _qdq_nodes_equal(node1, node2)

    def test_multiple_attrs_same(self):
        """Nodes with multiple same attributes should be equal."""
        node1 = make_qdq_node(
            "QuantizeLinear",
            "q1",
            scale=0.1,
            zero_point=128,
            attrs={"axis": 1, "saturate": 1},
        )
        node2 = make_qdq_node(
            "QuantizeLinear",
            "q2",
            scale=0.1,
            zero_point=128,
            attrs={"axis": 1, "saturate": 1},
        )
        assert _qdq_nodes_equal(node1, node2)

    def test_multiple_attrs_different(self):
        """Nodes with multiple attributes where one differs should not be equal."""
        node1 = make_qdq_node(
            "QuantizeLinear",
            "q1",
            scale=0.1,
            zero_point=128,
            attrs={"axis": 1, "saturate": 0},
        )
        node2 = make_qdq_node(
            "QuantizeLinear",
            "q2",
            scale=0.1,
            zero_point=128,
            attrs={"axis": 1, "saturate": 1},
        )
        assert not _qdq_nodes_equal(node1, node2)


def _make_back_to_back_qdq_model(
    pair1_scale=0.1,
    pair1_zp=128,
    pair2_scale=0.1,
    pair2_zp=128,
    pair1_attrs=None,
    pair2_attrs=None,
):
    """
    Create an ONNX model with back-to-back Q->DQ->Q->DQ pattern.

    (input) -> Q1 -> DQ1 -> Q2 -> DQ2 -> (output)
    """
    pair1_attrs = pair1_attrs or {}
    pair2_attrs = pair2_attrs or {}

    model = onnx.helper.make_model(
        ir_version=10,
        opset_imports=[onnx.helper.make_operatorsetid("", 21)],
        graph=onnx.helper.make_graph(
            name="back_to_back_qdq",
            inputs=[
                onnx.helper.make_tensor_value_info(
                    "input", onnx.TensorProto.FLOAT, shape=[1, 3, 4, 4]
                )
            ],
            outputs=[
                onnx.helper.make_tensor_value_info(
                    "output", onnx.TensorProto.FLOAT, shape=[1, 3, 4, 4]
                )
            ],
            nodes=[
                onnx.helper.make_node(
                    "QuantizeLinear",
                    inputs=["input", "scale1", "zp1"],
                    outputs=["q1_out"],
                    name="q1",
                    **pair1_attrs,
                ),
                onnx.helper.make_node(
                    "DequantizeLinear",
                    inputs=["q1_out", "scale1", "zp1"],
                    outputs=["dq1_out"],
                    name="dq1",
                    **pair1_attrs,
                ),
                onnx.helper.make_node(
                    "QuantizeLinear",
                    inputs=["dq1_out", "scale2", "zp2"],
                    outputs=["q2_out"],
                    name="q2",
                    **pair2_attrs,
                ),
                onnx.helper.make_node(
                    "DequantizeLinear",
                    inputs=["q2_out", "scale2", "zp2"],
                    outputs=["output"],
                    name="dq2",
                    **pair2_attrs,
                ),
            ],
            initializer=[
                onnx.helper.make_tensor(
                    "scale1", onnx.TensorProto.FLOAT, [], [pair1_scale]
                ),
                onnx.helper.make_tensor("zp1", onnx.TensorProto.UINT8, [], [pair1_zp]),
                onnx.helper.make_tensor(
                    "scale2", onnx.TensorProto.FLOAT, [], [pair2_scale]
                ),
                onnx.helper.make_tensor("zp2", onnx.TensorProto.UINT8, [], [pair2_zp]),
            ],
        ),
    )
    return model


class TestRemoveDuplicateQDQPairsPass:
    """Tests for RemoveDuplicateQDQPairsPass."""

    def _count_nodes(self, model, op_type):
        return sum(1 for n in model.graph.node if n.op_type == op_type)

    def _run_passes(self, model):
        """Run RemoveDuplicateQDQPairsPass followed by RemoveUnusedNodesPass."""
        ir_model = onnx_ir.from_proto(model)
        result = RemoveDuplicateQDQPairsPass().call(ir_model)
        # RemoveUnusedNodesPass removes dead code created by the rewiring
        onnx_ir.passes.common.RemoveUnusedNodesPass().call(ir_model)
        return result, onnx_ir.to_proto(ir_model)

    def test_same_params_removes_duplicate(self):
        """Back-to-back QDQ with same params should be optimized."""
        model = _make_back_to_back_qdq_model()
        assert self._count_nodes(model, "QuantizeLinear") == 2
        assert self._count_nodes(model, "DequantizeLinear") == 2

        result, optimized = self._run_passes(model)

        assert result.modified
        assert self._count_nodes(optimized, "QuantizeLinear") == 1
        assert self._count_nodes(optimized, "DequantizeLinear") == 1

    def test_different_scale_no_optimization(self):
        """Back-to-back QDQ with different scale should not be optimized."""
        model = _make_back_to_back_qdq_model(pair1_scale=0.1, pair2_scale=0.2)

        result, optimized = self._run_passes(model)

        assert not result.modified
        assert self._count_nodes(optimized, "QuantizeLinear") == 2
        assert self._count_nodes(optimized, "DequantizeLinear") == 2

    def test_different_zero_point_no_optimization(self):
        """Back-to-back QDQ with different zero_point should not be optimized."""
        model = _make_back_to_back_qdq_model(pair1_zp=128, pair2_zp=0)

        result, _ = self._run_passes(model)

        assert not result.modified

    @pytest.mark.parametrize(
        "pair1_attrs, pair2_attrs, expect_optimization",
        [
            # Same attributes - should optimize
            ({}, {}, True),
            ({"saturate": 1}, {"saturate": 1}, True),
            # Different attributes - should not optimize
            ({"saturate": 0}, {"saturate": 1}, False),
            ({"saturate": 1}, {}, False),
            ({}, {"saturate": 1}, False),
        ],
    )
    def test_attribute_comparison(self, pair1_attrs, pair2_attrs, expect_optimization):
        """Test that attribute differences prevent optimization."""
        model = _make_back_to_back_qdq_model(
            pair1_attrs=pair1_attrs, pair2_attrs=pair2_attrs
        )

        result, _ = self._run_passes(model)

        assert result.modified == expect_optimization

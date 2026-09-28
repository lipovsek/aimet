# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import itertools
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from unittest.mock import MagicMock, patch
import torch
from aimet_onnx.common.connected_graph.connectedgraph_utils import (
    get_all_input_ops,
    get_all_ops_with_constant_inputs,
)
from aimet_onnx.meta.connectedgraph import (
    ConnectedGraph,
    CONSTANT_TYPE,
    _get_matmul_add_bias_idx,
)
from .models import models_for_tests

DIM = 4
TRIP = 3


def _ti(name, shape, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


def _model(graph):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


def _flat_model():
    """No control flow at all: the shape every existing model in the suite has."""
    weight = numpy_helper.from_array(np.eye(DIM, dtype=np.float32), name="w")
    return _model(
        helper.make_graph(
            [
                helper.make_node("MatMul", ["x", "w"], ["mm"], name="the_matmul"),
                helper.make_node("Relu", ["mm"], ["y"], name="the_relu"),
            ],
            "flat",
            inputs=[_ti("x", [DIM, DIM])],
            outputs=[_ti("y", [DIM, DIM])],
            initializer=[weight],
        )
    )


def _scan_model(num_carries: int = 1, with_capture: bool = False):
    """A Scan with num_carries carries and one scanned input."""
    weight = numpy_helper.from_array(np.eye(DIM, dtype=np.float32), name="bw")

    nodes, body_inputs, body_outputs = [], [], []
    for k in range(num_carries):
        body_inputs.append(_ti(f"b_c{k}", [DIM, DIM]))
        nodes.append(
            helper.make_node("Add", [f"b_c{k}", "b_x"], [f"b_co{k}"], name=f"b_add{k}")
        )
        body_outputs.append(_ti(f"b_co{k}", [DIM, DIM]))
    body_inputs.append(_ti("b_x", [DIM, DIM]))

    last = f"b_co{num_carries - 1}" if num_carries else "b_x"
    if with_capture:
        # A weight owned by the enclosing graph but read only inside the body.
        nodes.append(
            helper.make_node("MatMul", [last, "bw"], ["b_scaled"], name="b_matmul")
        )
        last = "b_scaled"
    nodes.append(helper.make_node("Identity", [last], ["b_y"], name="b_y_id"))
    body_outputs.append(_ti("b_y", [DIM, DIM]))

    body = helper.make_graph(nodes, "scan_body", body_inputs, body_outputs)

    scan_inputs = [f"init{k}" for k in range(num_carries)] + ["xs"]
    scan_outputs = [f"final{k}" for k in range(num_carries)] + ["ys"]
    graph = helper.make_graph(
        [
            helper.make_node(
                "Scan",
                scan_inputs,
                scan_outputs,
                name="the_scan",
                body=body,
                num_scan_inputs=1,
            )
        ],
        "outer",
        inputs=[_ti(f"init{k}", [DIM, DIM]) for k in range(num_carries)]
        + [_ti("xs", [TRIP, DIM, DIM])],
        outputs=[_ti(f"final{k}", [DIM, DIM]) for k in range(num_carries)]
        + [_ti("ys", [TRIP, DIM, DIM])],
        initializer=[weight] if with_capture else [],
    )
    return _model(graph)


def _captured_activation_scan_model():
    """A Scan whose body reads a top-level activation by name."""
    body = helper.make_graph(
        [
            helper.make_node("Relu", ["b_x"], ["b_pre"], name="b_first"),
            helper.make_node("Add", ["b_pre", "captured"], ["b_co"], name="b_second"),
        ],
        "scan_body",
        inputs=[_ti("b_c", [DIM, DIM]), _ti("b_x", [DIM, DIM])],
        outputs=[_ti("b_co", [DIM, DIM])],
    )
    return _model(
        helper.make_graph(
            [
                helper.make_node("Relu", ["x"], ["captured"], name="pre"),
                helper.make_node("Relu", ["captured"], ["state_in"], name="pre2"),
                helper.make_node(
                    "Scan",
                    ["state_in", "xs"],
                    ["state_out"],
                    name="the_scan",
                    body=body,
                    num_scan_inputs=1,
                ),
            ],
            "outer",
            inputs=[_ti("x", [DIM, DIM]), _ti("xs", [TRIP, DIM, DIM])],
            outputs=[_ti("state_out", [DIM, DIM])],
        )
    )


def _wrapped_scan_model():
    """A Scan with top-level compute on both sides of it."""
    pre = numpy_helper.from_array(np.eye(DIM, dtype=np.float32), name="w_pre")
    post = numpy_helper.from_array(np.eye(DIM, dtype=np.float32), name="w_post")
    body = helper.make_graph(
        [
            helper.make_node("Add", ["b_c", "b_x"], ["b_sum"], name="b_add"),
            helper.make_node("Identity", ["b_sum"], ["b_y"], name="b_y_id"),
        ],
        "scan_body",
        inputs=[_ti("b_c", [DIM, DIM]), _ti("b_x", [DIM, DIM])],
        outputs=[_ti("b_sum", [DIM, DIM]), _ti("b_y", [DIM, DIM])],
    )
    return _model(
        helper.make_graph(
            [
                helper.make_node("MatMul", ["x", "w_pre"], ["state_in"], name="pre"),
                helper.make_node(
                    "Scan",
                    ["state_in", "xs"],
                    ["state_out", "ys"],
                    name="the_scan",
                    body=body,
                    num_scan_inputs=1,
                ),
                helper.make_node("MatMul", ["state_out", "w_post"], ["y"], name="post"),
            ],
            "outer",
            inputs=[_ti("x", [DIM, DIM]), _ti("xs", [TRIP, DIM, DIM])],
            outputs=[_ti("y", [DIM, DIM]), _ti("ys", [TRIP, DIM, DIM])],
            initializer=[pre, post],
        )
    )


def _nested_scan_model():
    """A Scan whose body contains another Scan."""
    inner_body = helper.make_graph(
        [helper.make_node("Add", ["i_c", "i_x"], ["i_co"], name="inner_add")],
        "inner_body",
        inputs=[_ti("i_c", [DIM, DIM]), _ti("i_x", [DIM, DIM])],
        outputs=[_ti("i_co", [DIM, DIM])],
    )
    outer_body = helper.make_graph(
        [
            helper.make_node(
                "Scan",
                ["o_c", "o_xs"],
                ["o_inner_out"],
                name="inner_scan",
                body=inner_body,
                num_scan_inputs=1,
            ),
            helper.make_node("Relu", ["o_inner_out"], ["o_co"], name="outer_relu"),
        ],
        "outer_body",
        inputs=[_ti("o_c", [DIM, DIM]), _ti("o_xs", [TRIP, DIM, DIM])],
        outputs=[_ti("o_co", [DIM, DIM])],
    )
    return _model(
        helper.make_graph(
            [
                helper.make_node(
                    "Scan",
                    ["init", "xss"],
                    ["final"],
                    name="outer_scan",
                    body=outer_body,
                    num_scan_inputs=1,
                )
            ],
            "top",
            inputs=[_ti("init", [DIM, DIM]), _ti("xss", [TRIP, TRIP, DIM, DIM])],
            outputs=[_ti("final", [DIM, DIM])],
        )
    )


def _sibling_scans_model(collide: str = ""):
    """Two Scan nodes side by side, so their bodies are sibling scopes."""

    def scan(tag):
        inner = "t" if collide == "tensor" else f"{tag}_t"
        node_name = "add" if collide == "node" else f"{tag}_add"
        body = helper.make_graph(
            [
                helper.make_node(
                    "Add", [f"{tag}_c", f"{tag}_x"], [inner], name=node_name
                )
            ],
            f"{tag}_body",
            inputs=[_ti(f"{tag}_c", [DIM, DIM]), _ti(f"{tag}_x", [DIM, DIM])],
            outputs=[_ti(inner, [DIM, DIM])],
        )
        return helper.make_node(
            "Scan",
            [f"init_{tag}", "xs"],
            [f"final_{tag}"],
            name=f"scan_{tag}",
            body=body,
            num_scan_inputs=1,
        )

    return _model(
        helper.make_graph(
            [scan("a"), scan("b")],
            "outer",
            inputs=[
                _ti("init_a", [DIM, DIM]),
                _ti("init_b", [DIM, DIM]),
                _ti("xs", [TRIP, DIM, DIM]),
            ],
            outputs=[_ti("final_a", [DIM, DIM]), _ti("final_b", [DIM, DIM])],
        )
    )


def _if_model():
    """If model."""

    def branch(name, op_type):
        return helper.make_graph(
            [helper.make_node(op_type, ["x"], [f"{name}_o"], name=f"{name}_op")],
            name,
            inputs=[],
            outputs=[_ti(f"{name}_o", [DIM])],
        )

    return _model(
        helper.make_graph(
            [
                helper.make_node(
                    "If",
                    ["cond"],
                    ["y"],
                    name="the_if",
                    then_branch=branch("thn", "Relu"),
                    else_branch=branch("els", "Neg"),
                )
            ],
            "outer",
            inputs=[_ti("x", [DIM]), _ti("cond", [1], TensorProto.BOOL)],
            outputs=[_ti("y", [DIM])],
        )
    )


def _all_nodes(model):
    """Every node in the model, at every depth, counting only Scan bodies."""

    def walk(graph):
        for node in graph.node:
            yield node
            if node.op_type != "Scan":
                continue
            for attr in node.attribute:
                if attr.name == "body":
                    yield from walk(attr.g)

    return list(walk(model.graph))


class TestConnectedGraph:
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
    def test_model_representation(self, model):
        """
        Given: A ConnectedGraph constructed for a given model
        Then: 1) All non-constant nodes should be represented as ops in the CG
              2) All tensors should be represented as products in the CG
        """
        nodes = {node.name for node in model.graph.node if node.op_type != "Constant"}
        tensors = {
            tensor
            for node in model.graph.node
            for tensor in itertools.chain(node.input, node.output)
            if tensor
        }
        cg = ConnectedGraph(model)
        ops = cg.get_all_ops()
        assert ops.keys() == nodes
        products = cg.get_all_products()
        assert products.keys() == tensors

        """
        All ops should appear in cg.ordered_ops
        """
        for op in ops.values():
            assert op in cg.ordered_ops

        for _, product in products.items():
            for node in model.graph.node:
                if node.op_type == "Constant":
                    continue
                """
                When: A tensor is the output of a node
                Then: The corresponding Op should be the product's producer
                """
                if product.name in node.output:
                    assert node == product.producer.get_module()

                """
                When: A tensor is the input of a node
                Then: The corresponding Op should appear in the product's consumers
                """
                if product.name in node.input:
                    assert node in [op.get_module() for op in product.consumers]

            """
            When: A tensor is a model input
            Then: The corresponding product should have product.is_model_input set to True
            """
            if product.name in {t.name for t in model.graph.input}:
                assert product.is_model_input
                assert not product.producer
            else:
                assert not product.is_model_input

            """
            When: A tensor has no producer
            Then: The tensor is either a model input, constant, or parameter
            """
            if not product.producer:
                assert product.is_model_input or product.is_const or product.is_parm
            else:
                assert not (
                    product.is_model_input or product.is_const or product.is_parm
                )

        for op_name, op in ops.items():
            node = op.get_module()
            """
            When: A tensor is input idx of node
            Then: The tensor's product should be input idx of the corresponding op
            """
            for idx, tensor in enumerate(node.input):
                assert op.inputs[idx] is products[tensor]

            """
            When: A tensor is output[0] of a node
            Then: The tensor's product should be the corresponding op's output
            """
            for idx, output in enumerate(op.outputs):
                assert output is products[node.output[idx]]

    def test_single_residual_model(self):
        model = models_for_tests.single_residual_model()
        conn_graph = ConnectedGraph(model)
        operator_names = {
            node.name for node in model.nodes() if node.op_type not in CONSTANT_TYPE
        }
        assert operator_names == conn_graph.get_all_ops().keys()

        model_weights = {
            node.input[1] for node in model.graph().node if node.op_type == "Conv"
        }
        products = conn_graph.get_all_products()

        for weight in model_weights:
            assert products[weight].is_parm

        input_ops = get_all_input_ops(conn_graph)
        assert len(input_ops) == 1
        for op in conn_graph.ordered_ops:
            if op.type == "Gemm":
                assert op.transposed_params

    def test_multi_inputs_model(self):
        model = models_for_tests.multi_input_model()
        conn_graph = ConnectedGraph(model)
        input_ops = get_all_input_ops(conn_graph)
        assert len(input_ops) == 2

    def test_concat_model(self):
        model = models_for_tests.concat_model()
        conn_graph = ConnectedGraph(model)
        ops = conn_graph.get_all_ops()
        assert len(ops["/Concat"].inputs) == 3

    def test_hierarchical_model(self):
        model = models_for_tests.hierarchical_model()
        conn_graph = ConnectedGraph(model)
        ordered_ops = conn_graph.ordered_ops
        name_to_index = {}
        for index, op in enumerate(ordered_ops):
            name_to_index[op.name] = index

        # Check in the graph that if A & B are connected and A comes before B in the graph then that should be the case
        # in ordered graphs as well
        assert name_to_index["/conv1/conv/Conv"] < name_to_index["/nm1/tm1/Reshape"]
        assert (
            name_to_index["/sq/seq_list/seq_list.0/Conv"]
            < name_to_index["/sq/seq_list/seq_list.5/Conv"]
        )
        assert name_to_index["/conv2/conv/Conv"] < name_to_index["/nm2/tm1/conv3/Conv"]

    def test_matmul_layer_param_creation(self):
        torch.manual_seed(10)
        torch_model = models_for_tests.BNBeforeFlattenLinear()

        torch_model.eval()

        input_shape = (2, 10, 24, 24)

        model = models_for_tests._convert_to_onnx_no_fold(
            torch_model, torch.randn(input_shape)
        )

        cg = ConnectedGraph(model)
        for op in cg.ordered_ops:
            if op.type == "MatMul":
                assert "fc2.weight" in op.parameters
                break
        else:
            assert False

    def test_constant_elementwise_inputs(self):
        """Test that constant inputs to elementwise ops are identified correctly"""
        model = models_for_tests.elementwise_op_model()
        cg = ConnectedGraph(model)

        assert len(get_all_ops_with_constant_inputs(cg)) == 2
        for product in cg.ordered_ops[0].inputs:
            if product.name == "input":
                assert not product.is_const
                assert product.is_model_input
            else:
                assert product.is_const

        for product in cg.ordered_ops[1].inputs:
            assert not product.is_model_input
            if product.producer == cg.ordered_ops[0]:
                assert not product.is_const
            else:
                assert product.is_const

    def test_instance_norm_model(self):
        model = models_for_tests.instance_norm_model()
        cg = ConnectedGraph(model)
        assert cg.ordered_ops[-2].type == "InstanceNormalization"

    def test_layer_norm_model(self):
        model = models_for_tests.layernorm_model()
        cg = ConnectedGraph(model)
        layernorm_cg_op = cg.ordered_ops[-1]
        assert layernorm_cg_op.type == "LayerNormalization"
        assert ["layernorm.scale", "layernorm.bias"] == list(
            layernorm_cg_op.parameters.keys()
        )

    def test_malformed_model(self):
        model = models_for_tests.layernorm_model()
        model.graph.node.pop(1)  # Remove constant node
        with pytest.raises(RuntimeError):
            cg = ConnectedGraph(model)

    def test_get_matmul_add_bias_idx(self):
        """Identify bias index for MatMul_1 -> MatMul_2 -> Add pattern"""
        bias_param = MagicMock()
        bias_param.dims = [64]

        # Only patch within the scope of the test
        # Use fully qualified path to `ParamUtils.get_param_by_name`
        with patch(
            "aimet_onnx.utils.ParamUtils.get_param_by_name",
            return_value=bias_param,
        ):
            # Create Products
            matmul1_output = MagicMock()
            matmul2_output = MagicMock()
            bias_input = MagicMock()
            bias_input.name = "bias"

            # Create Add op
            add_op = MagicMock()
            add_op.type = "Add"
            add_op.inputs = [matmul2_output, bias_input]

            # Create MatMul_2 op
            matmul2_op = MagicMock()
            matmul2_op.type = "MatMul"
            matmul2_op.inputs = [matmul1_output]
            matmul2_op.outputs = [matmul2_output]
            matmul2_output.producer = matmul2_op
            matmul2_output.consumers = [add_op]

            # Create MatMul_1 op
            matmul1_op = MagicMock()
            matmul1_op.type = "MatMul"
            matmul1_op.outputs = [matmul1_output]
            matmul1_output.producer = matmul1_op
            matmul1_output.consumers = [matmul2_op]

            # Test MatMul_1 → MatMul_2 → Add
            bias_idx_1 = _get_matmul_add_bias_idx(matmul1_op, MagicMock())
            bias_idx_2 = _get_matmul_add_bias_idx(matmul2_op, MagicMock())

            # MatMul_1 should not be associated with the Add
            assert not bias_idx_1

            # MatMul_2 should be correctly associated with the Add
            assert bias_idx_2 == 1  # bias is second input to Add

    def test_ops_and_products_match_the_nodes_and_tensors(self):
        """A model with no control flow"""
        model = _flat_model()
        cg = ConnectedGraph(model)

        """
        When: A ConnectedGraph is constructed for it
        Then: 1) Its ops are exactly the model's nodes
              2) They are ordered as the model declares them
              3) No op holds a body, there being no control flow to hold one
        """
        assert set(cg.get_all_ops()) == {node.name for node in model.graph.node}
        assert [op.name for op in cg.ordered_ops] == ["the_matmul", "the_relu"]
        assert not any(op.subgraph_ops for op in cg.get_all_ops().values())


class TestSubgraphOpsAreRepresented:
    def test_body_nodes_become_ops_held_by_the_op_running_them(self):
        """A model whose Scan body holds two nodes (Add, Identity)"""
        model = _scan_model()
        cg = ConnectedGraph(model)
        print(f"All ops: {cg.get_all_ops()}, Ordered ops: {cg.ordered_ops}")

        """
        When: A ConnectedGraph is constructed for it
        Then: The body's nodes are ops alongside the Scan rather than omitted, and the Scan is
              what says they belong to it -- a body is reached through the op running it
        """
        assert set(cg.get_all_ops()) == {node.name for node in _all_nodes(model)}
        scan = cg.get_all_ops()["the_scan"]
        assert sorted(op.name for op in scan.subgraph_ops) == ["b_add0", "b_y_id"]

    def test_a_body_is_held_by_its_own_op_at_every_depth(self):
        """A model with a Scan nested inside another Scan's body"""
        cg = ConnectedGraph(_nested_scan_model())
        ops = cg.get_all_ops()

        """
        When: A ConnectedGraph is constructed for it
        Then: Each body belongs to the op running it and to no other, so a nested body is
              reached one op at a time rather than by naming the graph it came from
        """
        assert [op.name for op in ops["outer_scan"].subgraph_ops] == [
            "inner_scan",
            "outer_relu",
        ]
        assert [op.name for op in ops["inner_scan"].subgraph_ops] == ["inner_add"]
        assert ops["outer_relu"].subgraph_ops == []

    def test_connectivity_inside_a_body(self):
        """A model with compute either side of a Scan"""
        cg = ConnectedGraph(_wrapped_scan_model())

        add = cg.get_all_ops()["b_add"]
        identity = cg.get_all_ops()["b_y_id"]
        print(f"add: {add}, Identity: {identity}")

        """
        When: A ConnectedGraph is constructed for it
        Then: A body op is joined to the other ops of its body by the product the body itself
              declares
        """
        assert "b_y_id" in [op.name for op in add.output_ops]

        edge = cg.get_all_products()["b_sum"]
        assert add.output is edge
        assert identity.inputs == [edge]
        assert edge.producer is add
        assert edge.consumers == [identity]

    def test_a_carry_stays_a_separate_product_from_the_one_outside_the_body(self):
        """A Scan whose carry-in is produced by a top-level op"""
        cg = ConnectedGraph(_wrapped_scan_model())
        products = cg.get_all_products()
        print(f"products: {products}")

        """
        When: A ConnectedGraph is constructed for it
        Then: The boundaries between Scan input(s) and body input(s) are disjoint
        """
        assert products["b_c"] is not products["state_in"]
        assert products["state_in"].producer.name == "pre"
        assert products["b_c"].producer is None

        """
        Then: And so are the boundaries between body output(s) and Scan output(s).
        """
        assert products["b_sum"] is not products["state_out"]
        assert products["b_y"] is not products["ys"]
        assert products["b_sum"].producer.name == "b_add"
        assert products["b_y"].producer.name == "b_y_id"
        assert products["state_out"].producer.name == "the_scan"
        assert products["ys"].producer.name == "the_scan"
        assert [c.name for c in products["state_out"].consumers] == ["post"]

    def test_a_scanned_input_keeps_the_shape_the_body_sees(self):
        """A Scan with a scanned input, which it slices along its leading axis"""
        cg = ConnectedGraph(_scan_model())
        products = cg.get_all_products()

        """
        When: A ConnectedGraph is constructed for it
        Then: The slice stays a separate product a rank below the stacked tensor
        """
        assert products["xs"].shape == [TRIP, DIM, DIM]
        assert products["b_x"].shape == [DIM, DIM]
        assert products["b_x"] is not products["xs"]

    def test_a_body_input_is_not_a_model_input(self):
        """A Scan whose body declares a scanned input, fed by the Scan rather than the caller"""
        cg = ConnectedGraph(_scan_model(num_carries=0))
        products = cg.get_all_products()

        """
        When: A ConnectedGraph is constructed for it
        Then: Only the top-level graph's inputs are model inputs, and only the Scan is an
              input op.
        """
        assert products["xs"].is_model_input
        assert not products["b_x"].is_model_input
        assert [op.name for op in get_all_input_ops(cg)] == ["the_scan"]

    def test_a_capture_is_a_parameter_of_the_body_op_that_reads_it(self):
        """A weight defined outside a Scan body but consumed by a MatMul inside it"""

        cg = ConnectedGraph(_scan_model(with_capture=True))

        """
        When: A ConnectedGraph is constructed for it
        Then: It resolves as that op's weight, because parameter lookup searches the
              enclosing graph's initializers and that is where an outer-scope capture lives
        """
        matmul = cg.get_all_ops()["b_matmul"]
        assert {
            name for name, (_, kind) in matmul.parameters.items() if kind == "weight"
        } == {"bw"}
        assert cg.get_all_products()["bw"].is_parm


class TestOrderedOps:
    @pytest.mark.parametrize(
        "model",
        (
            _flat_model(),
            _scan_model(num_carries=0),
            _scan_model(num_carries=1),
            _scan_model(num_carries=2),
            _scan_model(with_capture=True),
            _captured_activation_scan_model(),
            _wrapped_scan_model(),
            _nested_scan_model(),
            _sibling_scans_model(),
            _if_model(),
        ),
    )
    def test_every_op_is_ordered(self, model):
        cg = ConnectedGraph(model)

        """
        When: A ConnectedGraph is constructed for it
        Then: No op is dropped -- get_ordered_ops walks forward from the ops with no
              producer and silently omits whatever it cannot reach, so a missing boundary
              link shows up here as a short list rather than as an error
        """
        print(f"All ops: {cg.get_all_ops()} and ordered ops: {cg.ordered_ops}")
        assert {op.name for op in cg.ordered_ops} == set(cg.get_all_ops())

    def test_a_body_is_placed_by_the_op_that_runs_it_not_by_any_edge(self):
        """A model with compute either side of a Scan"""
        cg = ConnectedGraph(_wrapped_scan_model())
        order = [op.name for op in cg.ordered_ops]
        body_op = cg.get_all_ops()["b_add"]

        """
        When: A ConnectedGraph is constructed for it
        Then: The body is sorted ahead of the Scan, and the Scan ahead of what reads it
        """
        assert order.index("pre") < order.index("b_add")
        assert order.index("b_add") < order.index("the_scan")
        assert order.index("the_scan") < order.index("post")

        """
        Then: And that placement rests on containment alone, because the body has no incoming
              edge to rest on.
        """
        assert [product.name for product in body_op.inputs] == ["b_c", "b_x"]
        assert all(product.producer is None for product in body_op.inputs)
        assert not body_op.input_ops

        """
        Then: So only the top-level op starts the graph.
        """
        print(f"startin ops: {cg.starting_ops}")
        assert [op.name for op in cg.starting_ops] == ["pre"]

    def test_a_capture_stays_tied_to_the_ops_reading_it(self):
        """A Scan whose body reads a top-level activation by name"""
        cg = ConnectedGraph(_captured_activation_scan_model())

        """
        When: A ConnectedGraph is constructed for it
        Then: The capture is one shared product, held by the ops reading it alone -- the op
              running the body reads nothing extra for it
        """
        product = cg.get_product("captured")
        assert sorted(op.name for op in product.consumers) == ["b_second", "pre2"]
        assert product not in cg.get_all_ops()["the_scan"].inputs

    def test_a_capture_does_not_unseat_the_op_running_the_body(self):
        """A Scan reading model inputs, whose body captures an outer-scope weight"""
        cg = ConnectedGraph(_scan_model(with_capture=True))

        """
        When: A ConnectedGraph is constructed for it
        Then: The Scan still starts the graph. Its own operands have no producer, so callers
              walking forward from here -- batch norm folding does -- would never reach it, nor
              anything downstream of it, if the capture were allowed to unseat it
        """
        assert [op.name for op in cg.starting_ops] == ["the_scan"]


class TestSubgraphOps:
    @pytest.mark.parametrize(
        "model",
        (
            _flat_model(),
            _scan_model(num_carries=1),
            _scan_model(num_carries=2),
            _scan_model(with_capture=True),
            _captured_activation_scan_model(),
            _wrapped_scan_model(),
            _nested_scan_model(),
            _sibling_scans_model(),
        ),
    )
    def test_the_graph_holds_together_whatever_the_model(self, model):
        cg = ConnectedGraph(model)
        position = {op.name: index for index, op in enumerate(cg.ordered_ops)}

        """
        When: A ConnectedGraph is constructed for it
        Then: Every op holds one input product per entry in node.input.
        """
        for op in cg.ordered_ops:
            node = op.get_module()
            print(f"CG op inputs: {op.inputs}, node inputs: {node.input}")
            assert len(op.inputs) == len(node.input), op.name

        """
        Then: Every consumer of a product holds that product among its inputs
        """
        for product in cg.get_all_products().values():
            for consumer in product.consumers:
                assert product in consumer.inputs

        """
        Then: Every op a control-flow op contains comes before it
        """
        for op in cg.get_all_ops().values():
            for body_op in op.subgraph_ops:
                assert position[body_op.name] < position[op.name]

        """
        Then: And every producer precedes its consumer
        """
        for op in cg.get_all_ops().values():
            for producer in op.input_ops:
                assert position[producer.name] < position[op.name]

    def test_a_scan_owns_its_body_ops_without_a_tensor_edge_to_them(self):
        """A model with compute either side of a Scan"""
        cg = ConnectedGraph(_wrapped_scan_model())
        scan = cg.get_all_ops()["the_scan"]

        """
        When: A ConnectedGraph is constructed for it
        Then: The Scan names the ops of its body, ordered within the body, and only the
              control-flow op names any
        """
        assert [op.name for op in scan.subgraph_ops] == ["b_add", "b_y_id"]
        owners = {name for name, op in cg.get_all_ops().items() if op.subgraph_ops}
        assert owners == {"the_scan"}

        """
        Then: And that containment is the only relation between them -- no body tensor reaches
              the Scan from either side. The Scan's operands are still just the tensors
              node.input lists, so it has no body op among its producers, and a body output
              lists no consumer outside the body.
        """
        assert [product.name for product in scan.inputs] == ["state_in", "xs"]
        assert [op.name for op in scan.input_ops] == ["pre"]
        assert [op.name for op in cg.get_product("b_sum").consumers] == ["b_y_id"]
        assert cg.get_product("b_y").consumers == []

    def test_a_body_is_ordered_within_itself_not_from_where_a_capture_enters(self):
        """
        Given: A Scan whose body reads a top-level activation by name, and does so in its
               *second* op, so the op the capture reaches is not the body's first
        """
        model = _captured_activation_scan_model()

        """
        When: A ConnectedGraph is constructed for it
        Then: The body comes out in its own order. A capture is one shared product
        """
        ordered = [op.name for op in ConnectedGraph(model).ordered_ops]
        print(f"ordered ops: {ordered}")
        assert ordered == ["pre", "pre2", "b_first", "b_second", "the_scan"]

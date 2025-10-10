# /usr/bin/env python
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

import pytest
from unittest.mock import MagicMock
import torch
import copy
import json
import numpy as np
import os
import itertools
from onnx.utils import Extractor
import logging

from aimet_common import libquant_info
from aimet_common.utils import AimetLogger
from aimet_onnx import apply_seq_mse
from aimet_onnx.sequential_mse.dependency_graph import (
    DependencyGraph,
    SUPPORTED_MODULES,
)
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.sequential_mse.seq_mse import SeqMseParams, _add_value_info
from aimet_onnx.sequential_mse.seq_mse import SequentialMse
from aimet_common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.utils import make_dummy_input
from aimet_onnx.qc_quantize_op import (
    QcQuantizeOp,
    OpMode,
    TensorQuantizerParams,
)

from .models.test_models import single_linear_layer_model
from .models.test_models import single_conv_layer_model
from .models.test_models import model_with_split
from .models.test_models import single_residual_model
from .models import models_for_tests
from .models.test_models_onnx import model_with_multiple_inputs
from .models.test_models_onnx import model_with_multiple_outputs


AimetLogger.set_level_for_all_areas(logging.DEBUG)


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(0)
    torch.manual_seed(42)


def _get_config_file(
    is_symmetric: bool, strict_symmetric: bool, unsigned_symmetric: bool, pcq: bool
) -> str:
    """Temporary fix until the config file can be read from beq_config directory"""

    def get_bool_str(in_bool: bool) -> str:
        if in_bool:
            return "True"
        else:
            return "False"

    beq_per_channel_config = {
        "defaults": {
            "ops": {
                "is_output_quantized": "True",
                "is_symmetric": get_bool_str(is_symmetric),
            },
            "params": {
                "is_quantized": "True",
                "is_symmetric": get_bool_str(is_symmetric),
            },
            "strict_symmetric": get_bool_str(strict_symmetric),
            "unsigned_symmetric": get_bool_str(unsigned_symmetric),
            "per_channel_quantization": get_bool_str(pcq),
        },
        "params": {"bias": {"is_quantized": "True"}},
        "op_type": {"PRelu": {"params": {"weight": {"is_quantized": "False"}}}},
        "supergroups": [
            {"op_list": ["Gemm", "PRelu"]},
            {"op_list": ["Gemm", "Sigmoid"]},
            {"op_list": ["Conv", "PRelu"]},
            {"op_list": ["Conv", "Sigmoid"]},
        ],
        "model_input": {"is_input_quantized": "True"},
        "model_output": {},
    }

    if not os.path.exists("data"):
        os.mkdir("data")
    file_name = "./data/beq_per_channel_config.json"
    with open(file_name, "w") as f:
        json.dump(beq_per_channel_config, f)

    return file_name


def _create_input_dict(input_names, cached_data):
    input_dict = {}

    if not input_names:
        return cached_data
    for input_name in input_names:
        input_dict[input_name] = cached_data[input_name]
    return input_dict


def _build_session(model):
    """
    Build and return onnxruntime inference session
    :param providers: providers to execute onnxruntime
    """
    from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel

    sess_options = SessionOptions()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
    session = InferenceSession(
        path_or_bytes=model.SerializeToString(),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    return session


def data_loader(dummy_input):
    return [dummy_input for _ in range(10)]


def _get_weight_param_name(cg_op):
    param_names = [
        param_name
        for param_name, (_, param_type) in cg_op.parameters.items()
        if param_type == "weight"
    ]
    if len(param_names) == 1:
        return param_names[0]
    return None


@pytest.mark.parametrize(
    "granularity, shape, channel_axis, block_axis",
    [
        ("per_block", (30, 15), 0, 1),
        ("per_channel", (30, 15), 0, 1),
        ("per_tensor", (30, 15), 0, 1),
        ("per_block", (300, 150), 1, 0),  # index shift by +1
        ("per_channel", (300, 150), 1, 0),
        ("per_tensor", (300, 150), 1, 0),
        ("per_block", (30, 15, 3, 3), 0, 1),
        ("per_channel", (30, 15, 3, 3), 0, 1),
        ("per_tensor", (30, 15, 3, 3), 0, 1),
        ("per_block", (300, 150, 3, 3), 1, 0),  # index shift by +1
        ("per_channel", (300, 150, 3, 3), 1, 0),
        ("per_tensor", (300, 150, 3, 3), 1, 0),
    ],
)
def test_min_max_for_candidate_selection(granularity, shape, channel_axis, block_axis):
    """
    Test _get_min_and_max_for_candidate_selection which returns absolute min/max
    values for different quantization granularity.
    """
    # Mock dependency node and graph
    mock_dep_node = MagicMock()
    mock_dep_graph = MagicMock()
    mock_dep_graph.get_param_name.return_value = "mock_weight"

    calibration_tensor = np.random.randn(*shape).astype(np.float32)
    mock_dep_graph.get_param_value.return_value = calibration_tensor

    # Create quantizer
    tensor_quantizer_params = TensorQuantizerParams(shape, channel_axis, block_axis)
    quant_info = libquant_info.QcQuantizeInfo()
    bitwidth = 4
    quantizer = QcQuantizeOp(
        quant_info,
        bitwidth=bitwidth,
        op_mode=OpMode.updateStats,
        quant_scheme=QuantScheme.post_training_tf,
        tensor_quantizer_params=tensor_quantizer_params,
        use_symmetric_encodings=True,
    )
    # Set granularity
    if granularity == "per_channel":
        quantizer.enable_per_channel_quantization()
    elif granularity == "per_block":
        block_size = 3
        quantizer._enable_blockwise_quantization(block_size=block_size)
    else:
        pass
    quantizer.update_encoding_stats(calibration_tensor)
    quantizer.compute_encodings()

    # Mock sim object
    mock_sim = MagicMock()
    mock_sim._quant_scheme = QuantScheme.post_training_tf
    mock_sim.qc_quantize_op_dict = {"mock_weight": quantizer}

    # Mock seq_mse object w/o calling the __init__
    seq_mse = object.__new__(SequentialMse)
    seq_mse.sim = mock_sim
    seq_mse.dependency_graph = mock_dep_graph

    _, max_val = seq_mse._get_min_and_max_for_candidate_selection(mock_dep_node)
    if granularity == "per_tensor":  # per-tensor and per-channel are handled similarly
        max_val = np.max(max_val)

    encodings = quantizer.get_encodings()
    enc_max = np.array([enc.max for enc in encodings])
    enc_min = np.array([enc.min for enc in encodings])
    delta = (enc_max - enc_min) / (2**bitwidth - 1)

    assert np.allclose(
        enc_max, max_val.flatten(), atol=delta
    )  # Allow 1-tick difference


@pytest.mark.parametrize("loss_fn", ["mse"])
@pytest.mark.parametrize(
    "param_bw, enable_pcq, best_indices",
    [
        (4, True, np.array([11, 16, 16, 18, 19, 16, 17, 16, 16, 17])),
        (31, True, np.array([19, 19, 19, 19, 19, 19, 19, 19, 19, 19])),
        (4, False, np.array([[16, 15, 16, 18, 19, 15, 17, 16, 14, 17]])),
        (31, False, np.array([19, 19, 19, 19, 19, 19, 19, 19, 19, 19])),
    ],
)
def test_apply_seq_mse_for_conv(loss_fn, param_bw, enable_pcq, best_indices):
    model = single_conv_layer_model()
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf,
        default_activation_bw=8,
        default_param_bw=param_bw,
        providers=["CPUExecutionProvider"],
        config_file=_get_config_file(
            is_symmetric=True,
            strict_symmetric=False,
            unsigned_symmetric=False,
            pcq=enable_pcq,
        ),
    )
    all_param_ops = []
    for op in sim.connected_graph.ordered_ops:
        param_name = _get_weight_param_name(op)
        if param_name is not None:
            all_param_ops.append(param_name)

    assert len(all_param_ops) == 1

    quantizer = sim.qc_quantize_op_dict[all_param_ops[0]]
    assert not quantizer.is_encoding_frozen()

    inputs = [make_dummy_input(model.model) for _ in range(10)]
    num_candidates = 20
    seq_params = SeqMseParams(num_candidates=num_candidates, num_batches=2)
    seq_params.loss_fn = loss_fn
    seq_mse = SequentialMse(model, sim, seq_params, inputs)
    seq_mse.apply_seq_mse_algo()

    # Encodings are frozen
    assert quantizer.is_encoding_frozen()

    dep_node = list(seq_mse.dependency_graph._name_to_node.values())[0]
    _, init_max = seq_mse._get_min_and_max_for_candidate_selection(dep_node)

    """
    When: Given best indices
    Then: Expected max should be max_tensor / num_candidates * (indices + 1)
    """

    encodings = quantizer.get_encodings()
    actual_max = np.array([enc.max for enc in encodings]).reshape(
        quantizer._encoding_shape()
    )

    expected_max = init_max / num_candidates * (best_indices + 1)
    if not enable_pcq:
        expected_max = np.amax(expected_max)

    print(f"actual_max={actual_max}, expected_max={expected_max}")
    assert np.allclose(actual_max, expected_max)


@pytest.mark.parametrize("param_bw", [2, 31])
@pytest.mark.parametrize("loss_fn", ["mse", "l1", "sqnr"])
@pytest.mark.parametrize("enable_pcq", [True, False])
def test_static_apply_seq_mse(param_bw, loss_fn, enable_pcq):
    model = single_conv_layer_model()
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf,
        default_activation_bw=8,
        default_param_bw=param_bw,
        providers=["CPUExecutionProvider"],
        config_file=_get_config_file(
            is_symmetric=True,
            strict_symmetric=False,
            unsigned_symmetric=False,
            pcq=enable_pcq,
        ),
    )
    inputs = [make_dummy_input(model.model) for _ in range(10)]
    if loss_fn != "mse":
        seq_params = SeqMseParams(num_batches=1)
        seq_params.loss_fn = loss_fn
        SequentialMse.apply_seq_mse(model, sim, seq_params, inputs)
    else:
        apply_seq_mse(sim, inputs[:1])


@pytest.mark.parametrize("param_bw", [2, 31])
@pytest.mark.parametrize("loss_fn", ["mse", "l1", "sqnr"])
@pytest.mark.parametrize("enable_pcq", [True, False])
def test_apply_seq_mse_for_split(param_bw, loss_fn, enable_pcq):
    model = model_with_split()
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf,
        default_activation_bw=8,
        default_param_bw=param_bw,
        providers=["CPUExecutionProvider"],
        config_file=_get_config_file(
            is_symmetric=True,
            strict_symmetric=False,
            unsigned_symmetric=False,
            pcq=enable_pcq,
        ),
    )
    inputs = [make_dummy_input(model.model) for _ in range(10)]
    if loss_fn != "mse":
        seq_params = SeqMseParams(num_batches=1)
        seq_params.loss_fn = loss_fn
        seq_mse = SequentialMse(model, sim, seq_params, inputs)
        seq_mse.apply_seq_mse_algo()
    else:
        apply_seq_mse(sim, inputs[:1])

    weight_quantizer_conv_1 = sim.qc_quantize_op_dict["conv1.weight"]
    weight_quantizer_conv_2 = sim.qc_quantize_op_dict["conv2.weight"]
    weight_quantizer_conv_3 = sim.qc_quantize_op_dict["conv3.weight"]

    assert weight_quantizer_conv_1.is_encoding_frozen()
    assert weight_quantizer_conv_2.is_encoding_frozen()
    assert weight_quantizer_conv_3.is_encoding_frozen()


def test_dependency_graph():
    model = model_with_split()
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf,
        default_activation_bw=8,
        default_param_bw=4,
        providers=["CPUExecutionProvider"],
        config_file=_get_config_file(
            is_symmetric=True,
            strict_symmetric=False,
            unsigned_symmetric=False,
            pcq=True,
        ),
    )
    seq_params = SeqMseParams(num_batches=1)
    inputs = [make_dummy_input(model.model) for _ in range(10)]
    seq_mse = SequentialMse(model, sim, seq_params, inputs)

    assert seq_mse.dependency_graph._name_to_node["/conv1/Conv"].in_degree == 0
    assert seq_mse.dependency_graph._name_to_node["/conv1/Conv"].out_degree == 2
    assert seq_mse.dependency_graph._name_to_node["/conv1/Conv"].op_input_names == [
        "input"
    ]
    assert seq_mse.dependency_graph._name_to_node["/conv1/Conv"].op_output_names == [
        "/conv1/Conv_output_0"
    ]

    assert seq_mse.dependency_graph._name_to_node["/conv2/Conv"].in_degree == 1
    assert seq_mse.dependency_graph._name_to_node["/conv2/Conv"].out_degree == 0
    assert seq_mse.dependency_graph._name_to_node["/conv2/Conv"].op_input_names == [
        "/conv1/Conv_output_0"
    ]
    assert seq_mse.dependency_graph._name_to_node["/conv2/Conv"].op_output_names == [
        "/conv2/Conv_output_0"
    ]

    assert seq_mse.dependency_graph._name_to_node["/conv3/Conv"].in_degree == 1
    assert seq_mse.dependency_graph._name_to_node["/conv3/Conv"].out_degree == 0
    assert seq_mse.dependency_graph._name_to_node["/conv3/Conv"].op_input_names == [
        "/conv1/Conv_output_0"
    ]
    assert seq_mse.dependency_graph._name_to_node["/conv3/Conv"].op_output_names == [
        "/conv3/Conv_output_0"
    ]


def test_residual_model_dependency_graph():
    model = single_residual_model()
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf,
        default_activation_bw=8,
        default_param_bw=4,
        providers=["CPUExecutionProvider"],
        config_file=_get_config_file(
            is_symmetric=True,
            strict_symmetric=False,
            unsigned_symmetric=False,
            pcq=True,
        ),
    )
    seq_params = SeqMseParams(num_batches=1)
    inputs = [make_dummy_input(model.model) for _ in range(10)]
    seq_mse = SequentialMse(model, sim, seq_params, inputs)

    assert seq_mse.dependency_graph._name_to_node["/conv1/Conv"].in_degree == 0
    assert seq_mse.dependency_graph._name_to_node["/conv1/Conv"].out_degree == 2
    assert seq_mse.dependency_graph._name_to_node["/conv1/Conv"].op_input_names == [
        "input"
    ]
    assert seq_mse.dependency_graph._name_to_node["/conv1/Conv"].op_output_names == [
        "/conv1/Conv_output_0"
    ]

    assert seq_mse.dependency_graph._name_to_node["/conv4/Conv"].in_degree == 1
    assert seq_mse.dependency_graph._name_to_node["/conv4/Conv"].out_degree == 1
    assert seq_mse.dependency_graph._name_to_node["/conv4/Conv"].op_input_names == [
        "/maxpool/MaxPool_output_0"
    ]
    assert seq_mse.dependency_graph._name_to_node["/conv4/Conv"].op_output_names == [
        "/conv4/Conv_output_0"
    ]

    assert seq_mse.dependency_graph._name_to_node["/conv2/Conv"].in_degree == 1
    assert seq_mse.dependency_graph._name_to_node["/conv2/Conv"].out_degree == 1
    assert seq_mse.dependency_graph._name_to_node["/conv2/Conv"].op_input_names == [
        "/maxpool/MaxPool_output_0"
    ]
    assert seq_mse.dependency_graph._name_to_node["/conv2/Conv"].op_output_names == [
        "/conv2/Conv_output_0"
    ]

    assert seq_mse.dependency_graph._name_to_node["/conv3/Conv"].in_degree == 1
    assert seq_mse.dependency_graph._name_to_node["/conv3/Conv"].out_degree == 1
    assert seq_mse.dependency_graph._name_to_node["/conv3/Conv"].op_input_names == [
        "/relu2/Relu_output_0"
    ]
    assert seq_mse.dependency_graph._name_to_node["/conv3/Conv"].op_output_names == [
        "/conv3/Conv_output_0"
    ]


@pytest.mark.parametrize("param_bw", [2, 31])
@pytest.mark.parametrize("loss_fn", ["mse", "l1", "sqnr"])
@pytest.mark.parametrize("enable_pcq", [True, False])
def test_apply_seq_mse_for_residual_model(param_bw, loss_fn, enable_pcq):
    model = single_residual_model()
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf,
        default_activation_bw=8,
        default_param_bw=param_bw,
        providers=["CPUExecutionProvider"],
        config_file=_get_config_file(
            is_symmetric=True,
            strict_symmetric=False,
            unsigned_symmetric=False,
            pcq=enable_pcq,
        ),
    )
    inputs = [make_dummy_input(model.model) for _ in range(10)]
    if loss_fn != "mse":
        seq_params = SeqMseParams(num_batches=2)
        seq_params.loss_fn = loss_fn
        seq_mse = SequentialMse(model, sim, seq_params, inputs)
        seq_mse.apply_seq_mse_algo()
    else:
        apply_seq_mse(sim, inputs[:1])

    for conn_graph_op in sim.connected_graph.ordered_ops:
        if conn_graph_op.type in SUPPORTED_MODULES:
            param_names = [
                param_name
                for param_name, (_, param_type) in conn_graph_op.parameters.items()
                if param_type == "weight"
            ]
            assert len(param_names) == 1
            quantizer = sim.qc_quantize_op_dict[param_names[0]]
            assert quantizer.is_encoding_frozen()


def test_model_with_multiple_inputs_dependency_graph_utils():
    model = model_with_multiple_inputs()
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf,
        default_activation_bw=8,
        default_param_bw=4,
        providers=["CPUExecutionProvider"],
        config_file=_get_config_file(
            is_symmetric=True,
            strict_symmetric=False,
            unsigned_symmetric=False,
            pcq=True,
        ),
    )
    seq_params = SeqMseParams(num_batches=1)
    inputs = [make_dummy_input(model) for _ in range(10)]
    seq_mse = SequentialMse(model, sim, seq_params, inputs)

    starting_ops_names = [
        op.name_op for op in seq_mse.dependency_graph.conn_graph.starting_ops
    ]

    assert starting_ops_names == ["Conv1"]
    assert seq_mse.dependency_graph._op_names_with_model_inputs == [
        "Conv1",
        "ADD_0",
        "ADD_1",
    ]


def test_model_with_multiple_outputs_value_info():
    model = model_with_multiple_outputs()
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf,
        default_activation_bw=8,
        default_param_bw=4,
        providers=["CPUExecutionProvider"],
        config_file=_get_config_file(
            is_symmetric=True,
            strict_symmetric=False,
            unsigned_symmetric=False,
            pcq=True,
        ),
    )
    apply_seq_mse(sim, [make_dummy_input(model) for _ in range(1)])
    assert sim.qc_quantize_op_dict["Conv1_W"].is_encoding_frozen()
    assert sim.qc_quantize_op_dict["Conv2_W"].is_encoding_frozen()


def test_concat_model():
    model = models_for_tests.concat_model()
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf,
        default_activation_bw=8,
        default_param_bw=4,
        config_file=_get_config_file(
            is_symmetric=True,
            strict_symmetric=False,
            unsigned_symmetric=False,
            pcq=True,
        ),
    )

    apply_seq_mse(sim, [make_dummy_input(model.model)])
    for cg_op in sim.connected_graph.ordered_ops:
        if cg_op.type in SUPPORTED_MODULES:
            param_names = [
                param_name
                for param_name, (_, param_type) in cg_op.parameters.items()
                if param_type == "weight"
            ]
            assert len(param_names) == 1
            quantizer = sim.qc_quantize_op_dict[param_names[0]]
            assert quantizer.is_encoding_frozen()


def test_disable_subgraph_quantizers():
    model = models_for_tests.build_dummy_model()
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        providers=["CPUExecutionProvider"],
        default_param_bw=4,
    )
    sim.compute_encodings(lambda sess: sess.run(None, make_dummy_input(model)))
    seq_params = SeqMseParams(num_batches=2)
    dataloader = [make_dummy_input(model) for _ in range(2)]
    seq_mse = SequentialMse(model, sim, seq_params, dataloader)

    enabled = {q for q in sim.qc_quantize_op_dict.values() if q.enabled}
    assert enabled

    with seq_mse._disable_subgraph_quantizers(sim.model.model):
        assert not any(q.enabled for q in sim.qc_quantize_op_dict.values())

    assert enabled == {q for q in sim.qc_quantize_op_dict.values() if q.enabled}

    subgraph = seq_mse._split_onnx_graph(seq_mse._extractor, ["4"], ["output"])
    with seq_mse._disable_subgraph_quantizers(subgraph):
        assert not sim.qc_quantize_op_dict["fc_w"].enabled
        assert not sim.qc_quantize_op_dict["4"].enabled


def test_add_value_info():
    model = models_for_tests.single_residual_model().model
    sim = QuantizationSimModel(
        model=model,
        providers=["CPUExecutionProvider"],
        default_param_bw=4,
    )
    orig_value_info = sim.model.model.graph.value_info
    with _add_value_info(sim.model.model):
        updated_value_info = copy.deepcopy(sim.model.model.graph.value_info)

    # Check that the original value info is restored
    assert sim.model.model.graph.value_info == orig_value_info

    tensors_with_info = {info.name for info in updated_value_info}

    io_tensors = {
        tensor.name
        for tensor in itertools.chain(sim.model.graph().input, sim.model.graph().output)
    }
    initializers = {tensor.name for tensor in sim.model.initializer()}
    all_tensors = {
        name
        for node in sim.model.nodes()
        for name in itertools.chain(node.input, node.output)
    }

    # Note: Value info does not include shapes for initializers or i/o tensors
    assert tensors_with_info == (all_tensors - (io_tensors | initializers))


def test_nodes_to_exclude():
    """Skip seq mse optimization if nodes to exclude provided"""
    model = single_residual_model()
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf,
        default_activation_bw=8,
        default_param_bw=4,
        providers=["CPUExecutionProvider"],
    )
    inputs = [make_dummy_input(model.model)]
    excluded_node = "/conv3/Conv"
    child_node = "/fc/Gemm"
    apply_seq_mse(sim, inputs, nodes_to_exclude=[excluded_node])

    for conn_graph_op in sim.connected_graph.ordered_ops:
        if conn_graph_op.type in SUPPORTED_MODULES:
            param_name = _get_weight_param_name(conn_graph_op)
            quantizer = sim.qc_quantize_op_dict[param_name]

            if conn_graph_op.name == excluded_node:
                assert not quantizer.is_encoding_frozen()
            elif conn_graph_op.name == child_node:
                assert quantizer.is_encoding_frozen()
            else:
                assert quantizer.is_encoding_frozen()


class TestDependencyGraph:
    @pytest.mark.parametrize(
        "model, cached_data",
        [
            (
                models_for_tests.single_residual_model(),
                {"input": np.random.randn(1, 3, 32, 32).astype(np.float32)},
            ),
            (
                models_for_tests.concat_model(),
                {
                    "input1": np.random.randn(1, 3, 8, 8).astype(np.float32),
                    "input2": np.random.randn(1, 3, 8, 8).astype(np.float32),
                    "input3": np.random.randn(1, 3, 8, 8).astype(np.float32),
                },
            ),
            (
                models_for_tests.multi_input_model(),
                {
                    "input1": np.random.randn(32, 1, 28, 28).astype(np.float32),
                    "input2": np.random.randn(32, 1, 28, 28).astype(np.float32),
                },
            ),
            (
                models_for_tests.mobilenetv2(),
                {"input": np.random.randn(1, 3, 32, 32).astype(np.float32)},
            ),
            (
                models_for_tests.resnet18(),
                {"input": np.random.randn(1, 3, 32, 32).astype(np.float32)},
            ),
        ],
    )
    def test_dependency_graph(self, model, cached_data):
        """Compare the one-shot and iterative outputs"""
        dl = [cached_data]
        dep_graph = DependencyGraph(ConnectedGraph(model), dl)

        sorted_nodes = dep_graph.get_topologically_sorted_nodes()
        model_outputs = [node.name for node in model.model.graph.output]

        session = _build_session(model.model)
        one_shot_output = session.run(None, input_feed=cached_data)[0]

        # Iterate over the topologically sorted nodes to gather intermediate outputs and
        # provide them as inputs to the subsequent subgraph.
        with _add_value_info(model.model):
            extractor = Extractor(model.model)
        for i in range(1, len(sorted_nodes)):
            subgraph_inp_names, subgraph_out_names = (
                dep_graph.get_subgraph_inp_out_names(sorted_nodes[i])
            )
            model_ = extractor.extract_model(subgraph_inp_names, subgraph_out_names)
            session = _build_session(model_)
            input_dict = _create_input_dict(subgraph_inp_names, cached_data)
            cached_data[subgraph_out_names[0]] = session.run(
                None, input_feed=input_dict
            )[0]

        # Final subgraph extraction and session run
        subgraph_inp_names = subgraph_out_names
        subgraph_out_names = model_outputs
        model_ = extractor.extract_model(subgraph_inp_names, subgraph_out_names)
        session = _build_session(model_)
        input_dict = _create_input_dict(subgraph_inp_names, cached_data)
        iterative_output = session.run(None, input_feed=input_dict)[0]

        assert np.all(iterative_output == one_shot_output)

    def test_nodes_to_exclude(self):
        """When nodes_to_exclude are provided, they should be excluded from dependency graph"""
        model = models_for_tests.single_residual_model().model
        dl = [{"input": np.random.randn(1, 3, 32, 32).astype(np.float32)}]
        dep_graph = DependencyGraph(
            ConnectedGraph(model), dl, nodes_to_exclude=["/conv1/Conv"]
        )
        sorted_order = dep_graph.get_topologically_sorted_nodes()

        for i, sorted_nodes in sorted_order.items():
            print(f"{i}: {sorted_nodes}")

        """
        When: Given node is excluded
        Then: The children of excluded nodes are still processed and added to dependency graph.
        """

        assert "/conv1/Conv" not in [
            node.cg_op.name for nodes in sorted_order.values() for node in nodes
        ]
        assert "/conv2/Conv" in [
            node.cg_op.name for nodes in sorted_order.values() for node in nodes
        ]
        assert "/fc/Gemm" in [
            node.cg_op.name for nodes in sorted_order.values() for node in nodes
        ]

# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

"""Unit tests for Adaround Weights"""

import copy
from unittest.mock import patch

import numpy as np
import torch
import pytest
from onnx import numpy_helper
from onnxsim import simplify
import onnx

import aimet_onnx
from aimet_onnx import apply_adaround, QuantizationSimModel
from aimet_onnx.adaround.utils import AdaroundSupportedModules, ModelData
from aimet_onnx.utils import make_dummy_input, ParamUtils
from .models import models_for_tests
from .models.models_for_tests import conv_prelu_model


class TestAdaround:
    """
    AdaRound Weights Unit Test Cases
    """

    @pytest.mark.parametrize(
        "providers",
        (["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )
    def test_apply_adaround(self, providers):
        if "CUDAExecutionProvider" in providers and not torch.cuda.is_available():
            pytest.skip("Cuda not available")
        np.random.seed(0)
        torch.manual_seed(0)
        model = models_for_tests.single_residual_model()
        dummy_input = {"input": np.random.rand(1, 3, 32, 32).astype(np.float32)}

        sim = QuantizationSimModel(
            copy.deepcopy(model),
            providers=providers,
            param_type=aimet_onnx.int4,
            activation_type=aimet_onnx.int16,
        )
        sim.compute_encodings([dummy_input])
        out_before_ada = sim.session.run(None, dummy_input)
        apply_adaround(sim, [dummy_input for _ in range(2)], 5)
        out_after_ada = sim.session.run(None, dummy_input)
        assert not np.array_equal(out_before_ada[0], out_after_ada[0])

        sim.remove_quantizers(sim.model.model)
        for node in sim.model.nodes():
            if node.op_type in AdaroundSupportedModules:
                assert sim.qc_quantize_op_dict[node.input[1]]._is_encoding_frozen

    @pytest.mark.parametrize("pre_calibrate", [True, False])
    @pytest.mark.parametrize(
        "model",
        [
            models_for_tests.single_residual_model().model,
            models_for_tests.depthwise_conv_model().model,
            models_for_tests.add_matmul_model(),
            models_for_tests.model_with_split_matmul(),
        ],
    )
    @pytest.mark.parametrize(
        "providers",
        (["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]),
    )
    def test_apply_adaround_2(self, providers, model, pre_calibrate):
        if "CUDAExecutionProvider" in providers and not torch.cuda.is_available():
            pytest.skip("Cuda not available")

        inputs = [make_dummy_input(model) for _ in range(2)]
        sim = QuantizationSimModel(copy.deepcopy(model), providers=providers)
        graph_outputs = sim.model.graph().output
        is_enabled = {name: q.enabled for name, q in sim.qc_quantize_op_dict.items()}

        adaroundable_ops = [
            op
            for op in sim.connected_graph.ordered_ops
            if op.type in ("Conv", "MatMul", "Gemm")
        ]
        weight_tensors = {
            t.name: copy.deepcopy(onnx.numpy_helper.to_array(t.tensor))
            for op in adaroundable_ops
            for t, param_type in op.parameters.values()
            if param_type == "weight"
        }

        assert weight_tensors

        if pre_calibrate:
            sim.compute_encodings(inputs)

        apply_adaround(sim, inputs, num_iterations=5)

        # Quantizer enabled state must be restored after adaround
        for name, q in sim.qc_quantize_op_dict.items():
            assert q.enabled == is_enabled[name]

        # Only optimized weights should have frozen encodings
        for name, quantizer in sim.qc_quantize_op_dict.items():
            assert quantizer.is_encoding_frozen() == (name in weight_tensors)

        # Optimized weight should not be equal to original weight
        for name, old_weight in weight_tensors.items():
            new_weight = onnx.numpy_helper.to_array(
                ParamUtils.get_param_by_name(sim.model.model, name)
            )
            assert not np.all(old_weight == new_weight)

        assert graph_outputs == sim.model.graph().output

    @pytest.mark.parametrize(
        "providers",
        (["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )
    def test_apply_adaround_for_custom_op(self, providers):
        if "CUDAExecutionProvider" in providers and not torch.cuda.is_available():
            pytest.skip("Cuda not available")
        from onnxruntime_extensions import get_library_path

        model = models_for_tests.custom_add_model()
        onnx_library = get_library_path()
        np.random.seed(0)
        dummy_input = {"input": np.random.rand(1, 3, 64, 64).astype(np.float32)}
        sim = QuantizationSimModel(
            copy.deepcopy(model),
            providers=providers,
            param_type=aimet_onnx.int4,
            activation_type=aimet_onnx.int16,
            user_onnx_libs=[onnx_library],
        )
        model_data = ModelData(sim)
        orig_weight = torch.from_numpy(
            numpy_helper.to_array(
                model_data.module_to_info["conv"].params["weight"].tensor
            )
        )
        apply_adaround(sim, [dummy_input for _ in range(2)], 5)
        model_data = ModelData(sim)
        updated_weight = torch.from_numpy(
            numpy_helper.to_array(
                model_data.module_to_info["conv"].params["weight"].tensor
            )
        )
        assert not torch.equal(orig_weight, updated_weight)
        sim.compute_encodings([dummy_input])
        sim.remove_quantizers(sim.model.model)
        for node in sim.model.nodes():
            if node.op_type in AdaroundSupportedModules:
                assert sim.qc_quantize_op_dict[node.input[1]]._is_encoding_frozen

    @pytest.mark.parametrize(
        "model, input_shape",
        [
            (models_for_tests.weight_gemm_model(10, 20, True), (1, 10)),
            (models_for_tests.weight_gemm_model(10, 20, False), (1, 10)),
            (models_for_tests.weight_matmul_model(10, 20), (1, 10, 10)),
        ],
    )
    @pytest.mark.parametrize(
        "providers",
        (["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )
    def test_adaround_matmul_gemm(self, model, input_shape, tmpdir, providers):
        if "CUDAExecutionProvider" in providers and not torch.cuda.is_available():
            pytest.skip("Cuda not available")

        sim = QuantizationSimModel(
            copy.deepcopy(model),
            providers=providers,
            param_type=aimet_onnx.int4,
            activation_type=aimet_onnx.int16,
        )

        dummy_input = {"input": np.random.rand(*input_shape).astype(np.float32)}
        apply_adaround(sim, [dummy_input for _ in range(2)], 5)
        sim.remove_quantizers(sim.model.model)
        for node in sim.model.nodes():
            if node.op_type in AdaroundSupportedModules:
                assert sim.qc_quantize_op_dict[node.input[1]]._is_encoding_frozen

    @pytest.mark.parametrize(
        "model, input_shape", [(models_for_tests.dynamic_matmul_model(1), (1, 10))]
    )
    def test_adaround_dynamic_matmul(self, model, input_shape, tmpdir):
        """
        AdaRound should not error-out if there is a dynamic matmul
        """
        sim = QuantizationSimModel(
            copy.deepcopy(model),
            providers=["CPUExecutionProvider"],
            param_type=aimet_onnx.int4,
            activation_type=aimet_onnx.int16,
        )
        dummy_input = {"input": np.random.rand(*input_shape).astype(np.float32)}
        apply_adaround(sim, [dummy_input for _ in range(2)], 5)

    @pytest.mark.parametrize(
        "model, input_shape", [(models_for_tests.simplifiable_model(1), (1, 10))]
    )
    def test_adaround_simplifiable_model(self, model, input_shape, tmpdir):
        """
        AdaRound should not error-out for models which need simplification
        """
        model, _ = simplify(model)
        dummy_input = {"input": np.random.rand(*input_shape).astype(np.float32)}
        sim = QuantizationSimModel(
            copy.deepcopy(model),
            providers=["CPUExecutionProvider"],
            param_type=aimet_onnx.int4,
            activation_type=aimet_onnx.int16,
        )
        apply_adaround(sim, [dummy_input for _ in range(2)], 5)

    @pytest.mark.parametrize(
        "model_factory, input_shape",
        [
            (models_for_tests.pointwise_conv1d, (1, 10, 32)),
            (models_for_tests.pointwise_conv3d, (1, 10, 8, 8, 8)),
            (models_for_tests.pointwise_convtranspose1d, (1, 10, 32)),
            (models_for_tests.pointwise_convtranspose3d, (1, 10, 8, 4, 3)),
            (models_for_tests.padded_convtranspose2d, (1, 10, 32, 32)),
        ],
    )
    def test_adaround_convNd_model(self, model_factory, input_shape, tmpdir):
        """
        AdaRound should not error-out for non-2d Conv/ConvTranspose layers
        """
        model = model_factory(input_shape)

        dummy_input = {"input": np.random.rand(*input_shape).astype(np.float32)}
        sim = QuantizationSimModel(
            copy.deepcopy(model),
            providers=["CPUExecutionProvider"],
            param_type=aimet_onnx.int4,
            activation_type=aimet_onnx.int16,
        )
        apply_adaround(sim, [dummy_input for _ in range(2)], 5)

    @pytest.mark.parametrize(
        "config",
        [
            # model, ops_to_optimize input to pass, expected result
            (
                models_for_tests.single_residual_model().model,
                [],
                [
                    "/conv1/Conv",
                    "/conv2/Conv",
                    "/conv3/Conv",
                    "/conv4/Conv",
                    "/fc/Gemm",
                ],
            ),
            (
                models_for_tests.single_residual_model().model,
                ["/conv1/Conv"],
                ["/conv1/Conv"],
            ),
            (
                models_for_tests.single_residual_model().model,
                ["/conv2/Conv"],
                ["/conv2/Conv"],
            ),
            (
                models_for_tests.single_residual_model().model,
                [
                    "/conv1/Conv",
                    "/conv4/Conv",
                    "/conv2/Conv",
                    "/conv3/Conv",
                ],
                ["/conv1/Conv", "/conv4/Conv", "/conv2/Conv", "/conv3/Conv"],
            ),
        ],
    )
    def test_whitelist_functionality(self, config):
        model, whitelist_ops, expected = config
        inputs = [make_dummy_input(model) for _ in range(2)]
        sim = QuantizationSimModel(
            copy.deepcopy(model),
            providers=["CPUExecutionProvider"],
        )

        param_to_op_name_dict = {}
        for cg_op in sim.connected_graph.get_all_ops().values():
            if cg_op.type in AdaroundSupportedModules:
                param_to_op_name_dict[cg_op.inputs[1].name] = cg_op.name

        ops_processed = []

        def mock_adaround_module(module, *args, **kwargs):
            ops_processed.append(param_to_op_name_dict[module.params["weight"].name])

        with patch(
            "aimet_onnx.adaround.adaround_optimizer.AdaroundOptimizer.adaround_module",
            mock_adaround_module,
        ):
            apply_adaround(
                sim, inputs, num_iterations=5, node_names_to_optimize=whitelist_ops
            )

            print([name for name in ops_processed])
            assert ops_processed.sort() == expected.sort()

    def test_activation_with_param(self):
        if not torch.cuda.is_available():
            pytest.skip("Cuda not available")

        model = conv_prelu_model().model
        inputs = [make_dummy_input(model) for _ in range(2)]
        sim = QuantizationSimModel(
            copy.deepcopy(model), providers=["CUDAExecutionProvider"]
        )
        apply_adaround(sim, inputs, 10)
        # check adaround went through fine
        assert sim.qc_quantize_op_dict["conv1.weight"]._is_encoding_frozen == True


def dataloader(input_shape: tuple, batch_size=2):
    class DataLoader:
        """
        Example of a Dataloader which can be used for running AMPv2
        """

        def __init__(self, batch_size: int, input_shape: tuple):
            """
            :param batch_size: batch size for data loader
            """
            self.batch_size = batch_size
            self.input_shape = input_shape

        def __iter__(self):
            """Iterates over dataset"""
            dummy_input = np.random.rand(*self.input_shape).astype(np.float32)
            yield dummy_input

        def __len__(self):
            return 4

    dummy_dataloader = DataLoader(batch_size=batch_size, input_shape=input_shape)
    return dummy_dataloader

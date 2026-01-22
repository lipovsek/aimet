# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import os
import tempfile
import pytest

import onnx
import torch

from aimet_onnx.common.defs import QuantizationDataType
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.amp.quantizer_groups import find_quantizer_group, QuantizerGroup
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from .models.test_models import (
    model_small_mnist,
    model_with_split,
    single_residual_model,
    concat_model,
    linear_layer_model,
    ConvTransposeConvModel,
)
from .models import models_for_tests


class TestQuantizerGroups:
    def test_find_quantizer_groups(self):
        model = single_residual_model()
        sim = QuantizationSimModel(model.model)
        _, quantizer_groups = find_quantizer_group(sim)
        assert len(quantizer_groups) == 11
        conv3_group = avg_pool_group = None
        for group in quantizer_groups:
            if "conv3.weight" in group.parameter_quantizers:
                conv3_group = group
            if "/avgpool/AveragePool_output_0" in group.activation_quantizers:
                avg_pool_group = group

        assert conv3_group.activation_quantizers[0] == "/relu2/Relu_output_0"
        assert avg_pool_group.parameter_quantizers[0] == "fc.weight"

    def test_find_quantizer_groups_first_param_quantizer_disabled(self):
        model = single_residual_model()
        sim = QuantizationSimModel(model.model)
        # disable first param quantizer
        list(sim.qc_quantize_op_dict.values())[0].enabled = False
        _, quantizer_groups = find_quantizer_group(sim)
        input_group = conv3_group = avg_pool_group = None
        for group in quantizer_groups:
            if "input" in group.activation_quantizers:
                input_group = group
            if "conv3.weight" in group.parameter_quantizers:
                conv3_group = group
            if "/avgpool/AveragePool_output_0" in group.activation_quantizers:
                avg_pool_group = group

        assert len(quantizer_groups) == 11
        assert len(input_group.parameter_quantizers) == 0
        assert conv3_group.activation_quantizers[0] == "/relu2/Relu_output_0"
        assert avg_pool_group.parameter_quantizers[0] == "fc.weight"

    def test_set_and_get_bitwidth_quantizer_groups(self):
        model = single_residual_model()
        sim = QuantizationSimModel(model.model)
        op_name_to_quantizer_dict, quantizer_groups = find_quantizer_group(sim)
        quantizer_group = None
        for group in quantizer_groups:
            if "conv3.weight" in group.parameter_quantizers:
                quantizer_group = group
                break
        candidate = ((8, QuantizationDataType.int), (16, QuantizationDataType.float))
        quantizer_group.set_quantizers_to_candidate(
            op_name_to_quantizer_dict, candidate
        )
        found_candidate = quantizer_group.get_candidate(op_name_to_quantizer_dict)
        assert candidate == found_candidate

        number_of_active_quantizers = len(
            quantizer_group.get_active_quantizers(op_name_to_quantizer_dict)
        )
        assert number_of_active_quantizers == 2

    def test_linear_layer_model(self):
        model = linear_layer_model()
        sim = QuantizationSimModel(model)
        _, quantizer_groups = find_quantizer_group(sim)
        assert len(quantizer_groups) == 3
        output_group = [
            qg
            for qg in quantizer_groups
            if "/layers.1/Gemm_output_0" in qg.activation_quantizers
        ]
        assert len(output_group) == 1
        assert output_group[0].activation_quantizers[0] == "/layers.1/Gemm_output_0"

    def test_transpose_layer_model(self):
        model = ConvTransposeConvModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = os.path.join(tmpdir, "convTransposeConvModel.onnx")
            torch.onnx.export(
                model,
                torch.randn(1, 3, 9, 9),
                save_dir,
                input_names=["input.1"],
                output_names=["outputs"],
                dynamo=False,
            )
            sim = QuantizationSimModel(onnx.load(save_dir))
            _, quantizer_groups = find_quantizer_group(sim)

            assert len(quantizer_groups) == 3

            # Quantizer group should exist for "/conv1/Conv_output_0" and "conv2.weight"
            for quantizer_group in quantizer_groups:
                if "/conv1/Conv_output_0" in quantizer_group.activation_quantizers:
                    break
            else:
                assert False, "/conv1/Conv_output_0 not found in any quantizer group"

            assert len(quantizer_group.activation_quantizers) == 1
            assert (
                len(quantizer_group.parameter_quantizers) == 1
                and "conv2.weight" == quantizer_group.parameter_quantizers[0]
            )

    def test_binary_input_model(self):
        """
        When: Model has no parameters
        Then: All tensors should be in their own quantizer group
        """
        model = models_for_tests.elementwise_op_model().model
        sim = QuantizationSimModel(model)
        _, quantizer_groups = find_quantizer_group(sim)
        for group in quantizer_groups:
            assert len(group.activation_quantizers) == 1
            assert len(group.parameter_quantizers) == 0

    def test_weight_matmul_model(self):
        """
        When: Model has parameters
        Then: Parameters should be grouped with other inputs to the same op
        """
        model = models_for_tests.weight_matmul_model()
        sim = QuantizationSimModel(model)
        _, quantizer_groups = find_quantizer_group(sim)

        expected = {
            QuantizerGroup(("weight",), ("input",)),
            QuantizerGroup((), ("output",)),
        }
        assert set(quantizer_groups) == expected

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
            models_for_tests.shared_stat_batchnorm_model(),
            models_for_tests.mobilenetv2(),
            models_for_tests.gather_concat_model(),
            models_for_tests.weight_matmul_model(),
            model_small_mnist().model,
            model_with_split().model,
            concat_model().model,
        ),
    )
    def test_quantizer_group_correctness(self, model):
        """
        When: Find quantizer groups for a sim
        Then: All enabled quantizers exist in exactly one quantizer group
        """
        sim = QuantizationSimModel(model)
        _, quantizer_groups = find_quantizer_group(sim)

        enabled_quantizers = {
            name
            for name, quantizer in sim.qc_quantize_op_dict.items()
            if quantizer.enabled
        }

        quantizer_group_quantizers = [
            q
            for group in quantizer_groups
            for q in group.parameter_quantizers + group.activation_quantizers
        ]

        # Verify no duplicate quantizers
        assert len(enabled_quantizers) == len(quantizer_group_quantizers)

        # Verify all quantizers are included in a group
        assert enabled_quantizers == set(quantizer_group_quantizers)

        group_param_quantizers = {
            q for group in quantizer_groups for q in group.parameter_quantizers
        }
        group_act_quantizers = {
            q for group in quantizer_groups for q in group.activation_quantizers
        }

        # Verify that all quantizers are properly labeled as param or activation
        assert group_param_quantizers == {
            name for name in sim.param_names if sim.qc_quantize_op_dict[name].enabled
        }
        assert group_act_quantizers == {
            name
            for name in sim.activation_names
            if sim.qc_quantize_op_dict[name].enabled
        }

        # If there is no parameter input, all activations must be in their own group
        for quantizer_group in quantizer_groups:
            if not quantizer_group.parameter_quantizers:
                assert len(quantizer_group.activation_quantizers) == 1

        for op in sim.connected_graph.ordered_ops:
            quantized_inputs = [
                inp.name for inp in op.inputs if inp.name in enabled_quantizers
            ]
            # If op has parameters, all inputs should be in the same quantizer group
            if any(
                sim.qc_quantize_op_dict[name].enabled for name in op.parameters.keys()
            ):
                quantizer_group = [
                    qg
                    for qg in quantizer_groups
                    if quantized_inputs[0]
                    in qg.parameter_quantizers + qg.activation_quantizers
                ][0]
                qg_quantizers = set(
                    quantizer_group.activation_quantizers
                    + quantizer_group.parameter_quantizers
                )
                assert set(quantized_inputs).issubset(qg_quantizers)

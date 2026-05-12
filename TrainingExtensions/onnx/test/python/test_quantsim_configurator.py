# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import json
import os
import numpy as np
import onnx
import torch
import pytest
import torch
import onnx
from aimet_onnx.common.defs import QuantizationDataType, qtype
from aimet_onnx.common.quantsim_config.utils import (
    get_path_for_per_channel_config,
    get_path_for_per_tensor_config,
)
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.quantsim import QuantizationSimModel, QuantSimConfigurator
from .models import models_for_tests
from .utils import tmp_dir
import tempfile


class TestQuantSimConfig:
    """Tests for applying config to QuantizationSimModel"""

    def test_qs_config_dummy_model(self):
        model = models_for_tests.build_dummy_model()
        sim = QuantizationSimModel(model, providers=["CPUExecutionProvider"])
        assert sim.qc_quantize_op_dict["conv_w"].enabled == True
        assert sim.qc_quantize_op_dict["conv_b"].enabled == False
        assert sim.qc_quantize_op_dict["fc_w"].enabled == True
        assert sim.qc_quantize_op_dict["fc_b"].enabled == False
        assert sim.qc_quantize_op_dict["input"].enabled == True
        assert sim.qc_quantize_op_dict["3"].enabled == False
        assert sim.qc_quantize_op_dict["4"].enabled == True
        assert sim.qc_quantize_op_dict["5"].enabled == False  # Maxpool disabled
        assert sim.qc_quantize_op_dict["output"].enabled == True

    def test_default_config(self, tmp_dir):
        model = models_for_tests.build_dummy_model()

        quantsim_config = {
            "defaults": {
                "ops": {"is_output_quantized": "True", "is_symmetric": "False"},
                "params": {"is_quantized": "False", "is_symmetric": "True"},
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {},
        }
        data_dir = os.path.join(tmp_dir, "data")
        config_path = os.path.join(data_dir, "quantsim_config.json")
        os.makedirs(data_dir, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(
            model,
            config_file=config_path,
            providers=["CPUExecutionProvider"],
        )
        for name in ["3", "4", "output"]:
            assert sim.qc_quantize_op_dict[name].enabled == True
            assert sim.qc_quantize_op_dict[name].use_symmetric_encodings == False

    def test_param_config(self, tmp_dir):
        model = models_for_tests.build_dummy_model()

        quantsim_config = {
            "defaults": {
                "ops": {"is_output_quantized": "True", "is_symmetric": "False"},
                "params": {"is_quantized": "False", "is_symmetric": "True"},
            },
            "params": {
                "weight": {"is_quantized": "True", "is_symmetric": "True"},
            },
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {},
        }
        data_dir = os.path.join(tmp_dir, "data")
        config_path = os.path.join(data_dir, "quantsim_config.json")
        os.makedirs(data_dir, exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(
            model,
            config_file=config_path,
            providers=["CPUExecutionProvider"],
        )
        for name in ["conv_w", "fc_w"]:
            assert sim.qc_quantize_op_dict[name].enabled == True
            assert sim.qc_quantize_op_dict[name].use_symmetric_encodings == True

        for name in ["conv_b", "fc_b"]:
            assert sim.qc_quantize_op_dict[name].enabled == False
            assert sim.qc_quantize_op_dict[name].use_symmetric_encodings == True

    def test_op_level_config_and_model_output(self, tmp_dir):
        model = models_for_tests.build_dummy_model()

        quantsim_config = {
            "defaults": {
                "ops": {"is_output_quantized": "True", "is_symmetric": "False"},
                "params": {"is_quantized": "False", "is_symmetric": "True"},
            },
            "params": {},
            "op_type": {
                "Conv": {
                    "is_input_quantized": "True",
                    "is_symmetric": "False",
                    "params": {
                        "weight": {"is_quantized": "True", "is_symmetric": "False"}
                    },
                }
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {
                "is_output_quantized": "True",
            },
        }
        data_dir = os.path.join(tmp_dir, "data")
        config_path = os.path.join(data_dir, "quantsim_config.json")
        os.makedirs(data_dir, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(
            model,
            config_file=config_path,
            providers=["CPUExecutionProvider"],
        )

        assert sim.qc_quantize_op_dict["conv_w"].enabled == True
        assert sim.qc_quantize_op_dict["conv_w"].use_symmetric_encodings == False
        assert sim.qc_quantize_op_dict["input"].enabled == True
        assert sim.qc_quantize_op_dict["input"].use_symmetric_encodings == False
        assert sim.qc_quantize_op_dict["output"].enabled == True
        assert sim.qc_quantize_op_dict["5"].enabled == False  # Disable for Maxpool

    def test_config_for_model_input(self, tmp_dir):
        model = models_for_tests.build_dummy_model()

        quantsim_config = {
            "defaults": {"ops": {}, "params": {}},
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {},
        }

        data_dir = os.path.join(tmp_dir, "data")
        config_path = os.path.join(data_dir, "quantsim_config.json")
        os.makedirs(data_dir, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(
            model,
            config_file=config_path,
            providers=["CPUExecutionProvider"],
        )
        assert sim.qc_quantize_op_dict["input"].enabled == True

    def test_parse_config_file_supergroups_pass_list(self, tmp_dir):
        """
        Test that supergroup pass list is set correctly in configuration file
        """
        quantsim_config = {
            "defaults": {
                "ops": {"is_output_quantized": "True", "is_symmetric": "False"},
                "params": {"is_quantized": "False", "is_symmetric": "False"},
            },
            "params": {},
            "op_type": {},
            "supergroup_pass_list": ["LayerNormalization"],
            "supergroups": [
                {"op_list": ["Conv", "Relu"]},
                {"op_list": ["Relu", "MaxPool"]},
            ],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {},
        }

        data_dir = os.path.join(tmp_dir, "data")
        config_path = os.path.join(data_dir, "quantsim_config.json")
        os.makedirs(data_dir, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(quantsim_config, f)

        qsim_config = QuantSimConfigurator(
            config_file=config_path,
            param_type=qtype.int(8),
            activation_type=qtype.int(8),
        )
        assert qsim_config._get_supergroup_pass_list() == ["LayerNormalization"]

    @pytest.mark.parametrize("strict, unsigned", ((True, False), (False, True)))
    def test_parse_config_file_symmetric_modes(self, strict, unsigned, tmp_dir):
        """Test that model output quantization parameters are set correctly when using json config file"""
        model = models_for_tests.build_dummy_model()

        quantsim_config = {
            "defaults": {
                "ops": {},
                "params": {"is_symmetric": "True"},
                "per_channel_quantization": "True",
                "strict_symmetric": str(strict),
                "unsigned_symmetric": str(unsigned),
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {},
        }
        data_dir = os.path.join(tmp_dir, "data")
        config_path = os.path.join(data_dir, "quantsim_config.json")
        os.makedirs(data_dir, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(
            model,
            config_file=config_path,
            providers=["CPUExecutionProvider"],
        )

        for quantizer in sim.qc_quantize_op_dict.values():
            assert quantizer.use_strict_symmetric == strict
            assert quantizer.use_unsigned_symmetric == unsigned

    def test_generate_and_apply_op_level_config(self, tmp_dir):
        model = models_for_tests.build_dummy_model()

        quantsim_config = {
            "defaults": {
                "ops": {"is_output_quantized": "True", "is_symmetric": "False"},
                "params": {"is_quantized": "False", "is_symmetric": "True"},
            },
            "params": {},
            "op_type": {
                "Conv": {
                    "is_input_quantized": "True",
                    "is_symmetric": "False",
                    "params": {
                        "weight": {"is_quantized": "True", "is_symmetric": "False"}
                    },
                    "per_channel_quantization": "True",
                }
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {
                "is_output_quantized": "True",
            },
        }

        data_dir = os.path.join(tmp_dir, "data")
        config_path = os.path.join(data_dir, "quantsim_config.json")
        os.makedirs(data_dir, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(
            model,
            config_file=config_path,
            providers=["CPUExecutionProvider"],
        )
        assert sim.qc_quantize_op_dict["conv_w"].quant_info.usePerChannelMode == True
        assert sim.qc_quantize_op_dict["fc_w"].quant_info.usePerChannelMode == False

    def test_supported_kernels(self, tmp_dir):
        """
        Tests the generated supported_kernels
        """
        model = models_for_tests.single_residual_model()
        quantsim_config = {
            "defaults": {
                "ops": {"is_output_quantized": "True", "is_symmetric": "False"},
                "params": {"is_quantized": "False", "is_symmetric": "True"},
                "hw_version": "V01",
                "supported_kernels": [
                    {
                        "activation": {"bitwidth": 16, "dtype": "float"},
                        "param": {"bitwidth": 16, "dtype": "float"},
                    }
                ],
                "per_channel_quantization": "True",
            },
            "params": {},
            "op_type": {
                "Conv": {
                    "supported_kernels": [
                        {
                            "activation": {"bitwidth": 16, "dtype": "int"},
                            "param": {"bitwidth": 8, "dtype": "int"},
                        }
                    ],
                    "per_channel_quantization": "False",
                }
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {},
        }

        data_dir = os.path.join(tmp_dir, "data")
        config_path = os.path.join(data_dir, "quantsim_config.json")
        os.makedirs(data_dir, exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(model, config_file=config_path)
        op_to_supported_kernels = sim._op_to_supported_kernel
        for op_name in op_to_supported_kernels:
            assert len(op_to_supported_kernels[op_name]) == 1
            if "Conv" in op_name:
                assert op_to_supported_kernels[op_name] == [
                    ((16, QuantizationDataType.int), (8, QuantizationDataType.int))
                ]
            else:
                assert op_to_supported_kernels[op_name] == [
                    ((16, QuantizationDataType.float), (16, QuantizationDataType.float))
                ]

        expected_supported_kernels = [
            {
                "activation": {"bitwidth": 16, "dtype": QuantizationDataType.int},
                "param": {"bitwidth": 8, "dtype": QuantizationDataType.int},
            }
        ]
        supported_kernels_conv = sim.get_supported_kernels()["Conv"]
        assert len(supported_kernels_conv) == 1
        assert supported_kernels_conv == expected_supported_kernels

    def test_matmul_perchannel_config(self):
        model = models_for_tests.weight_matmul_model(in_features=10, out_features=20)
        sim = QuantizationSimModel(model, config_file=get_path_for_per_tensor_config())
        assert not sim.qc_quantize_op_dict["weight"].quant_info.usePerChannelMode

    @pytest.mark.parametrize("config", (None, get_path_for_per_channel_config()))
    def test_disable_batchnorm_stats_quantization(self, config):
        model = models_for_tests.batchnorm_model()
        sim = QuantizationSimModel(model, config_file=config)
        assert not sim.qc_quantize_op_dict["batchnorm.input_mean"].enabled
        assert not sim.qc_quantize_op_dict["batchnorm.input_var"].enabled
        assert not sim.qc_quantize_op_dict["batchnorm.bias"].enabled
        assert sim.qc_quantize_op_dict["batchnorm.weight"].enabled
        assert sim.qc_quantize_op_dict["output"].enabled

    def test_input_qtzr_upon_residual_connection(self, tmp_path):
        """
        Given: A residual connection in the model graph

          input -+-> MatMul --> Mul --> output
                 +---------------^

          where MatMul-Mul is defined as supergroup

        When: Create QuantizationSimModel with htp_v81 config
        Then: Intermediate output between MatMul and Mul should not be quantized

          input -Q--+-> MatMul --> Mul -Q-> output
                    +---------------^
        """
        config = {
            "defaults": {
                "ops": {"is_output_quantized": "True", "is_symmetric": "False"},
                "params": {"is_quantized": "True", "is_symmetric": "True"},
            },
            "params": {"bias": {"is_quantized": "False"}},
            "op_type": {},
            "supergroups": [
                {"op_list": ["MatMul", "Mul"]},
            ],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {},
        }
        with open(tmp_path / "config.json", "w") as f:
            json.dump(config, f)

        class Model(torch.nn.Module):
            def forward(self, x):
                return (x @ x) * x

        model = Model()
        torch.onnx.export(
            model,
            torch.randn(10, 10),
            tmp_path / "residual.onnx",
            input_names=["input"],
            output_names=["output"],
            opset_version=13,
            dynamo=False,
        )
        model = onnx.load(tmp_path / "residual.onnx")
        sim = QuantizationSimModel(model, config_file=tmp_path / "config.json")
        assert sim.qc_quantize_op_dict["input"].enabled
        assert sim.qc_quantize_op_dict["output"].enabled
        assert not sim.qc_quantize_op_dict["/MatMul_output_0"].enabled

    def test_ambiguous_supergroup(self, tmp_path):
        """
        Given:
          * model:       LayerNormalization ----------> Relu
                 (decomposed into elementwise ops)
          * config: Both LayerNormalization and Add-Relu are specified as supergroups
        When: Create QuantizationSimModel
        Then: supergroup_pass_list must take precedence over supergroups

            LayerNormalization -> Q -> Relu -> Q
        """
        config = {
            "defaults": {
                "ops": {"is_output_quantized": "True", "is_symmetric": "False"},
                "params": {"is_quantized": "True", "is_symmetric": "True"},
            },
            "params": {"bias": {"is_quantized": "False"}},
            "op_type": {},
            "supergroup_pass_list": ["LayerNormalization"],
            "supergroups": [
                {"op_list": ["Add", "Relu"]},
            ],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {},
        }
        with open(tmp_path / "quantsim_config.json", "w") as f:
            json.dump(config, f)

        model = torch.nn.Sequential(
            torch.nn.LayerNorm((10, 10)),
            torch.nn.ReLU(),
        )
        torch.onnx.export(
            model,
            torch.randn(10, 10),
            tmp_path / "model.onnx",
            input_names=["input"],
            output_names=["output"],
            opset_version=13,
            dynamo=False,
        )
        model = onnx.load_model(tmp_path / "model.onnx")
        sim = QuantizationSimModel(model, config_file=tmp_path / "quantsim_config.json")
        sim.compute_encodings([{"input": np.random.randn(10, 10).astype(np.float32)}])

        # LayerNormalization takes precedence over Add-Relu
        expected_qdq = ("input", "0.weight", "/0/Add_1_output_0", "output")

        for tensor_name, qtzr in sim.qc_quantize_op_dict.items():
            if tensor_name in expected_qdq:
                assert qtzr.enabled, (
                    f"Quantizer for {tensor_name} is disabled but expected to be enabled"
                )
            else:
                assert not qtzr.enabled, (
                    f"Quantizer for {tensor_name} is enabled but expected to be disabled"
                )

        """
        Given:
          * model: Conv -> Add -> Relu
          * config: Both Conv-Add and Add-Relu are specified as supergroups
        When: Create QuantizationSimModel
        Then: Whatever supergroup comes first in the config file must take precedence

            Conv -> Add -> Q -> Relu -> Q
        """
        config = {
            "defaults": {
                "ops": {"is_output_quantized": "True", "is_symmetric": "False"},
                "params": {"is_quantized": "True", "is_symmetric": "True"},
            },
            "params": {"bias": {"is_quantized": "False"}},
            "op_type": {},
            "supergroups": [
                {"op_list": ["Conv", "Add"]},
                {"op_list": ["Add", "Relu"]},
            ],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {},
        }
        with open(tmp_path / "quantsim_config.json", "w") as f:
            json.dump(config, f)

        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv(x)
                x = x + torch.ones_like(x)
                x = self.relu(x)
                return x

        model = Model()
        torch.onnx.export(
            model,
            torch.randn(1, 3, 10, 10),
            tmp_path / "model.onnx",
            input_names=["input"],
            output_names=["output"],
            opset_version=13,
            dynamo=False,
        )
        model = onnx.load_model(tmp_path / "model.onnx")
        sim = QuantizationSimModel(model, config_file=tmp_path / "quantsim_config.json")
        sim.compute_encodings(
            [{"input": np.random.randn(1, 3, 10, 10).astype(np.float32)}]
        )

        # Conv-Add takes precedence over Add-Relu
        expected_qdq = (
            "input",
            "conv.weight",
            "/Constant_output_0",  # second input to Add
            "/Add_output_0",
            "output",
        )

        for tensor_name, qtzr in sim.qc_quantize_op_dict.items():
            if tensor_name in expected_qdq:
                assert qtzr.enabled, (
                    f"Quantizer for {tensor_name} is disabled but expected to be enabled"
                )
            else:
                assert not qtzr.enabled, (
                    f"Quantizer for {tensor_name} is enabled but expected to be disabled"
                )

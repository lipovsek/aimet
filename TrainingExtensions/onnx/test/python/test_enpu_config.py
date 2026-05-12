# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import itertools
from pathlib import Path
import pytest
import torch
from torch import nn
import onnx

import aimet_onnx
from aimet_onnx.quantsim import _apply_constraints
from .models.test_models import RMSNorm


class Add(nn.Module):
    def forward(self, x, y):
        return x + y


class Sub(nn.Module):
    def forward(self, x, y):
        return x - y


class Mul(nn.Module):
    def forward(self, x, y):
        return x * y


class Sequential(nn.Sequential):
    def forward(self, *x):
        for module in self:
            if isinstance(x, tuple):
                x = module(*x)
            else:
                x = module(x)
        return x


class TestEnpuV6Config:
    @torch.no_grad()
    @pytest.mark.parametrize(
        "model, input",
        [
            (nn.Conv1d(3, 3, 3), torch.randn(1, 3, 10)),
            (nn.ConvTranspose1d(3, 3, 3), torch.randn(1, 3, 10)),
            (nn.Conv2d(3, 3, 3), torch.randn(1, 3, 10, 10)),
            (nn.ConvTranspose2d(3, 3, 3), torch.randn(1, 3, 10, 10)),
            (nn.Conv3d(3, 3, 3), torch.randn(1, 3, 5, 5, 5)),
            (nn.ConvTranspose3d(3, 3, 3), torch.randn(1, 3, 5, 5, 5)),
            (nn.Linear(10, 5, bias=True), torch.randn(1, 10)),
            (nn.Linear(10, 5, bias=True), torch.randn(1, 10, 10)),
            (nn.Linear(10, 5, bias=False), torch.randn(1, 10, 10)),
        ],
    )
    def test_conv(self, tmp_path: Path, model: nn.Module, input: torch.Tensor):
        torch.onnx.export(
            model,
            input,
            tmp_path / "model.onnx",
            input_names=["input"],
            output_names=["output"],
            opset_version=13,
            dynamo=False,
        )
        onnx_model = onnx.load(tmp_path / "model.onnx")
        weight_name = next(
            node.input[1]
            for node in onnx_model.graph.node
            if node.op_type in ["Conv", "ConvTranspose", "Gemm", "MatMul"]
        )

        sim = aimet_onnx.QuantizationSimModel(
            onnx_model,
            config_file="enpu_v6",
        )
        assert sim.qc_quantize_op_dict["input"].enabled
        assert sim.qc_quantize_op_dict["input"].use_symmetric_encodings
        assert sim.qc_quantize_op_dict["output"].enabled
        assert sim.qc_quantize_op_dict["output"].use_symmetric_encodings
        assert sim.qc_quantize_op_dict[weight_name].enabled
        assert sim.qc_quantize_op_dict[weight_name].use_symmetric_encodings
        # Per-tensor quantization is the default in eNPU
        assert not sim.qc_quantize_op_dict[weight_name].quant_info.usePerChannelMode

        if model.bias is not None:
            assert not sim.qc_quantize_op_dict["bias"].enabled

    @torch.no_grad()
    @pytest.mark.parametrize(
        "model, input",
        [
            (nn.BatchNorm1d(3), torch.randn(1, 3, 10)),
            (nn.BatchNorm2d(3), torch.randn(1, 3, 10, 10)),
            (nn.BatchNorm3d(3), torch.randn(1, 3, 5, 5, 5)),
        ],
    )
    def test_batchnorm_params(
        self, tmp_path: Path, model: nn.Module, input: torch.Tensor
    ):
        """
        Batchnorm parameters should not be quantized
        """
        model.running_mean.copy_(torch.randn(3))
        model.running_var.copy_(torch.randn(3).abs())

        torch.onnx.export(
            model.eval(),
            input,
            tmp_path / "model.onnx",
            input_names=["input"],
            output_names=["output"],
            opset_version=13,
            dynamo=False,
        )
        sim = aimet_onnx.QuantizationSimModel(
            onnx.load(tmp_path / "model.onnx"),
            config_file="enpu_v6",
        )
        assert sim.qc_quantize_op_dict["input"].enabled
        assert sim.qc_quantize_op_dict["input"].use_symmetric_encodings
        assert sim.qc_quantize_op_dict["output"].enabled
        assert sim.qc_quantize_op_dict["output"].use_symmetric_encodings
        assert not sim.qc_quantize_op_dict["weight"].enabled
        assert not sim.qc_quantize_op_dict["bias"].enabled
        assert not sim.qc_quantize_op_dict["running_mean"].enabled
        assert not sim.qc_quantize_op_dict["running_var"].enabled

    @torch.no_grad()
    @pytest.mark.parametrize("tie_quantizers", [True, False])
    @pytest.mark.parametrize(
        "model",
        [
            Sequential(nn.Conv2d(3, 3, 3), nn.ReLU()),
            Sequential(nn.Conv2d(3, 3, 3), nn.ReLU6()),
            Sequential(nn.Conv2d(3, 3, 3), nn.Sigmoid()),
            Sequential(nn.Conv2d(3, 3, 3), nn.Tanh()),
            Sequential(nn.Conv2d(3, 3, 3), nn.PReLU()),
            Sequential(nn.Conv2d(3, 3, 3), nn.LeakyReLU()),
            Sequential(nn.Conv2d(3, 3, 3), nn.Hardswish()),
            Sequential(nn.Conv2d(3, 3, 3), nn.GELU()),
            Sequential(nn.Conv2d(3, 3, 3), nn.BatchNorm2d(3)),
            Sequential(nn.Conv2d(3, 3, 3), nn.BatchNorm2d(3), nn.ReLU()),
            Sequential(nn.Conv2d(3, 3, 3), nn.BatchNorm2d(3), nn.ReLU6()),
            Sequential(nn.Conv2d(3, 3, 3), nn.BatchNorm2d(3), nn.Sigmoid()),
            Sequential(nn.Conv2d(3, 3, 3), nn.BatchNorm2d(3), nn.Tanh()),
            Sequential(nn.Conv2d(3, 3, 3), nn.BatchNorm2d(3), nn.GELU()),
            Sequential(nn.ConvTranspose2d(3, 3, 3), nn.ReLU()),
            Sequential(nn.ConvTranspose2d(3, 3, 3), nn.ReLU6()),
            Sequential(nn.ConvTranspose2d(3, 3, 3), nn.Sigmoid()),
            Sequential(nn.ConvTranspose2d(3, 3, 3), nn.Tanh()),
            Sequential(nn.ConvTranspose2d(3, 3, 3), nn.PReLU()),
            Sequential(nn.ConvTranspose2d(3, 3, 3), nn.LeakyReLU()),
            Sequential(nn.ConvTranspose2d(3, 3, 3), nn.Hardswish()),
            Sequential(nn.ConvTranspose2d(3, 3, 3), nn.GELU()),
            Sequential(nn.ConvTranspose2d(3, 3, 3), nn.BatchNorm2d(3)),
            Sequential(nn.ConvTranspose2d(3, 3, 3), nn.BatchNorm2d(3), nn.ReLU()),
            Sequential(nn.ConvTranspose2d(3, 3, 3), nn.BatchNorm2d(3), nn.ReLU6()),
            Sequential(nn.ConvTranspose2d(3, 3, 3), nn.BatchNorm2d(3), nn.Sigmoid()),
            Sequential(nn.ConvTranspose2d(3, 3, 3), nn.BatchNorm2d(3), nn.Tanh()),
            Sequential(nn.ConvTranspose2d(3, 3, 3), nn.BatchNorm2d(3), nn.GELU()),
            Sequential(nn.Linear(10, 10, bias=True), nn.ReLU()),
            Sequential(nn.Linear(10, 10, bias=True), nn.ReLU6()),
            Sequential(nn.Linear(10, 10, bias=True), nn.Sigmoid()),
            Sequential(nn.Linear(10, 10, bias=True), nn.Tanh()),
            Sequential(nn.Linear(10, 10, bias=True), nn.LeakyReLU()),
            Sequential(nn.Linear(10, 10, bias=True), nn.Hardswish()),
            Sequential(nn.Linear(10, 10, bias=True), nn.GELU()),
            Sequential(nn.Linear(10, 10, bias=False), nn.ReLU()),
            Sequential(nn.Linear(10, 10, bias=False), nn.ReLU6()),
            Sequential(nn.Linear(10, 10, bias=False), nn.Sigmoid()),
            Sequential(nn.Linear(10, 10, bias=False), nn.Tanh()),
            Sequential(nn.Linear(10, 10, bias=False), nn.LeakyReLU()),
            Sequential(nn.Linear(10, 10, bias=False), nn.Hardswish()),
            Sequential(nn.Linear(10, 10, bias=False), nn.GELU()),
            Sequential(nn.AvgPool2d(3), nn.ReLU()),
            Sequential(nn.AvgPool2d(3), nn.ReLU6()),
            Sequential(nn.AvgPool2d(3), nn.Sigmoid()),
            Sequential(nn.AvgPool2d(3), nn.Tanh()),
            Sequential(nn.AvgPool2d(3), nn.PReLU()),
            Sequential(nn.AvgPool2d(3), nn.LeakyReLU()),
            Sequential(nn.AvgPool2d(3), nn.Hardswish()),
            Sequential(nn.AvgPool2d(3), nn.GELU()),
            Sequential(nn.AvgPool2d(3), nn.BatchNorm2d(3)),
            Sequential(nn.AvgPool2d(3), nn.BatchNorm2d(3), nn.ReLU()),
            Sequential(nn.AvgPool2d(3), nn.BatchNorm2d(3), nn.ReLU6()),
            Sequential(nn.AvgPool2d(3), nn.BatchNorm2d(3), nn.Sigmoid()),
            Sequential(nn.AvgPool2d(3), nn.BatchNorm2d(3), nn.Tanh()),
            Sequential(nn.AvgPool2d(3), nn.BatchNorm2d(3), nn.GELU()),
            Sequential(nn.MaxPool2d(3), nn.ReLU()),
            Sequential(nn.MaxPool2d(3), nn.ReLU6()),
            Sequential(nn.MaxPool2d(3), nn.Sigmoid()),
            Sequential(nn.MaxPool2d(3), nn.Tanh()),
            Sequential(nn.MaxPool2d(3), nn.PReLU()),
            Sequential(nn.MaxPool2d(3), nn.LeakyReLU()),
            Sequential(nn.MaxPool2d(3), nn.Hardswish()),
            Sequential(nn.MaxPool2d(3), nn.GELU()),
            Sequential(nn.MaxPool2d(3), nn.BatchNorm2d(3)),
            Sequential(nn.MaxPool2d(3), nn.BatchNorm2d(3), nn.ReLU()),
            Sequential(nn.MaxPool2d(3), nn.BatchNorm2d(3), nn.ReLU6()),
            Sequential(nn.MaxPool2d(3), nn.BatchNorm2d(3), nn.Sigmoid()),
            Sequential(nn.MaxPool2d(3), nn.BatchNorm2d(3), nn.Tanh()),
            Sequential(nn.MaxPool2d(3), nn.BatchNorm2d(3), nn.GELU()),
            Sequential(Add(), nn.ReLU()),
            Sequential(Add(), nn.ReLU6()),
            Sequential(Add(), nn.Sigmoid()),
            Sequential(Add(), nn.Tanh()),
            Sequential(Add(), nn.LeakyReLU()),
            Sequential(Add(), nn.Hardswish()),
            Sequential(Add(), nn.GELU()),
            Sequential(Sub(), nn.ReLU()),
            Sequential(Sub(), nn.ReLU6()),
            Sequential(Sub(), nn.Sigmoid()),
            Sequential(Sub(), nn.Tanh()),
            Sequential(Sub(), nn.LeakyReLU()),
            Sequential(Sub(), nn.Hardswish()),
            Sequential(Sub(), nn.GELU()),
            Sequential(Mul(), nn.ReLU()),
            Sequential(Mul(), nn.ReLU6()),
            Sequential(Mul(), nn.Sigmoid()),
            Sequential(Mul(), nn.Tanh()),
            Sequential(Mul(), nn.LeakyReLU()),
            Sequential(Mul(), nn.Hardswish()),
            Sequential(Mul(), nn.GELU()),
            *(
                nn.LayerNorm(
                    normalized_shape=10,
                    bias=bias,
                )
                for bias in (True, False)
            ),
            *(
                RMSNorm(
                    dim=10,
                    elementwise_affine=elementwise_affine,
                    mul_for_pow=mul_for_pow,
                    mul_rsqrt_pattern=mul_rsqrt_pattern,
                )
                for elementwise_affine in (True, False)
                for mul_for_pow in (True, False)
                for mul_rsqrt_pattern in (
                    "mul_rsqrt",
                    "div_sqrt",
                    "mul_reciprocal_sqrt",
                )
            ),
        ],
    )
    def test_supergroup(self, tmp_path: Path, model: nn.Module, tie_quantizers: bool):
        input = (torch.randn(1, 3, 10, 10),)

        if isinstance(model, Sequential):
            if isinstance(model[0], (Add, Sub, Mul)):
                input = (torch.randn(1, 3, 10, 10), torch.randn(1, 3, 10, 10))
            elif isinstance(model[0], nn.Linear):
                if model[0].bias is not None:
                    input = (torch.randn(1, 10),)  # Will be exported as Gemm
                else:
                    input = (torch.randn(1, 3, 10, 10),)  # Will be exported as MatMul

        opset_version = (
            20
            if any(
                isinstance(module, (nn.Hardswish, nn.GELU))
                for module in model.modules()
            )
            else 13
        )
        torch.onnx.export(
            model.eval(),
            input,
            tmp_path / "model.onnx",
            input_names=["input"],
            output_names=["output"],
            opset_version=opset_version,
            do_constant_folding=False,
            dynamo=False,
        )
        onnx_model = onnx.load(tmp_path / "model.onnx")

        input_names = set(inp.name for inp in onnx_model.graph.input)
        output_names = set(out.name for out in onnx_model.graph.output)
        param_names = set(init.name for init in onnx_model.graph.initializer)
        param_names |= set(
            node.output[0]
            for node in onnx_model.graph.node
            if node.op_type == "Constant"
        )

        with _apply_constraints(tie_quantizers):
            sim = aimet_onnx.QuantizationSimModel(
                onnx_model,
                config_file="enpu_v6",
            )
        for name in itertools.chain(input_names, output_names):
            assert sim.qc_quantize_op_dict[name].enabled

        intermediate_activations = (
            sim.qc_quantize_op_dict.keys() - param_names - input_names - output_names
        )

        for name in intermediate_activations:
            qtzr = sim.qc_quantize_op_dict[name]
            assert not qtzr.enabled, (
                f"Output quantizer {name} should be disabled in supergroup"
            )

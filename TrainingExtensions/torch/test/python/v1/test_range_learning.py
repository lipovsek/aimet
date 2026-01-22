# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import pytest
import torch

from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_torch.v1.tensor_quantizer import LearnedGridTensorQuantizer


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *x):
        return torch.round(*x)

    @staticmethod
    def backward(ctx, *output_grad):
        return output_grad


class RangeLearningAsymAutograd(torch.nn.Module):
    def __init__(self, bitwidth):
        self.n_steps = 2**bitwidth - 1

    def forward(self, x, emin, emax):
        scale = (emax - emin) / self.n_steps
        offset = emin / scale
        x_q = torch.clamp(
            STE.apply(x / scale) - STE.apply(offset),
            self.n_steps - self.n_steps,
            self.n_steps,
        )
        x_dq = (x_q + STE.apply(offset)) * scale
        return x_dq


class RangeLearningSymAutograd(RangeLearningAsymAutograd):
    def forward(self, x, emin, emax):
        scale = emax / torch.div(self.n_steps, 2, rounding_mode="floor")
        offset = -torch.div(self.n_steps, 2, rounding_mode="floor") - 1
        x_q = torch.clamp(
            STE.apply(x / scale) - offset,
            self.n_steps - self.n_steps,
            self.n_steps,
        )
        x_dq = (x_q + offset) * scale
        return x_dq


class TestRangeLearning:
    def get_gradients(self, autograd_fn, encoding_min, encoding_max, input_max=10):
        torch.manual_seed(0)
        encoding_min = torch.nn.Parameter(
            torch.FloatTensor([encoding_min]), requires_grad=True
        )
        encoding_max = torch.nn.Parameter(
            torch.FloatTensor([encoding_max]), requires_grad=True
        )
        x = torch.FloatTensor(20, 10, 30, 50).uniform_(-input_max, input_max)
        y = autograd_fn(x, encoding_min, encoding_max)
        y.sum().backward()

        def get_detached_grad(var):
            if var.grad is not None:
                return var.grad.detach()
            else:
                return torch.zeros_like(var)

        return (
            get_detached_grad(encoding_min),
            get_detached_grad(encoding_max),
            y.detach(),
        )

    @pytest.mark.parametrize("encoding_min, encoding_max", [(-5, 0), (-5, 5), (0, 5)])
    def test_asymmetric_quantizer(self, encoding_min, encoding_max):
        """
        check if it has identical gradient compared to autograd function
        """
        tensor_quantizer = LearnedGridTensorQuantizer(
            bitwidth=8,
            round_mode="nearest",
            quant_scheme=QuantScheme.training_range_learning_with_tf_init,
            use_symmetric_encodings=False,
            enabled_by_default=True,
            data_type=QuantizationDataType.int,
        )

        tensor_quantizer.use_unsigned_symmetric = True
        auto_quantizer = RangeLearningAsymAutograd(bitwidth=8)

        aimet_grad_min, aimet_grad_max, aimet_xq = self.get_gradients(
            tensor_quantizer.quantize_dequantize, encoding_min, encoding_max
        )
        auto_grad_min, auto_grad_max, auto_xq = self.get_gradients(
            auto_quantizer.forward, encoding_min, encoding_max
        )

        assert torch.allclose(aimet_xq, auto_xq)
        assert torch.allclose(aimet_grad_min, auto_grad_min)
        assert torch.allclose(aimet_grad_max, auto_grad_max)

    def test_symmetric_quantizer(self):
        tensor_quantizer = LearnedGridTensorQuantizer(
            bitwidth=8,
            round_mode="nearest",
            quant_scheme=QuantScheme.training_range_learning_with_tf_init,
            use_symmetric_encodings=True,
            enabled_by_default=True,
            data_type=QuantizationDataType.int,
        )

        tensor_quantizer.use_unsigned_symmetric = False
        auto_quantizer = RangeLearningSymAutograd(bitwidth=8)

        init_encoding_min = -5
        init_encoding_max = 5
        aimet_grad_min, aimet_grad_max, aimet_xq = self.get_gradients(
            tensor_quantizer.quantize_dequantize, init_encoding_min, init_encoding_max
        )
        auto_grad_min, auto_grad_max, auto_xq = self.get_gradients(
            auto_quantizer.forward, init_encoding_min, init_encoding_max
        )

        assert torch.allclose(aimet_xq, auto_xq)
        assert torch.allclose(aimet_grad_min, -auto_grad_max)
        assert torch.allclose(aimet_grad_max, auto_grad_max)

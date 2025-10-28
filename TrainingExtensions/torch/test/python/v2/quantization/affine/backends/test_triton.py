# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

if torch.cuda.is_available():
    from aimet_torch.v2.quantization.affine.backends.triton import (
        TritonQuantize,
        TritonDequantize,
        TritonQuantizeDequantize,
    )
    from aimet_torch.v2.quantization.affine.backends.torch_builtins import (
        quantize,
        dequantize,
        quantize_dequantize,
    )

    @pytest.mark.parametrize("seed", range(5))
    def test_quantize_per_tensor(seed):
        """
        Triton quantize kernel should should produce close-enough output
        as PyTorch built-in quantize function.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        input = torch.randn((512, 512), dtype=torch.float32, device="cuda")
        scale = torch.tensor(0.1, dtype=torch.float32, device="cuda")
        offset = torch.tensor(0, dtype=torch.float32, device="cuda")
        output_torch = quantize(input, scale, offset, -128, 127)
        output_triton = TritonQuantize.apply(input, scale, offset, -128, 127)
        assert torch.allclose(output_triton, output_torch, atol=1)

    @pytest.mark.parametrize("seed", range(5))
    def test_dequantize_per_tensor(seed):
        """
        Triton dequantize kernel should should produce close-enough output
        as PyTorch built-in dequantize function.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        input = torch.randn((512, 512), dtype=torch.float32, device="cuda")
        scale = torch.tensor(0.1, dtype=torch.float32, device="cuda")
        offset = torch.tensor(0, dtype=torch.float32, device="cuda")
        output_torch = dequantize(input, scale, offset)
        output_triton = TritonDequantize.apply(input, scale, offset)
        assert torch.allclose(output_triton, output_torch)

    @pytest.mark.parametrize("seed", range(5))
    def test_quantize_dequantize_per_tensor(seed):
        """
        Triton quantize_dequantize kernel should should produce close-enough output
        as PyTorch built-in quantize_dequantize function.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        input = torch.randn(
            (512, 512), dtype=torch.float32, device="cuda", requires_grad=True
        )
        scale = torch.tensor(
            0.1, dtype=torch.float32, device="cuda", requires_grad=True
        )
        offset = torch.tensor(0, dtype=torch.float32, device="cuda", requires_grad=True)
        output_triton = TritonQuantizeDequantize.apply(input, scale, offset, -128, 127)
        loss = torch.nn.functional.mse_loss(output_triton, input.detach())
        loss.backward()

        input_ = input.clone().detach().requires_grad_(True)
        scale_ = scale.clone().detach().requires_grad_(True)
        offset_ = offset.clone().detach().requires_grad_(True)
        output_torch = quantize_dequantize(input_, scale_, offset_, -128, 127)
        loss = torch.nn.functional.mse_loss(output_torch, input_.detach())
        loss.backward()

        assert torch.allclose(output_triton, output_torch, atol=scale.item())
        assert torch.equal(input.grad, input_.grad)
        assert torch.allclose(scale.grad, scale_.grad)
        assert torch.allclose(offset.grad, offset_.grad)

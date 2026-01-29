# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from packaging.version import parse
import pytest
import itertools
import torch
import numpy as np

try:
    import triton
except ImportError:
    triton = None

if torch.cuda.is_available() and triton and parse(triton.__version__) >= parse("3.0.0"):
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
    import aimet_torch.experimental.pgs

    @pytest.fixture(params=[True, False], scope="function")
    def enable_pgs(request):
        enable = request.param

        if enable:
            try:
                aimet_torch.experimental.pgs.enable_pgs(eps=0.1, multiplier=3.0)
                yield
            finally:
                aimet_torch.experimental.pgs.disable_pgs()
        else:
            aimet_torch.experimental.pgs.disable_pgs()
            yield

    @pytest.mark.parametrize("seed", range(5))
    def test_quantize_per_tensor(seed):
        """
        Triton quantize kernel should should produce close-enough output
        as PyTorch built-in quantize function.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        input = torch.randn((512, 512), dtype=torch.float32, device="cuda")
        scale = torch.tensor(0.01, dtype=torch.float32, device="cuda")
        offset = torch.tensor(0, dtype=torch.float32, device="cuda")
        output_torch = quantize(input, scale, offset, -128, 127)
        output_triton = TritonQuantize.apply(input, scale, offset, -128, 127, None)
        assert torch.allclose(output_triton, output_torch, atol=1)

    @pytest.mark.parametrize("channel_axis", range(4))
    @pytest.mark.parametrize("seed", range(5))
    def test_quantize_per_channel(seed, channel_axis: int):
        """
        Triton quantize kernel should should produce close-enough output
        as PyTorch built-in quantize function.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        input = torch.randn((32, 32, 32, 32), dtype=torch.float32, device="cuda")
        scale_shape = [-1 if axis == channel_axis else 1 for axis in range(input.dim())]
        scale = torch.arange(
            0.01,
            0.33,
            step=0.01,
            dtype=torch.float32,
            device="cuda",
        ).view(*scale_shape)
        offset = torch.zeros(32, dtype=torch.float32, device="cuda").view(*scale_shape)
        output_torch = quantize(input, scale, offset, -128, 127)
        output_triton = TritonQuantize.apply(input, scale, offset, -128, 127, None)
        assert torch.allclose(output_triton, output_torch, atol=1)

    @pytest.mark.parametrize(
        "channel_axis, block_axis", itertools.combinations(range(4), 2)
    )
    @pytest.mark.parametrize("seed", range(5))
    def test_quantize_per_block(seed, channel_axis: int, block_axis: int):
        """
        Triton quantize kernel should should produce close-enough output
        as PyTorch built-in quantize function.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        input = torch.randn((32, 32, 32, 32), dtype=torch.float32, device="cuda")
        block_size = tuple(
            dim // 2 if axis == block_axis else 1 if axis == channel_axis else dim
            for axis, dim in enumerate(input.shape)
        )
        scale_shape = tuple(
            dim // block_size for dim, block_size in zip(input.shape, block_size)
        )
        scale = torch.arange(
            0.01,
            0.65,
            step=0.01,
            dtype=torch.float32,
            device="cuda",
        ).view(*scale_shape)
        offset = torch.zeros(64, dtype=torch.float32, device="cuda").view(*scale_shape)
        output_torch = quantize(input, scale, offset, -128, 127, block_size=block_size)
        output_triton = TritonQuantize.apply(
            input, scale, offset, -128, 127, block_size
        )
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
        scale = torch.tensor(0.01, dtype=torch.float32, device="cuda")
        offset = torch.tensor(0, dtype=torch.float32, device="cuda")
        output_torch = dequantize(input, scale, offset)
        output_triton = TritonDequantize.apply(input, scale, offset, None)
        assert torch.allclose(output_triton, output_torch)

    @pytest.mark.parametrize("channel_axis", range(4))
    @pytest.mark.parametrize("seed", range(5))
    def test_dequantize_per_channel(seed: int, channel_axis: int):
        """
        Triton dequantize kernel should should produce close-enough output
        as PyTorch built-in dequantize function.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        input = torch.randn((32, 32, 32, 32), dtype=torch.float32, device="cuda")
        scale_shape = [-1 if axis == channel_axis else 1 for axis in range(input.dim())]
        scale = torch.arange(
            0.01,
            0.33,
            step=0.01,
            dtype=torch.float32,
            device="cuda",
        ).view(*scale_shape)
        offset = torch.zeros(32, dtype=torch.float32, device="cuda").view(*scale_shape)
        output_torch = dequantize(input, scale, offset)
        output_triton = TritonDequantize.apply(input, scale, offset, None)
        assert torch.allclose(output_triton, output_torch)

    @pytest.mark.parametrize(
        "channel_axis, block_axis", itertools.combinations(range(4), 2)
    )
    @pytest.mark.parametrize("seed", range(5))
    def test_dequantize_per_block(seed, channel_axis: int, block_axis: int):
        """
        Triton dequantize kernel should should produce close-enough output
        as PyTorch built-in dequantize function.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        input = torch.randn((32, 32, 32, 32), dtype=torch.float32, device="cuda")
        block_size = tuple(
            dim // 2 if axis == block_axis else 1 if axis == channel_axis else dim
            for axis, dim in enumerate(input.shape)
        )
        scale_shape = tuple(
            dim // block_size for dim, block_size in zip(input.shape, block_size)
        )
        scale = torch.arange(
            0.01,
            0.65,
            step=0.01,
            dtype=torch.float32,
            device="cuda",
        ).view(*scale_shape)
        offset = torch.zeros(64, dtype=torch.float32, device="cuda").view(*scale_shape)
        output_torch = dequantize(input, scale, offset, block_size=block_size)
        output_triton = TritonDequantize.apply(input, scale, offset, block_size)
        assert torch.allclose(output_triton, output_torch, atol=1)

    @pytest.mark.parametrize("input_requires_grad", [True, False])
    @pytest.mark.parametrize("scale_requires_grad", [True, False])
    @pytest.mark.parametrize("offset_requires_grad", [True, False])
    @pytest.mark.parametrize("seed", range(5))
    def test_quantize_dequantize_per_tensor(
        input_requires_grad: bool,
        scale_requires_grad: bool,
        offset_requires_grad: bool,
        enable_pgs,
        seed: int,
    ):
        """
        Triton quantize_dequantize kernel should should produce close-enough output
        as PyTorch built-in quantize_dequantize function.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        input = torch.randn(
            (512, 512),
            dtype=torch.float32,
            device="cuda",
            requires_grad=input_requires_grad,
        )
        scale = torch.tensor(
            0.01, dtype=torch.float32, device="cuda", requires_grad=scale_requires_grad
        )
        offset = torch.tensor(
            -1, dtype=torch.float32, device="cuda", requires_grad=offset_requires_grad
        )
        output_triton = TritonQuantizeDequantize.apply(
            input, scale, offset, -128, 127, None
        )
        loss = torch.nn.functional.mse_loss(output_triton, input.detach())
        if loss.requires_grad:
            loss.backward()

        input_ = input.clone().detach().requires_grad_(input_requires_grad)
        scale_ = scale.clone().detach().requires_grad_(scale_requires_grad)
        offset_ = offset.clone().detach().requires_grad_(offset_requires_grad)
        output_torch = quantize_dequantize(input_, scale_, offset_, -128, 127)
        loss = torch.nn.functional.mse_loss(output_torch, input_.detach())
        if loss.requires_grad:
            loss.backward()

        assert torch.allclose(output_triton, output_torch, atol=scale.item())

        if input_requires_grad:
            assert input.grad is not None
            output_eq = output_triton == output_torch

            if aimet_torch.experimental.pgs.is_pgs_enabled():
                # When PGS is enabled, the gradients may not be exactly equal
                # even where outputs are equal due to precision error near PGS boundaries
                pgs_multiplier = aimet_torch.experimental.pgs.get_pgs_multiplier()
                grad_eq = input.grad[output_eq] == input_.grad[output_eq]
                assert grad_eq.sum() / output_eq.sum() > 0.999
                assert torch.all(
                    grad_eq
                    | (input.grad[output_eq] == input_.grad[output_eq] * pgs_multiplier)
                    | (input.grad[output_eq] * pgs_multiplier == input_.grad[output_eq])
                )
            else:
                assert torch.equal(input.grad[output_eq], input_.grad[output_eq])

            # Given MSE loss,
            # `grad_x = 2 * (x_qdq - x) / x.numel()`,
            # where `x_qdq` can differ by at most `scale` between triton and torch.
            atol = scale.item() * 2 / input.numel()

            if aimet_torch.experimental.pgs.is_pgs_enabled():
                # When PGS is enabled, the gradients can further
                # differ by a factor of pgs_multiplier
                pgs_multiplier = aimet_torch.experimental.pgs.get_pgs_multiplier()
                atol *= pgs_multiplier
                isclose = torch.isclose(input.grad, input_.grad, atol=atol)
                assert isclose.sum() / input.numel() > 0.999
                assert torch.all(
                    isclose
                    | torch.isclose(input.grad, input_.grad * pgs_multiplier, atol=atol)
                    | torch.isclose(input.grad * pgs_multiplier, input_.grad, atol=atol)
                )
            else:
                torch.allclose(input.grad, input_.grad, atol=atol)

        else:
            assert input.grad is None

        if scale_requires_grad:
            assert scale.grad is not None
            assert torch.allclose(scale.grad, scale_.grad)
        else:
            assert scale.grad is None

        if offset_requires_grad:
            assert offset.grad is not None
            assert torch.allclose(offset.grad, offset_.grad)
        else:
            assert offset.grad is None

    @pytest.mark.parametrize("input_requires_grad", [True, False])
    @pytest.mark.parametrize("scale_requires_grad", [True, False])
    @pytest.mark.parametrize("offset_requires_grad", [True, False])
    @pytest.mark.parametrize("channel_axis", range(4))
    @pytest.mark.parametrize("seed", range(5))
    def test_quantize_dequantize_per_channel(
        channel_axis: int,
        input_requires_grad: bool,
        scale_requires_grad: bool,
        offset_requires_grad: bool,
        enable_pgs,
        seed: int,
    ):
        """
        Triton quantize_dequantize kernel should should produce close-enough output
        as PyTorch built-in quantize_dequantize function.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        input = torch.randn(
            (32, 32, 32, 32),
            dtype=torch.float32,
            device="cuda",
            requires_grad=input_requires_grad,
        )
        scale_shape = [-1 if axis == channel_axis else 1 for axis in range(input.dim())]
        scale = torch.arange(
            0.01,
            0.33,
            step=0.01,
            dtype=torch.float32,
            device="cuda",
        ).view(*scale_shape)
        scale.requires_grad_(scale_requires_grad)
        offset = -torch.ones(
            32,
            dtype=torch.float32,
            device="cuda",
        ).view(*scale_shape)
        offset.requires_grad_(offset_requires_grad)

        output_triton = TritonQuantizeDequantize.apply(
            input, scale, offset, -128, 127, None
        )
        loss = torch.nn.functional.mse_loss(output_triton, input.detach())
        if loss.requires_grad:
            loss.backward()

        input_ = input.clone().detach().requires_grad_(input_requires_grad)
        scale_ = scale.clone().detach().requires_grad_(scale_requires_grad)
        offset_ = offset.clone().detach().requires_grad_(offset_requires_grad)
        output_torch = quantize_dequantize(input_, scale_, offset_, -128, 127)
        loss = torch.nn.functional.mse_loss(output_torch, input_.detach())
        if loss.requires_grad:
            loss.backward()

        assert np.allclose(
            output_triton.cpu().detach().numpy(),
            output_torch.cpu().detach().numpy(),
            atol=scale.cpu().detach().numpy(),
        )

        if input_requires_grad:
            assert input.grad is not None
            output_eq = output_triton == output_torch

            if aimet_torch.experimental.pgs.is_pgs_enabled():
                # When PGS is enabled, the gradients may not be exactly equal
                # even where outputs are equal due to precision error near PGS boundaries
                pgs_multiplier = aimet_torch.experimental.pgs.get_pgs_multiplier()
                grad_eq = input.grad[output_eq] == input_.grad[output_eq]
                assert grad_eq.sum() / output_eq.sum() > 0.999
                assert torch.all(
                    grad_eq
                    | (input.grad[output_eq] == input_.grad[output_eq] * pgs_multiplier)
                    | (input.grad[output_eq] * pgs_multiplier == input_.grad[output_eq])
                )
            else:
                assert torch.equal(input.grad[output_eq], input_.grad[output_eq])

            # Given MSE loss,
            # `grad_x = 2 * (x_qdq - x) / x.numel()`,
            # where `x_qdq` can differ by at most `scale` between triton and torch.
            atol = scale.cpu().detach().numpy() * 2 / input.numel()
            input_grad = input.grad.cpu().detach().numpy()
            input_grad_ = input_.grad.cpu().detach().numpy()

            if aimet_torch.experimental.pgs.is_pgs_enabled():
                # When PGS is enabled, the gradients can further
                # differ by a factor of pgs_multiplier
                pgs_multiplier = aimet_torch.experimental.pgs.get_pgs_multiplier()
                atol *= pgs_multiplier
                isclose = np.isclose(input_grad, input_grad_, atol=atol)
                assert isclose.sum() / input.numel() > 0.999
                assert np.all(
                    isclose
                    | np.isclose(input_grad, input_grad_ * pgs_multiplier, atol=atol)
                    | np.isclose(input_grad * pgs_multiplier, input_grad_, atol=atol)
                )
            else:
                np.allclose(input_grad, input_grad_, atol=atol)

        else:
            assert input.grad is None

        if scale_requires_grad:
            assert scale.grad is not None
            assert torch.allclose(scale.grad, scale_.grad, rtol=1e-3)
        else:
            assert scale.grad is None

        if offset_requires_grad:
            assert offset.grad is not None
            assert torch.allclose(offset.grad, offset_.grad, rtol=1e-3)
        else:
            assert offset.grad is None

    @pytest.mark.parametrize("input_requires_grad", [False, True])
    @pytest.mark.parametrize("scale_requires_grad", [False, True])
    @pytest.mark.parametrize("offset_requires_grad", [False, True])
    @pytest.mark.parametrize(
        "channel_axis, block_axis", itertools.combinations(range(4), 2)
    )
    @pytest.mark.parametrize("seed", range(5))
    def test_quantize_dequantize_per_block(
        seed: int,
        channel_axis: int,
        block_axis: int,
        input_requires_grad: bool,
        scale_requires_grad: bool,
        offset_requires_grad: bool,
        enable_pgs,
    ):
        """
        Triton quantize_dequantize kernel should should produce close-enough output
        as PyTorch built-in quantize_dequantize function.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        input = torch.randn(
            (32, 32, 32, 32),
            dtype=torch.float32,
            device="cuda",
            requires_grad=input_requires_grad,
        )
        block_size = tuple(
            dim // 2 if axis == block_axis else 1 if axis == channel_axis else dim
            for axis, dim in enumerate(input.shape)
        )
        scale_shape = tuple(
            dim // block_size for dim, block_size in zip(input.shape, block_size)
        )
        scale = torch.arange(
            0.01,
            0.65,
            step=0.01,
            dtype=torch.float32,
            device="cuda",
        ).view(*scale_shape)
        scale.requires_grad_(scale_requires_grad)
        offset = -torch.ones(
            64,
            dtype=torch.float32,
            device="cuda",
        ).view(*scale_shape)
        offset.requires_grad_(offset_requires_grad)

        output_triton = TritonQuantizeDequantize.apply(
            input, scale, offset, -128, 127, block_size
        )
        loss = torch.nn.functional.mse_loss(output_triton, input.detach())
        if loss.requires_grad:
            loss.backward()

        input_ = input.clone().detach().requires_grad_(input_requires_grad)
        scale_ = scale.clone().detach().requires_grad_(scale_requires_grad)
        offset_ = offset.clone().detach().requires_grad_(offset_requires_grad)
        output_torch = quantize_dequantize(
            input_, scale_, offset_, -128, 127, block_size=block_size
        )
        loss = torch.nn.functional.mse_loss(output_torch, input_.detach())
        if loss.requires_grad:
            loss.backward()

        atol = scale.repeat_interleave(repeats=block_size[block_axis], dim=block_axis)
        assert np.allclose(
            output_triton.cpu().detach().numpy(),
            output_torch.cpu().detach().numpy(),
            atol=atol.cpu().detach().numpy(),
        )

        if input_requires_grad:
            assert input.grad is not None
            output_eq = output_triton == output_torch

            if aimet_torch.experimental.pgs.is_pgs_enabled():
                # When PGS is enabled, the gradients may not be exactly equal
                # even where outputs are equal due to precision error near PGS boundaries
                pgs_multiplier = aimet_torch.experimental.pgs.get_pgs_multiplier()
                grad_eq = input.grad[output_eq] == input_.grad[output_eq]
                assert grad_eq.sum() / output_eq.sum() > 0.999
                assert torch.all(
                    grad_eq
                    | (input.grad[output_eq] == input_.grad[output_eq] * pgs_multiplier)
                    | (input.grad[output_eq] * pgs_multiplier == input_.grad[output_eq])
                )
            else:
                assert torch.equal(input.grad[output_eq], input_.grad[output_eq])

            atol = atol.cpu().detach().numpy()
            input_grad = input.grad.cpu().detach().numpy()
            input_grad_ = input_.grad.cpu().detach().numpy()

            if aimet_torch.experimental.pgs.is_pgs_enabled():
                # When PGS is enabled, the gradients can further
                # differ by a factor of pgs_multiplier
                pgs_multiplier = aimet_torch.experimental.pgs.get_pgs_multiplier()
                atol *= pgs_multiplier
                isclose = np.isclose(input_grad, input_grad_, atol=atol)
                assert isclose.sum() / input.numel() > 0.999
                assert np.all(
                    isclose
                    | np.isclose(input_grad, input_grad_ * pgs_multiplier, atol=atol)
                    | np.isclose(input_grad * pgs_multiplier, input_grad_, atol=atol)
                )
            else:
                np.allclose(input_grad, input_grad_, atol=atol)
        else:
            assert input.grad is None

        if scale_requires_grad:
            assert scale.grad is not None
            assert torch.allclose(scale.grad, scale_.grad, rtol=1e-3)
        else:
            assert scale.grad is None

        if offset_requires_grad:
            assert offset.grad is not None
            assert torch.allclose(offset.grad, offset_.grad, rtol=1e-3)
        else:
            assert offset.grad is None

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=redefined-builtin, import-error, abstract-method, arguments-differ, no-member
from typing import Optional
import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def quantize_per_tensor(
    input_ptr,
    scale: tl.float32,
    offset: tl.float32,
    qmin: tl.int32,
    qmax: tl.int32,
    output_ptr,
    n_elements: tl.uint64,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start_index = pid * BLOCK_SIZE
    idx = start_index + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    input = tl.load(input_ptr + idx, mask=mask)
    input = input / scale - offset
    rounded = tl.floor(input + 0.5)
    clamped = tl.clamp(rounded, qmin, qmax)
    tl.store(output_ptr + idx, clamped, mask=mask)


@triton.jit
def quantize_per_channel(
    input_ptr,
    scale_ptr,
    offset_ptr,
    qmin: tl.int32,
    qmax: tl.int32,
    output_ptr,
    I: tl.uint64,
    J: tl.uint64,
    K: tl.uint64,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Quantize input of shape (I, J, K) with per-channel scale and offset
    where J is the channel dimension.

    input:                   scale:
          ┌───────────────┐     ┌┐
          ├───────────────┤     ├┤
          ├───────────────┤     ├┤
        J ├───────────────┤   J ├┤
          ├───────────────┤     ├┤
          ├───────────────┤     ├┤
          └───────────────┘     └┘
                  K

    Args:
        input_ptr: Pointer to the input tensor.
        scale_ptr: Pointer to the quantization scale (1D tensor of size J).
        offset_ptr: Pointer to the quantization offset (1D tensor of size J).
        qmin: Minimum value in the quantization grid.
        qmax: Maximum value in the quantization grid.
        output_ptr: Pointer to the output buffer.
        I: Size of the first dimension of the input tensor.
        J: Size of the second dimension of the input tensor (channel dimension).
        K: Size of the third dimension of the input tensor.
        BLOCK_SIZE: Number of elements to process per block.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.uint64)
    mask = idx < (I * J * K)
    j = (idx % (J * K)) // K
    input = tl.load(input_ptr + idx, mask=mask)
    scale = tl.load(scale_ptr + j)
    offset = tl.load(offset_ptr + j)
    input = input / scale - offset
    rounded = tl.floor(input + 0.5)
    clamped = tl.clamp(rounded, qmin, qmax)
    tl.store(output_ptr + idx, clamped, mask=mask)


@triton.jit
def quantize_dequantize_per_tensor(
    input_ptr,
    scale: tl.float32,
    offset: tl.float32,
    qmin: tl.int32,
    qmax: tl.int32,
    output_ptr,
    mask_ptr,
    n_elements: tl.uint64,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Args:
        input_ptr: Pointer to the input tensor.
        scale: Quantization scale (scalar).
        offset: Quantizaiton offset (scalar).
        qmin: Minimum value in the quantization grid.
        qmax: Maximum value in the quantization grid.
        output_ptr: Pointer to the output buffer.
        mask_ptr: If not None, store a boolean mask indicating whether each element was clamped.
        n_elements: Number of elements in the input tensor.
        BLOCK_SIZE: Number of elements to process per block.
    """
    pid = tl.program_id(0)
    start_index = pid * BLOCK_SIZE
    idx = start_index + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    input = tl.load(input_ptr + idx, mask=mask)
    input = input / scale - offset
    rounded = tl.floor(input + 0.5)
    clamped = tl.clamp(rounded, qmin, qmax)
    tl.store(output_ptr + idx, (clamped + offset) * scale, mask=mask)

    if mask_ptr is not None:
        tl.store(mask_ptr + idx, (clamped == rounded), mask=mask)


@triton.jit
def quantize_dequantize_per_channel(
    input_ptr,
    scale_ptr,
    offset_ptr,
    qmin: tl.int32,
    qmax: tl.int32,
    output_ptr,
    mask_ptr,
    I: tl.uint64,
    J: tl.uint64,
    K: tl.uint64,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Quantize-dequantize input of shape (I, J, K) with per-channel scale and offset
    where J is the channel dimension.

    input:                   scale:
          ┌───────────────┐     ┌┐
          ├───────────────┤     ├┤
          ├───────────────┤     ├┤
        J ├───────────────┤   J ├┤
          ├───────────────┤     ├┤
          ├───────────────┤     ├┤
          └───────────────┘     └┘
                  K

    Args:
        input_ptr: Pointer to the input tensor.
        scale_ptr: Pointer to the quantization scale (1D tensor of size J).
        offset_ptr: Pointer to the quantization offset (1D tensor of size J).
        qmin: Minimum value in the quantization grid.
        qmax: Maximum value in the quantization grid.
        output_ptr: Pointer to the output buffer.
        mask_ptr: If not None, store a boolean mask indicating whether each element was clamped.
        I: Size of the first dimension of the input tensor.
        J: Size of the second dimension of the input tensor (channel dimension).
        K: Size of the third dimension of the input tensor.
        BLOCK_SIZE: Number of elements to process per block.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.uint64)
    mask = idx < (I * J * K)
    j = (idx % (J * K)) // K
    input = tl.load(input_ptr + idx, mask=mask)
    scale = tl.load(scale_ptr + j)
    offset = tl.load(offset_ptr + j)
    input = input / scale - offset
    rounded = tl.floor(input + 0.5)
    clamped = tl.clamp(rounded, qmin, qmax)
    tl.store(output_ptr + idx, (clamped + offset) * scale, mask=mask)

    if mask_ptr is not None:
        tl.store(mask_ptr + idx, (clamped == rounded), mask=mask)


@triton.jit
def quantize_dequantize_per_tensor_backward(
    output_grad_ptr,
    input_ptr,
    scale_ptr,
    offset_ptr,
    mask_ptr,
    input_grad_ptr,
    scale_grad_ptr,
    offset_grad_ptr,
    qmin: tl.int32,
    qmax: tl.int32,
    n_elements: tl.uint64,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start_index = pid * BLOCK_SIZE
    idx = start_index + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements

    output_grad = tl.load(output_grad_ptr + idx, mask=mask)
    grad_mask = tl.load(mask_ptr + idx, mask=mask)

    if input_grad_ptr is not None:
        input_grad = output_grad * grad_mask
        tl.store(input_grad_ptr + idx, input_grad, mask=mask)

    if scale_grad_ptr is not None:
        input = tl.load(input_ptr + idx, mask=mask)
        scale = tl.load(scale_ptr)
        offset = tl.load(offset_ptr)
        scaled = input / scale - offset
        rounded = tl.floor(scaled + 0.5)
        clamped = tl.clamp(rounded, qmin, qmax)
        scale_grad = (output_grad * (clamped - scaled * grad_mask)).sum()
        tl.atomic_add(scale_grad_ptr, scale_grad)

    if offset_grad_ptr is not None:
        scale = tl.load(scale_ptr)
        offset_grad = (output_grad * scale * ~grad_mask).sum()
        tl.atomic_add(offset_grad_ptr, offset_grad)


@triton.jit
def quantize_dequantize_per_channel_backward(
    output_grad_ptr,
    input_ptr,
    scale_ptr,
    offset_ptr,
    mask_ptr,
    input_grad_ptr,
    scale_grad_ptr,
    offset_grad_ptr,
    qmin: tl.int32,
    qmax: tl.int32,
    I: tl.uint64,
    J: tl.uint64,
    K: tl.uint64,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.uint64)
    mask = idx < (I * J * K)
    j = (idx % (J * K)) // K

    output_grad = tl.load(output_grad_ptr + idx, mask=mask)
    grad_mask = tl.load(mask_ptr + idx, mask=mask)

    if input_grad_ptr is not None:
        input_grad = output_grad * grad_mask
        tl.store(input_grad_ptr + idx, input_grad, mask=mask)

    if scale_grad_ptr is not None:
        input = tl.load(input_ptr + idx, mask=mask)
        scale = tl.load(scale_ptr + j)
        offset = tl.load(offset_ptr + j)
        scaled = input / scale - offset
        rounded = tl.floor(scaled + 0.5)
        clamped = tl.clamp(rounded, qmin, qmax)
        scale_grad = output_grad * (clamped - scaled * grad_mask)
        tl.store(scale_grad_ptr + idx, scale_grad, mask=mask)

    if offset_grad_ptr is not None:
        scale = tl.load(scale_ptr + j)
        offset_grad = output_grad * scale * ~grad_mask
        tl.store(offset_grad_ptr + idx, offset_grad, mask=mask)


@triton.jit
def dequantize_per_tensor(
    input_ptr,
    scale: tl.float32,
    offset: tl.float32,
    output_ptr,
    n_elements: tl.uint64,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start_index = pid * BLOCK_SIZE
    idx = start_index + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    input = tl.load(input_ptr + idx, mask=mask)
    tl.store(output_ptr + idx, (input + offset) * scale, mask=mask)


@triton.jit
def dequantize_per_channel(
    input_ptr,
    scale_ptr,
    offset_ptr,
    output_ptr,
    I: tl.uint64,
    J: tl.uint64,
    K: tl.uint64,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Dequantize input of shape (I, J, K) with per-channel scale and offset
    where J is the channel dimension.

    input:                   scale:
          ┌───────────────┐     ┌┐
          ├───────────────┤     ├┤
          ├───────────────┤     ├┤
        J ├───────────────┤   J ├┤
          ├───────────────┤     ├┤
          ├───────────────┤     ├┤
          └───────────────┘     └┘
                  K

    Args:
        input_ptr: Pointer to the input tensor.
        scale_ptr: Pointer to the quantization scale (1D tensor of size J).
        offset_ptr: Pointer to the quantization offset (1D tensor of size J).
        output_ptr: Pointer to the output buffer.
        I: Size of the first dimension of the input tensor.
        J: Size of the second dimension of the input tensor (channel dimension).
        K: Size of the third dimension of the input tensor.
        BLOCK_SIZE: Number of elements to process per block.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.uint64)
    mask = idx < (I * J * K)
    j = (idx % (J * K)) // K
    input = tl.load(input_ptr + idx, mask=mask)
    scale = tl.load(scale_ptr + j)
    offset = tl.load(offset_ptr + j)
    tl.store(output_ptr + idx, (input + offset) * scale, mask=mask)


def _get_channel_axis(scale: torch.Tensor):
    non_singleton_axes = [axis for axis, size in enumerate(scale.shape) if size != 1]

    if not non_singleton_axes:
        return None

    if len(non_singleton_axes) > 1:
        raise NotImplementedError(
            "Only per-tensor and per-channel quantization are supported."
        )

    (channel_axis,) = non_singleton_axes
    return channel_axis


def _validate_channel_axis(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    channel_axis: Optional[int],
):
    if channel_axis != _get_channel_axis(offset):
        raise RuntimeError(
            "Scale and offset must have same non-singleton axis for per-channel quantization. "
            f"Got scale shape {scale.shape} and offset shape {offset.shape}."
        )

    if channel_axis is not None:
        if scale.shape[channel_axis] != offset.shape[channel_axis]:
            raise RuntimeError(
                "Scale and offset must have same size along the channel axis for per-channel quantization. "
                f"Got scale shape {tuple(scale.shape)} and offset shape {tuple(offset.shape)}."
            )

        if scale.numel() != tensor.shape[channel_axis]:
            raise RuntimeError(
                f"Input shape {tuple(tensor.shape)} is incompatible with "
                f"scale shape {tuple(scale.shape)} "
                f"for per-channel quantization on axis {channel_axis}."
            )


class TritonQuantize(torch.autograd.Function):
    @staticmethod
    def forward(
        _,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        offset: torch.Tensor,
        qmin: int,
        qmax: int,
    ):
        channel_axis = _get_channel_axis(scale)
        _validate_channel_axis(tensor, scale, offset, channel_axis)

        output = torch.empty_like(tensor)
        BLOCK_SIZE = 1024
        NUM_BLOCKS = tensor.numel() // BLOCK_SIZE + 1

        if channel_axis is None:
            quantize_per_tensor[(NUM_BLOCKS,)](
                tensor,
                scale,
                offset,
                qmin,
                qmax,
                output,
                tensor.numel(),
                BLOCK_SIZE,
            )
        else:
            I = int(np.prod(tensor.shape[:channel_axis]))
            J = tensor.shape[channel_axis]
            K = int(np.prod(tensor.shape[channel_axis + 1 :]))

            quantize_per_channel[(NUM_BLOCKS,)](
                tensor,
                scale,
                offset,
                qmin,
                qmax,
                output,
                I,
                J,
                K,
                BLOCK_SIZE,
            )

        return output


class TritonDequantize(torch.autograd.Function):
    @staticmethod
    def forward(
        _,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        offset: torch.Tensor,
    ):
        channel_axis = _get_channel_axis(scale)
        _validate_channel_axis(tensor, scale, offset, channel_axis)

        output = torch.empty_like(tensor, dtype=torch.float32)
        BLOCK_SIZE = 1024
        NUM_BLOCKS = tensor.numel() // BLOCK_SIZE + 1

        if channel_axis is None:
            dequantize_per_tensor[(NUM_BLOCKS,)](
                tensor,
                scale,
                offset,
                output,
                tensor.numel(),
                BLOCK_SIZE,
            )
        else:
            I = int(np.prod(tensor.shape[:channel_axis]))
            J = tensor.shape[channel_axis]
            K = int(np.prod(tensor.shape[channel_axis + 1 :]))

            dequantize_per_channel[(NUM_BLOCKS,)](
                tensor,
                scale,
                offset,
                output,
                I,
                J,
                K,
                BLOCK_SIZE,
            )
        return output


class TritonQuantizeDequantize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        offset: torch.Tensor,
        qmin: int,
        qmax: int,
    ):
        channel_axis = _get_channel_axis(scale)
        _validate_channel_axis(tensor, scale, offset, channel_axis)

        if tensor.requires_grad or scale.requires_grad or offset.requires_grad:
            mask = torch.empty_like(tensor, dtype=torch.bool)
        else:
            mask = None

        output = torch.empty_like(tensor, dtype=torch.float32)
        BLOCK_SIZE = 1024
        NUM_BLOCKS = tensor.numel() // BLOCK_SIZE + 1

        if channel_axis is None:
            quantize_dequantize_per_tensor[(NUM_BLOCKS,)](
                tensor,
                scale,
                offset,
                qmin,
                qmax,
                output,
                mask,
                tensor.numel(),
                BLOCK_SIZE,
            )
        else:
            I = int(np.prod(tensor.shape[:channel_axis]))
            J = tensor.shape[channel_axis]
            K = int(np.prod(tensor.shape[channel_axis + 1 :]))

            quantize_dequantize_per_channel[(NUM_BLOCKS,)](
                tensor,
                scale,
                offset,
                qmin,
                qmax,
                output,
                mask,
                I,
                J,
                K,
                BLOCK_SIZE,
            )
            ctx.I = I
            ctx.J = J
            ctx.K = K

        ctx.save_for_backward(
            tensor if scale.requires_grad else None,
            scale if scale.requires_grad or offset.requires_grad else None,
            offset if scale.requires_grad else None,
            mask,
        )
        ctx.qmin = qmin
        ctx.qmax = qmax
        ctx.channel_axis = channel_axis
        ctx.tensor_requires_grad = tensor.requires_grad
        ctx.scale_requires_grad = scale.requires_grad
        ctx.offset_requires_grad = offset.requires_grad
        return output

    @staticmethod
    def backward(ctx, grad):
        input, scale, offset, mask = ctx.saved_tensors
        channel_axis = ctx.channel_axis

        input_grad = torch.empty_like(grad) if ctx.tensor_requires_grad else None

        BLOCK_SIZE = 1024
        NUM_BLOCKS = grad.numel() // BLOCK_SIZE + 1

        if channel_axis is None:
            scale_grad = torch.zeros_like(scale) if ctx.scale_requires_grad else None
            offset_grad = torch.zeros_like(scale) if ctx.offset_requires_grad else None
            quantize_dequantize_per_tensor_backward[(NUM_BLOCKS,)](
                grad,
                input,
                scale,
                offset,
                mask,
                input_grad,
                scale_grad,
                offset_grad,
                ctx.qmin,
                ctx.qmax,
                grad.numel(),
                BLOCK_SIZE,
            )
        else:
            I = ctx.I
            J = ctx.J
            K = ctx.K
            scale_grad = (
                torch.empty(grad.shape, dtype=scale.dtype, device=scale.device)
                if ctx.scale_requires_grad
                else None
            )
            offset_grad = (
                torch.empty(grad.shape, dtype=scale.dtype, device=scale.device)
                if ctx.offset_requires_grad
                else None
            )
            quantize_dequantize_per_channel_backward[(NUM_BLOCKS,)](
                grad,
                input,
                scale,
                offset,
                mask,
                input_grad,
                scale_grad,
                offset_grad,
                ctx.qmin,
                ctx.qmax,
                I,
                J,
                K,
                BLOCK_SIZE,
            )

        return input_grad, scale_grad, offset_grad, None, None

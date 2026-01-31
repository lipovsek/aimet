# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=all

import math
from itertools import chain, repeat
from typing import overload, Union, Tuple, Optional
from aimet_torch.v2.quantization.affine.backends import torch_builtins
import torch
from .utils import *
from aimet_torch.v2.utils import _torch_compiler_is_exporting


@overload
def quantize(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    bitwidth: Union[int, float],
    signed: bool = False,
    block_size: Optional[Tuple[int, ...]] = None,
): ...


@overload
def quantize(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    *,
    num_steps: int,
    signed: bool = False,
    block_size: Optional[Tuple[int, ...]] = None,
): ...


@overload
def quantize(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    qmin: int,
    qmax: int,
    block_size: Optional[Tuple[int, ...]] = None,
): ...


def quantize(
    tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, *args, **kwargs
):
    r"""
    Applies quantization to the input.

    Precisely,

    .. math::
        out = clamp\left(\left\lceil\frac{input}{scale}\right\rfloor - offset, qmin, qmax\right)

    If block size :math:`B = \begin{pmatrix} B_0  & B_1  & \cdots & B_{D-1} \end{pmatrix}` is specified,
    this equation will be further generalized as

    .. math::
        out_{j_0 \cdots j_{D-1}} & = clamp\left(
            \left\lceil\frac{input_{j_0 \cdots j_{D-1}}}{scale_{i_0 \cdots i_{D-1}}}\right\rfloor
            - offset_{i_0 \cdots i_{D-1}}, qmin, qmax\right)\\

        \text{where} \quad \forall_{0 \leq d < D} \quad i_d = \left\lfloor \frac{j_d}{B_d} \right\rfloor

    This function is overloaded with the signatures listed below:


    .. function:: quantize(tensor, scale, offset, bitwidth, signed=False, block_size=None)
       :noindex:

       Equivalent to:

       .. math::
           qmin=
           \begin{cases}
               -\left\lceil\frac{2^{bitwidth}-1}{2}\right\rceil,& \text{if } signed\\
               0,                                               & \text{otherwise   (default)}
           \end{cases}
           qmax=
           \begin{cases}
               \left\lfloor\frac{2^{bitwidth}-1}{2}\right\rfloor,& \text{if } signed\\
               2^{bitwidth}-1,                                   & \text{otherwise   (default)}
           \end{cases}

       :param Tensor tensor: Tensor to quantize
       :param Tensor scale: Scale for quantization
       :param Tensor offset: Offset for quantization
       :param int bitwidth: Bitwidth of quantized tensor based on which :math:`qmin` and :math:`qmax` will be derived
       :param bool signed: If false, the output will be mapped to positive integers only.
           Otherwise, it will range over both positive and negative integers.
       :param block_size: Block size
       :type block_size: Tuple[int, ...], optional

    .. function:: quantize(tensor, scale, offset, *, num_steps, signed=False, block_size=None)
       :noindex:

       Equivalent to:

       .. math::
           qmin=
           \begin{cases}
               -\left\lceil\frac{num\_steps}{2}\right\rceil,& \text{if } signed\\
               0,                                           & \text{otherwise   (default)}
           \end{cases}
           qmax=
           \begin{cases}
               \left\lfloor\frac{num\_steps}{2}\right\rfloor,& \text{if } signed\\
               num\_steps,                                   & \text{otherwise   (default)}
           \end{cases}


       :param Tensor tensor: Tensor to quantize
       :param Tensor scale: Scale for quantization
       :param Tensor offset: Offset for quantization
       :param int num_steps: The number of steps in the quantization range based on which :math:`qmin` and :math:`qmax` will be derived
       :param bool signed: If false, the output will be mapped to positive integers only.
           Otherwise, it will range over both positive and negative integers.
       :param block_size: Block size
       :type block_size: Tuple[int, ...], optional

    .. function:: quantize(tensor, scale, offset, *, qmin, qmax, block_size=None)
       :noindex:

       :param Tensor tensor: Tensor to quantize
       :param Tensor scale: Scale for quantization
       :param Tensor offset: Offset for quantization
       :param int qmin: Minimum value of the quantization range
       :param int qmax: Maximum value of the quantization range
       :param block_size: Block size
       :type block_size: Tuple[int, ...], optional


    Examples:

        >>> import aimet_torch.v2.quantization as Q
        >>> input = torch.arange(start=-0.3, end=1.3, step=0.05)
        >>> print(input)
        tensor([-3.0000e-01, -2.5000e-01, -2.0000e-01, -1.5000e-01, -1.0000e-01,
                -5.0000e-02, -1.1921e-08,  5.0000e-02,  1.0000e-01,  1.5000e-01,
                2.0000e-01,  2.5000e-01,  3.0000e-01,  3.5000e-01,  4.0000e-01,
                4.5000e-01,  5.0000e-01,  5.5000e-01,  6.0000e-01,  6.5000e-01,
                7.0000e-01,  7.5000e-01,  8.0000e-01,  8.5000e-01,  9.0000e-01,
                9.5000e-01,  1.0000e+00,  1.0500e+00,  1.1000e+00,  1.1500e+00,
                1.2000e+00,  1.2500e+00])
        >>> scale = torch.tensor(1/15)
        >>> offset = torch.tensor(0.0)
        >>> Q.affine.quantize(input, scale, offset, bitwidth=4)
        tensor([ 0.,  0.,  0.,  0.,  0.,  0., -0.,  1.,  2.,  2.,  3.,  4.,  4.,  5.,
                 6.,  7.,  7.,  8.,  9., 10., 10., 11., 12., 13., 13., 14., 15., 15.,
                 15., 15., 15., 15.])
        >>> Q.affine.quantize(input, scale, offset, num_steps=15)
        tensor([ 0.,  0.,  0.,  0.,  0.,  0., -0.,  1.,  2.,  2.,  3.,  4.,  4.,  5.,
                 6.,  7.,  7.,  8.,  9., 10., 10., 11., 12., 13., 13., 14., 15., 15.,
                 15., 15., 15., 15.])
        >>> Q.affine.quantize(input, scale, offset, qmin=0, qmax=15)
        tensor([ 0.,  0.,  0.,  0.,  0.,  0., -0.,  1.,  2.,  2.,  3.,  4.,  4.,  5.,
                 6.,  7.,  7.,  8.,  9., 10., 10., 11., 12., 13., 13., 14., 15., 15.,
                 15., 15., 15., 15.])
    """
    qmin, qmax, block_size, zero_point_shift = _parse_args(args, kwargs)
    if zero_point_shift != 0.0:
        raise RuntimeError("Nonzero zero_point_shift not supported for quantize()")

    if _torch_compiler_is_exporting() or torch.onnx.is_in_onnx_export():
        backend = torch_builtins
    else:
        backend = get_backend()

    return backend.quantize(tensor, scale, offset, qmin, qmax, block_size)


@overload
def quantize_dequantize(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    bitwidth: Union[int, float],
    signed: bool = False,
    block_size: Optional[Tuple[int, ...]] = None,
    zero_point_shift: Optional[float] = None,
): ...


@overload
def quantize_dequantize(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    *,
    num_steps: int,
    signed: bool = False,
    block_size: Optional[Tuple[int, ...]] = None,
    zero_point_shift: Optional[float] = None,
): ...


@overload
def quantize_dequantize(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    qmin: int,
    qmax: int,
    block_size: Optional[Tuple[int, ...]] = None,
    zero_point_shift: Optional[float] = None,
): ...


def quantize_dequantize(
    tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, *args, **kwargs
):
    r"""
    Applies fake-quantization by quantizing and dequantizing the input.

    Precisely,

    .. math::
        out = (\overline{input} + offset) * scale

    where

    .. math::
        \overline{input} = clamp\left(\left\lceil\frac{input}{scale}\right\rfloor - offset, qmin, qmax\right)


    If block size :math:`B = \begin{pmatrix} B_0  & B_1  & \cdots & B_{D-1} \end{pmatrix}` is specified,
    this equation will be further generalized as

    .. math::
        out_{j_0 \cdots j_{D-1}} &= (\overline{input}_{j_0 \cdots j_{D-1}} + offset_{i_0 \cdots i_{D-1}}) * scale_{i_0 \cdots i_{D-1}}\\
        \overline{input}_{j_0 \cdots j_{D-1}} &= clamp\left(
            \left\lceil\frac{input_{j_0 \cdots j_{D-1}}}{scale_{i_0 \cdots i_{D-1}}}\right\rfloor
            - offset_{i_0 \cdots i_{D-1}}, qmin, qmax\right)\\

        \text{where } \quad \forall_{0 \leq d < D} \quad i_d = \left\lfloor \frac{j_d}{B_d} \right\rfloor


    This function is overloaded with the signatures listed below:


    .. function:: quantize_dequantize(tensor, scale, offset, bitwidth, signed=False, block_size=None)
       :noindex:

       Equivalent to:

       .. math::
           qmin=
           \begin{cases}
               -\left\lceil\frac{2^{bitwidth}-1}{2}\right\rceil,& \text{if } signed\\
               0,                                               & \text{otherwise   (default)}
           \end{cases}
           qmax=
           \begin{cases}
               \left\lfloor\frac{2^{bitwidth}-1}{2}\right\rfloor,& \text{if } signed\\
               2^{bitwidth}-1,                                   & \text{otherwise   (default)}
           \end{cases}

       :param Tensor tensor: Tensor to quantize
       :param Tensor scale: Scale for quantization
       :param Tensor offset: Offset for quantization
       :param int bitwidth: Bitwidth of quantized tensor based on which :math:`qmin` and :math:`qmax` will be derived
       :param bool signed: If false, :math:`\overline{input}` will be mapped to positive integers only.
           Otherwise, :math:`\overline{input}` will range over both positive and negative integers.
       :param block_size: Block size
       :type block_size: Tuple[int, ...], optional

    .. function:: quantize_dequantize(tensor, scale, offset, *, num_steps, signed=False, block_size=None)
       :noindex:

       Equivalent to:

       .. math::
           qmin=
           \begin{cases}
               -\left\lceil\frac{num\_steps}{2}\right\rceil,& \text{if } signed\\
               0,                                           & \text{otherwise   (default)}
           \end{cases}
           qmax=
           \begin{cases}
               \left\lfloor\frac{num\_steps}{2}\right\rfloor,& \text{if } signed\\
               num\_steps,                                   & \text{otherwise   (default)}
           \end{cases}


       :param Tensor tensor: Tensor to quantize
       :param Tensor scale: Scale for quantization
       :param Tensor offset: Offset for quantization
       :param int num_steps: The number of steps in the quantization range based on which :math:`qmin` and :math:`qmax` will be derived
       :param bool signed: If false, :math:`\overline{input}` will be mapped to positive integers only.
           Otherwise, :math:`\overline{input}` will range over both positive and negative integers.
       :param block_size: Block size
       :type block_size: Tuple[int, ...], optional

    .. function:: quantize_dequantize(tensor, scale, offset, *, qmin, qmax, block_size=None)
       :noindex:

       :param Tensor tensor: Tensor to quantize
       :param Tensor scale: Scale for quantization
       :param Tensor offset: Offset for quantization
       :param int qmin: Minimum value of the quantization range
       :param int qmax: Maximum value of the quantization range
       :param block_size: Block size
       :type block_size: Tuple[int, ...], optional


    Examples:

        >>> import aimet_torch.v2.quantization as Q
        >>> input = torch.arange(start=-0.3, end=1.3, step=0.05)
        >>> print(input)
        tensor([-3.0000e-01, -2.5000e-01, -2.0000e-01, -1.5000e-01, -1.0000e-01,
                -5.0000e-02, -1.1921e-08,  5.0000e-02,  1.0000e-01,  1.5000e-01,
                2.0000e-01,  2.5000e-01,  3.0000e-01,  3.5000e-01,  4.0000e-01,
                4.5000e-01,  5.0000e-01,  5.5000e-01,  6.0000e-01,  6.5000e-01,
                7.0000e-01,  7.5000e-01,  8.0000e-01,  8.5000e-01,  9.0000e-01,
                9.5000e-01,  1.0000e+00,  1.0500e+00,  1.1000e+00,  1.1500e+00,
                1.2000e+00,  1.2500e+00])
        >>> scale = torch.tensor(1/15)
        >>> offset = torch.tensor(0.0)
        >>> Q.affine.quantize_dequantize(input, scale, offset, bitwidth=4)
        tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0667, 0.1333,
                0.1333, 0.2000, 0.2667, 0.2667, 0.3333, 0.4000, 0.4667, 0.4667, 0.5333,
                0.6000, 0.6667, 0.6667, 0.7333, 0.8000, 0.8667, 0.8667, 0.9333, 1.0000,
                1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
        >>> Q.affine.quantize_dequantize(input, scale, offset, num_steps=15)
        tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0667, 0.1333,
                0.1333, 0.2000, 0.2667, 0.2667, 0.3333, 0.4000, 0.4667, 0.4667, 0.5333,
                0.6000, 0.6667, 0.6667, 0.7333, 0.8000, 0.8667, 0.8667, 0.9333, 1.0000,
                1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
        >>> Q.affine.quantize_dequantize(input, scale, offset, qmin=0, qmax=15)
        tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0667, 0.1333,
                0.1333, 0.2000, 0.2667, 0.2667, 0.3333, 0.4000, 0.4667, 0.4667, 0.5333,
                0.6000, 0.6667, 0.6667, 0.7333, 0.8000, 0.8667, 0.8667, 0.9333, 1.0000,
                1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
    """
    qmin, qmax, block_size, zero_point_shift = _parse_args(args, kwargs)

    #                                             torch.onnx.is_in_onnx_export
    #                      |                True                   |         False
    #                ------|---------------------------------------|----------------------
    #                 True |  torch.onnx.export(..., dynamo=True)  |  torch.export.export
    #  torch               |     (Dynamo-based ONNX export)        |   (ExportedProgram)
    # .compiler      ------|---------------------------------------|----------------------
    # .is_exporting  False |  torch.onnx.export(..., dynamo=False) |    not in export
    #                      |    (TorchScript-based ONNX export)    |

    if _torch_compiler_is_exporting() and torch.onnx.is_in_onnx_export():
        # Dynamo-based ONNX export (torch.onnx.export(..., dynamo=True))
        # Call torch.ops.aimet.quantize_dequantize that dynamo tracer can
        # capture as a single torch.ops.aimet.quantize_dequantize node
        backend = torch.ops.aimet
    elif _torch_compiler_is_exporting() or torch.onnx.is_in_onnx_export():
        # TorchScript-based ONNX export (torch.onnx.export(..., dynamo=False))
        # or ExportedProgram export (torch.export.export)
        # Fall back to torch builtins backend which is exportable to ONNX/ExportedProgram
        backend = torch_builtins
    else:
        # Not in export mode. Use the globally set backend
        backend = get_backend()

    return backend.quantize_dequantize(
        tensor, scale, offset, qmin, qmax, block_size, zero_point_shift
    )


def dequantize(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    block_size: Optional[Tuple[int, ...]] = None,
):
    r"""
    Applies dequantization to the input.

    Precisely,

    .. math::
        out = (input + offset) * scale

    If block size :math:`B = \begin{pmatrix} B_0  & B_1  & \cdots & B_{D-1} \end{pmatrix}` is specified,
    this equation will be further generalized as

    .. math::
        out_{j_0 \cdots j_{D-1}} & = (input_{j_0 \cdots j_{D-1}} + offset_{i_0 \cdots i_{D-1}}) * scale_{i_0 \cdots i_{D-1}}

        \text{where} \quad \forall_{0 \leq d < D} \quad i_d = \left\lfloor \frac{j_d}{B_d} \right\rfloor

    :param Tensor tensor: Tensor to dequantize
    :param Tensor scale: Scale for dequantization
    :param Tensor offset: Offset for dequantization
    :param block_size: Block size
    :type block_size: Tuple[int, ...], optional
    """
    if _torch_compiler_is_exporting() or torch.onnx.is_in_onnx_export():
        backend = torch_builtins
    else:
        backend = get_backend()

    return backend.dequantize(tensor, scale, offset, block_size)


def _parse_args(args, kwargs) -> Tuple[int, int, Optional[Tuple[int, ...]], float]:
    bitwidth = num_steps = signed = qmin = qmax = None

    # Pad positional args with None's such that len(args) == 4
    args = tuple(chain(args, repeat(None, 4 - len(args))))
    arg0 = kwargs.get("qmin", kwargs.get("bitwidth", args[0]))
    arg1 = kwargs.get("qmax", kwargs.get("signed", args[1]))
    block_size = kwargs.get("block_size", None) or args[2]
    zero_point_shift = args[3] or kwargs.get("zero_point_shift", 0.0)

    if arg0 is None:
        num_steps = kwargs["num_steps"]
        signed = kwargs["signed"]
        qmin, qmax = _derive_qmin_qmax(num_steps=num_steps, signed=signed)
    elif arg1 is None or isinstance(arg1, bool):
        bitwidth, signed = arg0, bool(arg1)
        qmin, qmax = _derive_qmin_qmax(bitwidth=bitwidth, signed=signed)
    else:
        qmin, qmax = arg0, arg1

    assert qmin is not None
    assert qmax is not None

    return qmin, qmax, block_size, zero_point_shift


def _derive_qmin_qmax(*, bitwidth: int = None, num_steps: int = None, signed: bool):
    if bitwidth is not None:
        num_steps = 2**bitwidth - 1

    if signed:
        qmin = -math.ceil(num_steps / 2)
        qmax = math.floor(num_steps / 2)
    else:
        qmin = 0
        qmax = num_steps

    return qmin, qmax

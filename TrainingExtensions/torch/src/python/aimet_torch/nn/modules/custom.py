# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=missing-module-docstring, disable=wildcard-import, unused-wildcard-import
from ..._base.nn.modules.custom import *
from typing import Optional
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from aimet_torch._base.nn.modules.custom import *  # pylint: disable=wildcard-import, unused-wildcard-import
from aimet_torch.quantization.tensor import QuantizedTensorBase
from ..true_quant import (
    QuantizationMixin,
    _DispatchMixin,
    _quantize_if_applicable,
    _quantize_dequantize_if_applicable,
)

# NOTE: Disabling due to pylint false alarm in ModuleList
# pylint: disable=not-callable
# pylint: disable=arguments-differ


@QuantizationMixin.implements(Sin)
class QuantizedSin(_DispatchMixin, QuantizationMixin, Sin):
    """Quantized Sin"""

    _builtin_torch_fn = torch.sin


@QuantizationMixin.implements(Cos)
class QuantizedCos(_DispatchMixin, QuantizationMixin, Cos):
    """Quantized Cos"""

    _builtin_torch_fn = torch.cos


@QuantizationMixin.implements(AvgPool2d)
class QuantizedAvgPool2d(_DispatchMixin, QuantizationMixin, AvgPool2d):
    """Quantized AvgPool2d"""

    _builtin_torch_fn = F.avg_pool2d

    @classmethod
    def _is_dynamo_traceable(cls) -> Tuple[bool, Optional[str]]:
        return False, "F.avg_pool2d isn't dynamo-traceable"


@QuantizationMixin.implements(Reshape)
class QuantizedReshape(_DispatchMixin, QuantizationMixin, Reshape):
    """Quantized Reshape"""

    _builtin_torch_fn = torch.reshape

    @classmethod
    def _is_dynamo_traceable(cls) -> Tuple[bool, Optional[str]]:
        return False, "torch.reshape isn't dynamo-traceable"


@QuantizationMixin.implements(RSqrt)
class QuantizedRSqrt(_DispatchMixin, QuantizationMixin, RSqrt):
    """Quantized RSqrt"""

    _builtin_torch_fn = torch.rsqrt


@QuantizationMixin.implements(MatMul)
class QuantizedMatMul(_DispatchMixin, QuantizationMixin, MatMul):
    """Quantized MatMul"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.matmul


@QuantizationMixin.implements(Add)
class QuantizedAdd(_DispatchMixin, QuantizationMixin, Add):
    """Quantized Add"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.add


@QuantizationMixin.implements(Multiply)
class QuantizedMultiply(_DispatchMixin, QuantizationMixin, Multiply):
    """Quantized Multiply"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.mul


@QuantizationMixin.implements(Subtract)
class QuantizedSubtract(_DispatchMixin, QuantizationMixin, Subtract):
    """Quantized Subtract"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.sub


@QuantizationMixin.implements(Divide)
class QuantizedDivide(_DispatchMixin, QuantizationMixin, Divide):
    """Quantized Divide"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.div


@QuantizationMixin.implements(Outer)
class QuantizedOuter(_DispatchMixin, QuantizationMixin, Outer):
    """Quantized Outer"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.outer


@QuantizationMixin.implements(Concat)
class QuantizedConcat(_DispatchMixin, QuantizationMixin, Concat):
    """Quantized Concat"""

    _builtin_torch_fn = torch.cat

    def _is_dispatch_necessary(self) -> bool:
        return True

    def _builtin_torch_fn_helper(self, fn: Callable[..., Tensor]):
        def cat(tensors, dim=0, *, out=None):
            input_quantizers = (
                self.input_quantizers[idx] if idx < len(self.input_quantizers) else None
                for idx, _ in enumerate(tensors)
            )
            tensors = tuple(
                _quantize_dequantize_if_applicable(x, quantizer)
                for x, quantizer in zip(tensors, input_quantizers)
            )
            output = fn(tensors, dim=dim, out=out)
            return _quantize_dequantize_if_applicable(output, self.output_quantizers[0])

        return cat

    def _custom_kernel_helper(self, fn: Callable[..., QuantizedTensorBase]):
        def cat(tensors, dim=0, *, out=None):
            input_quantizers = (
                self.input_quantizers[idx] if idx < len(self.input_quantizers) else None
                for idx, _ in enumerate(tensors)
            )
            tensors = tuple(
                _quantize_if_applicable(x, quantizer)
                for x, quantizer in zip(tensors, input_quantizers)
            )
            output_encodings = (
                self.output_quantizers[0].get_encodings()
                if self.output_quantizers[0]
                else None
            )
            return fn(tensors, dim=dim, out=out, output_encodings=output_encodings)

        return cat

    @classmethod
    def _supports_dynamic_input_count(cls) -> bool:
        """
        Returns true if the module takes a dynamic number of inputs
        """
        return True


@QuantizationMixin.implements(FloorDivide)
class QuantizedFloorDivide(_DispatchMixin, QuantizationMixin, FloorDivide):
    """Quantized FloorDivide"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.floor_divide


@QuantizationMixin.implements(Norm)
class QuantizedNorm(_DispatchMixin, QuantizationMixin, Norm):
    """Quantized Norm"""

    _builtin_torch_fn = torch.norm


@QuantizationMixin.implements(Exponential)
class QuantizedExponential(_DispatchMixin, QuantizationMixin, Exponential):
    """Quantized Exponential"""

    _builtin_torch_fn = torch.exp


@QuantizationMixin.implements(Erf)
class QuantizedErf(_DispatchMixin, QuantizationMixin, Erf):
    """Quantized Erf"""

    _builtin_torch_fn = torch.erf


@QuantizationMixin.implements(Sqrt)
class QuantizedSqrt(_DispatchMixin, QuantizationMixin, Sqrt):
    """Quantized Sqrt"""

    _builtin_torch_fn = torch.sqrt


@QuantizationMixin.implements(Maximum)
class QuantizedMaximum(_DispatchMixin, QuantizationMixin, Maximum):
    """Quantized Maximum"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.maximum


@QuantizationMixin.implements(Max)
class QuantizedMax(_DispatchMixin, QuantizationMixin, Max):
    """Quantized Max"""

    __quant_init__ = QuantizationMixin.__unary__
    _builtin_torch_fn = torch.max


@QuantizationMixin.implements(AMax)
class QuantizedAMax(_DispatchMixin, QuantizationMixin, AMax):
    """Quantized AMax"""

    __quant_init__ = QuantizationMixin.__unary__
    _builtin_torch_fn = torch.amax


@QuantizationMixin.implements(Minimum)
class QuantizedMinimum(_DispatchMixin, QuantizationMixin, Minimum):
    """Quantized Minimum"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.minimum


@QuantizationMixin.implements(Min)
class QuantizedMin(_DispatchMixin, QuantizationMixin, Min):
    """Quantized Min"""

    __quant_init__ = QuantizationMixin.__unary__
    _builtin_torch_fn = torch.min


@QuantizationMixin.implements(AMin)
class QuantizedAMin(_DispatchMixin, QuantizationMixin, AMin):
    """Quantized AMin"""

    __quant_init__ = QuantizationMixin.__unary__
    _builtin_torch_fn = torch.amin


#
#
# @QuantizationMixin.implements(Where)
# class QuantizedWhere(_DispatchMixin, QuantizationMixin, Where):
#     """ Quantized Where """
#     _builtin_torch_fn = torch.where
#
#
# @QuantizationMixin.implements(Greater)
# class QuantizedGreater(_DispatchMixin, QuantizationMixin, Greater):
#     """ Quantized Greater """
#     _builtin_torch_fn = torch.gt
#
#
# @QuantizationMixin.implements(Less)
# class QuantizedLess(_DispatchMixin, QuantizationMixin, Less):
#     """ Quantized Less """
#     _builtin_torch_fn = torch.lt
#
#
# @QuantizationMixin.implements(GreaterEqual)
# class QuantizedGreaterEqual(_DispatchMixin, QuantizationMixin, GreaterEqual):
#     """ Quantized GreaterEqual """
#     _builtin_torch_fn = torch.ge
#
#
# @QuantizationMixin.implements(LessEqual)
# class QuantizedLessEqual(_DispatchMixin, QuantizationMixin, LessEqual):
#     """ Quantized LessEqual """
#     _builtin_torch_fn = torch.le
#
#
# @QuantizationMixin.implements(NotEqual)
# class QuantizedNotEqual(_DispatchMixin, QuantizationMixin, NotEqual):
#     """ Quantized NotEqual """
#     _builtin_torch_fn = torch.ne
#
#
# @QuantizationMixin.implements(Equal)
# class QuantizedEqual(_DispatchMixin, QuantizationMixin, Equal):
#     """ Quantized Equal """
#     _builtin_torch_fn = torch.eq


@QuantizationMixin.implements(Bmm)
class QuantizedBmm(_DispatchMixin, QuantizationMixin, Bmm):
    """Quantized Bmm"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.bmm


@QuantizationMixin.implements(CumSum)
class QuantizedCumSum(_DispatchMixin, QuantizationMixin, CumSum):
    """Quantized CumSum"""

    _builtin_torch_fn = torch.cumsum

    @classmethod
    def _is_dynamo_traceable(cls) -> Tuple[bool, Optional[str]]:
        return False, "torch.cumsum isn't dynamo-traceable"


@QuantizationMixin.implements(MaskedFill)
class QuantizedMaskedFill(QuantizationMixin, MaskedFill):
    """Quantized MaskedFill"""

    __quant_init__ = QuantizationMixin.__ternary__

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        value: torch.Tensor | float,
    ):
        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        # mask need not be quantized

        if isinstance(value, torch.Tensor) and self.input_quantizers[2]:
            value = self.input_quantizers[2](value)

        out = super().forward(x, mask, value)

        if self.output_quantizers[0]:
            out = self.output_quantizers[0](out)

        return out


@QuantizationMixin.implements(Mean)
class QuantizedMean(_DispatchMixin, QuantizationMixin, Mean):
    """Quantized Mean"""

    __quant_init__ = QuantizationMixin.__unary__
    _builtin_torch_fn = torch.mean


@QuantizationMixin.implements(Sum)
class QuantizedSum(_DispatchMixin, QuantizationMixin, Sum):
    """Quantized Sum"""

    __quant_init__ = QuantizationMixin.__unary__
    _builtin_torch_fn = torch.sum


@QuantizationMixin.implements(Prod)
class QuantizedProd(_DispatchMixin, QuantizationMixin, Prod):
    """Quantized Prod"""

    __quant_init__ = QuantizationMixin.__unary__
    _builtin_torch_fn = torch.prod


@QuantizationMixin.implements(Log)
class QuantizedLog(_DispatchMixin, QuantizationMixin, Log):
    """Quantized Log"""

    _builtin_torch_fn = torch.log


@QuantizationMixin.implements(Abs)
class QuantizedAbs(_DispatchMixin, QuantizationMixin, Abs):
    """Quantized Abs"""

    _builtin_torch_fn = torch.abs


@QuantizationMixin.implements(Neg)
class QuantizedNeg(_DispatchMixin, QuantizationMixin, Neg):
    """Quantized Neg"""

    _builtin_torch_fn = torch.neg


@QuantizationMixin.implements(Argmin)
class QuantizedArgmin(_DispatchMixin, QuantizationMixin, Argmin):
    """Quantized Argmin"""

    __quant_init__ = QuantizationMixin.__unary__
    _builtin_torch_fn = torch.argmin


@QuantizationMixin.implements(Argmax)
class QuantizedArgmax(_DispatchMixin, QuantizationMixin, Argmax):
    """Quantized Argmax"""

    __quant_init__ = QuantizationMixin.__unary__
    _builtin_torch_fn = torch.argmax


@QuantizationMixin.implements(ElementwiseCeil)
class QuantizedElementwiseCeil(_DispatchMixin, QuantizationMixin, ElementwiseCeil):
    """Quantized ElementwiseCeil"""

    __quant_init__ = QuantizationMixin.__unary__
    _builtin_torch_fn = torch.ceil


@QuantizationMixin.implements(ElementwiseFloor)
class QuantizedElementwiseFloor(_DispatchMixin, QuantizationMixin, ElementwiseFloor):
    """Quantized ElementwiseFloor"""

    __quant_init__ = QuantizationMixin.__unary__
    _builtin_torch_fn = torch.floor


@QuantizationMixin.implements(Asin)
class QuantizedAsin(_DispatchMixin, QuantizationMixin, Asin):
    """Quantized Asin"""

    __quant_init__ = QuantizationMixin.__unary__
    _builtin_torch_fn = torch.asin


@QuantizationMixin.implements(Atan)
class QuantizedAtan(_DispatchMixin, QuantizationMixin, Atan):
    """Quantized Atan"""

    __quant_init__ = QuantizationMixin.__unary__
    _builtin_torch_fn = torch.atan


@QuantizationMixin.implements(Round)
class QuantizedRound(_DispatchMixin, QuantizationMixin, Round):
    """Quantized Round"""

    __quant_init__ = QuantizationMixin.__unary__
    _builtin_torch_fn = torch.round


@QuantizationMixin.implements(Gather)
class QuantizedGather(_DispatchMixin, QuantizationMixin, Gather):
    """Quantized Gather"""

    __quant_init__ = QuantizationMixin.__unary__
    _builtin_torch_fn = torch.gather

    @classmethod
    def _is_dynamo_traceable(cls) -> Tuple[bool, Optional[str]]:
        return False, None


@QuantizationMixin.implements(LogicalOr)
class QuantizedLogicalOr(_DispatchMixin, QuantizationMixin, LogicalOr):
    """Quantized LogicalOr"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.logical_or


@QuantizationMixin.implements(LogicalAnd)
class QuantizedLogicalAnd(_DispatchMixin, QuantizationMixin, LogicalAnd):
    """Quantized LogicalAnd"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.logical_and


@QuantizationMixin.implements(LogicalNot)
class QuantizedLogicalNot(_DispatchMixin, QuantizationMixin, LogicalNot):
    """Quantized LogicalNot"""

    __quant_init__ = QuantizationMixin.__unary__
    _builtin_torch_fn = torch.logical_not


@QuantizationMixin.implements(Split)
class QuantizedSplit(QuantizationMixin, Split):
    """Quantized Split"""

    __quant_init__ = QuantizationMixin.__unary__

    # pylint: disable=arguments-differ
    def forward(
        self, tensor: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, ...]:
        """
        Quantized forward impl for custom.Split.
        """
        if tensor.is_floating_point() and self.input_quantizers[0]:
            tensor = self.input_quantizers[0](tensor)

        outputs = super().forward(tensor, *args, **kwargs)

        if self.output_quantizers[0]:
            # Use same output quantizer for all the output tensors
            output_qtzr = self.output_quantizers[0]
            outputs = tuple(output_qtzr(output) for output in outputs)

        return outputs

    @classmethod
    def _is_dynamo_traceable(cls) -> Tuple[bool, Optional[str]]:
        return False, "torch.split isn't dynamo-traceable"


@QuantizationMixin.implements(Permute)
class QuantizedPermute(_DispatchMixin, QuantizationMixin, Permute):
    """Quantized Permute"""

    __quant_init__ = QuantizationMixin.__unary__
    _builtin_torch_fn = torch.permute

    @classmethod
    def _is_dynamo_traceable(cls) -> Tuple[bool, Optional[str]]:
        return False, None


@QuantizationMixin.implements(Remainder)
class QuantizedRemainder(_DispatchMixin, QuantizationMixin, Remainder):
    """Quantized Remainder"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.remainder


@QuantizationMixin.implements(IndexSelect)
class QuantizedIndexSelect(_DispatchMixin, QuantizationMixin, IndexSelect):
    """Quantized IndexSelect"""

    __quant_init__ = QuantizationMixin.__unary__
    _builtin_torch_fn = torch.index_select

    @classmethod
    def _is_dynamo_traceable(cls) -> Tuple[bool, Optional[str]]:
        return False, None


@QuantizationMixin.implements(Fmod)
class QuantizedFmod(_DispatchMixin, QuantizationMixin, Fmod):
    """Quantized Fmod"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.fmod


# @QuantizationMixin.implements(NonZero)
# class QuantizedNonZero(_DispatchMixin, QuantizationMixin, NonZero):
#     """ Quantized NonZero """
#     _builtin_torch_fn = torch.nonzero
#
#
# @QuantizationMixin.implements(TopK)
# class QuantizedTopK(_DispatchMixin, QuantizationMixin, TopK):
#     """ Quantized TopK """
#     _builtin_torch_fn = torch.topk


QuantizationMixin.ignore(Shape)


@QuantizationMixin.implements(Tile)
class QuantizedTile(_DispatchMixin, QuantizationMixin, Tile):
    """Quantized Tile"""

    __quant_init__ = QuantizationMixin.__unary__
    _builtin_torch_fn = torch.tile


QuantizationMixin.ignore(ElementwiseUnarySign)


@QuantizationMixin.implements(Baddbmm)
class QuantizedBaddbmm(_DispatchMixin, QuantizationMixin, Baddbmm):
    """Quantized Baddbmm"""

    __quant_init__ = QuantizationMixin.__ternary__
    _builtin_torch_fn = torch.baddbmm


@QuantizationMixin.implements(Addmm)
class QuantizedAddmm(_DispatchMixin, QuantizationMixin, Addmm):
    """Quantized Addmm"""

    __quant_init__ = QuantizationMixin.__ternary__
    _builtin_torch_fn = torch.addmm


@QuantizationMixin.implements(RmsNorm)
class QuantizedRmsNorm(QuantizationMixin, RmsNorm):
    """Custom module for RmsNorm"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RmsNorm
        """
        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        with self._patch_quantized_parameters():
            out = super().forward(x)

        if self.output_quantizers[0]:
            out = self.output_quantizers[0](out)

        return out


@QuantizationMixin.implements(HadamardRotation)
class QuantizedHadamardRotation(QuantizationMixin, HadamardRotation):
    """Custom module for HadamardRotation"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for HadamardRotation
        """
        # Quantize input tensors
        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        # Run forward with quantized inputs and parameters
        with self._patch_quantized_parameters():
            ret = super().forward(x)

        # Quantize output tensors
        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)

        return ret


@QuantizationMixin.implements(Square)
class QuantizedSquare(_DispatchMixin, QuantizationMixin, Square):
    """Quantized Square"""

    __quant_init__ = QuantizationMixin.__unary__
    _builtin_torch_fn = torch.square


@QuantizationMixin.implements(Select)
class QuantizedSelect(_DispatchMixin, QuantizationMixin, Select):
    """Quantized Select"""

    __quant_init__ = QuantizationMixin.__unary__
    _builtin_torch_fn = torch.select

    @classmethod
    def _is_dynamo_traceable(cls) -> Tuple[bool, Optional[str]]:
        return False, None


# modules for functional operations defined under torch.nn.functional package
@QuantizationMixin.implements(Interpolate)
class QuantizedInterpolate(_DispatchMixin, QuantizationMixin, Interpolate):
    """Quantized Interpolate"""

    __quant_init__ = QuantizationMixin.__unary__
    _builtin_torch_fn = torch.nn.functional.interpolate


@QuantizationMixin.implements(MaxPool2d)
class QuantizedMaxPool2d(_DispatchMixin, QuantizationMixin, MaxPool2d):
    """Quantized MaxPool2d"""

    __quant_init__ = QuantizationMixin.__unary__
    _builtin_torch_fn = torch.nn.functional.max_pool2d

    @classmethod
    def _is_dynamo_traceable(cls) -> Tuple[bool, Optional[str]]:
        return False, None


@QuantizationMixin.implements(AdaptiveAvgPool2d)
class QuantizedAdaptiveAvgPool2d(_DispatchMixin, QuantizationMixin, AdaptiveAvgPool2d):
    """Quantized AdaptiveAvgPool2d"""

    __quant_init__ = QuantizationMixin.__unary__
    _builtin_torch_fn = torch.nn.functional.adaptive_avg_pool2d

    @classmethod
    def _is_dynamo_traceable(cls) -> Tuple[bool, Optional[str]]:
        return False, None


@QuantizationMixin.implements(BatchNorm)
class QuantizedBatchNorm(_DispatchMixin, QuantizationMixin, BatchNorm):
    """Quantized BatchNorm"""

    _builtin_torch_fn = torch.nn.functional.batch_norm

    def __quant_init__(self):
        super().__quant_init__()
        # pylint: disable=attribute-defined-outside-init
        self.input_quantizers = nn.ModuleList([None, None, None, None, None])

    def _is_dispatch_necessary(self) -> bool:
        return True

    def _builtin_torch_fn_helper(self, fn: Callable[..., Tensor]):
        # pylint: disable=redefined-builtin
        def batch_norm_wrapper(
            input: Tensor,
            running_mean: Optional[Tensor],
            running_var: Optional[Tensor],
            weight: Optional[Tensor] = None,
            bias: Optional[Tensor] = None,
            training: bool = False,
            momentum: float = 0.1,
            eps: float = 1e-5,
        ) -> Tensor:
            if training:
                if (
                    self.input_quantizers[1] is not None
                    or self.input_quantizers[2] is not None
                ):
                    raise RuntimeError(
                        f"{self.__class__} doesn't support quantizing running_mean or running_var in training mode"
                    )

            input = _quantize_dequantize_if_applicable(input, self.input_quantizers[0])
            running_mean = _quantize_dequantize_if_applicable(
                running_mean, self.input_quantizers[1]
            )
            running_var = _quantize_dequantize_if_applicable(
                running_var, self.input_quantizers[2]
            )
            weight = _quantize_dequantize_if_applicable(
                weight, self.input_quantizers[3]
            )
            bias = _quantize_dequantize_if_applicable(bias, self.input_quantizers[4])

            # PyTorch doesn't support gradient calculation of running_mean/var
            output = fn(
                input,
                running_mean.detach(),
                running_var.detach(),
                weight,
                bias,
                training,
                momentum,
                eps,
            )

            return _quantize_dequantize_if_applicable(output, self.output_quantizers[0])

        return batch_norm_wrapper

    def _custom_kernel_helper(self, fn: Callable[..., Tensor]):
        # pylint: disable=redefined-builtin
        def batch_norm_wrapper(
            input: Tensor,
            running_mean: Optional[Tensor],
            running_var: Optional[Tensor],
            weight: Optional[Tensor] = None,
            bias: Optional[Tensor] = None,
            training: bool = False,
            momentum: float = 0.1,
            eps: float = 1e-5,
        ) -> Tensor:
            if training:
                if (
                    self.input_quantizers[1] is not None
                    or self.input_quantizers[2] is not None
                ):
                    raise RuntimeError(
                        f"{self.__class__} doesn't support quantizing running_mean or running_var in training mode"
                    )

            input = _quantize_if_applicable(input, self.input_quantizers[0])
            running_mean = _quantize_if_applicable(
                running_mean, self.input_quantizers[1]
            )
            running_var = _quantize_if_applicable(running_var, self.input_quantizers[2])
            weight = _quantize_if_applicable(weight, self.input_quantizers[3])
            bias = _quantize_if_applicable(bias, self.input_quantizers[4])

            # PyTorch doesn't support gradient calculation of running_mean/var
            output = fn(
                input,
                running_mean.detach(),
                running_var.detach(),
                weight,
                bias,
                training,
                momentum,
                eps,
            )
            return _quantize_if_applicable(output, self.output_quantizers[0])

        return batch_norm_wrapper


@QuantizationMixin.implements(GroupNorm)
class QuantizedGroupNorm(_DispatchMixin, QuantizationMixin, GroupNorm):
    """Quantized GroupNorm"""

    _builtin_torch_fn = F.group_norm

    @classmethod
    def _is_dynamo_traceable(cls) -> Tuple[bool, Optional[str]]:
        return False, "F.group_norm isn't dynamo-traceable"


@QuantizationMixin.implements(Normalize)
class QuantizedNormalize(_DispatchMixin, QuantizationMixin, Normalize):
    """Quantized Normalize"""

    _builtin_torch_fn = torch.nn.functional.normalize


@QuantizationMixin.implements(NullRequant)
class QuantizedNullRequant(QuantizationMixin, NullRequant):
    """Quantized module for NullRequant"""

    def forward(self, x: torch.Tensor, shape: list) -> torch.Tensor:
        """
        Forward pass for NullRequant
        """
        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        with self._patch_quantized_parameters():
            out = super().forward(x, shape)

        if self.output_quantizers[0]:
            out = self.output_quantizers[0](out)

        return out


@QuantizationMixin.implements(Pad)
class QuantizedPad(_DispatchMixin, QuantizationMixin, Pad):
    """Quantized Pad"""

    __quant_init__ = QuantizationMixin.__unary__
    _builtin_torch_fn = torch.nn.functional.pad


@QuantizationMixin.implements(GridSample)
class QuantizedGridSample(_DispatchMixin, QuantizationMixin, GridSample):
    """Quantized GridSample"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.nn.functional.grid_sample


@QuantizationMixin.implements(DynamicConv2d)
class QuantizedDynamicConv2d(_DispatchMixin, QuantizationMixin, DynamicConv2d):
    """Quantized DynamicConv2d"""

    __quant_init__ = QuantizationMixin.__ternary__
    _builtin_torch_fn = F.conv2d


@QuantizationMixin.implements(Pow)
class QuantizedPow(QuantizationMixin, Pow):
    """Quantized Pow"""

    __quant_init__ = QuantizationMixin.__binary__

    def forward(self, x, y):
        if (
            isinstance(x, torch.Tensor)
            and x.is_floating_point()
            and self.input_quantizers[0]
        ):
            x = self.input_quantizers[0](x)

        if (
            isinstance(y, torch.Tensor)
            and y.is_floating_point()
            and self.input_quantizers[1]
        ):
            y = self.input_quantizers[1](y)

        out = super().forward(x, y)

        if (
            isinstance(out, torch.Tensor)
            and out.is_floating_point()
            and self.output_quantizers[0]
        ):
            out = self.output_quantizers[0](out)

        return out


@QuantizationMixin.implements(CustomSiLU)
class QuantizedCustomSiLU(QuantizationMixin, CustomSiLU):
    """Quantized CustomSiLU"""

    __quant_init__ = QuantizationMixin.__unary__

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (input_qtzr,) = self.input_quantizers
        (output_qtzr,) = self.output_quantizers

        if input_qtzr:
            x = input_qtzr(x)

        out = super().forward(x)

        if output_qtzr:
            out = output_qtzr(out)

        return out


@QuantizationMixin.implements(StridedSlice)
class QuantizedStridedSlice(QuantizationMixin, StridedSlice):
    """Quantized StridedSlice"""

    __quant_init__ = QuantizationMixin.__unary__

    def forward(self, *args) -> torch.Tensor:
        x, *args = args

        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        out = super().forward(x, *args)

        if self.output_quantizers[0]:
            out = self.output_quantizers[0](out)

        return out

    @classmethod
    def _is_dynamo_traceable(cls) -> Tuple[bool, Optional[str]]:
        return False, None


@QuantizationMixin.implements(ChannelShuffle)
class QuantizedChannelShuffle(QuantizationMixin, ChannelShuffle):
    """Quantized ChannelShuffle"""

    __quant_init__ = QuantizationMixin.__unary__

    def forward(self, *args) -> torch.Tensor:
        x, *args = args

        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        out = super().forward(x, *args)

        if self.output_quantizers[0]:
            out = self.output_quantizers[0](out)

        return out


QuantizationMixin.ignore(Cast)


@QuantizationMixin.implements(CustomGather)
class QuantizedCustomGather(QuantizationMixin, CustomGather):
    """Quantized CustomGather"""

    __quant_init__ = QuantizationMixin.__unary__

    def forward(
        self, data: torch.Tensor, indices: torch.Tensor, axis: int = 0
    ) -> torch.Tensor:
        """
        Forward-pass routine for ONNX Gather op
        """
        if self.input_quantizers[0]:
            data = self.input_quantizers[0](data)

        out = super().forward(data, indices, axis)

        if self.output_quantizers[0]:
            out = self.output_quantizers[0](out)

        return out


@QuantizationMixin.implements(DepthToSpaceCRDMode)
class QuantizedDepthToSpaceCRDMode(QuantizationMixin, DepthToSpaceCRDMode):
    """Quantized DepthToSpaceCRDMode"""

    __quant_init__ = QuantizationMixin.__unary__

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        out = super().forward(x)

        if self.output_quantizers[0]:
            out = self.output_quantizers[0](out)

        return out


@QuantizationMixin.implements(DepthToSpaceDCRMode)
class QuantizedDepthToSpaceDCRMode(QuantizationMixin, DepthToSpaceDCRMode):
    """Quantized DepthToSpaceDCRMode"""

    __quant_init__ = QuantizationMixin.__unary__

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        out = super().forward(x)

        if self.output_quantizers[0]:
            out = self.output_quantizers[0](out)

        return out


# @QuantizationMixin.implements(CustomSparseConv3DLayer)
# class QuantizedCustomSparseConv3DLayer(QuantizationMixin, CustomSparseConv3DLayer):
#     """ Quantized CustomSparseConv3DLayer """
#
#
# @QuantizationMixin.implements(SparseTensorWrapper)
# class QuantizedSparseTensorWrapper(QuantizationMixin, SparseTensorWrapper):
#     """ Quantized SparseTensorWrapper """
#
#
# @QuantizationMixin.implements(ScatterDense)
# class QuantizedScatterDense(QuantizationMixin, ScatterDense):
#     """ Quantized ScatterDense """


@QuantizationMixin.implements(ScatterND)
class QuantizedScatterND(QuantizationMixin, ScatterND):
    """Quantized ScatterND"""

    __quant_init__ = QuantizationMixin.__ternary__

    def forward(
        self, data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor
    ) -> torch.Tensor:
        if self.input_quantizers[0]:
            data = self.input_quantizers[0](data)

        # index need not be quantized

        if self.input_quantizers[2]:
            updates = self.input_quantizers[2](updates)

        out = super().forward(data, indices, updates)

        if self.output_quantizers[0]:
            out = self.output_quantizers[0](out)

        return out


@QuantizationMixin.implements(RoiAlign)
class QuantizedRoiAlign(QuantizationMixin, RoiAlign):
    """Quantized RoiAlign"""

    __quant_init__ = QuantizationMixin.__binary__

    def forward(
        self, inp: torch.Tensor, roi: torch.Tensor, batch_indices: torch.Tensor
    ) -> torch.Tensor:
        if self.input_quantizers[0]:
            inp = self.input_quantizers[0](inp)

        if self.input_quantizers[1]:
            roi = self.input_quantizers[1](roi)

        out = super().forward(inp, roi, batch_indices)

        if self.output_quantizers[0]:
            out = self.output_quantizers[0](out)

        return out


# @QuantizationMixin.implements(NonMaxSuppression)
# class QuantizedNonMaxSuppression(QuantizationMixin, NonMaxSuppression):
#     """ Quantized NonMaxSuppression """


@QuantizationMixin.implements(GatherNd)
class QuantizedGatherNd(QuantizationMixin, GatherNd):
    """Quantized GatherNd"""

    __quant_init__ = QuantizationMixin.__unary__

    def forward(self, data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Forward-pass routine for GatherNd op
        """
        if self.input_quantizers[0]:
            data = self.input_quantizers[0](data)

        out = super().forward(data, indices)

        if self.output_quantizers[0]:
            out = self.output_quantizers[0](out)

        return out


@QuantizationMixin.implements(ScatterElements)
class QuantizedScatterElements(QuantizationMixin, ScatterElements):
    """Quantized ScatterElements"""

    __quant_init__ = QuantizationMixin.__ternary__

    def forward(
        self,
        x: Union[torch.Tensor, list],
        index: Union[torch.Tensor, list],
        src: Union[torch.Tensor, list],
    ):
        if isinstance(index, list):
            index = torch.tensor(index, dtype=torch.int64)
        if isinstance(src, list):
            src = torch.tensor(src)
        if isinstance(x, list):
            x = torch.tensor(x, dtype=src.dtype)

        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        # index need not be quantized

        if self.input_quantizers[2]:
            src = self.input_quantizers[2](src)

        out = super().forward(x, index, src)

        if self.output_quantizers[0]:
            out = self.output_quantizers[0](out)

        return out


@QuantizationMixin.implements(OneHot)
class QuantizedOneHot(QuantizationMixin, OneHot):
    """Quantized OneHot"""

    __quant_init__ = QuantizationMixin.__unary__

    def forward(self, *args, **kwargs) -> torch.Tensor:
        out = super().forward(*args, **kwargs)

        if (
            isinstance(out, torch.Tensor)
            and out.is_floating_point()
            and self.output_quantizers[0]
        ):
            out = self.output_quantizers[0](out)

        return out


@QuantizationMixin.implements(Expand)
class QuantizedExpand(QuantizationMixin, Expand):
    """Quantized Expand"""

    __quant_init__ = QuantizationMixin.__unary__

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        out = super().forward(x, *args)

        if self.output_quantizers[0]:
            out = self.output_quantizers[0](out)

        return out


@QuantizationMixin.implements(DynamicLinear)
class QuantizedDynamicLinear(_DispatchMixin, QuantizationMixin, DynamicLinear):
    """Quantized DynamicLinear"""

    __quant_init__ = QuantizationMixin.__ternary__
    _builtin_torch_fn = F.linear

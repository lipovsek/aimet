# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import pytest
import torch
from aimet_torch.v2.nn import QuantizationMixin


class CustomOp(torch.nn.Module):
    """Dummy custom module"""

    def forward(self, input):
        return input * 2 + 1


class TestQuantizedCustomOp:
    def test_custom_op_from_module_unregistered(self):
        with pytest.raises(RuntimeError):
            _ = QuantizationMixin.from_module(CustomOp())

    def test_custom_op_from_module_registered(self):
        try:

            @QuantizationMixin.implements(CustomOp)
            class QuantizedCustomOp(QuantizationMixin, CustomOp):
                def quantized_forward(self, x):
                    x = super().forward(x)
                    return self.output_quantizers[0](x)

            quantized_custom_op = QuantizationMixin.from_module(CustomOp())
            assert isinstance(quantized_custom_op, QuantizedCustomOp)

            quantized_custom_op_ = QuantizationMixin.from_module(CustomOp())
            assert isinstance(quantized_custom_op_, QuantizedCustomOp)

        finally:
            # Unregister CustomOp so as not to affect other test functions
            QuantizationMixin.cls_to_qcls.pop(CustomOp)

    def test_custom_op_wrap_registered(self):
        try:

            @QuantizationMixin.implements(CustomOp)
            class QuantizedCustomOp(QuantizationMixin, CustomOp):
                def quantized_forward(self, x):
                    x = super().forward(x)
                    return self.output_quantizers[0](x)

            quantized_custom_op_cls = QuantizationMixin.wrap(CustomOp)
            assert quantized_custom_op_cls is QuantizedCustomOp

            quantized_custom_op_cls_ = QuantizationMixin.wrap(CustomOp)
            assert quantized_custom_op_cls_ is QuantizedCustomOp

        finally:
            # Unregister CustomOp so as not to affect other test functions
            QuantizationMixin.cls_to_qcls.pop(CustomOp)

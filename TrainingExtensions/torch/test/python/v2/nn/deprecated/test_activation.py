# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import pytest
import torch
import torch.nn.functional as F
from aimet_torch.v2.quantization.affine.backends import quantize_dequantize
from aimet_torch.v2.quantization.affine import QuantizeDequantize
from aimet_torch.v2.nn.fake_quant._legacy_impl import (
    FakeQuantizedSoftmax,
    FakeQuantizedReshape,
)
from aimet_torch.v2.quantization.affine.encoding import AffineEncoding
from aimet_torch.v2.quantization.encoding_analyzer import MinMaxEncodingAnalyzer
from aimet_torch.v2.quantization.tensor import DequantizedTensor


@pytest.fixture
def input():
    return torch.arange(100).view(10, 10) / 100


class TestFakeQuantizedSoftmax:
    def test_no_qtzr(self, input):
        """
        Given: Fake-quantized softmax module without any quantizer
        """

        """
        When: Inspect `*_quantizer` attributes
        Then: All of them are set to `None`
        """
        quant_softmax = FakeQuantizedSoftmax()
        assert quant_softmax.input_quantizers[0] is None
        assert quant_softmax.output_quantizers[0] is None

        """
        When: Run forward with an input x
        Then: The output should be equal to that of the base FP module.
        """
        expected_output = F.softmax(input, quant_softmax.dim)
        output = quant_softmax(input)
        assert torch.equal(output, expected_output)

        """
        Given: Fake-quantized reshape module without any quantizer
        When: Run forward with an input x
        Then: The output should hold the same encoding as input
        """
        input = input.as_subclass(DequantizedTensor)
        input.encoding = AffineEncoding(
            scale=torch.ones(()), offset=torch.zeros(()), bitwidth=8
        )
        output = FakeQuantizedReshape()(input, (100,))

        assert isinstance(output, DequantizedTensor)
        assert torch.equal(input.encoding.scale, output.encoding.scale)
        assert torch.equal(input.encoding.offset, output.encoding.offset)
        assert input.encoding.bitwidth == output.encoding.bitwidth
        assert input.encoding.symmetry == output.encoding.symmetry
        assert input.encoding.signed == output.encoding.signed

    def test_input_qtzn(self, input):
        """
        Given: Instantiate a fake-quantized module with input quantizer spec specified
        """
        quant_softmax = FakeQuantizedSoftmax()
        quant_softmax.input_quantizers[0] = QuantizeDequantize(
            (),
            bitwidth=8,
            symmetric=False,
            encoding_analyzer=MinMaxEncodingAnalyzer(()),
        )

        """
        When: Inspect `input_quantizer` attribute.
        Then: `input_quantizer` is set to `QuantizeDequantize` as a submodule
        """
        assert isinstance(quant_softmax.input_quantizers[0], QuantizeDequantize)
        assert quant_softmax.output_quantizers[0] is None

        """
        When: Invoke forward before the encodings are initialized with `compute_encodings()`
        Then: Throw runtime error
        """
        with pytest.raises(RuntimeError):
            _ = quant_softmax(input)

        """
        When: Invoke forward with input x after encodings are initialized
              with `compute_encodings()`
        Then: The output should be equal to FP softmax of quantize-dequantized x
        """
        with quant_softmax.compute_encodings():
            _ = quant_softmax(input)

        quant_output = quant_softmax(input)

        scale = quant_softmax.input_quantizers[0].get_scale()
        offset = quant_softmax.input_quantizers[0].get_offset()
        bitwidth = quant_softmax.input_quantizers[0].bitwidth
        input_qdq = quantize_dequantize(input, scale, offset, bitwidth)

        expected_output = F.softmax(input_qdq, quant_softmax.dim)
        assert torch.equal(quant_output, expected_output)

    def test_output_qtzn(self, input):
        """
        Given: Instantiate a fake-quantized module with output quantizer spec specified
        """
        quant_softmax = FakeQuantizedSoftmax()
        quant_softmax.output_quantizers[0] = QuantizeDequantize(
            (),
            bitwidth=8,
            symmetric=False,
            encoding_analyzer=MinMaxEncodingAnalyzer(()),
        )

        """
        When: Inspect `output_quantizer` attribute.
        Then: `output_quantizer` is set to `QuantizeDequantize` as a submodule
        """
        assert quant_softmax.input_quantizers[0] is None
        assert isinstance(quant_softmax.output_quantizers[0], QuantizeDequantize)

        """
        When: Invoke forward before the encodings are initialized with `compute_encodings()`
        Then: Throw runtime error
        """
        with pytest.raises(RuntimeError):
            _ = quant_softmax(input)

        """
        When: Invoke forward with input x after encodings are initialized
              with `compute_encodings()`
        Then: The output should be equal to quantize-dequantized FP softmax output
        """
        with quant_softmax.compute_encodings():
            _ = quant_softmax(input)

        quant_output = quant_softmax(input)

        scale = quant_softmax.output_quantizers[0].get_scale()
        offset = quant_softmax.output_quantizers[0].get_offset()
        bitwidth = quant_softmax.output_quantizers[0].bitwidth

        fp_output = F.softmax(input, quant_softmax.dim)
        expected_output = quantize_dequantize(fp_output, scale, offset, bitwidth)
        assert torch.equal(quant_output, expected_output)

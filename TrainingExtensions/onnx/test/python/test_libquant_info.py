# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for libquant_info module."""

import pytest

from aimet_onnx.common import libpymo
from aimet_onnx.common import libquant_info


class TestQcQuantizeInfo:
    """Test QcQuantizeInfo class."""

    def test_create_qc_quantize_info(self):
        """Test creating QcQuantizeInfo."""
        quant_info = libquant_info.QcQuantizeInfo()
        assert quant_info is not None

    def test_name_attribute(self):
        """Test name attribute."""
        quant_info = libquant_info.QcQuantizeInfo()
        quant_info.name = "test_quantizer"
        assert quant_info.name == "test_quantizer"

    def test_enabled_attribute(self):
        """Test enabled attribute."""
        quant_info = libquant_info.QcQuantizeInfo()
        quant_info.enabled = True
        assert quant_info.enabled is True
        quant_info.enabled = False
        assert quant_info.enabled is False

    def test_use_symmetric_encoding_attribute(self):
        """Test useSymmetricEncoding attribute."""
        quant_info = libquant_info.QcQuantizeInfo()
        quant_info.useSymmetricEncoding = True
        assert quant_info.useSymmetricEncoding is True
        quant_info.useSymmetricEncoding = False
        assert quant_info.useSymmetricEncoding is False

    def test_use_per_channel_mode_attribute(self):
        """Test usePerChannelMode attribute."""
        quant_info = libquant_info.QcQuantizeInfo()
        quant_info.usePerChannelMode = True
        assert quant_info.usePerChannelMode is True
        quant_info.usePerChannelMode = False
        assert quant_info.usePerChannelMode is False

    def test_is_int_data_type_attribute(self):
        """Test isIntDataType attribute."""
        quant_info = libquant_info.QcQuantizeInfo()
        quant_info.isIntDataType = True
        assert quant_info.isIntDataType is True
        quant_info.isIntDataType = False
        assert quant_info.isIntDataType is False

    def test_channel_axis_attribute(self):
        """Test channelAxis attribute."""
        quant_info = libquant_info.QcQuantizeInfo()
        quant_info.channelAxis = 0
        assert quant_info.channelAxis == 0
        quant_info.channelAxis = 1
        assert quant_info.channelAxis == 1

    def test_block_axis_attribute(self):
        """Test blockAxis attribute."""
        quant_info = libquant_info.QcQuantizeInfo()
        quant_info.blockAxis = 0
        assert quant_info.blockAxis == 0
        quant_info.blockAxis = 2
        assert quant_info.blockAxis == 2

    def test_block_size_attribute(self):
        """Test blockSize attribute."""
        quant_info = libquant_info.QcQuantizeInfo()
        quant_info.blockSize = 32
        assert quant_info.blockSize == 32
        quant_info.blockSize = 64
        assert quant_info.blockSize == 64

    def test_op_mode_attribute(self):
        """Test opMode attribute."""
        quant_info = libquant_info.QcQuantizeInfo()
        quant_info.opMode = libpymo.TensorQuantizerOpMode.passThrough
        assert quant_info.opMode == libpymo.TensorQuantizerOpMode.passThrough

        quant_info.opMode = libpymo.TensorQuantizerOpMode.updateStats
        assert quant_info.opMode == libpymo.TensorQuantizerOpMode.updateStats

        quant_info.opMode = libpymo.TensorQuantizerOpMode.quantizeDequantize
        assert quant_info.opMode == libpymo.TensorQuantizerOpMode.quantizeDequantize

    def test_tensor_quantizer_ref(self):
        """Test tensorQuantizerRef attribute with BlockTensorQuantizer."""
        quant_info = libquant_info.QcQuantizeInfo()

        # Create a BlockTensorQuantizer
        tensor_quantizer = libpymo.BlockTensorQuantizer(
            [],  # scalar shape
            8,  # bitwidth
            libpymo.QuantizationMode.QUANTIZATION_TF,
        )

        quant_info.tensorQuantizerRef = tensor_quantizer
        assert quant_info.tensorQuantizerRef is not None

    def test_encoding_property_with_tensor_quantizer(self):
        """Test encoding property when tensorQuantizer is set."""
        quant_info = libquant_info.QcQuantizeInfo()

        # Create and set a BlockTensorQuantizer
        tensor_quantizer = libpymo.BlockTensorQuantizer(
            [], 8, libpymo.QuantizationMode.QUANTIZATION_TF
        )
        quant_info.tensorQuantizerRef = tensor_quantizer

        # Create encoding
        enc = libpymo.TfEncoding()
        enc.min = -1.0
        enc.max = 1.0
        enc.delta = 2.0 / 255.0
        enc.offset = -128
        enc.bw = 8

        # Set and get encoding
        quant_info.encoding = [enc]
        retrieved = quant_info.encoding

        assert len(retrieved) == 1
        assert retrieved[0].min == enc.min
        assert retrieved[0].max == enc.max
        assert retrieved[0].bw == enc.bw

    def test_encoding_without_tensor_quantizer_raises(self):
        """Test that setting encoding without tensorQuantizer raises error."""
        quant_info = libquant_info.QcQuantizeInfo()

        enc = libpymo.TfEncoding()
        enc.min = -1.0
        enc.max = 1.0
        enc.delta = 2.0 / 255.0
        enc.offset = -128
        enc.bw = 8

        with pytest.raises(RuntimeError):
            quant_info.encoding = [enc]


class TestLibquantInfoIntegration:
    """Integration tests for libquant_info with libpymo."""

    def test_full_workflow(self):
        """Test full workflow: create QcQuantizeInfo, set up quantizer, set encoding."""
        # Create QcQuantizeInfo
        quant_info = libquant_info.QcQuantizeInfo()
        quant_info.name = "conv1_weight"
        quant_info.enabled = True
        quant_info.useSymmetricEncoding = False
        quant_info.usePerChannelMode = False
        quant_info.isIntDataType = True
        quant_info.channelAxis = 0
        quant_info.opMode = libpymo.TensorQuantizerOpMode.quantizeDequantize

        # Create and assign BlockTensorQuantizer
        tensor_quantizer = libpymo.BlockTensorQuantizer(
            [], 8, libpymo.QuantizationMode.QUANTIZATION_TF
        )
        quant_info.tensorQuantizerRef = tensor_quantizer

        # Create and set encoding
        enc = libpymo.TfEncoding()
        enc.min = -0.5
        enc.max = 0.5
        enc.delta = 1.0 / 255.0
        enc.offset = -128
        enc.bw = 8

        quant_info.encoding = [enc]

        # Verify everything is set correctly
        assert quant_info.name == "conv1_weight"
        assert quant_info.enabled is True
        assert quant_info.opMode == libpymo.TensorQuantizerOpMode.quantizeDequantize
        assert len(quant_info.encoding) == 1
        assert quant_info.encoding[0].min == -0.5

    def test_per_channel_setup(self):
        """Test per-channel quantization setup."""
        quant_info = libquant_info.QcQuantizeInfo()
        quant_info.usePerChannelMode = True
        quant_info.channelAxis = 0

        # Create per-channel BlockTensorQuantizer (3 channels)
        tensor_quantizer = libpymo.BlockTensorQuantizer(
            [3, 1, 1], 8, libpymo.QuantizationMode.QUANTIZATION_TF
        )
        quant_info.tensorQuantizerRef = tensor_quantizer

        # Create per-channel encodings
        encodings = []
        for i in range(3):
            enc = libpymo.TfEncoding()
            enc.min = -1.0 - i * 0.1
            enc.max = 1.0 + i * 0.1
            enc.delta = (enc.max - enc.min) / 255.0
            enc.offset = -128
            enc.bw = 8
            encodings.append(enc)

        quant_info.encoding = encodings

        # Verify
        assert quant_info.usePerChannelMode is True
        assert len(quant_info.encoding) == 3
        assert quant_info.tensorQuantizerRef.getShape() == [3, 1, 1]

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import aimet_onnx.common._libpymo as libpymo
import aimet_onnx.common.py_libpymo as py_libpymo


class TestLibpymoApiParity:
    """Test that py_libpymo has all public APIs from libpymo."""

    def test_libpymo_api(self):
        """Test public APIs from libpymo are part of py_libpymo."""
        for attribute in dir(libpymo):
            if not attribute.startswith("_"):
                assert hasattr(py_libpymo, attribute), (
                    f"py_libpymo missing: {attribute}"
                )


class TestEnums:
    """Test enum types exposed by libpymo."""

    def test_enums_exist(self):
        """Test that all enum types exist."""
        assert hasattr(libpymo, "ComputationMode")
        assert hasattr(libpymo, "QuantizationMode")
        assert hasattr(libpymo, "LayerInOut")
        assert hasattr(libpymo, "RoundingMode")
        assert hasattr(libpymo, "TensorQuantizerOpMode")

    def test_enum_values(self):
        """Test that enum values are accessible via enum classes."""
        # ComputationMode
        assert libpymo.ComputationMode.COMP_MODE_CPU is not None
        assert libpymo.ComputationMode.COMP_MODE_GPU is not None

        # QuantizationMode
        assert libpymo.QuantizationMode.QUANTIZATION_TF is not None
        assert libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED is not None

        # LayerInOut
        assert libpymo.LayerInOut.LAYER_INPUT is not None
        assert libpymo.LayerInOut.LAYER_OUTPUT is not None

        # RoundingMode
        assert libpymo.RoundingMode.ROUND_NEAREST is not None
        assert libpymo.RoundingMode.ROUND_STOCHASTIC is not None


class TestTfEncoding:
    """Test TfEncoding class."""

    def test_create_encoding(self):
        """Test creating TfEncoding."""
        enc = libpymo.TfEncoding()
        assert enc is not None

    def test_encoding_attributes(self):
        """Test TfEncoding attributes."""
        enc = libpymo.TfEncoding()
        enc.min = -1.0
        enc.max = 1.0
        enc.delta = 0.1
        enc.offset = 0.0
        enc.bw = 8

        assert enc.min == -1.0
        assert enc.max == 1.0
        assert enc.delta == 0.1
        assert enc.offset == 0.0
        assert enc.bw == 8


class TestEncodingAnalyzerForPython:
    """Test EncodingAnalyzerForPython class."""

    def test_create_analyzer(self):
        """Test creating EncodingAnalyzerForPython."""
        analyzer = libpymo.EncodingAnalyzerForPython(
            libpymo.QuantizationMode.QUANTIZATION_TF
        )
        assert analyzer is not None

    def test_update_stats_and_compute_encoding(self):
        """Test updateStats and computeEncoding."""
        analyzer = libpymo.EncodingAnalyzerForPython(
            libpymo.QuantizationMode.QUANTIZATION_TF
        )

        # Create test data
        data = np.random.randn(100).astype(np.float32)

        # Update stats
        analyzer.updateStats(data, False)  # use_cuda=False

        # Compute encoding
        encoding, is_valid = analyzer.computeEncoding(8, False, False, False)

        assert is_valid
        assert encoding.bw == 8
        assert encoding.min <= encoding.max


class TestTensorQuantizationSimForPython:
    """Test TensorQuantizationSimForPython class."""

    def test_create_sim(self):
        """Test creating TensorQuantizationSimForPython."""
        sim = libpymo.TensorQuantizationSimForPython()
        assert sim is not None

    def test_quantize_dequantize(self):
        """Test quantizeDequantize."""
        sim = libpymo.TensorQuantizationSimForPython()

        # Create encoding
        enc = libpymo.TfEncoding()
        enc.min = -1.0
        enc.max = 1.0
        enc.delta = 2.0 / 255.0
        enc.offset = -128
        enc.bw = 8

        # Create input data
        input_data = np.array([-0.5, 0.0, 0.5], dtype=np.float32)

        # Quantize-dequantize
        output = sim.quantizeDequantize(
            input_data, enc, libpymo.RoundingMode.ROUND_NEAREST, enc.bw, False
        )

        assert output.shape == input_data.shape
        assert output.dtype == np.float32


class TestTensorQuantizer:
    """Test TensorQuantizer class."""

    def test_create_quantizer(self):
        """Test creating TensorQuantizer."""
        quantizer = libpymo.TensorQuantizer(
            libpymo.QuantizationMode.QUANTIZATION_TF, libpymo.RoundingMode.ROUND_NEAREST
        )
        assert quantizer is not None

    def test_update_stats_and_compute_encoding(self):
        """Test updateStats and computeEncoding."""
        quantizer = libpymo.TensorQuantizer(
            libpymo.QuantizationMode.QUANTIZATION_TF, libpymo.RoundingMode.ROUND_NEAREST
        )

        # Update stats with test data
        data = np.random.randn(100).astype(np.float32)
        quantizer.updateStats(data, False)

        # Compute encoding
        encoding = quantizer.computeEncoding(8, False)

        assert encoding.bw == 8
        assert quantizer.isEncodingValid

    def test_quantize_dequantize(self):
        """Test quantizeDequantize."""
        quantizer = libpymo.TensorQuantizer(
            libpymo.QuantizationMode.QUANTIZATION_TF, libpymo.RoundingMode.ROUND_NEAREST
        )

        # Update stats and compute encoding
        data = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        quantizer.updateStats(data, False)
        encoding = quantizer.computeEncoding(8, False)

        # Quantize-dequantize
        input_tensor = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
        output_tensor = np.zeros_like(input_tensor)
        quantizer.quantizeDequantize(
            input_tensor, output_tensor, encoding.min, encoding.max, 8, False
        )

        assert output_tensor.shape == input_tensor.shape

    def test_quant_scheme_getter_setter(self):
        """Test setQuantScheme and getQuantScheme."""
        quantizer = libpymo.TensorQuantizer(
            libpymo.QuantizationMode.QUANTIZATION_TF, libpymo.RoundingMode.ROUND_NEAREST
        )

        assert quantizer.getQuantScheme() == libpymo.QuantizationMode.QUANTIZATION_TF

        quantizer.setQuantScheme(libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED)
        assert (
            quantizer.getQuantScheme()
            == libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED
        )

    def test_symmetric_flags(self):
        """Test strict/unsigned symmetric flags."""
        quantizer = libpymo.TensorQuantizer(
            libpymo.QuantizationMode.QUANTIZATION_TF, libpymo.RoundingMode.ROUND_NEAREST
        )

        # Test strict symmetric
        assert not quantizer.getStrictSymmetric()
        quantizer.setStrictSymmetric(True)
        assert quantizer.getStrictSymmetric()

        # Test unsigned symmetric
        assert not quantizer.getUnsignedSymmetric()
        quantizer.setUnsignedSymmetric(True)
        assert quantizer.getUnsignedSymmetric()

    def test_reset_encoding_stats(self):
        """Test resetEncodingStats clears collected statistics."""
        quantizer = libpymo.TensorQuantizer(
            libpymo.QuantizationMode.QUANTIZATION_TF, libpymo.RoundingMode.ROUND_NEAREST
        )

        # Collect stats and compute encoding
        data = np.random.randn(100).astype(np.float32)
        quantizer.updateStats(data, False)
        quantizer.computeEncoding(8, False)
        assert quantizer.isEncodingValid

        # Reset stats - encoding computed without new stats should be invalid
        quantizer.resetEncodingStats()
        quantizer.computeEncoding(8, False)
        assert not quantizer.isEncodingValid

        # After new stats, encoding should be valid again
        quantizer.updateStats(data, False)
        quantizer.computeEncoding(8, False)
        assert quantizer.isEncodingValid


class TestBlockTensorQuantizer:
    """Test BlockTensorQuantizer class."""

    def test_create_block_quantizer(self):
        """Test creating BlockTensorQuantizer."""
        quantizer = libpymo.BlockTensorQuantizer(
            [],  # scalar shape
            8,  # bitwidth
            libpymo.QuantizationMode.QUANTIZATION_TF,
        )
        assert quantizer is not None

    def test_block_quantizer_with_shape(self):
        """Test BlockTensorQuantizer with non-trivial shape."""
        quantizer = libpymo.BlockTensorQuantizer(
            [3, 1, 1],  # per-channel shape
            8,
            libpymo.QuantizationMode.QUANTIZATION_TF,
        )
        assert quantizer.getShape() == [3, 1, 1]
        assert quantizer.bitwidth == 8

    def test_update_stats_and_compute_encodings(self):
        """Test updateStats and computeEncodings."""
        quantizer = libpymo.BlockTensorQuantizer(
            [],  # scalar
            8,
            libpymo.QuantizationMode.QUANTIZATION_TF,
        )

        # Update stats
        data = np.random.randn(100).astype(np.float32)
        quantizer.updateStats(data)

        # Compute encodings
        encodings = quantizer.computeEncodings(False)
        assert len(encodings) == 1  # scalar shape -> 1 encoding
        assert encodings[0].bw == 8

    def test_set_get_encodings(self):
        """Test setEncodings and getEncodings."""
        quantizer = libpymo.BlockTensorQuantizer(
            [], 8, libpymo.QuantizationMode.QUANTIZATION_TF
        )

        # Create encoding
        enc = libpymo.TfEncoding()
        enc.min = -1.0
        enc.max = 1.0
        enc.delta = 2.0 / 255.0
        enc.offset = -128
        enc.bw = 8

        quantizer.setEncodings([enc])
        retrieved = quantizer.getEncodings()
        assert len(retrieved) == 1
        assert retrieved[0].min == enc.min
        assert retrieved[0].max == enc.max

    def test_quant_scheme(self):
        """Test quant scheme getter/setter."""
        quantizer = libpymo.BlockTensorQuantizer(
            [], 8, libpymo.QuantizationMode.QUANTIZATION_TF
        )

        assert quantizer.getQuantScheme() == libpymo.QuantizationMode.QUANTIZATION_TF
        quantizer.setQuantScheme(libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED)
        assert (
            quantizer.getQuantScheme()
            == libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED
        )

    def test_zero_point_shift(self):
        """Test zeroPointShift getter/setter."""
        quantizer = libpymo.BlockTensorQuantizer(
            [], 8, libpymo.QuantizationMode.QUANTIZATION_TF
        )

        assert quantizer.getZeroPointShift() == 0.0
        quantizer.setZeroPointShift(0.5)
        assert quantizer.getZeroPointShift() == 0.5


class TestModuleFunctions:
    """Test module-level functions."""

    def test_getScaleFactor(self):
        """Test getScaleFactor returns correct exponent and mantissa."""
        # Test with 1.0
        exponent, mantissa = libpymo.getScaleFactor(1.0, 8)
        assert isinstance(exponent, int)
        assert isinstance(mantissa, int)
        assert exponent == 1  # For 1.0: e = 127 - 126 = 1

        # Test with powers of 2
        exp2, _ = libpymo.getScaleFactor(2.0, 8)
        exp4, _ = libpymo.getScaleFactor(4.0, 8)
        assert exp4 == exp2 + 1  # 4.0 has exponent 1 higher than 2.0

        # Test different mantissa bits give same exponent
        exp1, _ = libpymo.getScaleFactor(1.0, 4)
        exp2, _ = libpymo.getScaleFactor(1.0, 8)
        assert exp1 == exp2

    def test_getRescaledOutputAndBias_exists(self):
        """Test that getRescaledOutputAndBias is exposed."""
        assert hasattr(libpymo, "getRescaledOutputAndBias")
        assert callable(libpymo.getRescaledOutputAndBias)

    def test_PtrToInt64_exists(self):
        """Test that PtrToInt64 is exposed."""
        assert hasattr(libpymo, "PtrToInt64")
        assert callable(libpymo.PtrToInt64)

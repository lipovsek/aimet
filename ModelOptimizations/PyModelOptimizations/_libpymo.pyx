# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# cython: language_level=3
# distutils: language=c++

"""
Cython wrapper for libpymo module.
Provides Python bindings for quantization operations.
"""

import numpy as _np
from libc.stdint cimport uint64_t, uint8_t, int64_t
from libc.stddef cimport size_t
from libcpp cimport bool as cbool
from libcpp.memory cimport shared_ptr, make_shared
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from cython.operator cimport dereference as deref

from typing import List as _List, Tuple as _Tuple

cimport _libpymo as cpp

# Import shared enums (sibling module in installed package)
from ._quant_enums import (
    ComputationMode,
    QuantizationMode,
    LayerInOut,
    RoundingMode,
    TensorQuantizerOpMode,
)


# ============================================================================
# TfEncoding
# ============================================================================

# Note: TfEncoding is a cdef class (Cython extension type) that needs to be
# defined in each module that uses it with cdef helper functions. Unlike Python
# classes, cdef classes cannot be easily shared across Cython modules at compile
# time. Both _libpymo and libquant_info define their own TfEncoding class.
cdef class TfEncoding:
    """Python wrapper for TfEncoding struct."""
    cdef public double min
    cdef public double max
    cdef public double delta
    cdef public double offset
    cdef public int bw

    def __init__(self, double min=0.0, double max=0.0, double delta=0.0,
                 double offset=0.0, int bw=8):
        self.min = min
        self.max = max
        self.delta = delta
        self.offset = offset
        self.bw = bw

    def __repr__(self):
        return f"TfEncoding(min={self.min}, max={self.max}, delta={self.delta}, offset={self.offset}, bw={self.bw})"


cdef TfEncoding _encoding_from_cpp(cpp.CppTfEncoding& enc):
    return TfEncoding(enc.min, enc.max, enc.delta, enc.offset, enc.bw)


cdef cpp.CppTfEncoding _encoding_to_cpp(TfEncoding enc):
    cdef cpp.CppTfEncoding result
    result.min = enc.min
    result.max = enc.max
    result.delta = enc.delta
    result.offset = enc.offset
    result.bw = enc.bw
    return result


# ============================================================================
# EncodingAnalyzerForPython
# ============================================================================

cdef class EncodingAnalyzerForPython:
    """
    Python wrapper that directly uses IQuantizationEncodingAnalyzer interface.
    No intermediate C++ wrapper class needed.
    """
    cdef unique_ptr[cpp.IQuantizationEncodingAnalyzer[float]] _analyzer
    cdef bint _is_encoding_valid

    def __cinit__(self, quant_mode):
        cdef int mode = int(quant_mode)
        self._analyzer = cpp.getEncodingAnalyzerInstance[float](<cpp.CppQuantizationMode>mode)
        self._is_encoding_valid = False

    def updateStats(self, tensor, bint use_cuda):
        """Update stats from tensor data (any dtype, converted to float32)."""
        flat = _np.ascontiguousarray(_np.asarray(tensor).ravel(), dtype=_np.float32)
        cdef const float* ptr = <const float*><size_t>flat.ctypes.data
        cdef size_t count = <size_t>flat.size
        cdef cpp.CppComputationMode cpu_gpu_mode = (
            cpp.COMP_MODE_GPU if use_cuda else cpp.COMP_MODE_CPU)
        deref(self._analyzer).updateStats(ptr, count, cpu_gpu_mode)
        self._is_encoding_valid = True

    def computeEncoding(self, unsigned int bitwidth, bint is_symmetric,
                        bint use_strict_symmetric=False, bint use_unsigned_symmetric=False):
        """Compute encoding from collected statistics.

        Returns:
            tuple: (TfEncoding, is_valid) where is_valid indicates if stats were collected
        """
        cdef cpp.CppTfEncoding enc
        if self._is_encoding_valid:
            enc = deref(self._analyzer).computeEncoding(
                bitwidth, is_symmetric, use_strict_symmetric, use_unsigned_symmetric)
        return (_encoding_from_cpp(enc), self._is_encoding_valid)

    def isEncodingValid(self) -> bool:
        """Check if encoding is valid (stats have been collected)."""
        return self._is_encoding_valid


# ============================================================================
# TensorQuantizationSimForPython
# ============================================================================

cdef class TensorQuantizationSimForPython:
    """
    Python wrapper that directly uses ITensorQuantizationSim interface.
    No intermediate C++ wrapper class needed.
    """
    cdef unique_ptr[cpp.ITensorQuantizationSim[float]] _sim

    def __cinit__(self):
        self._sim = cpp.getTensorQuantizationSim[float]()

    def quantizeDequantize(self, input_tensor, TfEncoding encoding,
                           rounding_mode, unsigned int bitwidth, bint use_cuda=False):
        """Quantize and dequantize tensor (any dtype, internally uses float32).

        Args:
            input_tensor: Input numpy array
            encoding: TfEncoding with min, max, delta, offset, bw
            rounding_mode: RoundingMode enum
            bitwidth: Quantization bitwidth
            use_cuda: Whether tensors are in CUDA memory

        Returns:
            Output numpy array with same shape as input
        """
        arr = _np.asarray(input_tensor)
        original_shape = arr.shape
        input_flat = _np.ascontiguousarray(arr.ravel(), dtype=_np.float32)
        output_flat = _np.zeros(input_flat.size, dtype=_np.float32)
        cdef const float* input_ptr = <const float*><size_t>input_flat.ctypes.data
        cdef float* output_ptr = <float*><size_t>output_flat.ctypes.data
        cdef size_t count = <size_t>input_flat.size
        cdef int rm = int(rounding_mode)
        deref(self._sim).quantizeDequantizeTensor(input_ptr, count, output_ptr,
                                                  encoding.min, encoding.max, bitwidth,
                                                  <cpp.CppRoundingMode>rm, use_cuda)
        return output_flat.reshape(original_shape)


# ============================================================================
# TensorQuantizer
# ============================================================================

cdef class TensorQuantizer:
    """Python wrapper for TensorQuantizer."""
    cdef cpp.CppTensorQuantizer* _ptr

    def __cinit__(self, quant_scheme, rounding_mode):
        cdef int qs = int(quant_scheme)
        cdef int rm = int(rounding_mode)
        self._ptr = new cpp.CppTensorQuantizer(
            <cpp.CppQuantizationMode>qs,
            <cpp.CppRoundingMode>rm)

    def __dealloc__(self):
        if self._ptr != NULL:
            del self._ptr

    def updateStats(self, tensor, bint use_cuda):
        """Update stats from numpy tensor (any dtype, converted to float32)."""
        flat = _np.ascontiguousarray(_np.asarray(tensor).ravel(), dtype=_np.float32)
        cdef const float* ptr = <const float*><size_t>flat.ctypes.data
        cdef size_t count = <size_t>flat.size
        self._ptr.updateStats(ptr, count, use_cuda)

    def computeEncoding(self, unsigned int bitwidth, bint use_symmetric) -> TfEncoding:
        """Compute encoding from collected statistics.

        Note: Only computes if valid stats have been collected via updateStats().
        Check isEncodingValid after calling to verify encoding was computed.
        """
        cdef cpp.CppTfEncoding enc = self._ptr.computeEncoding(bitwidth, use_symmetric)
        return _encoding_from_cpp(enc)

    def hasValidStats(self) -> bool:
        """Check if valid statistics have been collected."""
        return self._ptr.hasValidStats()

    def quantizeDequantize(self, input_tensor, output_tensor,
                           double encoding_min, double encoding_max,
                           unsigned int bitwidth, bint use_cuda):
        """Quantize and dequantize tensor (any dtype, internally uses float32)."""
        input_flat = _np.ascontiguousarray(_np.asarray(input_tensor).ravel(), dtype=_np.float32)
        output_flat = _np.zeros(input_flat.size, dtype=_np.float32)
        cdef const float* input_ptr = <const float*><size_t>input_flat.ctypes.data
        cdef float* output_ptr = <float*><size_t>output_flat.ctypes.data
        cdef size_t count = <size_t>input_flat.size
        self._ptr.quantizeDequantize(input_ptr, count, output_ptr,
                                     encoding_min, encoding_max, bitwidth, use_cuda)
        # Copy back to output tensor (converts back to original dtype)
        _np.copyto(output_tensor.ravel(), output_flat)

    def resetEncodingStats(self):
        self._ptr.resetEncodingStats()

    def setQuantScheme(self, scheme):
        cdef int s = int(scheme)
        self._ptr.setQuantScheme(<cpp.CppQuantizationMode>s)

    def getQuantScheme(self):
        cdef cpp.CppQuantizationMode cpp_mode = self._ptr.getQuantScheme()
        return QuantizationMode(<int>cpp_mode)

    def setStrictSymmetric(self, bint strict):
        self._ptr.setStrictSymmetric(strict)

    def getStrictSymmetric(self) -> bool:
        return self._ptr.getStrictSymmetric()

    def setUnsignedSymmetric(self, bint unsigned):
        self._ptr.setUnsignedSymmetric(unsigned)

    def getUnsignedSymmetric(self) -> bool:
        return self._ptr.getUnsignedSymmetric()

    def getStatsHistogram(self) -> _List[_Tuple[float, float]]:
        """Get histogram of statistics.

        Returns:
            List of (bucket_edge, pdf) tuples
        """
        cdef cpp.StatsHistogram cpp_hist = self._ptr.getStatsHistogram()
        cdef list result = []
        cdef size_t i
        for i in range(cpp_hist.size()):
            result.append((cpp.get0(cpp_hist[i]), cpp.get1(cpp_hist[i])))
        return result

    def setPercentileValue(self, float percentile):
        self._ptr.setPercentileValue(percentile)

    def getPercentileValue(self) -> float:
        return self._ptr.getPercentileValue()

    def computePartialEncoding(self, unsigned char bw, TfEncoding encoding,
                               bint use_symmetric, bint use_unsigned_symmetric,
                               bint use_strict_symmetric):
        cdef cpp.CppTfEncoding cpp_enc = _encoding_to_cpp(encoding)
        self._ptr.computePartialEncoding(bw, cpp_enc, use_symmetric,
                                         use_unsigned_symmetric, use_strict_symmetric)
        encoding.min = cpp_enc.min
        encoding.max = cpp_enc.max
        encoding.delta = cpp_enc.delta
        encoding.offset = cpp_enc.offset
        encoding.bw = cpp_enc.bw

    @property
    def roundingMode(self):
        cdef cpp.CppRoundingMode cpp_mode = self._ptr.roundingMode
        return RoundingMode(<int>cpp_mode)

    @roundingMode.setter
    def roundingMode(self, mode):
        cdef int m = int(mode)
        self._ptr.roundingMode = <cpp.CppRoundingMode>m

    @property
    def isEncodingValid(self) -> bool:
        return self._ptr.isEncodingValid

    @isEncodingValid.setter
    def isEncodingValid(self, bint value):
        self._ptr.isEncodingValid = value


# ============================================================================
# BlockTensorQuantizer
# ============================================================================

cdef class BlockTensorQuantizer:
    """Python wrapper for BlockTensorQuantizer."""
    cdef shared_ptr[cpp.CppBlockTensorQuantizer] _ptr

    def __cinit__(self):
        pass

    def __init__(self, shape, int bitwidth, quant_scheme=1):
        """Initialize BlockTensorQuantizer.

        Args:
            shape: List or tuple of dimensions
            bitwidth: Quantization bitwidth
            quant_scheme: QuantizationMode enum value (default: QUANTIZATION_TF_ENHANCED)
        """
        cdef vector[int64_t] cpp_shape
        cdef int qs = int(quant_scheme)
        for dim in shape:
            cpp_shape.push_back(<int64_t>dim)
        self._ptr = make_shared[cpp.CppBlockTensorQuantizer](
            cpp_shape, bitwidth, <cpp.CppQuantizationMode>qs)

    @staticmethod
    cdef BlockTensorQuantizer _from_shared_ptr(shared_ptr[cpp.CppBlockTensorQuantizer] ptr):
        cdef BlockTensorQuantizer obj = BlockTensorQuantizer.__new__(BlockTensorQuantizer)
        obj._ptr = ptr
        return obj

    def resetEncodingStats(self):
        deref(self._ptr).resetEncodingStats()

    def computeEncodings(self, bint use_symmetric) -> _List[TfEncoding]:
        cdef cpp.Encodings cpp_enc = deref(self._ptr).computeEncodings(use_symmetric)
        return [_encoding_from_cpp(cpp_enc[i]) for i in range(cpp_enc.size())]

    def setEncodings(self, list encodings):
        cdef cpp.Encodings cpp_enc
        cdef TfEncoding enc
        for enc in encodings:
            cpp_enc.push_back(_encoding_to_cpp(enc))
        deref(self._ptr).setEncodings(cpp_enc)

    def getEncodings(self) -> _List[TfEncoding]:
        cdef cpp.Encodings cpp_enc = deref(self._ptr).getEncodings()
        return [_encoding_from_cpp(cpp_enc[i]) for i in range(cpp_enc.size())]

    def updateStats(self, tensor):
        """Update stats from numpy tensor (any dtype, converted to float32)."""
        arr = _np.asarray(tensor)
        original_shape = arr.shape
        flat = _np.ascontiguousarray(arr.ravel(), dtype=_np.float32)
        cdef const float* ptr = <const float*><size_t>flat.ctypes.data
        cdef vector[int64_t] shape
        cdef int i
        for i in range(len(original_shape)):
            shape.push_back(<int64_t>original_shape[i])
        deref(self._ptr).updateStats(ptr, shape, False)

    def quantizeDequantize(self, input_tensor):
        """Quantize and dequantize tensor (any dtype), returns float32 output."""
        arr = _np.asarray(input_tensor)
        original_shape = arr.shape
        input_flat = _np.ascontiguousarray(arr.ravel(), dtype=_np.float32)
        output_flat = _np.zeros_like(input_flat)
        cdef const float* input_ptr = <const float*><size_t>input_flat.ctypes.data
        cdef float* output_ptr = <float*><size_t>output_flat.ctypes.data
        cdef vector[int64_t] shape
        cdef int i
        for i in range(len(original_shape)):
            shape.push_back(<int64_t>original_shape[i])
        deref(self._ptr).quantizeDequantize(input_ptr, output_ptr, shape, False)
        return output_flat.reshape(original_shape)

    def setQuantScheme(self, scheme):
        cdef int scheme_int = int(scheme)
        deref(self._ptr).setQuantScheme(<cpp.CppQuantizationMode>scheme_int)

    def getQuantScheme(self):
        cdef cpp.CppQuantizationMode cpp_mode = deref(self._ptr).getQuantScheme()
        return QuantizationMode(<int>cpp_mode)

    def setStrictSymmetric(self, bint strict):
        deref(self._ptr).setStrictSymmetric(strict)

    def getStrictSymmetric(self) -> bool:
        return deref(self._ptr).getStrictSymmetric()

    def setUnsignedSymmetric(self, bint unsigned):
        deref(self._ptr).setUnsignedSymmetric(unsigned)

    def getUnsignedSymmetric(self) -> bool:
        return deref(self._ptr).getUnsignedSymmetric()

    def getStatsHistogram(self) -> _List[_List[_Tuple[float, float]]]:
        """Get histogram of statistics for each block.

        Returns:
            List of histograms, where each histogram is a list of (bucket_edge, pdf) tuples
        """
        cdef cpp.BlockStatsHistogram cpp_hist = deref(self._ptr).getStatsHistogram()
        cdef list result = []
        cdef list inner
        cdef size_t i, j
        for i in range(cpp_hist.size()):
            inner = []
            for j in range(cpp_hist[i].size()):
                inner.append((cpp.get0(cpp_hist[i][j]), cpp.get1(cpp_hist[i][j])))
            result.append(inner)
        return result

    def setPercentileValue(self, float percentile):
        deref(self._ptr).setPercentileValue(percentile)

    def getPercentileValue(self) -> float:
        return deref(self._ptr).getPercentileValue()

    def setZeroPointShift(self, double shift):
        deref(self._ptr).setZeroPointShift(shift)

    def getZeroPointShift(self) -> float:
        return deref(self._ptr).getZeroPointShift()

    def getShape(self) -> _List[int]:
        cdef cpp.TensorDims shape = deref(self._ptr).getShape()
        return [shape[i] for i in range(shape.size())]

    def hasValidStats(self) -> bool:
        """Check if valid statistics have been collected."""
        return deref(self._ptr).hasValidStats()

    @property
    def bitwidth(self) -> int:
        return deref(self._ptr).bitwidth

    @bitwidth.setter
    def bitwidth(self, int value):
        deref(self._ptr).bitwidth = value

    @property
    def isEncodingValid(self) -> bool:
        return deref(self._ptr).isEncodingValid

    @isEncodingValid.setter
    def isEncodingValid(self, bint value):
        deref(self._ptr).isEncodingValid = value


# ============================================================================
# Module-level functions
# ============================================================================

def PtrToInt64(obj) -> int:
    """Convert object pointer to int64.

    Supports BlockTensorQuantizer and any object with _get_ptr_as_int64 method
    (e.g., QcQuantizeInfo from libquant_info).
    """
    cdef uint64_t ptr_val
    cdef BlockTensorQuantizer btq
    if isinstance(obj, BlockTensorQuantizer):
        btq = <BlockTensorQuantizer>obj
        ptr_val = <uint64_t><void*>(btq._ptr.get())
        return ptr_val
    # Duck typing: support objects with _get_ptr_as_int64 method (e.g., QcQuantizeInfo)
    if hasattr(obj, '_get_ptr_as_int64'):
        return obj._get_ptr_as_int64()
    raise TypeError(f"Cannot convert {type(obj)} to int64 pointer")


cdef extern from "DlQuantization/EncodingRescale.hpp" namespace "DlQuantization":
    pair[int, int] c_getScaleFactor "DlQuantization::getScaleFactor"(float x, int mbits)

    cdef cppclass CppConvSpecArgs "DlQuantization::ConvSpecArgs<float>":
        float out_encoding_delta
        float out_encoding_offset
        float input_scale
        unsigned char bw
        vector[float] weight_scale

    void c_getRescaledOutputAndBias "DlQuantization::getRescaledOutputAndBias<float>"(
        const float* bias_in, int count, CppConvSpecArgs& conv_args,
        float* bias_out, float* scaling_params, bint use_cuda, bint withOffsetWrap)


def getScaleFactor(float x, int mbits) -> _Tuple[int, int]:
    """Get scale factor as exponent and mantissa.

    Returns the exponent and mantissa of x, as a n-bit number.
    Constraint: iexpo must be in range -126..127
    Input must not be negative, inf, nan, zero, or denormal.

    Args:
        x: Float value to convert
        mbits: Number of mantissa bits

    Returns:
        Tuple of (exponent, mantissa)
    """
    cdef pair[int, int] result = c_getScaleFactor(x, mbits)
    return (result.first, result.second)


def getRescaledOutputAndBias(bias_in, conv_args_dict, bint use_cuda, bint with_offset_wrap):
    """Compute rescaled output and bias for on-device convolution simulation.

    Args:
        bias_in: Input bias numpy array (float32)
        conv_args_dict: Dictionary with keys:
            - out_encoding_delta: float
            - out_encoding_offset: float
            - input_scale: float
            - bw: int (bitwidth)
            - weight_scale: list of floats
        use_cuda: Whether to use CUDA
        with_offset_wrap: Whether to apply offset wrapping

    Returns:
        Tuple of (bias_out, scaling_params) numpy arrays
    """
    # Convert input to contiguous float32 array
    bias_arr = _np.ascontiguousarray(_np.asarray(bias_in).ravel(), dtype=_np.float32)
    cdef int count = <int>bias_arr.size

    # Prepare output arrays
    bias_out = _np.zeros(count, dtype=_np.float32)
    scaling_params = _np.zeros(count, dtype=_np.float32)

    # Set up ConvSpecArgs
    cdef CppConvSpecArgs conv_args
    conv_args.out_encoding_delta = <float>conv_args_dict['out_encoding_delta']
    conv_args.out_encoding_offset = <float>conv_args_dict['out_encoding_offset']
    conv_args.input_scale = <float>conv_args_dict['input_scale']
    conv_args.bw = <unsigned char>conv_args_dict['bw']

    cdef list weight_scale_list = conv_args_dict['weight_scale']
    cdef size_t i
    for i in range(len(weight_scale_list)):
        conv_args.weight_scale.push_back(<float>weight_scale_list[i])

    # Get pointers
    cdef const float* bias_in_ptr = <const float*><size_t>bias_arr.ctypes.data
    cdef float* bias_out_ptr = <float*><size_t>bias_out.ctypes.data
    cdef float* scaling_params_ptr = <float*><size_t>scaling_params.ctypes.data

    # Call C++ function
    c_getRescaledOutputAndBias(bias_in_ptr, count, conv_args,
                               bias_out_ptr, scaling_params_ptr,
                               use_cuda, with_offset_wrap)

    return (bias_out, scaling_params)

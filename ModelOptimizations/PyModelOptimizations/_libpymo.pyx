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
from cython.operator cimport dereference as deref

from typing import List as _List, Tuple as _Tuple

cimport _libpymo as cpp

# Import shared enums (sibling module in installed package)
from ._quant_enums import (
    QuantizationMode,
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



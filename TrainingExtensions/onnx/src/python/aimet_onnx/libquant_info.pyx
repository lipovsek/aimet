# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# cython: language_level=3
# distutils: language=c++

"""
Cython wrapper for QcQuantizeInfo and BlockTensorQuantizer.
Provides Python bindings via direct C++ wrapping.
"""

from libc.stdint cimport int64_t, uint64_t
from libc.stddef cimport size_t
from libcpp cimport bool as cbool
from libcpp.memory cimport shared_ptr, make_shared
from libcpp.string cimport string
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref

from typing import List, Tuple

cimport libquant_info as cpp

# Import shared enums (sibling module in installed package)
from ._quant_enums import (
    TensorQuantizerOpMode,
    QuantizationMode,
)


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
    """Convert C++ TfEncoding to Python TfEncoding."""
    return TfEncoding(enc.min, enc.max, enc.delta, enc.offset, enc.bw)


cdef cpp.CppTfEncoding _encoding_to_cpp(TfEncoding enc):
    """Convert Python TfEncoding to C++ TfEncoding."""
    cdef cpp.CppTfEncoding result
    result.min = enc.min
    result.max = enc.max
    result.delta = enc.delta
    result.offset = enc.offset
    result.bw = enc.bw
    return result


cdef class BlockTensorQuantizer:
    """Python wrapper for BlockTensorQuantizer."""
    cdef cpp.BlockTensorQuantizerPtr _ptr

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
    cdef BlockTensorQuantizer _from_shared_ptr(cpp.BlockTensorQuantizerPtr ptr):
        """Create wrapper from existing shared_ptr."""
        cdef BlockTensorQuantizer obj = BlockTensorQuantizer.__new__(BlockTensorQuantizer)
        obj._ptr = ptr
        return obj

    def resetEncodingStats(self):
        """Reset encoding statistics."""
        deref(self._ptr).resetEncodingStats()

    def computeEncodings(self, bint use_symmetric) -> List[TfEncoding]:
        """Compute encodings from collected statistics."""
        cdef cpp.Encodings cpp_enc = deref(self._ptr).computeEncodings(use_symmetric)
        return [_encoding_from_cpp(cpp_enc[i]) for i in range(cpp_enc.size())]

    def setEncodings(self, list encodings):
        """Set encodings."""
        cdef cpp.Encodings cpp_enc
        cdef TfEncoding enc
        for enc in encodings:
            cpp_enc.push_back(_encoding_to_cpp(enc))
        deref(self._ptr).setEncodings(cpp_enc)

    def getEncodings(self) -> List[TfEncoding]:
        """Get current encodings."""
        cdef cpp.Encodings cpp_enc = deref(self._ptr).getEncodings()
        return [_encoding_from_cpp(cpp_enc[i]) for i in range(cpp_enc.size())]

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

    def hasValidStats(self) -> bool:
        return deref(self._ptr).hasValidStats()

    def setQuantScheme(self, scheme):
        cdef int s = int(scheme)
        deref(self._ptr).setQuantScheme(<cpp.CppQuantizationMode>s)

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

    def getZeroPointShift(self) -> float:
        return deref(self._ptr).getZeroPointShift()

    def setZeroPointShift(self, double shift):
        deref(self._ptr).setZeroPointShift(shift)

    def getShape(self) -> List[int]:
        cdef cpp.TensorDims shape = deref(self._ptr).getShape()
        return [shape[i] for i in range(shape.size())]


cdef class QcQuantizeInfo:
    """Python wrapper for QcQuantizeInfo."""
    cdef cpp.CppQcQuantizeInfo* _ptr
    cdef object _tensor_quantizer  # Python reference to keep alive

    def __cinit__(self):
        self._ptr = new cpp.CppQcQuantizeInfo()
        self._tensor_quantizer = None

    def __dealloc__(self):
        if self._ptr != NULL:
            del self._ptr
            self._ptr = NULL

    def _get_ptr_as_int64(self) -> int:
        """Get pointer value as int64 for passing to custom ops."""
        return <int64_t><void*>self._ptr

    @property
    def tensorQuantizerRef(self):
        """Get the tensor quantizer."""
        return self._tensor_quantizer

    @tensorQuantizerRef.setter
    def tensorQuantizerRef(self, btq):
        """Set the tensor quantizer."""
        cdef BlockTensorQuantizer typed_btq
        self._tensor_quantizer = btq
        if btq is not None:
            typed_btq = <BlockTensorQuantizer>btq
            self._ptr.tensorQuantizer = typed_btq._ptr
        else:
            self._ptr.tensorQuantizer.reset()

    @property
    def encoding(self) -> List[TfEncoding]:
        """Get encodings from tensor quantizer."""
        cdef cpp.Encodings cpp_enc = self._ptr.getEncodings()
        return [_encoding_from_cpp(cpp_enc[i]) for i in range(cpp_enc.size())]

    @encoding.setter
    def encoding(self, list encodings):
        """Set encodings on tensor quantizer.

        Accepts TfEncoding objects from either libquant_info or _libpymo modules.
        """
        cdef cpp.Encodings cpp_enc
        cdef cpp.CppTfEncoding cpp_encoding
        for enc in encodings:
            # Duck-type: accept any object with min, max, delta, offset, bw attributes
            cpp_encoding.min = enc.min
            cpp_encoding.max = enc.max
            cpp_encoding.delta = enc.delta
            cpp_encoding.offset = enc.offset
            cpp_encoding.bw = enc.bw
            cpp_enc.push_back(cpp_encoding)
        self._ptr.setEncodings(cpp_enc)

    @property
    def opMode(self):
        cdef cpp.CppTensorQuantizerOpMode cpp_mode = self._ptr.opMode
        return TensorQuantizerOpMode(<int>cpp_mode)

    @opMode.setter
    def opMode(self, mode):
        cdef int m = int(mode)
        self._ptr.opMode = <cpp.CppTensorQuantizerOpMode>m

    @property
    def useSymmetricEncoding(self) -> bool:
        return self._ptr.useSymmetricEncoding

    @useSymmetricEncoding.setter
    def useSymmetricEncoding(self, bint value):
        self._ptr.useSymmetricEncoding = value

    @property
    def enabled(self) -> bool:
        return self._ptr.enabled

    @enabled.setter
    def enabled(self, bint value):
        self._ptr.enabled = value

    @property
    def isIntDataType(self) -> bool:
        return self._ptr.isIntDataType

    @isIntDataType.setter
    def isIntDataType(self, bint value):
        self._ptr.isIntDataType = value

    @property
    def usePerChannelMode(self) -> bool:
        return self._ptr.usePerChannelMode

    @usePerChannelMode.setter
    def usePerChannelMode(self, bint value):
        self._ptr.usePerChannelMode = value

    @property
    def channelAxis(self) -> int:
        return self._ptr.channelAxis

    @channelAxis.setter
    def channelAxis(self, int value):
        self._ptr.channelAxis = value

    @property
    def blockAxis(self) -> int:
        return self._ptr.blockAxis

    @blockAxis.setter
    def blockAxis(self, int value):
        self._ptr.blockAxis = value

    @property
    def blockSize(self) -> int:
        return self._ptr.blockSize

    @blockSize.setter
    def blockSize(self, size_t value):
        self._ptr.blockSize = value

    @property
    def name(self) -> str:
        return self._ptr.name.decode('utf-8')

    @name.setter
    def name(self, str value):
        self._ptr.name = value.encode('utf-8')


def PtrToInt64(obj) -> int:
    """Convert QcQuantizeInfo pointer to int64 for custom op attribute."""
    if isinstance(obj, QcQuantizeInfo):
        return (<QcQuantizeInfo>obj)._get_ptr_as_int64()
    raise TypeError(f"Cannot convert {type(obj)} to int64 pointer")

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# cython: language_level=3
# distutils: language=c++

"""C++ declarations for QcQuantizeInfo and related types."""

from libc.stdint cimport int64_t
from libcpp cimport bool as cbool
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

# Import shared declarations from common pxd file
from DlQuantization._quant_types cimport (
    CppQuantizationMode,
    QUANTIZATION_TF,
    QUANTIZATION_TF_ENHANCED,
    QUANTIZATION_RANGE_LEARNING,
    QUANTIZATION_PERCENTILE,
    CppTfEncoding,
    Encodings,
    CppTensorQuantizerOpMode,
    cpp_updateStats,
    cpp_oneShotQuantizeDequantize,
    cpp_quantizeDequantize,
    cpp_passThrough,
    TensorDims,
)


# BlockTensorQuantizer with methods needed by libquant_info
# (Note: declared here because the shared file doesn't include module-specific methods)
cdef extern from "DlQuantization/TensorQuantizer.h" namespace "DlQuantization":
    cdef cppclass CppBlockTensorQuantizer "DlQuantization::BlockTensorQuantizer":
        CppBlockTensorQuantizer(TensorDims shape, int bitwidth, CppQuantizationMode quantScheme) except +
        void resetEncodingStats()
        Encodings computeEncodings(cbool useSymmetricEncodings) except +
        void setEncodings(const Encodings& encodings) except +
        Encodings getEncodings()
        void setQuantScheme(CppQuantizationMode quantScheme)
        CppQuantizationMode getQuantScheme()
        cbool getStrictSymmetric()
        void setStrictSymmetric(cbool useStrictSymmetric)
        cbool getUnsignedSymmetric()
        void setUnsignedSymmetric(cbool useUnsignedSymmetric)
        double getZeroPointShift()
        void setZeroPointShift(double zeroPointShift)
        cbool hasValidStats()
        TensorDims getShape()
        cbool isEncodingValid
        int bitwidth


cdef extern from "QcQuantizeInfo.h":
    # Forward declare the shared_ptr type using the full C++ name
    ctypedef shared_ptr[CppBlockTensorQuantizer] BlockTensorQuantizerPtr "std::shared_ptr<DlQuantization::BlockTensorQuantizer>"

    cdef cppclass CppQcQuantizeInfo "QcQuantizeInfo":
        QcQuantizeInfo() except +
        void setEncodings(const Encodings& encodings) except +
        Encodings getEncodings()
        BlockTensorQuantizerPtr tensorQuantizer
        CppTensorQuantizerOpMode opMode
        cbool useSymmetricEncoding
        cbool enabled
        cbool isIntDataType
        cbool usePerChannelMode
        int channelAxis
        int blockAxis
        size_t blockSize
        string name

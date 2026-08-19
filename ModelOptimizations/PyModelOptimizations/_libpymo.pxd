# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# cython: language_level=3
# distutils: language=c++

"""C++ declarations for libpymo module."""

from libc.stdint cimport int64_t
from libcpp cimport bool as cbool
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.vector cimport vector

# Import shared declarations from common pxd file
from DlQuantization._quant_types cimport (
    CppQuantizationMode,
    CppQuantizationType,
    CppFloatQuantizationSpec,
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

# Declare C++ tuple type for histogram
cdef extern from "<tuple>" namespace "std" nogil:
    cdef cppclass tuple2dd "std::tuple<double, double>":
        pass
    # std::get for accessing tuple elements
    double get0 "std::get<0>"(tuple2dd&) nogil
    double get1 "std::get<1>"(tuple2dd&) nogil

ctypedef vector[tuple2dd] StatsHistogram
ctypedef vector[StatsHistogram] BlockStatsHistogram


# Additional declarations specific to libpymo
cdef extern from "DlQuantization/Quantization.hpp" namespace "DlQuantization":
    cdef enum CppRoundingMode "DlQuantization::RoundingMode":
        ROUND_NEAREST "DlQuantization::ROUND_NEAREST"
        ROUND_STOCHASTIC "DlQuantization::ROUND_STOCHASTIC"


# BlockTensorQuantizer with full method set including tensor operations and histogram
# (Note: declared here because it needs BlockStatsHistogram which is libpymo-specific)
cdef extern from "DlQuantization/TensorQuantizer.h" namespace "DlQuantization":
    cdef cppclass CppBlockTensorQuantizer "DlQuantization::BlockTensorQuantizer":
        CppBlockTensorQuantizer(TensorDims shape, int bitwidth, CppQuantizationMode quantScheme) except +
        CppBlockTensorQuantizer(TensorDims shape, CppQuantizationType qtype, CppQuantizationMode quantScheme) except +
        void resetEncodingStats()
        Encodings computeEncodings(cbool useSymmetricEncodings) except +
        void setEncodings(const Encodings& encodings) except +
        Encodings getEncodings()
        void updateStats(const float* tensor, const TensorDims& tensorShape, cbool useCuda) except +
        void quantizeDequantize(const float* input, float* output,
                                const TensorDims& tensorShape, cbool useCuda) except +
        void setQuantScheme(CppQuantizationMode quantScheme)
        CppQuantizationMode getQuantScheme()
        CppQuantizationType getQuantizationType()
        cbool getStrictSymmetric()
        void setStrictSymmetric(cbool useStrictSymmetric)
        cbool getUnsignedSymmetric()
        void setUnsignedSymmetric(cbool useUnsignedSymmetric)
        BlockStatsHistogram getStatsHistogram()
        void setPercentileValue(float percentile)
        float getPercentileValue()
        void setZeroPointShift(double zeroPointShift)
        double getZeroPointShift()
        TensorDims getShape()
        cbool hasValidStats()
        cbool isEncodingValid
        int bitwidth

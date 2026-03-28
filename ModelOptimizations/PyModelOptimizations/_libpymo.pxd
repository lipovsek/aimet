# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# cython: language_level=3
# distutils: language=c++

"""C++ declarations for libpymo module."""

from libc.stdint cimport int64_t
from libcpp cimport bool as cbool
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.vector cimport vector
from libcpp.pair cimport pair

# Import shared declarations from common pxd file
from DlQuantization._quant_types cimport (
    CppQuantizationMode,
    QUANTIZATION_TF,
    QUANTIZATION_TF_ENHANCED,
    QUANTIZATION_RANGE_LEARNING,
    QUANTIZATION_PERCENTILE,
    QUANTIZATION_MSE,
    QUANTIZATION_ENTROPY,
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
    cdef enum CppComputationMode "DlQuantization::ComputationMode":
        COMP_MODE_CPU "DlQuantization::COMP_MODE_CPU"
        COMP_MODE_GPU "DlQuantization::COMP_MODE_GPU"

    cdef enum CppLayerInOut "DlQuantization::LayerInOut":
        LAYER_INPUT "DlQuantization::LAYER_INPUT"
        LAYER_OUTPUT "DlQuantization::LAYER_OUTPUT"

    cdef enum CppRoundingMode "DlQuantization::RoundingMode":
        ROUND_NEAREST "DlQuantization::ROUND_NEAREST"
        ROUND_STOCHASTIC "DlQuantization::ROUND_STOCHASTIC"


cdef extern from "DlQuantization/IQuantizationEncodingAnalyzer.hpp" namespace "DlQuantization":
    cdef cppclass IQuantizationEncodingAnalyzer[T]:
        void updateStats(const T* tensor, size_t tensorSize, CppComputationMode tensorCpuGpuMode) except +
        void resetStats() except +
        CppTfEncoding computeEncoding(unsigned char bw, cbool useSymmetricEncodings,
                                      cbool useStrictSymmetric, cbool useUnsignedSymmetric) except +


cdef extern from "DlQuantization/ITensorQuantizationSim.h" namespace "DlQuantization":
    cdef cppclass ITensorQuantizationSim[T]:
        void quantizeDequantizeTensor(const T* inputTensorData, size_t inputTensorCount,
                                      T* outputTensorData, double encodingMin, double encodingMax,
                                      unsigned char bw, CppRoundingMode roundMode, cbool use_cuda) except +


cdef extern from "DlQuantization/QuantizerFactory.hpp" namespace "DlQuantization":
    unique_ptr[IQuantizationEncodingAnalyzer[float]] getEncodingAnalyzerInstance[T](CppQuantizationMode mode)
    unique_ptr[ITensorQuantizationSim[float]] getTensorQuantizationSim[T]()


# TensorQuantizer has additional methods beyond CppBlockTensorQuantizer in shared file
cdef extern from "DlQuantization/TensorQuantizer.h" namespace "DlQuantization":
    cdef cppclass CppTensorQuantizer "DlQuantization::TensorQuantizer":
        CppTensorQuantizer(CppQuantizationMode quantScheme, CppRoundingMode roundingMode) except +
        void resetEncodingStats()
        void updateStats(const float* tensor, size_t tensorSize, cbool useCuda) except +
        CppTfEncoding computeEncoding(unsigned int bitwidth, cbool useSymmetricEncoding) except +
        void quantizeDequantize(const float* input, size_t tensorSize, float* output,
                                double encodingMin, double encodingMax,
                                unsigned int bitwidth, cbool useCuda) except +
        void setQuantScheme(CppQuantizationMode quantScheme)
        CppQuantizationMode getQuantScheme()
        cbool getStrictSymmetric()
        void setStrictSymmetric(cbool useStrictSymmetric)
        cbool getUnsignedSymmetric()
        void setUnsignedSymmetric(cbool useUnsignedSymmetric)
        StatsHistogram getStatsHistogram()
        void setPercentileValue(float percentile)
        float getPercentileValue()
        void computePartialEncoding(unsigned char bw, CppTfEncoding& encoding,
                                    cbool useSymmetricEncodings, cbool useUnsignedSymmetric,
                                    cbool useStrictSymmetric)
        cbool hasValidStats()
        CppRoundingMode roundingMode
        cbool isEncodingValid

# BlockTensorQuantizer with full method set including tensor operations and histogram
# (Note: declared here because it needs BlockStatsHistogram which is libpymo-specific)
cdef extern from "DlQuantization/TensorQuantizer.h" namespace "DlQuantization":
    cdef cppclass CppBlockTensorQuantizer "DlQuantization::BlockTensorQuantizer":
        CppBlockTensorQuantizer(TensorDims shape, int bitwidth, CppQuantizationMode quantScheme) except +
        void resetEncodingStats()
        Encodings computeEncodings(cbool useSymmetricEncodings) except +
        void setEncodings(const Encodings& encodings) except +
        Encodings getEncodings()
        void updateStats(const float* tensor, const TensorDims& tensorShape, cbool useCuda) except +
        void quantizeDequantize(const float* input, float* output,
                                const TensorDims& tensorShape, cbool useCuda) except +
        void setQuantScheme(CppQuantizationMode quantScheme)
        CppQuantizationMode getQuantScheme()
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


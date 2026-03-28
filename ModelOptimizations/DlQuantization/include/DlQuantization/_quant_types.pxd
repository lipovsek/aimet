# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# cython: language_level=3
# distutils: language=c++

"""Shared C++ declarations for DlQuantization types.

This file contains common C++ type declarations used by both _libpymo and
libquant_info Cython modules. By centralizing these declarations, we avoid
duplication and ensure consistency across the codebase.

Note: CppBlockTensorQuantizer is declared separately in each module because
_libpymo needs additional methods (getStatsHistogram with BlockStatsHistogram)
that aren't used by libquant_info.
"""

from libc.stdint cimport int64_t
from libcpp cimport bool as cbool
from libcpp.vector cimport vector


cdef extern from "DlQuantization/Quantization.hpp" namespace "DlQuantization":
    cdef enum CppQuantizationMode "DlQuantization::QuantizationMode":
        QUANTIZATION_TF "DlQuantization::QUANTIZATION_TF"
        QUANTIZATION_TF_ENHANCED "DlQuantization::QUANTIZATION_TF_ENHANCED"
        QUANTIZATION_RANGE_LEARNING "DlQuantization::QUANTIZATION_RANGE_LEARNING"
        QUANTIZATION_PERCENTILE "DlQuantization::QUANTIZATION_PERCENTILE"
        QUANTIZATION_MSE "DlQuantization::QUANTIZATION_MSE"
        QUANTIZATION_ENTROPY "DlQuantization::QUANTIZATION_ENTROPY"

    cdef cppclass CppTfEncoding "DlQuantization::TfEncoding":
        CppTfEncoding() except +
        double min
        double max
        double delta
        double offset
        int bw

    ctypedef vector[CppTfEncoding] Encodings


cdef extern from "DlQuantization/TensorQuantizerOpFacade.h" namespace "DlQuantization":
    cdef enum CppTensorQuantizerOpMode "DlQuantization::TensorQuantizerOpMode":
        cpp_updateStats "DlQuantization::TensorQuantizerOpMode::updateStats"
        cpp_oneShotQuantizeDequantize "DlQuantization::TensorQuantizerOpMode::oneShotQuantizeDequantize"
        cpp_quantizeDequantize "DlQuantization::TensorQuantizerOpMode::quantizeDequantize"
        cpp_passThrough "DlQuantization::TensorQuantizerOpMode::passThrough"


cdef extern from "DlQuantization/TensorQuantizer.h" namespace "DlQuantization":
    ctypedef vector[int64_t] TensorDims

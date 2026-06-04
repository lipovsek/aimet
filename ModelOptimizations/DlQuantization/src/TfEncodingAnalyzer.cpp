// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <cassert>
#include <cstddef>
#include <vector>

#include "DlQuantization/Quantization.hpp"
#include "math_functions.hpp"
#include "quantization_utils.hpp"

#include "TfEncodingAnalyzer.h"

namespace DlQuantization
{

template <typename DTYPE>
std::vector<std::tuple<double, double>> TfEncodingAnalyzer<DTYPE>::getStatsHistogram() const
{
    // No real histogram data is kept for TF Encoding analyzer
    throw std::runtime_error("TfEncodingAnalyzer does not maintain histogram data");
}

template <typename DTYPE>
void TfEncodingAnalyzer<DTYPE>::updateStats(const DTYPE* tensor, const size_t tensorSize,
                                            ComputationMode tensorCpuGpuMode)
{
    this->_statsUpdated = true;
    // Compute stats for the tensor being passed in
    auto minmax = GetMinMax(tensor, tensorSize, tensorCpuGpuMode);
    double currentMin = std::get<0>(minmax);
    double currentMax = std::get<1>(minmax);

    // Update accumulated stats
    _accumulatedStats.min = std::min(_accumulatedStats.min, currentMin);
    _accumulatedStats.max = std::max(_accumulatedStats.max, currentMax);
}

template <typename DTYPE>
void TfEncodingAnalyzer<DTYPE>::updateStats(const DTYPE* tensor, const size_t tensorSize,
                                            ComputationMode tensorCpuGpuMode, IAllocator* allocator)
{
    updateStats(tensor, tensorSize, tensorCpuGpuMode);
}

template <typename DTYPE>
void TfEncodingAnalyzer<DTYPE>::resetStats()
{
    this->_accumulatedStats.min = 0.0;
    this->_accumulatedStats.max = 0.0;
}

template <typename DTYPE>
TfEncoding TfEncodingAnalyzer<DTYPE>::computeEncoding(uint8_t bw, bool useSymmetricEncodings, bool useStrictSymmetric,
                                                      bool useUnsignedSymmetric) const
{
    // If symmetric encodings are requested then strictSymmetric and unsignedSymmetric are exclusive modes
    if (useSymmetricEncodings)
        assert(!(useStrictSymmetric && useUnsignedSymmetric));

    // Make sure zero value is within the range
    double newMin = std::min(0.0, _accumulatedStats.min);
    double newMax = std::max(0.0, _accumulatedStats.max);
    return getComputedEncodings(bw, newMin, newMax, useSymmetricEncodings, useStrictSymmetric, useUnsignedSymmetric,
                                0.0);
}


template class TfEncodingAnalyzer<float>;

}   // namespace DlQuantization

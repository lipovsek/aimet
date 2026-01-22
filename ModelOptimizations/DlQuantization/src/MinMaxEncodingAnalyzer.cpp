// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <cassert>
#include <cstddef>
#include <vector>

#include "DlQuantization/Quantization.hpp"
#include "math_functions.hpp"
#include "quantization_utils.hpp"
#include "tensor_utils.hpp"

#include "MinMaxEncodingAnalyzer.h"

namespace DlQuantization
{


template <typename DTYPE>
MinMaxEncodingAnalyzer<DTYPE>::MinMaxEncodingAnalyzer(TensorDims shape)
{
    this->_shape     = shape;
    size_t numBlocks = getNumel(shape);
    _minStats.resize(numBlocks);
    _maxStats.resize(numBlocks);
    this->resetStats();
}

template <typename DTYPE>
void MinMaxEncodingAnalyzer<DTYPE>::updateStatsContiguous(const DTYPE* tensor, const TensorDims& shape,
                                                          size_t blockSize, ComputationMode tensorCpuGpuMode,
                                                          IAllocator* allocator, void* stream)
{
    size_t cnt                 = getNumel(shape);
    auto currMinMax            = GetMinMax(tensor, cnt, blockSize, tensorCpuGpuMode, allocator, stream);
    std::vector<DTYPE> currMin = std::get<0>(currMinMax);
    std::vector<DTYPE> currMax = std::get<1>(currMinMax);
    for (size_t idx = 0; idx < _minStats.size(); idx++)
    {
        _minStats[idx] = std::min(_minStats[idx], currMin[idx]);
        _maxStats[idx] = std::max(_maxStats[idx], currMax[idx]);
    }
}

template <typename DTYPE>
Encodings MinMaxEncodingAnalyzer<DTYPE>::computeEncoding(uint8_t bw, bool useSymmetricEncodings,
                                                         bool useStrictSymmetric, bool useUnsignedSymmetric,
                                                         double zeroPointShift) const
{
    // If symmetric encodings are requested then strictSymmetric and unsignedSymmetric are exclusive modes
    if (useSymmetricEncodings)
        assert(!(useStrictSymmetric && useUnsignedSymmetric));

    size_t numEncodings = _minStats.size();
    Encodings encodings(numEncodings);

    for (int idx = 0; idx < numEncodings; idx++)
    {
        // Make sure zero value is within the range
        double newMin = std::min(DTYPE(0.0), _minStats[idx]);
        double newMax = std::max(DTYPE(0.0), _maxStats[idx]);

        // When the min and max are too close together, nudge the maximum to meet the
        // minimum range requirement
        // This also handles the case where min==max==0 to avoid division by zero
        newMax         = std::max(newMax, newMin + MIN_RANGE);
        encodings[idx] = getComputedEncodings(bw, newMin, newMax, useSymmetricEncodings, useStrictSymmetric,
                                              useUnsignedSymmetric, zeroPointShift);
    }

    return encodings;
}

template <typename DTYPE>
std::vector<std::vector<std::tuple<double, double>>> MinMaxEncodingAnalyzer<DTYPE>::getStatsHistogram() const
{
    throw std::runtime_error("MinMaxEncodingAnalyzer does not have histogram stats");
}

template <typename DTYPE>
void MinMaxEncodingAnalyzer<DTYPE>::resetStats()
{
    for (size_t idx = 0; idx < _minStats.size(); idx++)
    {
        _minStats[idx] = std::numeric_limits<DTYPE>::max();
        _maxStats[idx] = std::numeric_limits<DTYPE>::lowest();
    }
}


template class MinMaxEncodingAnalyzer<double>;
template class MinMaxEncodingAnalyzer<float>;


}   // namespace DlQuantization

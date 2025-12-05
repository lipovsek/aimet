//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//  3. Neither the name of the copyright holder nor the names of its contributors
//     may be used to endorse or promote products derived from this software
//     without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//  SPDX-License-Identifier: BSD-3-Clause
//
//  @@-COPYRIGHT-END-@@
//
//==============================================================================

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

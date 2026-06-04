// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <cassert>
#include <cstddef>
#include <vector>

#include "DlQuantization/Quantization.hpp"
#include "math_functions.hpp"
#include "quantization_utils.hpp"
#include "tensor_utils.hpp"
#include <Eigen/Core>

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
template <typename T>
void MinMaxEncodingAnalyzer<DTYPE>::_updateStats(const T* tensor, const TensorDims& tensorShape,
                                                 ComputationMode tensorCpuGpuMode, IAllocator* allocator, void* stream)
{
    withContiguousBlocks(tensor, tensorShape, this->_shape, tensorCpuGpuMode, allocator, stream,
                         [this](const T* data, const TensorDims& shape, size_t blockSize, ComputationMode mode,
                                IAllocator* alloc, void* str)
                         {
                             size_t cnt      = getNumel(shape);
                             auto currMinMax = GetMinMax(data, cnt, blockSize, mode, alloc, str);
                             auto currMin    = std::get<0>(currMinMax);
                             auto currMax    = std::get<1>(currMinMax);
                             for (size_t idx = 0; idx < _minStats.size(); idx++)
                             {
                                 _minStats[idx] = std::min(_minStats[idx], DTYPE(currMin[idx]));
                                 _maxStats[idx] = std::max(_maxStats[idx], DTYPE(currMax[idx]));
                             }
                         });
}

template <typename DTYPE>
void MinMaxEncodingAnalyzer<DTYPE>::updateStats(const DTYPE* tensor, const TensorDims& tensorShape,
                                                ComputationMode tensorCpuGpuMode, IAllocator* allocator, void* stream)
{
    _updateStats(tensor, tensorShape, tensorCpuGpuMode, allocator, stream);
}

template <typename DTYPE>
void MinMaxEncodingAnalyzer<DTYPE>::updateStats(const Eigen::half* tensor, const TensorDims& tensorShape,
                                                ComputationMode tensorCpuGpuMode, IAllocator* allocator, void* stream)
{
    if (tensorCpuGpuMode == COMP_MODE_GPU)
    {
        _updateStats(tensor, tensorShape, tensorCpuGpuMode, allocator, stream);
    }
    else
    {
        size_t numel   = getNumel(tensorShape);
        float* fp32Buf = static_cast<float*>(allocator ? allocator->allocateRaw(sizeof(float) * numel)
                                                       : MemoryAllocation(tensorCpuGpuMode, sizeof(float) * numel));
        convertToFloat(tensor, fp32Buf, numel, tensorCpuGpuMode, stream);
        _updateStats(fp32Buf, tensorShape, tensorCpuGpuMode, allocator, stream);
        allocator ? allocator->deleteRaw(fp32Buf) : MemoryFree(tensorCpuGpuMode, fp32Buf);
    }
}

template <typename DTYPE>
void MinMaxEncodingAnalyzer<DTYPE>::updateStats(const Eigen::bfloat16* tensor, const TensorDims& tensorShape,
                                                ComputationMode tensorCpuGpuMode, IAllocator* allocator, void* stream)
{
    if (tensorCpuGpuMode == COMP_MODE_GPU)
    {
        _updateStats(tensor, tensorShape, tensorCpuGpuMode, allocator, stream);
    }
    else
    {
        size_t numel   = getNumel(tensorShape);
        float* fp32Buf = static_cast<float*>(allocator ? allocator->allocateRaw(sizeof(float) * numel)
                                                       : MemoryAllocation(tensorCpuGpuMode, sizeof(float) * numel));
        convertToFloat(tensor, fp32Buf, numel, tensorCpuGpuMode, stream);
        _updateStats(fp32Buf, tensorShape, tensorCpuGpuMode, allocator, stream);
        allocator ? allocator->deleteRaw(fp32Buf) : MemoryFree(tensorCpuGpuMode, fp32Buf);
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


template class MinMaxEncodingAnalyzer<float>;


}   // namespace DlQuantization

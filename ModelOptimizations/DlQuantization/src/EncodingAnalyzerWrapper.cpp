// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "EncodingAnalyzerWrapper.h"
#include "DlQuantization/QuantizerFactory.hpp"
#include "quantization_utils.hpp"
#include "tensor_utils.hpp"
#include <Eigen/Core>
#include <type_traits>


namespace DlQuantization
{

template <typename DTYPE>
EncodingAnalyzerWrapper<DTYPE>::EncodingAnalyzerWrapper(TensorDims shape, QuantizationMode mode)
{
    this->_shape     = shape;
    size_t numBlocks = getNumel(shape);
    _encodingAnalyzers.resize(numBlocks);
    for (auto& ptr: _encodingAnalyzers)
    {
        ptr = getEncodingAnalyzerInstance<DTYPE>(mode);
    }
}

template <typename DTYPE>
void EncodingAnalyzerWrapper<DTYPE>::updateStats(const DTYPE* tensor, const TensorDims& tensorShape,
                                                 ComputationMode tensorCpuGpuMode, IAllocator* allocator, void* stream)
{
    withContiguousBlocks(tensor, tensorShape, this->_shape, tensorCpuGpuMode, allocator, stream,
                         [this](const DTYPE* data, const TensorDims& shape, size_t blockSize, ComputationMode mode,
                                IAllocator* alloc, void* str)
                         { updateStatsContiguous(data, shape, blockSize, mode, alloc, str); });
}

template <typename DTYPE>
void EncodingAnalyzerWrapper<DTYPE>::updateStatsContiguous(const DTYPE* tensor, const TensorDims& shape,
                                                           size_t blockSize, ComputationMode tensorCpuGpuMode,
                                                           IAllocator* allocator, void* stream)
{
    synchronizeStream(tensorCpuGpuMode, stream);
    for (size_t idx = 0; idx < _encodingAnalyzers.size(); idx++)
    {
        _encodingAnalyzers[idx]->updateStats(tensor + idx * blockSize, blockSize, tensorCpuGpuMode, allocator);
    }
}

template <typename DTYPE>
void EncodingAnalyzerWrapper<DTYPE>::resetStats()
{
    for (auto& encodingAnalyzer: _encodingAnalyzers)
    {
        encodingAnalyzer->resetStats();
    }
}

template <typename DTYPE>
std::vector<TfEncoding>
EncodingAnalyzerWrapper<DTYPE>::computeEncoding(uint8_t bw, bool useSymmetricEncodings, bool useStrictSymmetric,
                                                bool useUnsignedSymmetric, double zeroPointShift) const
{
    if (zeroPointShift != 0.0)
    {
        throw std::runtime_error("Non-zero zeroPointShift is only supported for min-max quant scheme.");
    }
    std::vector<TfEncoding> encodings(_encodingAnalyzers.size());
    for (size_t idx = 0; idx < encodings.size(); idx++)
    {
        encodings[idx] = _encodingAnalyzers[idx]->computeEncoding(bw, useSymmetricEncodings, useStrictSymmetric,
                                                                  useUnsignedSymmetric);
    }
    return encodings;
}

template <typename DTYPE>
std::vector<std::vector<std::tuple<double, double>>> EncodingAnalyzerWrapper<DTYPE>::getStatsHistogram() const
{
    std::vector<std::vector<std::tuple<double, double>>> statsHistograms(_encodingAnalyzers.size());
    for (size_t idx = 0; idx < _encodingAnalyzers.size(); idx++)
    {
        statsHistograms[idx] = _encodingAnalyzers[idx]->getStatsHistogram();
    }
    return statsHistograms;
}

template <typename DTYPE>
void EncodingAnalyzerWrapper<DTYPE>::setPercentileValue(float percentile)
{
    for (auto& encodingAnalyzer: _encodingAnalyzers)
    {
        encodingAnalyzer->setPercentileValue(percentile);
    }
}

template <typename DTYPE>
float EncodingAnalyzerWrapper<DTYPE>::getPercentileValue()
{
    return _encodingAnalyzers[0]->getPercentileValue();
}

template <typename DTYPE>
void EncodingAnalyzerWrapper<DTYPE>::updateStats(const Eigen::half* tensor, const TensorDims& tensorShape,
                                                 ComputationMode tensorCpuGpuMode, IAllocator* allocator, void* stream)
{
    if constexpr (std::is_same_v<DTYPE, float>)
    {
        size_t numel   = getNumel(tensorShape);
        float* fp32Buf = static_cast<float*>(allocator ? allocator->allocateRaw(sizeof(float) * numel)
                                                       : MemoryAllocation(tensorCpuGpuMode, sizeof(float) * numel));
        convertHalfToFloat(tensor, fp32Buf, numel, tensorCpuGpuMode, stream);
        updateStats(fp32Buf, tensorShape, tensorCpuGpuMode, allocator, stream);
        allocator ? allocator->deleteRaw(fp32Buf) : MemoryFree(tensorCpuGpuMode, fp32Buf);
    }
    else
    {
        throw std::runtime_error("fp16 calibration only supported with float accumulators");
    }
}

template class EncodingAnalyzerWrapper<float>;

}   // namespace DlQuantization

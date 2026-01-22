// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "ContiguousEncodingAnalyzer.h"
#include "math_functions.hpp"
#include "tensor_utils.hpp"


namespace DlQuantization
{

template <typename DTYPE>
void ContiguousEncodingAnalyzerBase<DTYPE>::updateStats(const DTYPE* tensor, const TensorDims& tensorShape,
                                                        ComputationMode tensorCpuGpuMode, IAllocator* allocator,
                                                        void* stream)
{
    auto numBlocks = getNumel(_shape);
    auto numel     = getNumel(tensorShape);

    // Early exit for per-tensor mode
    if (numBlocks == 1)
    {
        return updateStatsContiguous(tensor, tensorShape, numel, tensorCpuGpuMode, allocator, stream);
    }

    // View tensor and encoding as broadcastable shapes
    auto bcShapes      = getBroadcastableShapes(tensorShape, _shape);
    auto bcTensorShape = std::get<0>(bcShapes);
    auto bcEncShape    = std::get<1>(bcShapes);

    size_t blockSize = numel / numBlocks;

    std::vector<size_t> broadcastDims, nonBroadcastDims;

    // Determine the dim ordering such that all indexes in a single quantization block are contiguous
    for (size_t i = 0; i < bcTensorShape.size(); i++)
    {
        if (bcEncShape[i] == 1 && bcTensorShape[i] != 1)
        {
            broadcastDims.push_back(i);
        }
        else
        {
            nonBroadcastDims.push_back(i);
        }
    }
    std::vector<size_t> dimOrder = nonBroadcastDims;
    dimOrder.insert(dimOrder.end(), broadcastDims.begin(), broadcastDims.end());

    // Permute the input to have contiguous blocks if necessary and update stats
    if (not hasContiguousBlocks(bcTensorShape, bcEncShape))
    {
        DTYPE* tempBuffer = static_cast<DTYPE*>(allocator ? allocator->allocateRaw(sizeof(DTYPE) * numel)
                                                          : MemoryAllocation(tensorCpuGpuMode, sizeof(DTYPE) * numel));
        permute(tensor, tempBuffer, bcTensorShape, dimOrder, tensorCpuGpuMode, stream);
        updateStatsContiguous(tempBuffer, tensorShape, blockSize, tensorCpuGpuMode, allocator, stream);
        allocator ? allocator->deleteRaw(tempBuffer) : MemoryFree(tensorCpuGpuMode, tempBuffer);
    }
    else
    {
        updateStatsContiguous(tensor, tensorShape, blockSize, tensorCpuGpuMode, allocator, stream);
    }
}

template <typename DTYPE>
TensorDims ContiguousEncodingAnalyzerBase<DTYPE>::getShape()
{
    return _shape;
}

template class ContiguousEncodingAnalyzerBase<double>;

template class ContiguousEncodingAnalyzerBase<float>;

}   // namespace DlQuantization
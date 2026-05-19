// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DL_QUANTIZATION_TENSOR_UTILS_H
#define DL_QUANTIZATION_TENSOR_UTILS_H

#include "math_functions.hpp"
#include <DlQuantization/Quantization.hpp>

namespace Eigen
{
struct half;
struct bfloat16;
}   // namespace Eigen

namespace DlQuantization
{


std::tuple<TensorDims, TensorDims> getBroadcastableShapes(const TensorDims& tensorShape,
                                                          const TensorDims& encodingShape);

size_t getNumel(const TensorDims& shape);

TensorDims shapeToStrides(const TensorDims& shape);

bool hasContiguousBlocks(const TensorDims& tensorShape, const TensorDims& encodingShape);

template <typename T>
void permute(const T* input, T* output, const TensorDims& inputShape, std::vector<size_t> order, ComputationMode mode,
             void* stream = nullptr);

template <typename T>
void permuteKernelCPU(const T* inTensor, T* outTensor, size_t numel, const TensorDims& inputStrides,
                      const TensorDims& outputStrides);

template <typename T>
void permuteKernelGPU(const T* inTensor, T* outTensor, size_t numel, const TensorDims& inputStrides,
                      const TensorDims& outputStrides, void* stream);

template <typename T>
void convertToFloat(const T* input, float* output, size_t count, ComputationMode mode, void* stream = nullptr);

#ifdef GPU_QUANTIZATION_ENABLED
template <typename T>
void convertToFloat_gpu(const T* in, size_t cnt, float* out, void* stream);
#endif

void synchronizeStream(ComputationMode mode, void* stream);

/**
 * @brief Rearrange tensor data so that each encoding block is contiguous in memory, then invoke the
 *        callback once with a pointer to the (possibly permuted) data.
 *
 * If the blocks are already contiguous (or there is only one block), the callback receives the
 * original pointer directly.  Otherwise a temporary buffer is allocated, the data is permuted into
 * it, and the buffer is freed after the callback returns.
 */
template <typename T, typename Callback>
void withContiguousBlocks(const T* tensor, const TensorDims& tensorShape, const TensorDims& encodingShape,
                          ComputationMode tensorCpuGpuMode, IAllocator* allocator, void* stream, Callback&& cb)
{
    auto numBlocks   = getNumel(encodingShape);
    auto numel       = getNumel(tensorShape);
    size_t blockSize = numel / numBlocks;

    if (numBlocks == 1)
    {
        cb(tensor, tensorShape, numel, tensorCpuGpuMode, allocator, stream);
        return;
    }

    auto bcShapes      = getBroadcastableShapes(tensorShape, encodingShape);
    auto bcTensorShape = std::get<0>(bcShapes);
    auto bcEncShape    = std::get<1>(bcShapes);

    if (not hasContiguousBlocks(bcTensorShape, bcEncShape))
    {
        std::vector<size_t> broadcastDims, nonBroadcastDims;
        for (size_t i = 0; i < bcTensorShape.size(); i++)
        {
            if (bcEncShape[i] == 1 && bcTensorShape[i] != 1)
                broadcastDims.push_back(i);
            else
                nonBroadcastDims.push_back(i);
        }
        std::vector<size_t> dimOrder = nonBroadcastDims;
        dimOrder.insert(dimOrder.end(), broadcastDims.begin(), broadcastDims.end());

        T* tempBuffer = static_cast<T*>(allocator ? allocator->allocateRaw(sizeof(T) * numel)
                                                  : MemoryAllocation(tensorCpuGpuMode, sizeof(T) * numel));
        permute(tensor, tempBuffer, bcTensorShape, dimOrder, tensorCpuGpuMode, stream);
        cb(tempBuffer, tensorShape, blockSize, tensorCpuGpuMode, allocator, stream);
        allocator ? allocator->deleteRaw(tempBuffer) : MemoryFree(tensorCpuGpuMode, tempBuffer);
    }
    else
    {
        cb(tensor, tensorShape, blockSize, tensorCpuGpuMode, allocator, stream);
    }
}

void synchronizeCudaStream(void* stream);

}   // namespace DlQuantization

#endif   // DL_QUANTIZATION_TENSOR_UTILS_H

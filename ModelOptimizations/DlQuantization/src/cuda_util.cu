// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "cuda_util.hpp"
#include <Eigen/Core>
#include <stdexcept>

namespace DlQuantization
{
bool CudaMemCpy(void* dest, const void* src, size_t bytes, CudaMemcpyDirection direction)
{
    if (CudaMemcpyDirection::DEVICE_TO_HOST == direction)
    {
        return cudaSuccess == cudaMemcpy(dest, src, bytes, cudaMemcpyDeviceToHost);
    }
    else
    {
        return cudaSuccess == cudaMemcpy(dest, src, bytes, cudaMemcpyHostToDevice);
    }
}

bool CudaSupportedHelper()
{
    int deviceCount;
    cudaError_t e = cudaGetDeviceCount(&deviceCount);
    if (cudaSuccess != e || 0 == deviceCount)
    {
        return false;
    }
    else
    {
        return true;
    }
}

bool CudaSynchronize()
{
    return cudaSuccess == cudaDeviceSynchronize();
}

void* CudaAllocator::allocateRaw(size_t bytes)
{
    void* ptr;
    cudaMalloc(&ptr, bytes);
    return ptr;
}

void CudaAllocator::deleteRaw(void* ptr)
{
    cudaFree(ptr);
}

template <typename T>
void copyTensorsCuda(T* outTensor, const T* inTensor, size_t count, void* stream)
{
    cudaError_t e = cudaMemcpyAsync(outTensor, inTensor, count * sizeof(T), cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream));
    if (e != cudaSuccess)
    {
        throw std::runtime_error("CUDA memcpy failed");
    }
}

// Explicit template instantiations
template void copyTensorsCuda<float>(float* outTensor, const float* inTensor, size_t count, void* stream);
template void copyTensorsCuda<Eigen::half>(Eigen::half* outTensor, const Eigen::half* inTensor, size_t count,
                                           void* stream);
template void copyTensorsCuda<Eigen::bfloat16>(Eigen::bfloat16* outTensor, const Eigen::bfloat16* inTensor,
                                               size_t count, void* stream);

}   // End of namespace DlQuantization

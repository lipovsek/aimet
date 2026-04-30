// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "cuda_util.hpp"
#include "tensor_utils.hpp"
#include <Eigen/Core>

namespace DlQuantization
{

template <typename DTYPE>
__global__ void permuteTensorKernel(const DTYPE* in, DTYPE* out, int numElements, int numDims,
                                    const TensorDim* inputStrides, const TensorDim* outputStrides)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numElements; i += blockDim.x * gridDim.x)
    {
        size_t outputIdx = 0;
        size_t remainder = i;
        for (auto dim = 0; dim < numDims; dim++)
        {
            size_t dimIdx = remainder / inputStrides[dim];
            remainder     = remainder - dimIdx * inputStrides[dim];
            outputIdx += outputStrides[dim] * dimIdx;
        }

        out[outputIdx] = in[i];
    }
}


template <typename T>
void permuteKernelGPU(const T* inTensor, T* outTensor, size_t numel, const TensorDims& inputStrides,
                      const TensorDims& outputStrides, void* stream)
{
    size_t numDims       = inputStrides.size();
    int64_t totalThreads = numel;
    int64_t gridSize     = CUDA_NUM_BLOCKS(totalThreads);
    TensorDim strideData[2][numDims];
    auto cuStream = static_cast<cudaStream_t>(stream);

    // Copy the stride information to the cuda device
    for (int i = 0; i < numDims; i++)
    {
        strideData[0][i] = inputStrides[i];
        strideData[1][i] = outputStrides[i];
    }
    TensorDim* deviceStrideData;
    cudaMalloc((void**) &deviceStrideData, 2 * numDims * sizeof(TensorDim));
    cudaMemcpyAsync(deviceStrideData, strideData, 2 * numDims * sizeof(TensorDim), cudaMemcpyHostToDevice, cuStream);

    // Launch the cuda kernel
    permuteTensorKernel<<<gridSize, CUDA_NUM_THREADS, 0, cuStream>>>(inTensor, outTensor, numel, numDims,
                                                                     deviceStrideData, deviceStrideData + numDims);

    // Free the device stride data
    cudaFree(deviceStrideData);
}

void synchronizeCudaStream(void* stream)
{
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
}

__global__ void convertHalfToFloatKernel(const __half* in, uint64_t cnt, float* out)
{
    CUDA_KERNEL_LOOP(i, cnt)
    {
        *(out + i) = __half2float(*(in + i));
    }
}

void convertHalfToFloat_gpu(const Eigen::half* in, size_t cnt, float* out, void* stream)
{
    convertHalfToFloatKernel<<<CUDA_NUM_BLOCKS(cnt), CUDA_NUM_THREADS, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __half*>(in), cnt, out);
}


template void permuteKernelGPU(const float* intensor, float* outTensor, size_t numel, const TensorDims& inputStrides,
                               const TensorDims& outputStrides, void* stream);

template void permuteKernelGPU(const Eigen::half* intensor, Eigen::half* outTensor, size_t numel,
                               const TensorDims& inputStrides, const TensorDims& outputStrides, void* stream);

}   // namespace DlQuantization

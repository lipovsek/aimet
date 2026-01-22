// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef UTIL_CUDA_UTIL_H_
#define UTIL_CUDA_UTIL_H_

#include "DlQuantization/Quantization.hpp"

namespace DlQuantization
{
enum class CudaMemcpyDirection : char
{
    DEVICE_TO_HOST,
    HOST_TO_DEVICE
};

/**
 * @brief Copy memory between host and device.
 * @return True if CUDA call succeeds.
 *
 * The memory allocation has to happen outside of this function.
 */
bool CudaMemCpy(void* dest, const void* src, size_t bytes, CudaMemcpyDirection direction);

/**
 * @brief Find out at runtime if there exists a GPU with CUDA support.
 * @return True if CUDA is supported, false otherwise.
 */
bool CudaSupportedHelper();

/**
 * @brief Make sure all kernels have finished.
 *
 * This allows for accurate timing measurements.
 */
bool CudaSynchronize();

// Always use 512 threads per block
const int CUDA_NUM_THREADS = 512;

// Compute the number of blocks based on the total number of threads.
inline size_t CUDA_NUM_BLOCKS(const size_t N)
{
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// Loop over data in kernel (stride: grid)
#define CUDA_KERNEL_LOOP(i, n) \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

}   // End of namespace DlQuantization

#endif   // UTIL_CUDA_UTIL_H_

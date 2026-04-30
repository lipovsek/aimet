// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <limits>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <cub/cub.cuh>
#include <cstdint>

#include "cuda_util.hpp"
#include "math_functions.hpp"
#include <Eigen/Core>


namespace DlQuantization
{

// Custom min/max reduction operators. Unlike cub::Min/cub::Max, these resolve comparisons through
// Eigen::half's operator< which provides __device__ implementations via __hlt/__hgt intrinsics,
// avoiding the ambiguous __half implicit conversion operators that break cub::Min/Max on CUDA 12.0.
struct ReduceMin
{
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T& a, const T& b) const
    {
        return (b < a) ? b : a;
    }
};

struct ReduceMax
{
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T& a, const T& b) const
    {
        return (a < b) ? b : a;
    }
};

template <typename DTYPE>
DTYPE GetMax_gpu(const DTYPE* data, uint64_t cnt)
{
    const thrust::device_ptr<const DTYPE> ptr = thrust::device_pointer_cast(data);
    return thrust::reduce(ptr, ptr + cnt, std::numeric_limits<DTYPE>::lowest(), thrust::maximum<DTYPE>());
}

template <typename DTYPE>
DTYPE GetMin_gpu(const DTYPE* data, uint64_t cnt)
{
    const thrust::device_ptr<const DTYPE> ptr = thrust::device_pointer_cast(data);
    return thrust::reduce(ptr, ptr + cnt, std::numeric_limits<DTYPE>::max(), thrust::minimum<DTYPE>());
}



struct _ApplyScale
{
    _ApplyScale(uint64_t scale) : scale(scale)
    {
    }

    __host__ __device__ uint64_t operator()(uint64_t x) const
    {
        return x * scale;
    }

private:
    uint64_t scale;
};


template <typename DTYPE>
std::tuple<std::vector<DTYPE>, std::vector<DTYPE>> GetMinMax_gpu(const DTYPE* data, uint64_t cnt, uint64_t blockSize,
                                                                 IAllocator* allocator, void* stream)
{
    auto computeStream         = static_cast<cudaStream_t>(stream);

    // Faster for per-tensor mode
    if (cnt == blockSize)
    {
        // Per-tensor reduce requires stream to be synchronized first
        cudaStreamSynchronize(computeStream);
        auto minMax = GetMinMax_gpu(data, cnt);
        return std::make_tuple<std::vector<DTYPE>, std::vector<DTYPE> >({std::get<0>(minMax)}, {std::get<1>(minMax)});
    }

    size_t numBlocks           = cnt / blockSize;
    void* dTempStorage         = nullptr;
    size_t tempStorageBytesMin = 0;
    size_t tempStorageBytesMax = 0;
    size_t memSize             = sizeof(DTYPE) * 2 * numBlocks;
    std::vector<DTYPE> minMaxOut(2 * numBlocks);
    DTYPE* dMinMaxOut;
    dMinMaxOut = static_cast<DTYPE*>(allocator ? allocator->allocateRaw(memSize) : MemoryAllocation_gpu(memSize));

    auto offsetIterator = thrust::make_transform_iterator(thrust::make_counting_iterator<uint64_t>(0), _ApplyScale(blockSize));

    // When dTempStorage is nullptr, this does not do any device computation, but sets tempStorageBytes to the size of
    // temporary storage necessary for computation.
    // Use the maximum storage needed for min and max calculations (these will likely be identical, this is just to be
    // safe)
    DTYPE initMin = std::numeric_limits<DTYPE>::max();
    DTYPE initMax = std::numeric_limits<DTYPE>::lowest();
    cub::DeviceSegmentedReduce::Reduce(dTempStorage, tempStorageBytesMin, data, dMinMaxOut, numBlocks, offsetIterator,
                                       offsetIterator + 1, ReduceMin(), initMin);
    cub::DeviceSegmentedReduce::Reduce(dTempStorage, tempStorageBytesMax, data, dMinMaxOut + numBlocks, numBlocks,
                                       offsetIterator, offsetIterator + 1, ReduceMax(), initMax);
    size_t tempStorageBytes = std::max(tempStorageBytesMin, tempStorageBytesMax);

    // Allocate the temporary device storage
    dTempStorage = static_cast<DTYPE*>(allocator ? allocator->allocateRaw(tempStorageBytes)
                                                 : MemoryAllocation_gpu(tempStorageBytes));

    // Perform the actual min/max reductions
    cub::DeviceSegmentedReduce::Reduce(dTempStorage, tempStorageBytes, data, dMinMaxOut, numBlocks, offsetIterator,
                                       offsetIterator + 1, ReduceMin(), initMin, computeStream);
    cub::DeviceSegmentedReduce::Reduce(dTempStorage, tempStorageBytes, data, dMinMaxOut + numBlocks, numBlocks,
                                       offsetIterator, offsetIterator + 1, ReduceMax(), initMax, computeStream);

    // Transfer reduced min/max to CPU
    cudaStreamSynchronize(computeStream);
    cudaMemcpy(minMaxOut.data(), dMinMaxOut, 2 * sizeof(DTYPE) * numBlocks, cudaMemcpyDeviceToHost);

    std::vector<DTYPE> minOut(minMaxOut.begin(), minMaxOut.begin() + numBlocks);
    std::vector<DTYPE> maxOut(minMaxOut.begin() + numBlocks, minMaxOut.end());

    // Free allocated device memory
    allocator ? allocator->deleteRaw(dTempStorage) : (void) MemoryFree_gpu(dTempStorage);
    allocator ? allocator->deleteRaw(dMinMaxOut) : (void) MemoryFree_gpu(dMinMaxOut);

    return std::make_tuple(std::move(minOut), std::move(maxOut));
}

template <typename DTYPE>
std::tuple<DTYPE, DTYPE> GetMinMax_gpu(const DTYPE* data, uint64_t cnt)
{
    DTYPE minMaxOut[2];
    DTYPE *dMinMaxOut;
    void *dTempStorage = nullptr;
    size_t tempStorageBytesMin = 0;
    size_t tempStorageBytesMax = 0;
    cudaMalloc(&dMinMaxOut, sizeof(DTYPE) * 2);

    // When dTempStorage is nullptr, this does not do any device computation, but sets tempStorageBytes to the size of
    // temporary storage necessary for computation.
    // Use the maximum storage needed for min and max calculations (these will likely be identical, this is just to be safe)
    DTYPE initMin = std::numeric_limits<DTYPE>::max();
    DTYPE initMax = std::numeric_limits<DTYPE>::lowest();
    cub::DeviceReduce::Reduce(dTempStorage, tempStorageBytesMin, data, dMinMaxOut, cnt, ReduceMin(), initMin);
    cub::DeviceReduce::Reduce(dTempStorage, tempStorageBytesMax, data, dMinMaxOut + 1, cnt, ReduceMax(), initMax);
    size_t tempStorageBytes = std::max(tempStorageBytesMin, tempStorageBytesMax);

    // Allocate the temporary device storage
    cudaMalloc(&dTempStorage, tempStorageBytes);

    // Perform the actual min/max reductions
    cub::DeviceReduce::Reduce(dTempStorage, tempStorageBytes, data, dMinMaxOut, cnt, ReduceMin(), initMin);
    cub::DeviceReduce::Reduce(dTempStorage, tempStorageBytes, data, dMinMaxOut + 1, cnt, ReduceMax(), initMax);

    // Transfer reduce min/max to CPU
    cudaMemcpy(minMaxOut, dMinMaxOut, 2 * sizeof(DTYPE), cudaMemcpyDeviceToHost);

    // Free allocated device memory
    cudaFree(dTempStorage);
    cudaFree(dMinMaxOut);
    return std::make_tuple(minMaxOut[0], minMaxOut[1]);
}

__global__ void ElementwiseMult_kernel(const float* in, size_t cnt, float factor, float* out)
{
    CUDA_KERNEL_LOOP(i, cnt)
    {
        out[i] = in[i] * factor;
    }
}

void ElementwiseMult_gpu(const float* in, size_t cnt, float factor, float* out)
{
    ElementwiseMult_kernel<<<CUDA_NUM_BLOCKS(cnt), CUDA_NUM_THREADS>>>(in, cnt, factor, out);
}

void* MemoryAllocation_gpu(size_t bytes)
{
    void* devPtr;
    auto status = cudaMalloc(&devPtr, bytes);

    if (cudaErrorMemoryAllocation == status) {
        throw std::runtime_error("CUDA OOM");
    }

    if (cudaSuccess != status) {
        throw std::runtime_error("cuda malloc failed");
    }

    return devPtr;
}

bool MemoryFree_gpu(void* data)
{
    return cudaSuccess == cudaFree(data);
}

template float GetMax_gpu(const float* data, uint64_t cnt);

template float GetMin_gpu(const float* data, uint64_t cnt);

template std::tuple<std::vector<float>, std::vector<float>> GetMinMax_gpu(const float* data, uint64_t cnt, uint64_t blockSize,
                                                                          IAllocator* allocator, void* stream);

template std::tuple<float, float> GetMinMax_gpu(const float* data, uint64_t cnt);

template std::tuple<std::vector<Eigen::half>, std::vector<Eigen::half>>
GetMinMax_gpu(const Eigen::half* data, uint64_t cnt, uint64_t blockSize, IAllocator* allocator, void* stream);

template std::tuple<Eigen::half, Eigen::half> GetMinMax_gpu(const Eigen::half* data, uint64_t cnt);

template <typename DTYPE>
void GetHistogram_gpu(const DTYPE* data,
                      uint64_t cnt,
                      uint32_t histogram[PDF_SIZE],
                      const double bucket_size,
                      const int pdf_offset,
                      const bool is_signed,
                      IAllocator* allocator)
{
    int num_levels    = PDF_SIZE + 1;   // Number of bin edges in histogram
    // Note: double precision needed here to represent: width = (upper_level - lower_level) / (num_levels - 1)
    double lower_level = pdf_offset * bucket_size;
    double upper_level = (pdf_offset + PDF_SIZE) * bucket_size;

    constexpr size_t hist_size = sizeof(uint32_t) * PDF_SIZE;
    auto histogram_gpu =
        static_cast<uint32_t*>(allocator ? allocator->allocateRaw(hist_size) : MemoryAllocation_gpu(hist_size));

    // When d_temp_storage is nullptr, this does not do any device computation, but sets temp_storage_bytes to the size
    // of temporary storage necessary for computation.
    void* d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, data, histogram_gpu, num_levels,
                                        lower_level, upper_level, cnt);

    // Allocate temporary storage
    d_temp_storage = allocator ? allocator->allocateRaw(temp_storage_bytes) : MemoryAllocation_gpu(temp_storage_bytes);

    // Perform the actual histogram computation
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, data, histogram_gpu, num_levels,
                                        lower_level, upper_level, cnt);

    // Transfer histogram to CPU
    cudaMemcpy(histogram, histogram_gpu, sizeof(uint32_t) * PDF_SIZE, cudaMemcpyDeviceToHost);

    allocator ? allocator->deleteRaw(d_temp_storage) : (void) MemoryFree_gpu(d_temp_storage);
    allocator ? allocator->deleteRaw(histogram_gpu) : (void) MemoryFree_gpu(histogram_gpu);
}


template void GetHistogram_gpu(const float* data,
                               uint64_t cnt,
                               uint32_t histogram[PDF_SIZE],
                               const double bucket_size,
                               const int pdf_offset,
                               const bool is_signed,
                               IAllocator* allocator);

}   // End of namespace DlQuantization

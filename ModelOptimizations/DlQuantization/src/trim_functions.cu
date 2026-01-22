// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "cuda_fp16.h"
#include "cuda_util.hpp"
#include "trim_functions.cuh"
#include "trim_functions.hpp"

namespace DlQuantization
{
template <typename DTYPE>
__global__ void quantizeDequantizeKernel(const DTYPE* in, uint64_t cnt, DTYPE* out,
                                         DTYPE encoding_min, DTYPE encoding_max,
                                         DTYPE encoding_delta, DTYPE encoding_offset,
                                         RoundingMode rounding_mode)
{
    CUDA_KERNEL_LOOP(i, cnt)
    {
        quantizeToFxpDevice<DTYPE>(in + i, out + i,
                                   encoding_min, encoding_max,
                                   encoding_delta, encoding_offset,
                                   rounding_mode, i);
        dequantizeFromFxpDevice<DTYPE>(out + i, encoding_delta, encoding_offset);
    }
}

template <typename DTYPE>
__global__ void quantizeToFxpKernel(const DTYPE* in, uint64_t cnt, DTYPE* out,
                                    DTYPE encoding_min, DTYPE encoding_max,
                                    DTYPE encoding_delta, DTYPE encoding_offset,
                                    RoundingMode rounding_mode, unsigned int shift)
{
    CUDA_KERNEL_LOOP(i, cnt)
    {
        quantizeToFxpDevice<DTYPE>(in + i, out + i,
                                   encoding_min, encoding_max,
                                   encoding_delta, encoding_offset,
                                   rounding_mode, i);
        *(out + i) -= shift;
    }
}

template <typename DTYPE>
__global__ void quantizeDequantizePerChannelKernel(const DTYPE* in, int numChannel, int numElement, int numElementPerChannel,
                                                   DTYPE* out, DTYPE* encodingMin, DTYPE* encodingMax, DTYPE* encodingDelta,
                                                   DTYPE* encodingOffset, RoundingMode roundingMode)
{
    CUDA_KERNEL_LOOP(i, numElement)
    {
        int channelIdx = (i / numElementPerChannel) % numChannel;
        quantizeToFxpDevice<DTYPE>(in + i, out + i,
                                   *(encodingMin + channelIdx), *(encodingMax + channelIdx),
                                   *(encodingDelta + channelIdx), *(encodingOffset + channelIdx),
                                   roundingMode, i);
        dequantizeFromFxpDevice<DTYPE>(out + i, *(encodingDelta + channelIdx), *(encodingOffset + channelIdx));
    }
}



template <typename DTYPE>
__global__ void quantizeDequantizeBroadcastKernel(const DTYPE* in, DTYPE* out, int64_t numElements, int64_t numDims,
                                                  const TensorDim* inputStrides, const TensorDim* encodingStrides,
                                                  const DTYPE* encodingMin, const DTYPE* encodingMax,
                                                  const DTYPE* encodingDelta, const DTYPE* encodingOffset)
{
    CUDA_KERNEL_LOOP(i, numElements)
    {
        int encodingIdx = 0;
        int remainder   = i;
        for (auto dim = 0; dim < numDims; dim++)
        {
            int dimIdx = remainder / inputStrides[dim];
            remainder = remainder - dimIdx * inputStrides[dim];
            // encodingStrides will be 0 along broadcast dimensions
            encodingIdx += encodingStrides[dim] * dimIdx;
        }

        auto delta  = *(encodingDelta + encodingIdx);
        auto offset = *(encodingOffset + encodingIdx);
        auto min    = *(encodingMin + encodingIdx);
        auto max    = *(encodingMax + encodingIdx);

        quantizeToFxpDevice<DTYPE>(in + i, out + i, min, max, delta, offset, ROUND_NEAREST, i);
        dequantizeFromFxpDevice<DTYPE>(out + i, delta, offset);
    }
}


template <typename DTYPE>
void quantizeDequantizeGpu(const DTYPE* in, uint64_t cnt, const TfEncoding& encoding, DTYPE* out, RoundingMode rounding_mode,
                           void* stream)
{
    quantizeDequantizeKernel<DTYPE>
        <<<CUDA_NUM_BLOCKS(cnt), CUDA_NUM_THREADS, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            in, cnt, out, encoding.min, encoding.max, encoding.delta, encoding.offset, rounding_mode);
}


__global__ void quantizeDequantizeFp16Kernel(const float* in, uint64_t cnt, float* out)
{
    CUDA_KERNEL_LOOP(i, cnt)
    {
        *(out + i) = __half2float(__float2half(*(in + i)));
    }
}


void quantizeDequantizeFp16ForGPU(const float* in, uint64_t cnt, float* out, void* stream)
{
    quantizeDequantizeFp16Kernel<<<CUDA_NUM_BLOCKS(cnt), CUDA_NUM_THREADS, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        in, cnt, out);
}

__global__ void convertFloatToFp16Kernel(const float* in, uint64_t cnt, __half* out)
{
    CUDA_KERNEL_LOOP(i, cnt)
    {
        *(out + i) = __float2half(*(in + i));
    }
}


void convertFloatToFp16KernelForGPU(const float* in, uint64_t cnt, __half* out, void* stream)
{
    convertFloatToFp16Kernel<<<CUDA_NUM_BLOCKS(cnt), CUDA_NUM_THREADS, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        in, cnt, out);
}


__global__ void convertFp16ToFloatKernel(const __half* in, uint64_t cnt, float* out)
{
    CUDA_KERNEL_LOOP(i, cnt)
    {
        *(out + i) = __half2float(*(in + i));
    }
}


void convertFp16ToFloatKernelForGPU(const __half* in, uint64_t cnt, float* out, void* stream)
{
    convertFp16ToFloatKernel<<<CUDA_NUM_BLOCKS(cnt), CUDA_NUM_THREADS, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        in, cnt, out);
}



template <typename DTYPE>
void quantizeToFxpGpu(const DTYPE* in, uint64_t cnt, const TfEncoding& encoding,
                      DTYPE* out, RoundingMode rounding_mode, bool shiftToSigned)
{
    unsigned int shift = 0;
    if (shiftToSigned) {
        shift = pow(2, encoding.bw - 1);
    }
    quantizeToFxpKernel<DTYPE><<<CUDA_NUM_BLOCKS(cnt), CUDA_NUM_THREADS>>>(
            in, cnt, out, encoding.min, encoding.max, encoding.delta,
            encoding.offset, rounding_mode, shift);
}

template <typename DTYPE>
void quantizeDequantizePerChannelGpu(const DTYPE* in, int numChannel, int numElement, int numElementPerChannel,
                                     DTYPE* out, DTYPE* encodingMin, DTYPE* encodingMax, DTYPE* encodingDelta,
                                     DTYPE* encodingOffset, RoundingMode roundingMode, void* stream)
{
    quantizeDequantizePerChannelKernel<DTYPE>
        <<<CUDA_NUM_BLOCKS(numElement), CUDA_NUM_THREADS, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            in, numChannel, numElement, numElementPerChannel, out, encodingMin, encodingMax, encodingDelta,
            encodingOffset, roundingMode);
}

template <typename DTYPE>
void quantizeDequantizeBroadcastGpu(const DTYPE* in, DTYPE* out, const Encodings& encodings,
                                    int64_t numElements, const TensorDims& inputStrides,
                                    const TensorDims& encodingStrides, void* stream)
{
    int64_t numEncodings = encodings.size();
    int64_t numDims      = inputStrides.size();
    std::vector<DTYPE> encVec(4 * numEncodings);

    for (int i = 0; i < numEncodings; i++)
    {
        encVec[i]                    = encodings[i].min;
        encVec[numEncodings + i]     = encodings[i].max;
        encVec[2 * numEncodings + i] = encodings[i].delta;
        encVec[3 * numEncodings + i] = encodings[i].offset;
    }

    // Allocate device memory for strides and encodings
    TensorDim* stridesDevice;
    DTYPE* encodingVectorDevice;
    cudaMalloc((void**) &stridesDevice, 2 * numDims * sizeof(TensorDim));
    cudaMalloc((void**) &encodingVectorDevice, 4 * numEncodings * sizeof(DTYPE));

    // Send encoding information to device
    cudaMemcpyAsync(encodingVectorDevice, encVec.data(), 4 * numEncodings * sizeof(DTYPE), cudaMemcpyHostToDevice,
                    static_cast<cudaStream_t>(stream));

    // Send stride information to device
    TensorDim* strideBuffer[2 * numDims];
    memcpy(strideBuffer, inputStrides.data(), numDims * sizeof(TensorDim));
    memcpy(strideBuffer + numDims, encodingStrides.data(), numDims * sizeof(TensorDim));
    cudaMemcpyAsync(stridesDevice, strideBuffer, 2 * numDims * sizeof(TensorDim), cudaMemcpyHostToDevice,
                    static_cast<cudaStream_t>(stream));

    DTYPE* encodingMin    = encodingVectorDevice;
    DTYPE* encodingMax    = encodingVectorDevice + numEncodings;
    DTYPE* encodingDelta  = encodingVectorDevice + 2 * numEncodings;
    DTYPE* encodingOffset = encodingVectorDevice + 3 * numEncodings;

    TensorDim* dTensorStrides   = stridesDevice;
    TensorDim* dEncodingStrides = stridesDevice + numDims;

    quantizeDequantizeBroadcastKernel<DTYPE>
        <<<CUDA_NUM_BLOCKS(numElements), CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
            in, out, numElements, numDims, dTensorStrides, dEncodingStrides, encodingMin, encodingMax, encodingDelta,
            encodingOffset);

    // Free device memory for strides and encodings
    cudaFree(stridesDevice);
    cudaFree(encodingVectorDevice);
}


// Explicit instantiations
template void quantizeDequantizeGpu(const double* in, uint64_t cnt, const TfEncoding& encoding, double* out,
                                    RoundingMode rounding_mode, void* stream);

template void quantizeDequantizeGpu(const float* in, uint64_t cnt, const TfEncoding& encoding, float* out,
                                    RoundingMode rounding_mode, void* stream);

template void quantizeToFxpGpu(const double* in, uint64_t cnt, const TfEncoding& encoding, double* out,
                               RoundingMode rounding_mode, bool shiftToSigned);


template void quantizeToFxpGpu(const float* in, uint64_t cnt, const TfEncoding& encoding, float* out,
                               RoundingMode rounding_mode, bool shiftToSigned);

template void quantizeDequantizePerChannelGpu(const float* in, int numChannel, int numElement, int numElementPerChannel,
                                              float* out, float* encodingMin, float* encodingMax, float* encodingDelta,
                                              float* encodingOffset, RoundingMode roundingMode, void* stream);

template void quantizeDequantizePerChannelGpu(const double* in, int numChannel, int numElement,
                                              int numElementPerChannel, double* out, double* encodingMin,
                                              double* encodingMax, double* encodingDelta, double* encodingOffset,
                                              RoundingMode roundingMode, void* stream);

template void quantizeDequantizeBroadcastGpu(const float* in, float* out, const Encodings& encodings,
                                             int64_t numElements, const TensorDims& inputStrides,
                                             const TensorDims& encodingStrides, void* stream);

}   // End of namespace DlQuantization

// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef UTIL_TRIM_FUNCTIONS_HPP_
#define UTIL_TRIM_FUNCTIONS_HPP_

#include <cstdint>
#include <vector>

#include "DlQuantization/Quantization.hpp"

#ifdef GPU_QUANTIZATION_ENABLED
#include <cuda_fp16.h>
#endif


namespace DlQuantization
{
inline double randUniformCpu();

template <typename DTYPE>
void quantizeDequantize(const DTYPE* in, uint64_t cnt, const TfEncoding& encoding, DTYPE* out, ComputationMode mode_cpu_gpu,
                        RoundingMode rounding_mode, void* stream);


void quantizeDequantizeFp16ForGPU(const float* in, uint64_t cnt, float* out, void* stream);

#ifdef ONNX_CUDA
void convertFloatToFp16KernelForGPU(const float* in, uint64_t cnt, __half* out, void* stream);

void convertFp16ToFloatKernelForGPU(const __half* in, uint64_t cnt, float* out, void* stream);
#endif

template <typename DTYPE>
void quantizeToFxp(const DTYPE* in, uint64_t cnt, const TfEncoding& encoding, DTYPE* out, ComputationMode mode_cpu_gpu,
                   RoundingMode rounding_mode, bool shiftToSigned);

template <typename DTYPE>
void quantizeToFxpPacked(const DTYPE* in, uint64_t cnt, const TfEncoding& encoding, uint8_t* out, size_t out_size,
                         ComputationMode mode_cpu_gpu, RoundingMode rounding_mode, bool shiftToSigned);

template <typename DTYPE>
void dequantizeFromPackedFxp(const uint8_t* input, uint64_t cnt, const TfEncoding& encoding, DTYPE* output,
                             ComputationMode mode_cpu_gpu, bool shiftToSigned);

template <typename DTYPE>
void quantizeDequantizeCpu(const DTYPE* in, uint64_t cnt, const TfEncoding& encoding, DTYPE* out,
                           RoundingMode rounding_mode);

template <typename DTYPE>
void quantizeToFxpCpu(const DTYPE* in, uint64_t cnt, const TfEncoding& encoding, DTYPE* out, RoundingMode rounding_mode,
                      bool shiftToSigned);

template <typename DTYPE>
void quantizeToFxpPackedCpu(const DTYPE* in, uint64_t cnt, const TfEncoding& encoding, DTYPE* out, size_t out_size,
                            RoundingMode rounding_mode, bool shiftToSigned);

// Multi-threading implementation
template <typename DTYPE>
void dequantizeFromPackedFxpCpuMt(const uint8_t* input, uint64_t cnt, const TfEncoding& encoding, DTYPE* output,
                                  bool shiftToSigned);

template <typename DTYPE>
void dequantizeFromPackedFxpCpu(const uint8_t* input, uint64_t cnt, const TfEncoding& encoding, DTYPE* output,
                                bool shiftToSigned);

double computeDelta(double encodingMin, double encodingMax, double numSteps);
double computeOffset(double encodingMin, double delta);

template <typename DTYPE>
void quantizeDequantizePerChannel(const DTYPE* in, int numChannel, int numElement, int numElementPerChannel, DTYPE* out,
                                  DTYPE* encodingMin, DTYPE* encodingMax, DTYPE* encodingDelta, DTYPE* encodingOffset,
                                  ComputationMode modeCpuGpu, RoundingMode roundingMode, void* stream);


template <typename DTYPE>
void quantizeDequantizeBroadcastCpu(const DTYPE* in, DTYPE* out, const Encodings& encodings,
                                    int64_t numElement, const TensorDims& inputStrides,
                                    const TensorDims& encodingStrides);


// GPU implementations ...
#ifdef GPU_QUANTIZATION_ENABLED

template <typename DTYPE>
void quantizeToFxpGpu(const DTYPE* in, uint64_t cnt, const TfEncoding& encoding, DTYPE* out, RoundingMode rounding_mode,
                      bool shiftToSigned);

template <typename DTYPE>
void quantizeDequantizeGpu(const DTYPE* in, uint64_t cnt, const TfEncoding& encoding, DTYPE* out, RoundingMode rounding_mode,
                           void* stream);

template <typename DTYPE>
void quantizeDequantizePerChannelGpu(const DTYPE* in, int numChannel, int numElement, int numElementPerChannel,
                                     DTYPE* out, DTYPE* encodingMin, DTYPE* encodingMax, DTYPE* encodingDelta,
                                     DTYPE* encodingOffset, RoundingMode roundingMode, void* stream);

template <typename DTYPE>
void quantizeDequantizeBroadcastGpu(const DTYPE* in, DTYPE* out, const Encodings& encodings,
                                    int64_t numElements, const TensorDims& inputStrides,
                                    const TensorDims& encodingStrides, void* stream);

#endif   // GPU_QUANTIZATION_ENABLED

}   // End of namespace DlQuantization

#endif   // UTIL_TRIM_FUNCTIONS_HPP_

// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef AIMET_MAIN_AIMETOPUTILS_H
#define AIMET_MAIN_AIMETOPUTILS_H

#include <numeric>
#include "DlQuantization/Fp16Quantization.hpp"
#include "DlQuantization/IForLoopRunner.h"
#include "DlQuantization/Quantization.hpp"
#include "DlQuantization/TensorQuantizer.h"
#include "DlQuantization/TensorQuantizerOpFacade.h"
#include "Eigen/Core"
#include "Eigen/src/Core/arch/Default/Half.h"
#include "OnnxOpUtils.h"

#ifdef ONNX_CUDA
#include <cuda_fp16.h>
#endif

#include "trim_functions.hpp"

#include <cstdint>
#include <stdexcept>
#include <type_traits>


template <typename T>
void copyInputTensorsToOutputTensors(const T* inTensor, size_t count, T* outTensor, bool useCuda, void* stream);

void quantizeDequantizeFp16Cpu(const float* in, uint64_t cnt, float* out);

/**
 * @brief Convert fp16 tensor data to fp32 into a caller-provided output buffer.
 */
inline void convertToFloat(const Eigen::half* inTensor, int64_t numElements, float* outBuffer, bool useCuda,
                           void* stream)
{
    if (useCuda)
    {
#ifdef ONNX_CUDA
        DlQuantization::convertFp16ToFloatKernelForGPU(reinterpret_cast<const __half*>(inTensor), numElements,
                                                       outBuffer, stream);
#else
        throw std::runtime_error("Not compiled for GPU mode.");
#endif
    }
    else
    {
        for (int64_t i = 0; i < numElements; i++)
            outBuffer[i] = static_cast<float>(inTensor[i]);
    }
}


template <typename T>
void modeSpecificActionBroadcastInt(const T* inTensor, T* outTensor, const std::vector<int64_t> inputShape,
                                    DlQuantization::BlockTensorQuantizer* tensorQuantizer,
                                    const DlQuantization::TensorQuantizerOpMode opMode,
                                    const bool useSymmetricEncoding, DlQuantization::IAllocator* allocator,
                                    bool useCuda, void* stream,
                                    DlQuantization::IForLoopRunner* runner = nullptr)
{
    int64_t numElements = std::accumulate(inputShape.begin(), inputShape.end(), int64_t(1), std::multiplies<int64_t>());

    switch (opMode)
    {
    case DlQuantization::TensorQuantizerOpMode::oneShotQuantizeDequantize:
    {
        tensorQuantizer->resetEncodingStats();
        if constexpr (std::is_same_v<T, float>)
        {
            tensorQuantizer->updateStats(inTensor, inputShape, useCuda, allocator, stream);
        }
        else
        {
            float* fp32Buf = static_cast<float*>(allocator->allocateRaw(numElements * sizeof(float)));
            convertToFloat(inTensor, numElements, fp32Buf, useCuda, stream);
            tensorQuantizer->updateStats(fp32Buf, inputShape, useCuda, allocator, stream);
            allocator->deleteRaw(fp32Buf);
        }
        auto computedEncodings = tensorQuantizer->computeEncodings(useSymmetricEncoding);
        tensorQuantizer->setEncodings(computedEncodings);
        // Continue to quantizeDequantize
    }
    case DlQuantization::TensorQuantizerOpMode::quantizeDequantize:
    {
        tensorQuantizer->quantizeDequantize(inTensor, outTensor, inputShape, useCuda, stream, runner);
        break;
    }
    case DlQuantization::TensorQuantizerOpMode::updateStats:
    {
        if constexpr (std::is_same_v<T, float>)
        {
            tensorQuantizer->updateStats(inTensor, inputShape, useCuda, allocator, stream);
        }
        else
        {
            float* fp32Buf = static_cast<float*>(allocator->allocateRaw(numElements * sizeof(float)));
            convertToFloat(inTensor, numElements, fp32Buf, useCuda, stream);
            tensorQuantizer->updateStats(fp32Buf, inputShape, useCuda, allocator, stream);
            allocator->deleteRaw(fp32Buf);
        }
        // Continue to passThrough
    }
    case DlQuantization::TensorQuantizerOpMode::passThrough:
    {
        copyInputTensorsToOutputTensors(inTensor, numElements, outTensor, useCuda, stream);
        break;
    }
    default:
    {
        throw std::exception();
    }
    }
}

template <typename T>
void modeSpecificActionFloat(const T* inTensor, size_t count, T* outTensor,
                             const DlQuantization::TensorQuantizerOpMode opMode, DlQuantization::IAllocator* allocator,
                             bool useCuda, void* stream)
{
    switch (opMode)
    {
    case DlQuantization::TensorQuantizerOpMode::oneShotQuantizeDequantize:
    case DlQuantization::TensorQuantizerOpMode::quantizeDequantize:
    {
        if (useCuda)
        {
            DlQuantization::quantizeDequantizeFp16Gpu(inTensor, count, outTensor, stream);
        }
        else
            quantizeDequantizeFp16Cpu(inTensor, count, outTensor);
        break;
    }
    case DlQuantization::TensorQuantizerOpMode::updateStats:
    case DlQuantization::TensorQuantizerOpMode::passThrough:
    {
        copyInputTensorsToOutputTensors(inTensor, count, outTensor, useCuda, stream);
        break;
    }
    default:
    {
        throw std::exception();
    }
    }
}

#endif   // AIMET_MAIN_AIMETOPUTILS_H

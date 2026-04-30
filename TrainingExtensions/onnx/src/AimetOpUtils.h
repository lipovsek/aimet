// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef AIMET_MAIN_AIMETOPUTILS_H
#define AIMET_MAIN_AIMETOPUTILS_H

#include "DlQuantization/Fp16Quantization.hpp"
#include "DlQuantization/IForLoopRunner.h"
#include "DlQuantization/Quantization.hpp"
#include "DlQuantization/TensorQuantizer.h"
#include "Eigen/Core"
#include "OnnxOpUtils.h"
#include <numeric>

#ifdef ONNX_CUDA
#include <cuda_fp16.h>
#endif

#include "trim_functions.hpp"

#include <cstdint>
#include <stdexcept>


template <typename T>
void copyInputTensorsToOutputTensors(const T* inTensor, size_t count, T* outTensor, bool useCuda, void* stream);

void quantizeDequantizeFp16Cpu(const float* in, uint64_t cnt, float* out);


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
        tensorQuantizer->updateStats(inTensor, inputShape, useCuda, allocator, stream);
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
        tensorQuantizer->updateStats(inTensor, inputShape, useCuda, allocator, stream);
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

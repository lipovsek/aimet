// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "AimetOpUtils.h"

template <typename T>
void copyInputTensorsToOutputTensors(const T* inTensor, size_t count, T* outTensor, bool useCuda, void* stream)
{
    // copy input_tensor to output_tensor
    if (useCuda)
    {
#ifdef ONNX_CUDA
        DlQuantization::copyTensorsCuda(outTensor, inTensor, count, stream);
#else
        throw std::runtime_error("Not compiled for GPU mode.");
#endif
    }
    else
    {
        std::copy(inTensor, inTensor + count, outTensor);
    }
}


void quantizeDequantizeFp16Cpu(const float* in, uint64_t cnt, float* out)
{
    for (uint64_t i = 0; i < cnt; ++i)
    {
        *(out + i) = Eigen::half_impl::half_to_float(Eigen::half_impl::float_to_half_rtne(*(in + i)));
    }
}


template void copyInputTensorsToOutputTensors(const float* inTensor, size_t count, float* outTensor, bool useCuda,
                                              void* stream);
template void copyInputTensorsToOutputTensors(const Eigen::half* inTensor, size_t count, Eigen::half* outTensor,
                                              bool useCuda, void* stream);
template void copyInputTensorsToOutputTensors(const Eigen::bfloat16* inTensor, size_t count, Eigen::bfloat16* outTensor,
                                              bool useCuda, void* stream);

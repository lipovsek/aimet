// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "DlQuantization/Fp16Quantization.hpp"
#include "trim_functions.hpp"
#include <stdexcept>
#include <cstdint>

namespace DlQuantization
{

void quantizeDequantizeFp16Gpu(const float* in, uint64_t cnt, float* out, void* stream)
{
#ifdef GPU_QUANTIZATION_ENABLED
    quantizeDequantizeFp16ForGPU(in, cnt, out, stream);
#else
    throw std::runtime_error("Not compiled for GPU mode.");
#endif
}
}   // namespace DlQuantization
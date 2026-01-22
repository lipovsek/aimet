// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef FP16_QUANTIZATION_HPP
#define FP16_QUANTIZATION_HPP

#include "DlQuantization/Quantization.hpp"
#include <cstdint>

namespace DlQuantization
{
/*
 * This function can be used for quantization and dequantization of float tensors
 * @param in pointer to the input tensor
 * @param cnt total size of input tensor
 * @param out pointer to the output tensor
 */
void quantizeDequantizeFp16Gpu(const float* in, uint64_t cnt, float* out, void* stream = nullptr);

}   // namespace DlQuantization

#endif
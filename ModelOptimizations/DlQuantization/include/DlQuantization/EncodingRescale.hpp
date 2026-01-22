// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ENCODING_RESCALE_HPP_
#define ENCODING_RESCALE_HPP_
#include <cstddef>
#include <iostream>

#include "DlQuantization/Quantization.hpp"

namespace DlQuantization
{

/**
 * @brief Arguments used for simulating on-device convolution
 */
template<typename DTYPE>
struct ConvSpecArgs
{
    // delta of output encoding of convolution
    float out_encoding_delta;
    // offset of output encoding of convolution
    float out_encoding_offset;
    // delta of input encoding of convolution
    float input_scale;
    // quantization bitwidths
    uint8_t bw;
    // weight scales of weight encodings of convolution, if the quantization scheme is perchannel, the length of
    // weight_scale is equal to the count, if the quantization scheme is pertensor, the length of weight_scale is 1.
    std::vector<DTYPE> weight_scale;
};

/**
 * @brief returns the exponent and mantissa of x, as a n-bit number
 *
 * Constraint: iexpo must be in range -126..127
 * Input must not be negative, inf, nan, zero, or denormal.
 */
inline std::pair<int32_t, int32_t> getScaleFactor(float x, int mbits)
{
    int32_t inval = *reinterpret_cast<int *>(&x);
    int MBITS = mbits;
    int32_t mask = (1 << MBITS) - 1;
    inval = (inval + (1 << (24 - MBITS - 1))) >> (24 - MBITS);
    int32_t m = ((inval & mask) | (1 << (MBITS - 1)));
    int32_t e = int32_t((inval >> (MBITS - 1)) & 0xFF) - 126;
    if (e < -23)
        e = -9999;
    return {e, m};
}

inline void setCpuGpuMode(bool use_cuda, DlQuantization::ComputationMode& cpu_gpu_mode)
{
    if (use_cuda)
        cpu_gpu_mode = DlQuantization::ComputationMode::COMP_MODE_GPU;
    else
        cpu_gpu_mode = DlQuantization::ComputationMode::COMP_MODE_CPU;
}

template <typename DTYPE>
void getRescaledOutputAndBias(const DTYPE* bias_in, const int count, ConvSpecArgs<DTYPE> &conv_args,
                       DTYPE* bias_out, DTYPE* scaling_params, bool use_cuda, bool withOffsetWrap);


} // end of namespace DlQuantization
#endif // end of ENCODING_RESCALE_HPP_

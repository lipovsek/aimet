// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SPEC_FUNCTIONS_HPP_
#define SPEC_FUNCTIONS_HPP_

#include <iostream>
#include <vector>

#include "DlQuantization/Quantization.hpp"
#include "DlQuantization/EncodingRescale.hpp"

namespace DlQuantization
{

template <typename DTYPE>
void getRescaledOutputAndBiasImpl(const DTYPE* bias_in, const int count, ConvSpecArgs<DTYPE> &conv_args, DTYPE* bias_out,
                           DTYPE* scaling_params, ComputationMode cpu_gpu_mode, bool withOffsetWrap);

template <typename DTYPE>
void getRescaledOutputAndBiasImplCpu(const DTYPE* bias_in, const int count, ConvSpecArgs<DTYPE> &conv_args, DTYPE* bias_out,
                              DTYPE* scaling_params, bool withOffsetWrap);

// GPU implementations ...
#ifdef GPU_QUANTIZATION_ENABLED
template <typename DTYPE>
void getRescaledOutputAndBiasImplGpu(const DTYPE* bias_in, const int count, ConvSpecArgs<DTYPE> &hw_conv_args, DTYPE* bias_out,
                              DTYPE* scaling_params, bool withOffsetWrap);

#endif //End of GPU_QUANTIZATION_ENABLED

} // end of namespace DlQuantization

#endif // SPEC_FUNCTIONS_HPP_

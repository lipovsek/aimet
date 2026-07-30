// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef QUANTIZATION_UTILS_H_
#define QUANTIZATION_UTILS_H_

#include <math.h>
#include <stdint.h>

#include "DlQuantization/Quantization.hpp"
#include "DlQuantization/QuantizationType.hpp"

namespace DlQuantization
{

static constexpr double EPSILON = 1e-5;

TfEncoding getComputedEncodings(uint8_t bw, double min, double max, bool useSymmetricEncodings, bool useStrictSymmetric,
                                bool useUnsignedSymmetric, double zeroPointShift);

// ensures min - max is not too close, by checking that max - min > epsilon
void gateMinMax(double& encodingMin, double& encodingMax);

void computeMinMaxRangeFromDeltaOffset(uint8_t bw, TfEncoding& encoding, bool useSymmetricEncodings,
                                       bool useUnsignedSymmetric, bool useStrictSymmetric);

void computeDeltaAndOffsetFromMinMax(uint8_t bw, TfEncoding& encoding, bool useSymmetricEncodings,
                                     bool useUnsignedSymmetric, bool useStrictSymmetric);

double computeFp8Scale(double amax, const FloatQuantizationSpec& fp8Spec);

// Function to slice a tensor along an axis, allocate and populate output buffers. Output shape will be the same for
// each slice.
template <typename DTYPE>
void slice(const DTYPE* data, const std::vector<uint32_t>& inputShape, int32_t axis,
           std::vector<std::vector<DTYPE>>& output, std::vector<uint32_t>& splitShape);

// Function to concatenate from slice along an axis. Should be the same shape as the original input shape to slice.
template <typename DTYPE>
void concat(const std::vector<std::vector<DTYPE>>& data, const std::vector<uint32_t>& inputShape, int32_t axis,
            DTYPE* output, std::vector<uint32_t>& outputShape);

template <typename DTYPE>
std::tuple<DTYPE, std::vector<int>> quantizeSingleChannelPerBlockScale(std::vector<DTYPE>& scale, int compressed_bw,
                                                                       int decompressed_bw);

}   // End of namespace DlQuantization

#endif   // QUANTIZATION_UTILS_H_
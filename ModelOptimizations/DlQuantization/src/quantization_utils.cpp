// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "quantization_utils.hpp"
#include "DlQuantization/Quantization.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "math_functions.hpp"
#ifdef GPU_QUANTIZATION_ENABLED
#include "cuda_util.hpp"
#endif


namespace DlQuantization
{
using namespace std;

TfEncoding getComputedEncodings(uint8_t bw, double min, double max, bool useSymmetricEncodings, bool useStrictSymmetric,
                                bool useUnsignedSymmetric, double zeroPointShift)
{
    TfEncoding encoding;

    double numSteps = pow(2, bw) - 1;
    if (useSymmetricEncodings && useStrictSymmetric)
    {
        numSteps -= 1;
    }
    encoding.bw = bw;

    // Use 0.99 of numeric limits to avoid floating point overflow
    double FLOAT32_MIN = std::numeric_limits<float>::lowest() * 0.99;
    double FLOAT32_MAX = std::numeric_limits<float>::max() * 0.99;

    min = std::clamp(min, FLOAT32_MIN, 0.0);
    max = std::clamp(max, 0.0, FLOAT32_MAX);

    // Unsigned symmetric is not applicable if min < 0
    useUnsignedSymmetric = useUnsignedSymmetric && (min >= 0.0);

    if (zeroPointShift != 0.0)
    {
        if (!useSymmetricEncodings)
        {
            throw std::runtime_error("zero_point_shift is only supported when useSymmetricEncodings is set to true");
        }
        if (useUnsignedSymmetric)
        {
            throw std::runtime_error("zero_point_shift is only supported when useUnsignedSymmetric is set to false");
        }
        if (useStrictSymmetric)
        {
            throw std::runtime_error("zero_point_shift is only supported when useStrictSymmetric is set to false");
        }
    }

    double fp32_eps = std::numeric_limits<float>::epsilon();
    double min_scale = std::min(0.01 / numSteps, fp32_eps);

    if (useSymmetricEncodings && !useUnsignedSymmetric)
    {
        double numPositiveSteps = std::floor(numSteps / 2) + zeroPointShift;
        double numNegativeSteps = numSteps - numPositiveSteps;

        // For low-bit quantization, using math.floor to compute num_pos_steps can result in a wasted bin on the negative side given a symmetrically distributed weight.
        // Using math.ceil instead trades off having some clipping error in return for being able to use all bins.
        // Checking for numSteps = 3 to account for strict symmetric grid.
        if (numSteps == 3 && (zeroPointShift == 0.0))
        {
            numPositiveSteps = numNegativeSteps;
        }
        encoding.delta  = std::max({max / numPositiveSteps, -min / numNegativeSteps, min_scale});
        encoding.offset = -numNegativeSteps;
    }
    else
    {
        encoding.delta = std::max((max - min) / numSteps, min_scale);
        encoding.offset = std::clamp(round(min / encoding.delta), -numSteps, 0.0);
    }

    // Clamp delta such that dequantized tensor always lie in range [FLOAT32_MIN, FLOAT32_MAX]
    // In other words, FLOAT32_MIN <= encoding.delta * encoding.offset
    //                              < encoding.delta * (encoding.offset + num_steps)
    //                              < FLOAT32_MAX
    if (encoding.offset < 0)
        encoding.delta = std::min(encoding.delta, FLOAT32_MIN / encoding.offset);

    if (encoding.offset + numSteps > 0)
        encoding.delta = std::min(encoding.delta, FLOAT32_MAX / (encoding.offset + numSteps));

    encoding.min = encoding.delta * encoding.offset;
    encoding.max = encoding.delta * (encoding.offset + numSteps);

    return encoding;
}

void gateMinMax(double& encodingMin, double& encodingMax)
{
    // Additional handling to retain zero in range
    // encodingMin can be at maximum 0.0
    encodingMin = std::min(encodingMin, 0.0);

    // encodingMax can be at minimum 0.0
    encodingMax = std::max(encodingMax, 0.0);

    // handle case where encodingMin == encodingMax
    encodingMax = std::max(encodingMax, encodingMin + EPSILON);
}

void computeMinMaxRangeFromDeltaOffset(uint8_t bw, TfEncoding& encoding, bool useSymmetricEncodings,
                                       bool useUnsignedSymmetric, bool useStrictSymmetric)
{
    auto origEncoding = encoding;
    if (encoding.bw == 0)
    {
        throw std::invalid_argument("Encodings must have a valid non-zero bitwidth");
    }

    if (origEncoding.min != 0 && origEncoding.max != 0)
    {
        throw std::invalid_argument("Encoding min and max must be zero to use this function");
    }

    if (origEncoding.delta == 0 && origEncoding.offset > 0)
    {
        throw std::invalid_argument("Encoding must have a valid non-zero delta/offset if min and max are zero");
    }


    auto numSteps = pow(2, bw) - 1;

    if (useSymmetricEncodings && useStrictSymmetric)
    {
        numSteps -= 1;
    }

    // set up min and max values
    // Note delta and offset are assumed to allow for zero to be quantizable
    encoding.min = encoding.offset * encoding.delta;

    // factor in symmetry into max calculation
    if (useSymmetricEncodings && ((encoding.min < 0.0) || (!useUnsignedSymmetric)))
    {
        auto numPositiveSteps = std::floor(numSteps / 2);
        encoding.max          = encoding.delta * numPositiveSteps;
    }
    else
    {
        encoding.max = encoding.delta * numSteps + encoding.min;
    }

    // check that min, max are not too close
    // It's unlikely but we can still gate if needed
    if ((encoding.max - encoding.min < EPSILON))
    {
        gateMinMax(encoding.min, encoding.max);
    }
}

void computeDeltaAndOffsetFromMinMax(uint8_t bw, TfEncoding& encoding, bool useSymmetricEncodings,
                                     bool useUnsignedSymmetric, bool useStrictSymmetric)
{
    auto origEncoding = encoding;
    if (encoding.bw == 0)
    {
        throw std::invalid_argument("Encodings must have a valid non-zero bitwidth");
    }

    if (origEncoding.delta != 0 && origEncoding.offset != 0)
    {
        throw std::invalid_argument("Encoding delta and offset must be zero to use this function");
    }

    // Compute delta and offset, which may also adjust min and max
    // Note min, max is retained
    encoding     = getComputedEncodings(bw, encoding.min, encoding.max, useSymmetricEncodings, useStrictSymmetric,
                                        useUnsignedSymmetric, 0.0);
    encoding.min = origEncoding.min;
    encoding.max = origEncoding.max;
}

double computeFp8Scale(double amax, const FloatQuantizationSpec& fp8Spec)
{
    if (!(amax > 0))
    {
        throw std::invalid_argument("FP8 amax must be positive");
    }

    return amax / fp8Spec.maxValue;
}

// Function to slice a tensor along an axis. Output shape will be the same for each slice.
template <typename DTYPE>
void slice(const DTYPE* input, const std::vector<uint32_t>& inputDim, int32_t axis,
           std::vector<std::vector<DTYPE>>& outputs, std::vector<uint32_t>& outputDim)
{
    // Account for negative axis
    uint32_t realAxis   = (axis >= 0) ? axis : inputDim.size() + axis;
    outputDim           = inputDim;
    outputDim[realAxis] = 1;

    // If input slice axis dimension size == 1, then it's already "sliced". Copy input->output as there is nothing to
    // slice
    uint32_t outputCnt = std::accumulate(outputDim.begin(), outputDim.end(), 1, std::multiplies<uint32_t>());
    if (inputDim[realAxis] == 1)
    {
        outputs.emplace_back(input, input + outputCnt);
        return;
    }

    // Add all the slices
    std::vector<uint32_t> slices;
    for (uint32_t i = 1; i < inputDim[realAxis]; ++i)
    {
        slices.push_back(i);
    }

    uint32_t sliceCnt = slices.size() + 1;

    // std::cout << "Slice axis: " << realAxis << std::endl;
    // std::cout << "# slices: " << sliceCnt << std::endl;

    outputs.resize(sliceCnt);
    for (uint32_t i = 0; i < outputs.size(); ++i)
    {
        outputs[i].resize(outputCnt);
    }

    // Compute input dim strides
    // For e.g. input dim = {6, 3, 4}
    // strides = {12, 4, 1}
    std::vector<uint32_t> inputDimStrides(inputDim.size());
    for (uint32_t i = inputDim.size(); i > 0; --i)
    {
        inputDimStrides[i - 1] = std::accumulate(inputDim.begin() + i, inputDim.end(), 1, std::multiplies<uint32_t>());
        // std::cout << "inputDimStrides[" << (i-1) << "]=" << inputDimStrides[i-1] << std::endl;
    }

    // Compute num of slice runs
    uint32_t numSliceRuns =
        std::accumulate(inputDim.begin(), inputDim.begin() + realAxis, 1, std::multiplies<uint32_t>());
    // std::cout << "No. of slice runs : " << numSliceRuns << std::endl;

    // Compute the distane to move during each run for a given slice.
    // It is the same for each slice.
    uint32_t sliceRunStride = (realAxis == 0) ? 0 : inputDimStrides[realAxis - 1];
    // std::cout << "Slice run stride : " <<  sliceRunStride << std::endl;

    typedef struct
    {
        uint32_t inputStartOffset;
        uint32_t size;
    } SliceInfo;

    std::vector<SliceInfo> sliceInfos;

    // Compute slice info for each slice, using slice points, axis and input dim
    for (uint32_t sliceIdx = 0; sliceIdx < slices.size() + 1; ++sliceIdx)
    {
        SliceInfo sinfo;

        if (sliceIdx == 0)   // First slice
        {
            sinfo.inputStartOffset = 0;
            sinfo.size             = slices[sliceIdx] * inputDimStrides[realAxis];
        }
        else if (sliceIdx == slices.size())   // Last slice
        {
            sinfo.inputStartOffset = inputDimStrides[realAxis] * slices[sliceIdx - 1];
            sinfo.size             = (inputDim[realAxis] - slices[sliceIdx - 1]) * inputDimStrides[realAxis];
        }
        else   // Middle slices
        {
            sinfo.inputStartOffset = inputDimStrides[realAxis] * slices[sliceIdx - 1];
            sinfo.size             = (slices[sliceIdx] - slices[sliceIdx - 1]) * inputDimStrides[realAxis];
        }
        sliceInfos.push_back(sinfo);
        // std::cout << "SliceInfo Idx " << sliceIdx << ", inputStartOffset: " << sinfo.inputStartOffset << ", size: "
        // << sinfo.size << std::endl;
    }

    for (uint32_t runIdx = 0; runIdx < numSliceRuns; ++runIdx)
    {
        for (uint32_t sliceIdx = 0; sliceIdx < sliceCnt; ++sliceIdx)
        {
            auto inPtr = input + sliceInfos[sliceIdx].inputStartOffset + runIdx * sliceRunStride;
            std::copy(inPtr, inPtr + sliceInfos[sliceIdx].size,
                      outputs[sliceIdx].data() + runIdx * sliceInfos[sliceIdx].size);
        }
    }
}

// Function to concatenate from slice along an axis. Output shape should be the same shape as the original input shape
// to slice.
template <typename DTYPE>
void concat(const std::vector<std::vector<DTYPE>>& inputs, const std::vector<uint32_t>& inputDim, int32_t axis,
            DTYPE* output, std::vector<uint32_t>& outputDim)
{
    uint32_t realAxis   = (axis >= 0) ? axis : inputDim.size() + axis;
    outputDim           = inputDim;
    outputDim[realAxis] = inputs.size();

    uint32_t a        = 0;
    uint32_t numUnits = 1;
    for (; a < realAxis; ++a)
    {
        numUnits = numUnits * inputDim[a];
    }

    uint32_t unitSize = 1;
    for (; a < inputDim.size(); ++a)
    {
        unitSize = unitSize * inputDim[a];
    }
    for (uint32_t i = 0; i < numUnits; ++i)
    {
        for (uint32_t u = 0; u < (uint32_t) inputs.size(); ++u)
        {
            const DTYPE* src = inputs[u].data() + unitSize * i;
            std::copy(src, src + unitSize, output);
            output += unitSize;
        }
    }
}


template <typename DTYPE>
std::tuple<DTYPE, std::vector<int>> quantizeSingleChannelPerBlockScale(std::vector<DTYPE>& scale, int compressed_bw,
                                                                       int decompressed_bw)
{
    typename std::vector<DTYPE>::iterator maxScale;
    maxScale = std::max_element(scale.begin(), scale.end());
    DTYPE perChannelScale = *maxScale / pow(2, decompressed_bw - compressed_bw);
    std::vector<int> perBlockIntScale;
    for (int i=0; i<scale.size(); i++)
    {
        perBlockIntScale.push_back(std::max(1, static_cast<int>(std::round(scale[i] / perChannelScale))));
    }
    return std::tuple<DTYPE, std::vector<int>>(perChannelScale, perBlockIntScale);
}

template void slice(const float* input, const std::vector<uint32_t>& inputDim, int32_t axis,
                    std::vector<std::vector<float>>& outputs, std::vector<uint32_t>& outputDim);
template void slice(const double* input, const std::vector<uint32_t>& inputDim, int32_t axis,
                    std::vector<std::vector<double>>& outputs, std::vector<uint32_t>& outputDim);
template void slice(const uint8_t* input, const std::vector<uint32_t>& inputDim, int32_t axis,
                    std::vector<std::vector<uint8_t>>& outputs, std::vector<uint32_t>& outputDim);

template void concat(const std::vector<std::vector<float>>& inputs, const std::vector<uint32_t>& inputDim, int32_t axis,
                     float* output, std::vector<uint32_t>& outputDim);
template void concat(const std::vector<std::vector<double>>& inputs, const std::vector<uint32_t>& inputDim,
                     int32_t axis, double* output, std::vector<uint32_t>& outputDim);
template void concat(const std::vector<std::vector<unsigned char>>& inputs, const std::vector<uint32_t>& inputDim,
                     int32_t axis, unsigned char* output, std::vector<uint32_t>& outputDim);

template std::tuple<float, std::vector<int>> quantizeSingleChannelPerBlockScale(std::vector<float>& scale,
                                                                                int compressed_bw,
                                                                                int decompressed_bw);
template std::tuple<double, std::vector<int>> quantizeSingleChannelPerBlockScale(std::vector<double>& scale,
                                                                                 int compressed_bw,
                                                                                 int decompressed_bw);

}   // End of namespace DlQuantization

// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>


#include "DlQuantization/Quantization.hpp"
#include "math_functions.hpp"
#include "quantization_utils.hpp"

#include "PercentileEncodingAnalyzer.h"

namespace DlQuantization
{
template <typename DTYPE>
std::vector<std::tuple<double, double>> PercentileEncodingAnalyzer<DTYPE>::getStatsHistogram() const
{
    // Return the collected histogram data.
    return getCollectedHistogram(this->_stats);
}

template <typename DTYPE>
void PercentileEncodingAnalyzer<DTYPE>::updateStats(const DTYPE* tensor, const size_t tensorSize,
                                                    ComputationMode tensorCpuGpuMode)
{
    updateStats(tensor, tensorSize, tensorCpuGpuMode, nullptr);
}

template <typename DTYPE>
void PercentileEncodingAnalyzer<DTYPE>::updateStats(const DTYPE* tensor, const size_t tensorSize,
                                                    ComputationMode tensorCpuGpuMode, IAllocator* allocator)
{
    this->_statsUpdated = true;

    // update pdf
    UpdatePdf(tensor, tensorSize, tensorCpuGpuMode, true, this->_stats, allocator);
}

template <typename DTYPE>
void PercentileEncodingAnalyzer<DTYPE>::resetStats()
{
    this->_stats.xLeft.clear();
    this->_stats.pdf.clear();
    this->_stats.iterations = 0;
}

template <typename DTYPE>
TfEncoding PercentileEncodingAnalyzer<DTYPE>::computeEncoding(uint8_t bw, bool useSymmetricEncodings,
                                                              bool useStrictSymmetric, bool useUnsignedSymmetric) const
{
    TfEncoding encoding = {0, 0, 0, 0, 0};
    DTYPE numSteps      = pow(2, bw) - 1;

    // For strict symmetric mode, we make even number of buckets
    if (useSymmetricEncodings && useStrictSymmetric)
    {
        numSteps -= 1;
    }

    if (this->_stats.xLeft.size() == 0)
    {
        if (this->_statsUpdated)
        {
            // Histogram has not been initialized yet, we have seen all zero data
            // We generate a valid encoding that covers float 0
            encoding.min    = -1;
            encoding.max    = 1;
            encoding.delta  = (encoding.max - encoding.min) / int(numSteps);
            encoding.offset = floor(encoding.min / encoding.delta);
            encoding.min    = encoding.offset * encoding.delta;
            encoding.max    = encoding.min + int(numSteps) * encoding.delta;
            encoding.bw     = bw;

            return encoding;
        }
        else
        {
            // Histogram has not been initialized yet because we have not seen any data
            // We return a zero encoding - which is a failure indicator
            return encoding;
        }
    }

    // Find the adjusted min and max
    DTYPE aMin, aMax;
    std::tie(aMin, aMax) = _computePercentileRange();

    // After Min and Max adjustment, the requirement that 0 be an exactly
    // representable value must be met.
    // There is a possibility that 0 may not be present in the Percentile
    // calibrated Min and Max range. Hence, extend the interval
    // [aMin, aMax] to ensure that it contains 0.
    aMin = std::min(aMin, DTYPE(0.f));
    aMax = std::max(aMax, DTYPE(0.f));

    assert(aMin <= aMax && "min must not be bigger than max");

    return getComputedEncodings(bw, aMin, aMax, useSymmetricEncodings, useStrictSymmetric, useUnsignedSymmetric,
                                0.0);
}

template <typename DTYPE>
std::tuple<DTYPE, DTYPE> PercentileEncodingAnalyzer<DTYPE>::_computePercentileRange() const
{
    // Number of histogram bins.
    const int numBins = PDF_SIZE;

    // Find the range of our collected stats
    DTYPE minVal, maxVal;
    std::tie(minVal, maxVal) = _findRangeOfAggregateStats();

    // Incase of percenitle value of 100 no need of calibration.
    if (this->_percentile == 100.0f)
    {
        return std::tuple<DTYPE, DTYPE>(minVal, maxVal);
    }

    const float histBinWidth = this->_stats.xLeft[1] - this->_stats.xLeft[0];
    DTYPE histMin            = this->_stats.xLeft[0];
    DTYPE histMax            = this->_stats.xLeft[PDF_SIZE - 1] + histBinWidth;

    DTYPE percentileMin = histMin;
    DTYPE percentileMax = histMax;

    // Copy the pdf collected and compute cdf.
    std::vector<double> cdf(this->_stats.pdf);
    for (auto i = 1; i < cdf.size(); i++)
    {
        cdf[i] += cdf[i - 1];
    }

    // Compute percentile calibration Min.
    float leftPercentile = 1 - this->_percentile / 100;
    for (auto i = 0; i < numBins; i++)
    {
        if (cdf[i] >= leftPercentile)
        {
            percentileMin = this->_stats.xLeft[i];
            break;
        }
    }

    // Compute percentile calibration Max.
    float rightPercentile = this->_percentile / 100;
    for (auto i = numBins - 1; i >= 0; i--)
    {
        // Ensure that percentileMax is not greater than the max value of the tensor.
        if (cdf[i] < rightPercentile && this->_stats.xLeft[i] < maxVal)
        {
            percentileMax = this->_stats.xLeft[i] + histBinWidth;
            break;
        }
    }

    // Enforce difference between percentileMin and percentileMax to be atleast
    // one bin width. This will ensure that percentileMin and percentileMax are
    // not equal in the scenarios where most of the tensor values are concentrated
    // in a single histogram bin. This will also ensure that percentileMin and
    // percentileMax are edges of the single bin where most of the tensor elements
    // are concentrated.
    if (percentileMin == percentileMax)
        percentileMax += histBinWidth;

    return std::tuple<DTYPE, DTYPE>(percentileMin, percentileMax);
}

template <typename DTYPE>
std::tuple<DTYPE, DTYPE> PercentileEncodingAnalyzer<DTYPE>::_findRangeOfAggregateStats() const
{
    return findOriginalRange<DTYPE>(this->_stats);
}

template <typename DTYPE>
void PercentileEncodingAnalyzer<DTYPE>::setPercentileValue(float percentile)
{
    this->_percentile = percentile;
}

template <typename DTYPE>
float PercentileEncodingAnalyzer<DTYPE>::getPercentileValue()
{
    return this->_percentile;
}

// Explicit instantiations
template class PercentileEncodingAnalyzer<double>;

template class PercentileEncodingAnalyzer<float>;

}   // namespace DlQuantization

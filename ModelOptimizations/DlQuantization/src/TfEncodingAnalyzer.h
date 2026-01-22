// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DL_QUANTIZATION_TF_ENCODING_ANALYZER_H
#define DL_QUANTIZATION_TF_ENCODING_ANALYZER_H

// This file contains code to analyze and calculate quantization encodings
// This code is specific for the TF quantization scheme

#include "math_functions.hpp"
#include <DlQuantization/IQuantizationEncodingAnalyzer.hpp>

namespace DlQuantization
{
template <typename DTYPE>
class TfEncodingAnalyzer : public IQuantizationEncodingAnalyzer<DTYPE>
{
public:
    void updateStats(const DTYPE* tensor, const size_t tensorSize, ComputationMode tensorCpuGpuMode) override;

    void updateStats(const DTYPE* tensor, const size_t tensorSize, ComputationMode tensorCpuGpuMode,
                     IAllocator* allocator) override;

    /**
     * @brief Reset running stats
     */
    void resetStats() override;

    /**
     * @brief Given a number distribution in CPU memory, compute the TensorFlow
     * encoding with the highest possible SQNR.
     *
     * To do so, we perform a grid search over different deltas and offsets.
     * This grid search optimizes the encoding to reduce the cost of quantization.
     * In this cost function, saturation errors are weighted higher than
     * quantization errors.
     */
    TfEncoding computeEncoding(uint8_t bw, bool useSymmetricEncodings, bool useStrictSymmetric,
                               bool useUnsignedSymmetric) const override;

    /**
     * @brief Returns a histogram that represents a PDF of tensor values seen by this encoding analyzer so far
     *
     * @return Histogram of statistics. The histogram returned is a vector of buckets. Each bucket is a tuple of
     * two values - the float value representing the left edge of the bucket and a PDF of the values in this bucket
     * relative to all the values seen across all buckets
     */
    std::vector<std::tuple<double, double>> getStatsHistogram() const override;

    // Minimum range of quantization
    static constexpr double MIN_RANGE = 0.01;


private:
    bool _statsUpdated = false;
    struct
    {
        double min = std::numeric_limits<double>::max();
        double max = -std::numeric_limits<double>::max();

    } _accumulatedStats;
};

}   // namespace DlQuantization

#endif   // DL_QUANTIZATION_TF_ENCODING_ANALYZER_H

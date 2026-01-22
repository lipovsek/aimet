// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DL_QUANTIZATION_ENTROPY_ENCODING_ANALYZER_H
#define DL_QUANTIZATION_ENTROPY_ENCODING_ANALYZER_H

#include "math_functions.hpp"
#include <DlQuantization/IQuantizationEncodingAnalyzer.hpp>

namespace DlQuantization
{
template <typename DTYPE>
class EntropyEncodingAnalyzer : public IQuantizationEncodingAnalyzer<DTYPE>
{
public:
    /**
     * Updates internal PDF stats given a tensor.
     * Intent is to keep a histogram of all the values that we have seen over multiple instances of a tensor
     * @param tensor Reference to a tensor
     * @param tensorSize Size of the tensor (number of elements)
     * @param tensorCpuGpuMode Indicates if the tensor is in CPU or GPU memory
     */
    void updateStats(const DTYPE* tensor, const size_t tensorSize, ComputationMode tensorCpuGpuMode) override;

    void updateStats(const DTYPE* tensor, const size_t tensorSize, ComputationMode tensorCpuGpuMode,
                     IAllocator* allocator) override;

    /**
     * @brief Reset running stats
     */
    void resetStats() override;

    /***
     * Compute the encodings using the collected histogram stats by minimizing the mean square
     * error due to quantization of the tensor.
     * @param bw Bitwidth to use for computing encodings
     * @param useSymmetricEncodings If true, compute symmetric encodings
     * @param useStrictSymmetric If true, compute symmetric encodings with even number of buckets
     * @param useUnsignedSymmetric If true, compute asymmetric encodings
     * @return Computed encoding
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

private:
    TensorProfilingParams _tensorProfilingParams;
    bool _statsUpdated = false;

    /**
     * Find range (min, max) of the aggregated stats
     * @return Tuple of min and max values
     */
    std::tuple<DTYPE, DTYPE> _findRangeOfAggregateStats() const;

    /**
     * @brief Adjust the min/max range of tensor values by minimizing KL divergence
     *
     * @param bw Bitwidth to use for computing encodings
     * @param useSymmetricEncodings If true, compute symmetric encodings
     * @param useStrictSymmetric If true, compute symmetric encodings with even number of buckets
     * @param useUnsignedSymmetric If true, compute asymmetric encodings
     * @return Tuple of <best min, best max> to be used for calculating the best delta and offset
     */
    std::tuple<DTYPE, DTYPE> _optimizeKL(uint8_t bw, bool useSymmetricEncodings, bool useStrictSymmetric,
                                         bool useUnsignedSymmetric) const;
};

}   // namespace DlQuantization

#endif   // DL_QUANTIZATION_ENTROPY_ENCODING_ANALYZER_H

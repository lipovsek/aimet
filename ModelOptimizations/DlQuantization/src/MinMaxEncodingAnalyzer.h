// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DL_QUANTIZATION_MIN_MAX_ENCODING_ANALYZER_H
#define DL_QUANTIZATION_MIN_MAX_ENCODING_ANALYZER_H

#include "ContiguousEncodingAnalyzer.h"


namespace DlQuantization
{

template <typename DTYPE>
class MinMaxEncodingAnalyzer : public ContiguousEncodingAnalyzerBase<DTYPE>
{
public:
    MinMaxEncodingAnalyzer(TensorDims shape);

    void resetStats() override;

    Encodings computeEncoding(uint8_t bw, bool useSymmetricEncodings, bool useStrictSymmetric,
                              bool useUnsignedSymmetric, double zeroPointShift = 0.0) const override;

    static constexpr DTYPE MIN_RANGE = 0.01;

    std::vector<std::vector<std::tuple<double, double>>> getStatsHistogram() const override;

protected:
    void updateStatsContiguous(const DTYPE* tensor, const TensorDims& shape, size_t blockSize,
                               ComputationMode tensorCpuGpuMode, IAllocator* allocator, void* stream) override;

private:
    std::vector<DTYPE> _minStats;
    std::vector<DTYPE> _maxStats;
};

}   // namespace DlQuantization

#endif   // DL_QUANTIZATION_MIN_MAX_ENCODING_ANALYZER_H

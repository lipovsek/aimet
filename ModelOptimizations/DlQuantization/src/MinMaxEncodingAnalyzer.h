// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DL_QUANTIZATION_MIN_MAX_ENCODING_ANALYZER_H
#define DL_QUANTIZATION_MIN_MAX_ENCODING_ANALYZER_H

#include "DlQuantization/IQuantizationEncodingAnalyzer.hpp"


namespace DlQuantization
{

template <typename DTYPE>
class MinMaxEncodingAnalyzer : public IBlockEncodingAnalyzer<DTYPE>
{
public:
    MinMaxEncodingAnalyzer(TensorDims shape);

    void updateStats(const DTYPE* tensor, const TensorDims& tensorShape, ComputationMode tensorCpuGpuMode,
                     IAllocator* allocator = nullptr, void* stream = nullptr) override;

    void updateStats(const Eigen::half* tensor, const TensorDims& tensorShape, ComputationMode tensorCpuGpuMode,
                     IAllocator* allocator = nullptr, void* stream = nullptr) override;

    void updateStats(const Eigen::bfloat16* tensor, const TensorDims& tensorShape, ComputationMode tensorCpuGpuMode,
                     IAllocator* allocator = nullptr, void* stream = nullptr) override;

    void resetStats() override;

    Encodings computeEncoding(uint8_t bw, bool useSymmetricEncodings, bool useStrictSymmetric,
                              bool useUnsignedSymmetric, double zeroPointShift = 0.0) const override;

    static constexpr DTYPE MIN_RANGE = 0.01;

    std::vector<std::tuple<double, double>> getObservedMinMax() const override;

    std::vector<std::vector<std::tuple<double, double>>> getStatsHistogram() const override;

private:
    template <typename T>
    void _updateStats(const T* tensor, const TensorDims& tensorShape, ComputationMode tensorCpuGpuMode,
                      IAllocator* allocator, void* stream);

    std::vector<DTYPE> _minStats;
    std::vector<DTYPE> _maxStats;
};

}   // namespace DlQuantization

#endif   // DL_QUANTIZATION_MIN_MAX_ENCODING_ANALYZER_H

// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DL_QUANTIZATION_WRAPPED_ENCODING_ANALYZER_H
#define DL_QUANTIZATION_WRAPPED_ENCODING_ANALYZER_H

#include <cstdint>
#include <memory>

#include "DlQuantization/IQuantizationEncodingAnalyzer.hpp"

namespace DlQuantization
{

/**
 * @class EncodingAnalyzerWrapper
 * @brief Wrapper over legacy encoding analyzers enabling blockwise behavior
 */
template <typename DTYPE>
class EncodingAnalyzerWrapper : public IBlockEncodingAnalyzer<DTYPE>
{
public:
    EncodingAnalyzerWrapper(TensorDims shape, QuantizationMode mode);

    void updateStats(const DTYPE* tensor, const TensorDims& tensorShape, ComputationMode tensorCpuGpuMode,
                     IAllocator* allocator = nullptr, void* stream = nullptr) override;

    void updateStats(const Eigen::half* tensor, const TensorDims& tensorShape, ComputationMode tensorCpuGpuMode,
                     IAllocator* allocator = nullptr, void* stream = nullptr) override;

    void resetStats() override;

    std::vector<TfEncoding> computeEncoding(uint8_t bw, bool useSymmetricEncodings, bool useStrictSymmetric,
                                            bool useUnsignedSymmetric, double zeroPointShift = 0.0) const override;

    std::vector<std::vector<std::tuple<double, double>>> getStatsHistogram() const override;

    void setPercentileValue(float percentile) override;

    float getPercentileValue() override;

private:
    void updateStatsContiguous(const DTYPE* tensor, const TensorDims& shape, size_t blockSize,
                               ComputationMode tensorCpuGpuMode, IAllocator* allocator = nullptr,
                               void* stream = nullptr);

    std::vector<std::unique_ptr<IQuantizationEncodingAnalyzer<DTYPE>>> _encodingAnalyzers;
};


}   // namespace DlQuantization


#endif   // DL_QUANTIZATION_WRAPPED_ENCODING_ANALYZER_H

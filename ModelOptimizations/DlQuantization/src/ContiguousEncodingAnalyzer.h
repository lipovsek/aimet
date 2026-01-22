// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef I_CONTIGUOUS_ENCODING_ANALYZER_HPP
#define I_CONTIGUOUS_ENCODING_ANALYZER_HPP

#include "DlQuantization/IQuantizationEncodingAnalyzer.hpp"


namespace DlQuantization
{

/**
 * @class ContiguousEncodingAnalyzerBase
 * @brief Base class for blockwise encoding analyzers which require data for each block
 *        to be contiguous in memory.
 */
template <typename DTYPE>
class ContiguousEncodingAnalyzerBase : public IBlockEncodingAnalyzer<DTYPE>
{
public:
    virtual ~ContiguousEncodingAnalyzerBase() = default;

    virtual void resetStats() = 0;

    void updateStats(const DTYPE* tensor, const TensorDims& tensorShape, ComputationMode tensorCpuGpuMode,
                     IAllocator* allocator = nullptr, void* stream = nullptr) override;

    virtual std::vector<TfEncoding> computeEncoding(uint8_t bw, bool useSymmetricEncodings, bool useStrictSymmetric,
                                                    bool useUnsignedSymmetric, double zeroPointShift) const = 0;

    virtual std::vector<std::vector<std::tuple<double, double>>> getStatsHistogram() const = 0;

    TensorDims getShape();

protected:
    virtual void updateStatsContiguous(const DTYPE* tensor, const TensorDims& shape, size_t blockSize,
                                       ComputationMode tensorCpuGpuMode, IAllocator* allocator = nullptr,
                                       void* stream = nullptr) = 0;

    TensorDims _shape;
};


}   // namespace DlQuantization


#endif   // I_CONTIGUOUS_ENCODING_ANALYZER_HPP

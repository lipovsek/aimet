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

    void updateStats(const DTYPE* tensor, const TensorDims& tensorShape, ComputationMode tensorCpuGpuMode,
                     IAllocator* allocator = nullptr, void* stream = nullptr) override;

    TensorDims getShape() override;

protected:
    virtual void updateStatsContiguous(const DTYPE* tensor, const TensorDims& shape, size_t blockSize,
                                       ComputationMode tensorCpuGpuMode, IAllocator* allocator = nullptr,
                                       void* stream = nullptr) = 0;

    TensorDims _shape;
};


}   // namespace DlQuantization


#endif   // I_CONTIGUOUS_ENCODING_ANALYZER_HPP

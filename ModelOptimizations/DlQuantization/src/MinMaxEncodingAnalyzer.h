//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2019 - 2022, Qualcomm Innovation Center, Inc. All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//  3. Neither the name of the copyright holder nor the names of its contributors
//     may be used to endorse or promote products derived from this software
//     without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//  SPDX-License-Identifier: BSD-3-Clause
//
//  @@-COPYRIGHT-END-@@
//
//==============================================================================

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

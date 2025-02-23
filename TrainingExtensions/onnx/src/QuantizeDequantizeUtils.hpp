//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

#ifndef AIMET_QUANTIZEDEQUANTIZEUTILS_HPP
#define AIMET_QUANTIZEDEQUANTIZEUTILS_HPP

#include "DlQuantization/TensorQuantizer.h"
#include "OnnxOpUtils.h"
#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef ONNX_CUDA
#include <cuda_runtime_api.h>
#endif


template <typename T>
void permuteTensorGPU(const T* inTensor, T* outTensor, int64_t numel, int64_t numDims, const int64_t* inputStrides,
                      const int64_t* outputStrides);

template <typename T>
void permuteTensorCPU(const T* inTensor, T* outTensor, int64_t numel, int64_t numDims, const int64_t* inputStrides,
                      const int64_t* outputStrides);

std::vector<int64_t> shapeToStrides(const std::vector<int64_t>& shape);

int64_t getNumElements(const std::vector<int64_t>& shape);


struct BroadcastShapeInfo
{
    BroadcastShapeInfo(const std::vector<int64_t>& inputShape, int channelAxis, int blockAxis, uint blockSize);

    bool hasContiguousBlocks() const;

    std::vector<int64_t> tensorShape;
    std::vector<int64_t> encodingShape;
    std::vector<int64_t> tensorStrides;
    std::vector<int64_t> encodingStrides;
    int64_t numElements;
    int64_t numEncodings;
    int64_t numDims;
};

// Permutes the input data so each entire encoding block is contiguous in memory
template <typename T>
void copyToContiguousBlockLayout(const T* inTensor, T* outTensor, const BroadcastShapeInfo& shapeInfo, bool useCuda);


#endif   // AIMET_QUANTIZEDEQUANTIZEUTILS_HPP

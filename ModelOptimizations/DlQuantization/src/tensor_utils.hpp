// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DL_QUANTIZATION_TENSOR_UTILS_H
#define DL_QUANTIZATION_TENSOR_UTILS_H

#include <DlQuantization/Quantization.hpp>

namespace DlQuantization
{


std::tuple<TensorDims, TensorDims> getBroadcastableShapes(const TensorDims& tensorShape,
                                                          const TensorDims& encodingShape);

size_t getNumel(const TensorDims& shape);

TensorDims shapeToStrides(const TensorDims& shape);

bool hasContiguousBlocks(const TensorDims& tensorShape, const TensorDims& encodingShape);

template <typename T>
void permute(const T* input, T* output, const TensorDims& inputShape, std::vector<size_t> order, ComputationMode mode,
             void* stream = nullptr);

template <typename T>
void permuteKernelCPU(const T* inTensor, T* outTensor, size_t numel, const TensorDims& inputStrides,
                      const TensorDims& outputStrides);

template <typename T>
void permuteKernelGPU(const T* inTensor, T* outTensor, size_t numel, const TensorDims& inputStrides,
                      const TensorDims& outputStrides, void* stream);

void synchronizeStream(ComputationMode mode, void* stream);

void synchronizeCudaStream(void* stream);

}   // namespace DlQuantization

#endif   // DL_QUANTIZATION_TENSOR_UTILS_H

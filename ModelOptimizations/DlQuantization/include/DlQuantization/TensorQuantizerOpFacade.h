// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef AIMET_TENSOR_QUANTIZER_OP_FACADE_H
#define AIMET_TENSOR_QUANTIZER_OP_FACADE_H

#include <DlQuantization/Quantization.hpp>

namespace DlQuantization
{


enum class TensorQuantizerOpMode
{
    updateStats,
    oneShotQuantizeDequantize,
    quantizeDequantize,
    passThrough
};


/**
 * This is a facade interface for the TensorQuantizer class. This facade only exposes the interfaces that are needed
 * by a C++ custom op (PyTorch). Specifically methods that require numpy tensors are omitted
 * as these are only intended to be invoked from Python code which has easy access to numpy variants of torch and tf
 * tensors.
 */
class TensorQuantizerOpFacade
{
public:
    /**
     * Reset stats being collected to compute encoding
     */
    virtual void resetEncodingStats() = 0;

    /**
     * Update stats being collected to compute encoding
     * @param tensor Tensor to update the stats with
     * @param tensorSize Size of the tensor (number of tensor elements)
     * @param useCuda If true, the tensor is assumed to be in CUDA memory
     */
    virtual void updateStats(const float* tensor, std::size_t tensorSize, bool useCuda)                    = 0;
    virtual void updateStats(const float* tensor, std::size_t tensorSize, bool useCuda, IAllocator* alloc) = 0;

    /**
     * Convert a tensor from float to quantized int and back to float
     * @param input Input tensor
     * @param tensorSize Size of the input tensor (number of tensor elements)
     * @param output Output tensor
     * @param encodingMin minimum value of encoding range
     * @param encodingMax maximum value of encoding range
     * @param bitwidth to be used
     * @param useCuda If true, both the input and output tensors are assumed to be in CUDA memory
     */
    virtual void quantizeDequantize(const float* input, std::size_t tensorSize, float* output, double encodingMin,
                                    double encodingMax, unsigned int bitwidth, bool useCuda) = 0;

    virtual void quantizeDequantize(const float* input, std::size_t tensorSize, float* output, double encodingMin,
                                    double encodingMax, unsigned int bitwidth, bool useCuda, void* stream) = 0;
    /**
     * Compute the encoding for this tensor using stats collected so far
     */
    virtual TfEncoding computeEncoding(unsigned int bitwidth, bool useSymmetricEncoding) = 0;

    virtual bool getStrictSymmetric()   = 0;
    virtual bool getUnsignedSymmetric() = 0;
};


}   // namespace DlQuantization

#endif   // AIMET_TENSOR_QUANTIZER_OP_FACADE_H

// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef AIMET_PY_TENSOR_QUANTIZER_H
#define AIMET_PY_TENSOR_QUANTIZER_H

#include <memory>
#include <pybind11/numpy.h>

#include "DlQuantization/TensorQuantizer.h"

namespace py = pybind11;

namespace DlQuantization
{
/**
 * This class sublasses a tensor quantizer and overloads two of its functions with pybind
 * alternatives
 */
class PyTensorQuantizer : public TensorQuantizer
{
public:
    /**
     * Constructor
     * @param quantScheme Quantization scheme (e.g. TF-Enhanced)
     * @param roundingMode Rounding mode to use during quantization
     */
    PyTensorQuantizer(QuantizationMode quantScheme, RoundingMode roundingMode);

    /**
     * Update stats being collected to compute encoding. Overloaded version that accepts a numpy tensor.
     * @param tensor Tensor to update the stats with
     * @param useCuda If true, the tensor is assumed to be in CUDA memory
     */
    void updateStats(py::array_t<float> tensor, bool useCuda);

    /**
     * Convert a tensor from float to quantized int and back to float. Overloaded version that accepts numpy tensors.
     * @param input Input tensor
     * @param output Output tensor
     * @param encodingMin minimum value of encoding range
     * @param encodingMax maximum value of encoding range
     * @param bitwidth bitwidth to be used
     * @param useCuda If true, both the input and output tensors are assumed to be in CUDA memory
     */
    void quantizeDequantize(py::array_t<float> inputTensor, py::array_t<float> outputTensor, double encodingMin,
                            double encodingMax, unsigned int bitwidth, bool useCuda);

    ~PyTensorQuantizer() = default;
};

void pyUpdateStats(BlockTensorQuantizer &self, py::array_t<float> tensor);

py::array_t<float> pyQuantizeDequantize(BlockTensorQuantizer &self, py::array_t<float> inputTensor);

}   // namespace DlQuantization

#endif   // AIMET_PY_TENSOR_QUANTIZER_H

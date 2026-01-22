// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "PyTensorQuantizer.hpp"

namespace DlQuantization
{

PyTensorQuantizer::PyTensorQuantizer(QuantizationMode quantScheme, RoundingMode roundingMode) :
    TensorQuantizer(quantScheme, roundingMode)
{
}

void PyTensorQuantizer::updateStats(py::array_t<float> tensor, bool useCuda)
{
    auto npArr = tensor.mutable_unchecked();

    size_t tensorSize = 1;
    for (int i = 0; i < npArr.ndim(); i++)
        tensorSize *= npArr.shape(i);

    // Get a pointer to the tensor data
    auto tensorPtr = (float*) npArr.mutable_data();

    // Delegate
    TensorQuantizer::updateStats(tensorPtr, tensorSize, useCuda);
}

void PyTensorQuantizer::quantizeDequantize(py::array_t<float> inputTensor, py::array_t<float> outputTensor,
                                           double encodingMin, double encodingMax, unsigned int bitwidth, bool useCuda)
{
    auto inputArr  = inputTensor.mutable_unchecked();
    auto outputArr = outputTensor.mutable_unchecked();

    size_t tensorSize = inputArr.size();

    auto inputTensorPtr  = static_cast<float*>(inputArr.mutable_data());
    auto outputTensorPtr = static_cast<float*>(outputArr.mutable_data());

    // Delegate
    TensorQuantizer::quantizeDequantize(inputTensorPtr, tensorSize, outputTensorPtr, encodingMin, encodingMax, bitwidth,
                                        useCuda);
}


void pyUpdateStats(BlockTensorQuantizer &self, py::array_t<float> tensor)
{
    py::buffer_info buf = tensor.request();
    TensorDims shape(buf.ndim);
    for (size_t i = 0; i < buf.ndim; i++)
    {
        shape[i] = buf.shape[i];
    }
    auto ptr = static_cast<float*>(buf.ptr);
    self.updateStats(ptr, shape, false);

}

py::array_t<float> pyQuantizeDequantize(BlockTensorQuantizer &self, py::array_t<float> inputTensor)
{
    // Ensure the input tensor is contiguous
    if (!(inputTensor.flags() & py::array::c_style)) {
        inputTensor = py::cast<py::array_t<float>>(inputTensor.attr("copy")());
    }

    py::buffer_info inputBuf = inputTensor.request();
    TensorDims shape(inputBuf.ndim);
    for (size_t i = 0; i < inputBuf.ndim; i++)
    {
        shape[i] = inputBuf.shape[i];
    }
    auto inputTensorPtr = static_cast<float*>(inputBuf.ptr);
    std::vector<size_t> outShape(inputBuf.shape.begin(), inputBuf.shape.end());
    py::array_t<float> outputTensor(outShape);
    py::buffer_info outputBuf = outputTensor.request();
    auto outputTensorPtr = static_cast<float*>(outputBuf.ptr);

    // Delegate quantizeDequantize method on BlockTensorQuantizer instance
    self.quantizeDequantize(inputTensorPtr, outputTensorPtr, shape, false);

    return outputTensor;
}

}   // namespace DlQuantization

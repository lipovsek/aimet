// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <DlQuantization/IQuantizer.hpp>
#include <DlQuantization/ITensorQuantizationSim.h>
#include <DlQuantization/Quantization.hpp>
#include <DlQuantization/QuantizerFactory.hpp>
#include <DlQuantization/TensorQuantizationSimForPython.h>
#include <iostream>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
namespace py = pybind11;

namespace DlQuantization
{
TensorQuantizationSimForPython::TensorQuantizationSimForPython()
{
    _tensorQuantizationSim = DlQuantization::getTensorQuantizationSim<float>();
}


py::array_t<float> TensorQuantizationSimForPython::quantizeDequantize(py::array_t<float> input,
                                                                      DlQuantization::TfEncoding& encoding,
                                                                      DlQuantization::RoundingMode roundingMode,
                                                                      unsigned int bitwidth, bool use_cuda)
{
    auto npArr        = input.mutable_unchecked<>();
    auto inputDataPtr = (float*) npArr.mutable_data();

    // Allocate an output tensor as the same shape as the input
    py::array_t<float> output = input;
    auto outNpArr             = output.mutable_unchecked<>();
    auto outputDataPtr        = (float*) outNpArr.mutable_data();

    size_t inputTensorSize = npArr.size();

    _tensorQuantizationSim->quantizeDequantizeTensor(inputDataPtr, inputTensorSize, outputDataPtr, encoding.min,
                                                     encoding.max, bitwidth, roundingMode, use_cuda, nullptr);

    return output;
}

}   // namespace DlQuantization

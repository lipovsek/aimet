// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <DlQuantization/EncodingAnalyzerForPython.h>
#include <DlQuantization/IQuantizationEncodingAnalyzer.hpp>
#include <DlQuantization/IQuantizer.hpp>
#include <DlQuantization/ITensorQuantizationSim.h>
#include <DlQuantization/Quantization.hpp>
#include <DlQuantization/QuantizerFactory.hpp>
#include <iostream>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
namespace py = pybind11;

namespace DlQuantization
{
EncodingAnalyzerForPython::EncodingAnalyzerForPython(DlQuantization::QuantizationMode quantizationScheme) :
    _quantizationScheme(quantizationScheme)
{
    _encodingAnalyzer = DlQuantization::getEncodingAnalyzerInstance<float>(quantizationScheme);
}

void EncodingAnalyzerForPython::updateStats(py::array_t<float> input, bool use_cuda)
{
    auto npArr = input.mutable_unchecked<>();

    // Set encoding as valid
    _isEncodingValid = true;

    size_t inputTensorSize = npArr.size();

    // Get a pointer to the tensor data
    auto inputDataPtr = (float*) npArr.mutable_data();

    DlQuantization::ComputationMode cpu_gpu_mode =
        use_cuda ? DlQuantization::ComputationMode::COMP_MODE_GPU : DlQuantization::ComputationMode::COMP_MODE_CPU;
    _encodingAnalyzer->updateStats(inputDataPtr, inputTensorSize, cpu_gpu_mode);
}


std::tuple<DlQuantization::TfEncoding, bool> EncodingAnalyzerForPython::computeEncoding(unsigned int bitwidth,
                                                                                        bool isSymmetric,
                                                                                        bool useStrictSymmetric,
                                                                                        bool useUnsignedSymmetric)
{
    DlQuantization::TfEncoding out_encoding;

    if (_isEncodingValid)
    {
        out_encoding =
            _encodingAnalyzer->computeEncoding(bitwidth, isSymmetric, useStrictSymmetric, useUnsignedSymmetric);
    }

    return std::make_tuple(out_encoding, _isEncodingValid);
}


}   // namespace DlQuantization
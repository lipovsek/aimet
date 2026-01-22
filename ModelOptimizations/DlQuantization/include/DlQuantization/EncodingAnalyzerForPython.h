// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef AIMET_ENCODING_ANALYZER_FOR_PYTHON_H
#define AIMET_ENCODING_ANALYZER_FOR_PYTHON_H

#include <DlQuantization/IQuantizationEncodingAnalyzer.hpp>
#include <DlQuantization/IQuantizer.hpp>
#include <DlQuantization/Quantization.hpp>
#include <DlQuantization/QuantizerFactory.hpp>
#include <iostream>
#include <pybind11/numpy.h>
#include <string>
#include <vector>

namespace py = pybind11;

namespace DlQuantization
{
class EncodingAnalyzerForPython

{
public:
    EncodingAnalyzerForPython(DlQuantization::QuantizationMode quantizationScheme);

    void updateStats(py::array_t<float> input, bool use_cuda);

    std::tuple<DlQuantization::TfEncoding, bool> computeEncoding(unsigned int bitwidth, bool isSymmetric,
                                                                 bool useStrictSymmetric, bool useUnsignedSymmetric);


private:
    bool _isEncodingValid;
    DlQuantization::QuantizationMode _quantizationScheme;
    std::unique_ptr<DlQuantization::IQuantizationEncodingAnalyzer<float>> _encodingAnalyzer;
};

}   // namespace DlQuantization

#endif   // AIMET_ENCODING_ANALYZER_FOR_PYTHON_H

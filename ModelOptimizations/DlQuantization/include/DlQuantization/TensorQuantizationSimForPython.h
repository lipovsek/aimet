// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef AIMET_TENSORQUANTIZATIONSIMFORPYTHON_H
#define AIMET_TENSORQUANTIZATIONSIMFORPYTHON_H

#include <DlQuantization/ITensorQuantizationSim.h>
#include <DlQuantization/Quantization.hpp>
#include <iostream>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace DlQuantization
{
class TensorQuantizationSimForPython
{
public:
    TensorQuantizationSimForPython();

    py::array_t<float> quantizeDequantize(py::array_t<float> input, DlQuantization::TfEncoding& encoding,
                                          DlQuantization::RoundingMode roundingMode, unsigned int bitwidth,
                                          bool use_cuda);

private:
    std::unique_ptr<DlQuantization::ITensorQuantizationSim<float>> _tensorQuantizationSim;
};

}   // namespace DlQuantization
#endif   // AIMET_TENSORQUANTIZATIONSIMFORPYTHON_H

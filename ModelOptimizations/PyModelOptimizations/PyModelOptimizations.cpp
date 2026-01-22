// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "pybind11/complex.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DlQuantization/EncodingAnalyzerForPython.h"
#include "DlQuantization/IQuantizationEncodingAnalyzer.hpp"
#include "DlQuantization/IQuantizer.hpp"
#include "DlQuantization/Quantization.hpp"
#include "DlQuantization/QuantizerFactory.hpp"
#include "DlQuantization/TensorQuantizationSimForPython.h"
#include "DlQuantization/TensorQuantizerOpFacade.h"
#include "DlQuantization/EncodingRescale.hpp"
#include "PyTensorQuantizer.hpp"

namespace py = pybind11;

using namespace DlQuantization;


PYBIND11_MODULE(_libpymo, m)
{
    py::options options;
    options.show_function_signatures();
    options.show_user_defined_docstrings();

    // Quantization python bindings
    py::enum_<ComputationMode>(m, "ComputationMode")
        .value("COMP_MODE_CPU", ComputationMode::COMP_MODE_CPU)
        .value("COMP_MODE_GPU", ComputationMode::COMP_MODE_GPU)
        .export_values();

    py::enum_<QuantizationMode>(m, "QuantizationMode")
        .value("QUANTIZATION_TF", QuantizationMode::QUANTIZATION_TF)
        .value("QUANTIZATION_TF_ENHANCED", QuantizationMode::QUANTIZATION_TF_ENHANCED)
        .value("QUANTIZATION_RANGE_LEARNING", QuantizationMode::QUANTIZATION_RANGE_LEARNING)
        .value("QUANTIZATION_PERCENTILE", QuantizationMode::QUANTIZATION_PERCENTILE)
        .value("QUANTIZATION_MSE", QuantizationMode::QUANTIZATION_MSE)
        .value("QUANTIZATION_ENTROPY", QuantizationMode::QUANTIZATION_ENTROPY)
        .export_values();

    py::enum_<LayerInOut>(m, "LayerInOut")
        .value("LAYER_INPUT", LayerInOut::LAYER_INPUT)
        .value("LAYER_OUTPUT", LayerInOut::LAYER_OUTPUT)
        .export_values();

    py::enum_<RoundingMode>(m, "RoundingMode")
        .value("ROUND_NEAREST", RoundingMode::ROUND_NEAREST)
        .value("ROUND_STOCHASTIC", RoundingMode::ROUND_STOCHASTIC)
        .export_values();

    py::class_<DlQuantization::TfEncoding>(m, "TfEncoding")
        .def(py::init<>())
        .def_readwrite("min", &DlQuantization::TfEncoding::min)
        .def_readwrite("max", &DlQuantization::TfEncoding::max)
        .def_readwrite("delta", &DlQuantization::TfEncoding::delta)
        .def_readwrite("offset", &DlQuantization::TfEncoding::offset)
        .def_readwrite("bw", &DlQuantization::TfEncoding::bw);

    py::class_<IQuantizationEncodingAnalyzer<float>>(m, "QuantizationEncodingAnalyzer");
    m.def("GetQuantizationEncodingAnalyzerInstance", &getEncodingAnalyzerInstance<float>);

    py::class_<DlQuantization::EncodingAnalyzerForPython>(m, "EncodingAnalyzerForPython")
        .def(py::init<DlQuantization::QuantizationMode>())
        .def("updateStats", &DlQuantization::EncodingAnalyzerForPython::updateStats)
        .def("computeEncoding", &DlQuantization::EncodingAnalyzerForPython::computeEncoding);

    py::class_<DlQuantization::TensorQuantizationSimForPython>(m, "TensorQuantizationSimForPython")
        .def(py::init<>())
        .def("quantizeDequantize",
             (py::array_t<float>(TensorQuantizationSimForPython::*)(py::array_t<float>, DlQuantization::TfEncoding&,
                                                                    DlQuantization::RoundingMode, unsigned int, bool)) &
                 DlQuantization::TensorQuantizationSimForPython::quantizeDequantize)
        .def("quantizeDequantize",
             (py::array_t<float>(TensorQuantizationSimForPython::*)(py::array_t<float>, DlQuantization::TfEncoding&,
                                                                    DlQuantization::RoundingMode, bool)) &
                 DlQuantization::TensorQuantizationSimForPython::quantizeDequantize);

    py::enum_<DlQuantization::TensorQuantizerOpMode>(m, "TensorQuantizerOpMode")
        .value("updateStats", DlQuantization::TensorQuantizerOpMode::updateStats)
        .value("oneShotQuantizeDequantize", DlQuantization::TensorQuantizerOpMode::oneShotQuantizeDequantize)
        .value("quantizeDequantize", DlQuantization::TensorQuantizerOpMode::quantizeDequantize)
        .value("passThrough", DlQuantization::TensorQuantizerOpMode::passThrough);

    py::class_<DlQuantization::BlockTensorQuantizer, std::shared_ptr<DlQuantization::BlockTensorQuantizer>>(m, "BlockTensorQuantizer")
        .def(py::init<TensorDims, int, QuantizationMode>())
        .def("resetEncodingStats", &DlQuantization::BlockTensorQuantizer::resetEncodingStats)
        .def("setEncodings", &DlQuantization::BlockTensorQuantizer::setEncodings)
        .def("getEncodings", &DlQuantization::BlockTensorQuantizer::getEncodings)
        .def("setQuantScheme", &DlQuantization::BlockTensorQuantizer::setQuantScheme)
        .def("getQuantScheme", &DlQuantization::BlockTensorQuantizer::getQuantScheme)
        .def("setStrictSymmetric", &DlQuantization::BlockTensorQuantizer::setStrictSymmetric)
        .def("getStrictSymmetric", &DlQuantization::BlockTensorQuantizer::getStrictSymmetric)
        .def("setUnsignedSymmetric", &DlQuantization::BlockTensorQuantizer::setUnsignedSymmetric)
        .def("getUnsignedSymmetric", &DlQuantization::BlockTensorQuantizer::getUnsignedSymmetric)
        .def("getStatsHistogram", &DlQuantization::BlockTensorQuantizer::getStatsHistogram)
        .def("setPercentileValue", &DlQuantization::BlockTensorQuantizer::setPercentileValue)
        .def("getPercentileValue", &DlQuantization::BlockTensorQuantizer::getPercentileValue)
        .def("setZeroPointShift", &DlQuantization::BlockTensorQuantizer::setZeroPointShift)
        .def("getZeroPointShift", &DlQuantization::BlockTensorQuantizer::getZeroPointShift)
        .def("computeEncodings", &DlQuantization::BlockTensorQuantizer::computeEncodings)
        .def("getShape", &DlQuantization::BlockTensorQuantizer::getShape)
        .def("updateStats", &pyUpdateStats)
        .def("quantizeDequantize", &pyQuantizeDequantize)
        .def_readwrite("bitwidth", &DlQuantization::BlockTensorQuantizer::bitwidth)
        .def_readwrite("isEncodingValid", &DlQuantization::BlockTensorQuantizer::isEncodingValid);

    py::class_<DlQuantization::PyTensorQuantizer>(m, "TensorQuantizer")
        .def(py::init<DlQuantization::QuantizationMode, DlQuantization::RoundingMode>())
        .def("updateStats",
             (void(PyTensorQuantizer::*)(py::array_t<float>, bool)) & DlQuantization::PyTensorQuantizer::updateStats)
        .def("computeEncoding", &DlQuantization::PyTensorQuantizer::computeEncoding)
        .def("quantizeDequantize",
             (void(PyTensorQuantizer::*)(py::array_t<float>, py::array_t<float>, double, double, unsigned int, bool)) &
                 DlQuantization::PyTensorQuantizer::quantizeDequantize)
        .def("resetEncodingStats", &DlQuantization::PyTensorQuantizer::resetEncodingStats)
        .def("setQuantScheme", &DlQuantization::PyTensorQuantizer::setQuantScheme)
        .def("getQuantScheme", &DlQuantization::PyTensorQuantizer::getQuantScheme)
        .def("setStrictSymmetric", &DlQuantization::PyTensorQuantizer::setStrictSymmetric)
        .def("getStrictSymmetric", &DlQuantization::PyTensorQuantizer::getStrictSymmetric)
        .def("setUnsignedSymmetric", &DlQuantization::PyTensorQuantizer::setUnsignedSymmetric)
        .def("getUnsignedSymmetric", &DlQuantization::PyTensorQuantizer::getUnsignedSymmetric)
        .def("getStatsHistogram", &DlQuantization::PyTensorQuantizer::getStatsHistogram)
        .def("setPercentileValue", &DlQuantization::PyTensorQuantizer::setPercentileValue)
        .def("getPercentileValue", &DlQuantization::PyTensorQuantizer::getPercentileValue)
        .def("computePartialEncoding", &DlQuantization::PyTensorQuantizer::computePartialEncoding)
        .def_readwrite("roundingMode", &DlQuantization::PyTensorQuantizer::roundingMode)
        .def_readwrite("isEncodingValid", &DlQuantization::PyTensorQuantizer::isEncodingValid);

    m.def("PtrToInt64", [](void* ptr) { return (uint64_t) ptr; });
    m.def("getScaleFactor", &getScaleFactor);
    m.def("getRescaledOutputAndBias", &getRescaledOutputAndBias<float>);
}

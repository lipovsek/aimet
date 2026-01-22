// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "QcQuantizeInfo.h"
#include "DlQuantization/Quantization.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(libquant_info, m)
{
    pybind11::class_<QcQuantizeInfo>(m, "QcQuantizeInfo")
        .def(py::init<>())
        .def_readwrite("tensorQuantizerRef", &QcQuantizeInfo::tensorQuantizer)
        .def_property("encoding", &QcQuantizeInfo::getEncodings, &QcQuantizeInfo::setEncodings)
        .def_readwrite("opMode", &QcQuantizeInfo::opMode)
        .def_readwrite("name", &QcQuantizeInfo::name)
        .def_readwrite("enabled", &QcQuantizeInfo::enabled)
        .def_readwrite("useSymmetricEncoding", &QcQuantizeInfo::useSymmetricEncoding)
        .def_readwrite("usePerChannelMode", &QcQuantizeInfo::usePerChannelMode)
        .def_readwrite("isIntDataType", &QcQuantizeInfo::isIntDataType)
        .def_readwrite("channelAxis", &QcQuantizeInfo::channelAxis)
        .def_readwrite("blockSize", &QcQuantizeInfo::blockSize)
        .def_readwrite("blockAxis", &QcQuantizeInfo::blockAxis);
}

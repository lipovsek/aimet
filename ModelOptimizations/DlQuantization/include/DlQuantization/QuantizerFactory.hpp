// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef QUANTIZER_FACTORY_HPP
#define QUANTIZER_FACTORY_HPP

#include <memory>
#include <string>
#include <vector>


#include "DlQuantization/IQuantizationEncodingAnalyzer.hpp"
#include "DlQuantization/IQuantizer.hpp"
#include "DlQuantization/ITensorQuantizationSim.h"
#include "DlQuantization/Quantization.hpp"

namespace DlQuantization
{

template <typename DTYPE>
std::unique_ptr<IQuantizationEncodingAnalyzer<DTYPE>> getEncodingAnalyzerInstance(QuantizationMode quantization_mode);

template <typename DTYPE>
std::unique_ptr<IBlockEncodingAnalyzer<DTYPE>> getBlockEncodingAnalyzerInstance(QuantizationMode quantization_mode, const TensorDims& shape);

template <typename DTYPE>
std::unique_ptr<ITensorQuantizationSim<DTYPE>> getTensorQuantizationSim();

}   // End of namespace DlQuantization

#endif   // QUANTIZER_FACTORY_HPP

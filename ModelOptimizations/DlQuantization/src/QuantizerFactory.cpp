// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "DlQuantization/IQuantizer.hpp"
#include "DlQuantization/Quantization.hpp"
#include "DlQuantization/QuantizerFactory.hpp"
#include "EntropyEncodingAnalyzer.h"
#include "MinMaxEncodingAnalyzer.h"
#include "MseEncodingAnalyzer.h"
#include "PercentileEncodingAnalyzer.h"
#include "TensorQuantizationSim.h"
#include "TfEncodingAnalyzer.h"
#include "TfEnhancedEncodingAnalyzer.h"
#include "EncodingAnalyzerWrapper.h"

namespace DlQuantization
{

template <typename DTYPE>
std::unique_ptr<IQuantizationEncodingAnalyzer<DTYPE>> getEncodingAnalyzerInstance(QuantizationMode quantization_mode)
{
    if (quantization_mode == QUANTIZATION_TF_ENHANCED)
    {
        return std::unique_ptr<IQuantizationEncodingAnalyzer<DTYPE>>(new TfEnhancedEncodingAnalyzer<DTYPE>);
    }
    else if (quantization_mode == QUANTIZATION_PERCENTILE)
    {
        return std::unique_ptr<IQuantizationEncodingAnalyzer<DTYPE>>(new PercentileEncodingAnalyzer<DTYPE>);
    }
    else if (quantization_mode == QUANTIZATION_MSE)
    {
        return std::unique_ptr<IQuantizationEncodingAnalyzer<DTYPE>>(new MseEncodingAnalyzer<DTYPE>);
    }
    else if (quantization_mode == QUANTIZATION_ENTROPY)
    {
        return std::unique_ptr<IQuantizationEncodingAnalyzer<DTYPE>>(new EntropyEncodingAnalyzer<DTYPE>);
    }
    else
    {
        return std::unique_ptr<IQuantizationEncodingAnalyzer<DTYPE>>(new TfEncodingAnalyzer<DTYPE>);
    }
}

template <typename DTYPE>
std::unique_ptr<IBlockEncodingAnalyzer<DTYPE>> getBlockEncodingAnalyzerInstance(QuantizationMode quantization_mode, const TensorDims& shape)
{
    if (quantization_mode == QUANTIZATION_TF)
    {
        return std::unique_ptr<IBlockEncodingAnalyzer<DTYPE>>(new MinMaxEncodingAnalyzer<DTYPE>(shape));
    }
    return std::unique_ptr<IBlockEncodingAnalyzer<DTYPE>>(new EncodingAnalyzerWrapper<DTYPE>(shape, quantization_mode));
}

template <typename DTYPE>
std::unique_ptr<ITensorQuantizationSim<DTYPE>> getTensorQuantizationSim()
{
    return std::unique_ptr<ITensorQuantizationSim<DTYPE>>(new TensorQuantizationSim<DTYPE>());
}

template std::unique_ptr<IQuantizationEncodingAnalyzer<float>>
getEncodingAnalyzerInstance(QuantizationMode quantization_mode);
template std::unique_ptr<IQuantizationEncodingAnalyzer<double>>
getEncodingAnalyzerInstance(QuantizationMode quantization_mode);

template std::unique_ptr<IBlockEncodingAnalyzer<float>>
getBlockEncodingAnalyzerInstance(QuantizationMode quantization_mode, const TensorDims& shape);
template std::unique_ptr<IBlockEncodingAnalyzer<double>>
getBlockEncodingAnalyzerInstance(QuantizationMode quantization_mode, const TensorDims& shape);

template std::unique_ptr<ITensorQuantizationSim<float>> getTensorQuantizationSim();
template std::unique_ptr<ITensorQuantizationSim<double>> getTensorQuantizationSim();

}   // End of namespace DlQuantization

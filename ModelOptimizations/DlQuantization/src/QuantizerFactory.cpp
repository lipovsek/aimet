// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <stdexcept>

#include "DlQuantization/Quantization.hpp"
#include "DlQuantization/QuantizerFactory.hpp"
#include "EncodingAnalyzerWrapper.h"
#include "MinMaxEncodingAnalyzer.h"
#include "PercentileEncodingAnalyzer.h"
#include "TfEncodingAnalyzer.h"
#include "TfEnhancedEncodingAnalyzer.h"

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

template std::unique_ptr<IQuantizationEncodingAnalyzer<float>>
getEncodingAnalyzerInstance(QuantizationMode quantization_mode);

template std::unique_ptr<IBlockEncodingAnalyzer<float>>
getBlockEncodingAnalyzerInstance(QuantizationMode quantization_mode, const TensorDims& shape);

}   // End of namespace DlQuantization

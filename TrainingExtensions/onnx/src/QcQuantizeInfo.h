// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "DlQuantization/IQuantizationEncodingAnalyzer.hpp"
#include "DlQuantization/QuantizerFactory.hpp"
#include "DlQuantization/TensorQuantizer.h"
#include <string>


struct QcQuantizeInfo
{
    void setEncodings(const DlQuantization::Encodings& encodings)
    {
        if (tensorQuantizer.get())
        {
            tensorQuantizer->setEncodings(encodings);
        }
        else
        {
            throw std::runtime_error("Cannot set encodings before instantiating tensor quantizer");
        }
    }
    DlQuantization::Encodings getEncodings()
    {
        if (tensorQuantizer.get())
        {
            return tensorQuantizer->getEncodings();
        }
        return DlQuantization::Encodings(0);
    }

    std::shared_ptr<DlQuantization::BlockTensorQuantizer> tensorQuantizer;
    DlQuantization::TensorQuantizerOpMode opMode;
    bool useSymmetricEncoding;
    bool enabled;
    bool isIntDataType;
    bool usePerChannelMode;
    int channelAxis;
    int blockAxis;
    size_t blockSize;
    std::string name;
};

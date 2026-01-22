// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

// This file contains code to analyze and calculate quantization encodings
// This code is specific for the TF Enhanced quantization scheme
// Support for other schemes is currently TBD

#ifndef QUANTIZATION_SIM_H
#define QUANTIZATION_SIM_H

#include "DlQuantization/ITensorQuantizationSim.h"
#include "DlQuantization/Quantization.hpp"
#include <cstddef>
#include <cstdint>

namespace DlQuantization
{
template <typename DTYPE>
class TensorQuantizationSim : public ITensorQuantizationSim<DTYPE>
{
public:
    explicit TensorQuantizationSim();

    void quantizeDequantizeTensor(const DTYPE* inputTensorData, size_t inputTensorCount, DTYPE* outputTensorData,
                                  double encodingMin, double encodingMax, uint8_t bw, RoundingMode roundMode,
                                  bool use_cuda) override;

    void quantizeDequantizeTensor(const DTYPE* inputTensorData, size_t inputTensorCount, DTYPE* outputTensorData,
                                  double encodingMin, double encodingMax, uint8_t bw, RoundingMode roundMode,
                                  bool use_cuda, void* stream) override;

    void quantizeTensor(const DTYPE* inputTensorData, size_t inputTensorCount, DTYPE* outputTensorData,
                        double encodingMin, double encodingMax, uint8_t bw, RoundingMode roundMode, bool use_cuda,
                        bool shiftToSigned) override;

    void quantizeTensorPacked(const DTYPE* inputTensorData, size_t inputTensorCount,
                              std::vector<uint8_t>& outputTensorData, double encodingMin, double encodingMax,
                              uint8_t bw, RoundingMode roundMode, bool useCuda, bool shiftToSigned) override;

    void quantizeDequantizePerChannelTensor(std::vector<std::vector<DTYPE>>& splits, std::vector<uint32_t> splitShape,
                                            uint32_t axis, DTYPE* outputTensorData,
                                            const std::vector<TfEncoding>& encodings, uint8_t bw,
                                            RoundingMode roundMode, bool useCuda) override;

    void quantizePerChannelTensorPacked(std::vector<std::vector<DTYPE>>& splits, std::vector<uint32_t> splitShape,
                                        uint32_t axis, std::vector<uint8_t>& outputTensorData,
                                        const std::vector<TfEncoding>& encodings, uint8_t bw, RoundingMode roundMode,
                                        bool useCuda, bool shiftToSigned) override;

    void dequantizeTensor(const uint8_t* inputTensorData, size_t inputTensorCount, DTYPE* output, double encodingMin,
                          double encodingMax, uint8_t bw, bool shiftToSigned) override;

    void dequantizePerChannelTensor(const uint8_t* inputTensorData, const std::vector<uint32_t>& inputShape,
                                    uint32_t axis, DTYPE* outputTensorData, uint8_t bw,
                                    const std::vector<TfEncoding>& encodings, bool shiftToSigned) override;

    void fillEncodingInfo(TfEncoding& encoding, uint8_t bw, double encodingMin, double encodingMax) override;

    void generateScaleOffset(double& encodingMin, double& encodingMax, uint8_t bw, double& encodingScale,
                             double& encodingOffset);

    void quantizeDequantizeTensorPerChannel(const DTYPE* inputTensorData, size_t numChannel, size_t numElement,
                                            size_t numElementPerChannel, DTYPE* outputTensorData, DTYPE* encodingMin,
                                            DTYPE* encodingMax, DTYPE* encodingDelta, DTYPE* encodingOffset,
                                            RoundingMode roundingMode, bool useCuda) override;

    void quantizeDequantizeTensorPerChannel(const DTYPE* inputTensorData, size_t numChannel, size_t numElement,
                                            size_t numElementPerChannel, DTYPE* outputTensorData, DTYPE* encodingMin,
                                            DTYPE* encodingMax, DTYPE* encodingDelta, DTYPE* encodingOffset,
                                            RoundingMode roundingMode, bool useCuda, void* stream) override;

    inline DlQuantization::ComputationMode getComputationMode(bool use_cuda)
    {
        return (use_cuda ? DlQuantization::ComputationMode::COMP_MODE_GPU
                         : DlQuantization::ComputationMode::COMP_MODE_CPU);
    }

    ~TensorQuantizationSim() = default;
};

}   // namespace DlQuantization

#endif   // QUANTIZATION_SIM_H

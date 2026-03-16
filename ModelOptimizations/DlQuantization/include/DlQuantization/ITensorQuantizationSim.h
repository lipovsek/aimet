// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef I_QUANTIZATION_SIM_H
#define I_QUANTIZATION_SIM_H

#include "Quantization.hpp"

namespace DlQuantization
{
template <typename DTYPE>
class ITensorQuantizationSim
{
public:
    virtual ~ITensorQuantizationSim() = default;

    virtual void quantizeDequantizeTensor(const DTYPE* inputTensorData, size_t inputTensorCount,
                                          DTYPE* outputTensorData, double encodingMin, double encodingMax, uint8_t bw,
                                          RoundingMode roundMode, bool use_cuda) = 0;

    virtual void quantizeDequantizeTensor(const DTYPE* inputTensorData, size_t inputTensorCount,
                                          DTYPE* outputTensorData, double encodingMin, double encodingMax, uint8_t bw,
                                          RoundingMode roundMode, bool use_cuda, void* stream) = 0;

    virtual void quantizeTensor(const DTYPE* inputTensorData, size_t inputTensorCount, DTYPE* outputTensorData,
                                double encodingMin, double encodingMax, uint8_t bw, RoundingMode roundMode,
                                bool use_cuda, bool shiftToSigned) = 0;
    /**
     * @brief Convert a tensor from DTYPE to quantized 8-bit packed format
     */
    virtual void quantizeTensorPacked(const DTYPE* inputTensorData, size_t inputTensorCount,
                                      std::vector<uint8_t>& outputTensorData, double encodingMin, double encodingMax,
                                      uint8_t bw, RoundingMode roundMode, bool useCuda, bool shiftToSigned) = 0;

    /**
     * @brief Convert a tensor from quantized 8-bit format into DTYPE
     */
    virtual void dequantizeTensor(const uint8_t* inputTensorData, size_t inputTensorCount, DTYPE* output,
                                  double encodingMin, double encodingMax, uint8_t bw, bool shiftToSigned) = 0;

    /**
     * @brief Performs per channel quantization for each split in splits, and concatenates the result into a quantized
     *        int output before de-quantizing back to float.
     * @relates quantizeDequantizeTensor
     */
    virtual void quantizeDequantizePerChannelTensor(std::vector<std::vector<DTYPE>>& splits,
                                                    std::vector<uint32_t> splitShape, uint32_t axis,
                                                    DTYPE* outputTensorData, const std::vector<TfEncoding>& encodings,
                                                    uint8_t bw, RoundingMode roundMode, bool useCuda) = 0;
    /**
     * @brief Performs per channel quantization for each split in splits, and concatenates the result into a quantized
     *         output before de-quantizing. Output is packed 8 bit quantized data.
     * @relates quantizeDequantizePerChannelTensor
     * @param[in/out] Unsigned 8 bit output tensor
     */
    virtual void quantizePerChannelTensorPacked(std::vector<std::vector<DTYPE>>& splits,
                                                std::vector<uint32_t> splitShape, uint32_t axis,
                                                std::vector<uint8_t>& outputTensorData,
                                                const std::vector<TfEncoding>& encodings, uint8_t bw,
                                                RoundingMode roundMode, bool useCuda, bool shiftToSigned) = 0;

    /**
     * @brief Convert a tensor from quantized 8-bit format into DTYPE, by splitting the data into channels and
     *        dequantizing independently, before concatenating the final result into the output tensor.
     */
    virtual void dequantizePerChannelTensor(const uint8_t* inputTensorData, const std::vector<uint32_t>& inputShape,
                                            uint32_t axis, DTYPE* outputTensorData, uint8_t bw,
                                            const std::vector<TfEncoding>& encodings, bool shiftToSigned) = 0;

    virtual void fillEncodingInfo(TfEncoding& encoding, uint8_t bw, double encodingMin, double encodingMax) = 0;

    virtual void generateScaleOffset(double& encodingMin, double& encodingMax, uint8_t bw, double& encodingScale,
                                     double& encodingOffset) = 0;

    //    Quantize-dequantize a tensor using vectorized method
    virtual void quantizeDequantizeTensorPerChannel(const DTYPE* inputTensorData, size_t numChannel, size_t numElement,
                                                    size_t numElementPerChannel, DTYPE* outputTensorData,
                                                    DTYPE* encodingMin, DTYPE* encodingMax, DTYPE* encodingDelta,
                                                    DTYPE* encodingOffset, RoundingMode roundingMode, bool useCuda) = 0;

    virtual void quantizeDequantizeTensorPerChannel(const DTYPE* inputTensorData, size_t numChannel, size_t numElement,
                                                    size_t numElementPerChannel, DTYPE* outputTensorData,
                                                    DTYPE* encodingMin, DTYPE* encodingMax, DTYPE* encodingDelta,
                                                    DTYPE* encodingOffset, RoundingMode roundingMode, bool useCuda,
                                                    void* stream) = 0;
};

}   // namespace DlQuantization

#endif   // I_QUANTIZATION_SIM_H

// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef AIMET_TENSOR_QUANTIZER_H
#define AIMET_TENSOR_QUANTIZER_H

#include <memory>

#include "DlQuantization/IForLoopRunner.h"
#include <DlQuantization/IQuantizationEncodingAnalyzer.hpp>
#include <DlQuantization/Quantization.hpp>


namespace DlQuantization
{

enum class TensorQuantizerOpMode
{
    updateStats,
    oneShotQuantizeDequantize,
    quantizeDequantize,
    passThrough
};

/**
 * @class BlockTensorQuantizer
 * @brief Performs simulated blocked quantization by dividing the input tensor into blocks, each with its own encodings.
 *
 * During blockwise quantization, the size of blocking along dimension i is TensorShape[i] / QuantizerShape[i].
 * Statistics and encodings are computed for each block of the input tensor independently.
 */
class BlockTensorQuantizer
{
public:
    /**
     * Constructor
     *
     * @param shape Shape of the quantizer's encoding vectors
     * @param bitwidth Bitwidth to use for quantization
     * @param quantScheme Quantization scheme (e.g. TF-Enhanced)
     */
    BlockTensorQuantizer(TensorDims shape, int bitwidth, QuantizationMode quantScheme);

    // TODO: Remove symmetric arg, use this->_symmetric
    Encodings computeEncodings(bool useSymmetricEncodings) const;

    /**
     * Reset stats being collected to compute encoding
     */
    void resetEncodingStats();

    /**
     * @brief Convert a tensor from float to quantized int and back to float
     *
     * @param input Input tensor
     * @param output Output tensor
     * @param tensorShape Shape of the input tensor
     * @param output Output tensor
     * @param useCuda If true, both the input and output tensors are assumed to be in CUDA memory
     * @param stream Cuda stream to use for GPU kernels
     */
    template <typename T>
    void quantizeDequantize(const T* input, T* output, const TensorDims& tensorShape, bool useCuda,
                            void* stream = nullptr, IForLoopRunner* runner = nullptr) const;

    /**
     * Update stats being collected to compute encoding
     * @param tensor Tensor to update the stats with
     * @param tensorShape Shape of the tensor
     * @param useCuda If true, the tensor is assumed to be in CUDA memory
     * @param alloc Allocator to use for memory allocations
     * @param stream Cuda stream to use for GPU kernels
     */
    void updateStats(const float* tensor, const TensorDims& tensorShape, bool useCuda, IAllocator* alloc = nullptr,
                     void* stream = nullptr);

    template <typename T>
    void updateStats(const T* tensor, const TensorDims& tensorShape, bool useCuda, IAllocator* alloc = nullptr,
                     void* stream = nullptr);

    /**
     * sets quantScheme and creates new encoding analyzer instance
     * @param quantScheme Quantization scheme (e.g. TF-Enhanced)
     */
    void setQuantScheme(QuantizationMode quantScheme);

    /**
     * gets quantScheme configured for this Tensor Quantizer
     * @return quantScheme as QuantizationMode
     */
    QuantizationMode getQuantScheme() const;

    /**
     * gets strict symmetric flag configured for this Tensor Quantizer
     * @return quantScheme as QuantizationMode
     */
    bool getStrictSymmetric() const;

    /**
     * sets strict symmetric flag
     * @param bool, True if strict symmetric, False otherwise
     */
    void setStrictSymmetric(bool useStrictSymmetric);

    /**
     * gets unsigned symmetric flag config for this Tensor Quantizer
     * @return bool, True if unsigned symmetric mode, False otherwise
     */
    bool getUnsignedSymmetric() const;

    /**
     * sets unsigned symmetric flag
     * @param bool, True or False
     */
    void setUnsignedSymmetric(bool useUnsignedsymmetric);

    /**
     * @brief Returns a list of histograms the a PDF of tensor values seen by the encoding analyzer for
     * each block
     *
     * @return List of histograms of statistics. Each histogram returned is a vector of buckets. Each bucket is a tuple
     * of two values - the float value representing the left edge of the bucket and a PDF of the values in this bucket
     * relative to all the values seen across all buckets
     */
    std::vector<std::vector<std::tuple<double, double>>> getStatsHistogram() const;

    /**
     * @brief Sets the specified percentile value for the encoding analyzer
     *
     * @param percentile Percentile value to set.
     */
    void setPercentileValue(float percentile);

    /**
     * @brief Fetches the percentile value for the encoding analyzer
     *
     * @return Percentile value of the encoding analyzer.
     */
    float getPercentileValue() const;

    bool hasValidStats()
    {
        return _validStats;
    }

    TensorDims getShape()
    {
        return _shape;
    }

    Encodings getEncodings()
    {
        return _encodings;
    }

    double getZeroPointShift() const
    {
        return _zeroPointShift;
    }

    void setZeroPointShift(double zeroPointShift)
    {
        _zeroPointShift = zeroPointShift;
    }

    void setEncodings(const Encodings& encodings);

    bool isEncodingValid;
    int bitwidth;

private:
    QuantizationMode _quantScheme;
    bool _useStrictSymmetric;
    bool _useUnsignedSymmetric;   // TODO: Remove
    bool _symmetric;
    bool _validStats;   // TODO: Move to EncodingAnalyzer
    double _zeroPointShift;
    Encodings _encodings;
    std::unique_ptr<IBlockEncodingAnalyzer<float>> _encodingAnalyzer;
    TensorDims _shape;
};


}   // namespace DlQuantization

#endif   // AIMET_TENSOR_QUANTIZER_H

// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "DlQuantization/TensorQuantizer.h"
#include "DlQuantization/QuantizerFactory.hpp"
#include "quantization_utils.hpp"
#include "tensor_utils.hpp"
#include "trim_functions.hpp"
#include <cassert>
#include <numeric>
#include <stdexcept>

namespace DlQuantization
{

TensorQuantizer::TensorQuantizer(QuantizationMode quantScheme, RoundingMode roundingMode) :
    _quantScheme(quantScheme),
    roundingMode(roundingMode),
    isEncodingValid(false),
    _useStrictSymmetric(false),
    _useUnsignedSymmetric(false),
    _validStats(false)
{
    _encodingAnalyzer      = getEncodingAnalyzerInstance<float>(quantScheme);
    _tensorQuantizationSim = getTensorQuantizationSim<float>();
}

QuantizationMode TensorQuantizer::getQuantScheme()
{
    return _quantScheme;
}

void TensorQuantizer::setQuantScheme(QuantizationMode quantScheme)
{
    // update quantScheme held by Tensor Quantizer
    _quantScheme = quantScheme;

    // create new encoding analyzer instance and reset associated flags
    // _validStats is tightly coupled with the encoding analyzer instance, needs reset
    resetEncodingStats();
}

bool TensorQuantizer::getStrictSymmetric()
{
    return _useStrictSymmetric;
}

void TensorQuantizer::setStrictSymmetric(bool useStrictSymmetric)
{
    // update strict symmetric flag held by Tensor Quantizer
    _useStrictSymmetric = useStrictSymmetric;
    // create new encoding analyzer instance and reset associated flags
    // _validStats is tightly coupled with the encoding analyzer instance, needs reset
    resetEncodingStats();
}

bool TensorQuantizer::getUnsignedSymmetric()
{
    return _useUnsignedSymmetric;
}

void TensorQuantizer::setUnsignedSymmetric(bool useUnsignedSymmetric)
{
    // update unsignedSymmetric flag held by Tensor Quantizer
    _useUnsignedSymmetric = useUnsignedSymmetric;
    // create new encoding analyzer instance and reset associated flags
    // _validStats is tightly coupled with the encoding analyzer instance, needs reset
    resetEncodingStats();
}

void TensorQuantizer::resetEncodingStats()
{
    _validStats     = false;
    isEncodingValid = false;

    // This is syntactic sugar provided by unique_ptr to call reset() - delete the underlying object
    _encodingAnalyzer = nullptr;
    _encodingAnalyzer = getEncodingAnalyzerInstance<float>(_quantScheme);
}

void TensorQuantizer::updateStats(const float* tensor, std::size_t tensorSize, bool useCuda)
{
    updateStats(tensor, tensorSize, useCuda, nullptr);
}

void TensorQuantizer::updateStats(const float* tensor, std::size_t tensorSize, bool useCuda, IAllocator* alloc)
{
    // Set encoding as valid
    _validStats = true;

    ComputationMode cpuGpuMode = useCuda ? ComputationMode::COMP_MODE_GPU : ComputationMode::COMP_MODE_CPU;
    _encodingAnalyzer->updateStats(tensor, tensorSize, cpuGpuMode, alloc);
}


TfEncoding TensorQuantizer::computeEncoding(unsigned int bitwidth, bool useSymmetricEncoding)
{
    TfEncoding encoding;

    if (_validStats)
    {
        encoding        = _encodingAnalyzer->computeEncoding(bitwidth, useSymmetricEncoding, _useStrictSymmetric,
                                                             _useUnsignedSymmetric);
        isEncodingValid = true;
    }

    return encoding;
}

void TensorQuantizer::computeEncodingFromData(uint8_t bw, const float* data, size_t count, TfEncoding& encoding,
                                              ComputationMode cpuGpuMode, bool useSymmetricEncodings,
                                              bool useUnsignedSymmetric, bool useStrictSymmetric)
{
    // Forget all settings except the choice of encoding analyzer
    if (encoding.delta == 0 && bw != 0)
    {
        resetEncodingStats();
        // Use data to compute min, max statistics (forget any accumulated/updated stats)
        // To avoid duplication use update stats since internal functions rely on this->_stats
        _encodingAnalyzer->updateStats(data, count, cpuGpuMode);

        encoding = _encodingAnalyzer->computeEncoding(bw, useSymmetricEncodings, useStrictSymmetric,
                                                      useUnsignedSymmetric);
    }
    else
    {
        throw std::runtime_error("This function is only valid when encodings must be computed"
                                 " from data");
    }
}

void TensorQuantizer::quantizeDequantize(const float* input, std::size_t tensorSize, float* output, double encodingMin,
                                         double encodingMax, unsigned int bitwidth, bool useCuda)
{
    quantizeDequantize(input, tensorSize, output, encodingMin, encodingMax, bitwidth, useCuda, nullptr);
}

void TensorQuantizer::quantizeDequantize(const float* input, std::size_t tensorSize, float* output, double encodingMin,
                                         double encodingMax, unsigned int bitwidth, bool useCuda, void* stream)
{
    assert(isEncodingValid);
    _tensorQuantizationSim->quantizeDequantizeTensor(input, tensorSize, output, encodingMin, encodingMax, bitwidth,
                                                     roundingMode, useCuda, stream);
}

void TensorQuantizer::quantizeTensorPacked(const float* input, std::size_t tensorSize, std::vector<uint8_t>& output,
                                           double encodingMin, double encodingMax, uint8_t bw, RoundingMode roundMode,
                                           bool useCuda, bool useStrictSymmetric)
{
    assert(isEncodingValid);
    _tensorQuantizationSim->quantizeTensorPacked(input, tensorSize, output, encodingMin, encodingMax, bw, roundMode,
                                                 useCuda, useStrictSymmetric);
}

void TensorQuantizer::quantizeDequantizePerChannelTensor(const float* input, const std::vector<uint32_t>& inputShape,
                                                         uint32_t axis, float* output,
                                                         std::vector<TfEncoding>& encodings, uint8_t bw,
                                                         RoundingMode roundMode, bool useCuda, bool useStrictSymmetric)
{
    std::vector<uint32_t> splitShape;
    std::vector<std::vector<float>> splits;

    this->setStrictSymmetric(useStrictSymmetric);   // currently we only support strict symmetric
    generatePerChannelEncodings(input, inputShape, axis, encodings, bw, splits, splitShape, useCuda);

    _tensorQuantizationSim->quantizeDequantizePerChannelTensor(splits, splitShape, axis, output, encodings, bw,
                                                               roundMode, useCuda);
}

void TensorQuantizer::quantizePerChannelTensorPacked(const float* input, const std::vector<uint32_t>& inputShape,
                                                     uint32_t axis, std::vector<uint8_t>& output,
                                                     std::vector<TfEncoding>& encodings, uint8_t bw,
                                                     RoundingMode roundMode, bool useCuda, bool useStrictSymmetric)
{
    std::vector<uint32_t> splitShape;
    std::vector<std::vector<float>> splits;

    // currently we only support strict symmetric for packed tensors
    this->setStrictSymmetric(useStrictSymmetric);
    generatePerChannelEncodings(input, inputShape, axis, encodings, bw, splits, splitShape, useCuda);
    _tensorQuantizationSim->quantizePerChannelTensorPacked(splits, splitShape, axis, output, encodings, bw, roundMode,
                                                           useCuda, useStrictSymmetric);
}

void TensorQuantizer::dequantize(const uint8_t* input, std::size_t tensorSize, double encodingMin, double encodingMax,
                                 uint8_t bw, float* output, bool useStrictSymmetric)
{
    _tensorQuantizationSim->dequantizeTensor(input, tensorSize, output, encodingMin, encodingMax, bw,
                                             useStrictSymmetric);
}

void TensorQuantizer::dequantizePerChannelTensor(const uint8_t* input, const std::vector<uint32_t>& inputShape,
                                                 uint32_t axis, const std::vector<TfEncoding>& encodings, uint8_t bw,
                                                 float* output, bool useStrictSymmetric)
{
    _tensorQuantizationSim->dequantizePerChannelTensor(input, inputShape, axis, output, bw, encodings,
                                                       useStrictSymmetric);
}

std::vector<std::tuple<double, double>> TensorQuantizer::getStatsHistogram()
{
    auto histogram = _encodingAnalyzer->getStatsHistogram();
    return histogram;
}

void TensorQuantizer::setPercentileValue(float percentile)
{
    // Set percentile value only when quant scheme is percentile.
    if (_quantScheme == DlQuantization::QuantizationMode::QUANTIZATION_PERCENTILE)
    {
        _encodingAnalyzer->setPercentileValue(percentile);
    }
}

float TensorQuantizer::getPercentileValue()
{
    if (_quantScheme == DlQuantization::QuantizationMode::QUANTIZATION_PERCENTILE)
    {
        return _encodingAnalyzer->getPercentileValue();
    }
    else
        throw std::runtime_error("Percentile Value only exists in case of percentile quant scheme.");
}

void TensorQuantizer::generatePerChannelEncodings(const float* input, const std::vector<uint32_t>& inputShape,
                                                  uint32_t axis, std::vector<TfEncoding>& encodings, uint32_t bw,
                                                  std::vector<std::vector<float>>& splits,
                                                  std::vector<uint32_t>& splitShape, bool useCuda)
{
    if (bw < 8)
    {
        throw std::runtime_error("Only bitwidths >= 8 supported for per-channel quantization");
    }

    if (inputShape.size() != 4)
    {
        throw std::runtime_error("Per-channel quantization only operates on 4 dimensional data!");
    }

    if (axis > 3)
    {
        throw std::runtime_error("Per-channel axis must be < 4");
    }

    if (encodings.size() != 0 && encodings.size() != inputShape[axis])
    {
        throw std::runtime_error("Must provide 0 or all encodings for per-channel quantization");
    }

    encodings.resize(inputShape[axis]);

    // Split the data by axis and perform quantization analysis
    slice(input, inputShape, axis, splits, splitShape);
    if (splits.size() != inputShape[axis])
    {
        throw std::runtime_error("Invalid slice count generated. Count must be equal to axis split on!");
    }

    uint32_t splitCount = std::accumulate(std::begin(splitShape), std::end(splitShape), 1, std::multiplies<uint32_t>());
    uint32_t outputCount =
        std::accumulate(std::begin(inputShape), std::end(inputShape), 1, std::multiplies<uint32_t>());
    if (outputCount != splitCount * splits.size())
    {
        throw std::runtime_error("Accumulated split count doesn't match original input count");
    }

    for (uint32_t i = 0; i < splits.size(); ++i)
    {
        auto& e     = encodings[i];
        auto& split = splits[i];

        if (split.size() != splitCount)
        {
            throw std::runtime_error("Tensor split size mismatch!");
        }

        if (e.bw != (double) bw)
        {
            e.bw    = (double) bw;
            e.delta = 0, e.offset = 0;
        }
        // compute encodings
        ComputationMode mode       = useCuda ? COMP_MODE_GPU : COMP_MODE_CPU;
        auto useSymmetricEncodings = _useStrictSymmetric || _useUnsignedSymmetric;

        if (!isEncodingValid)
        {
            this->computeEncodingFromData(bw, split.data(), split.size(), e, mode, useSymmetricEncodings,
                                          _useUnsignedSymmetric, _useStrictSymmetric);
        }
    }
}

void TensorQuantizer::computePartialEncoding(uint8_t bw, TfEncoding& encoding, bool useSymmetricEncodings,
                                             bool useUnsignedSymmetric, bool useStrictSymmetric)
{
    if (encoding.min == 0 && encoding.max == 0)
    {
        computeMinMaxRangeFromDeltaOffset(bw, encoding, useSymmetricEncodings, useUnsignedSymmetric,
                                          useStrictSymmetric);
    }
    else if (encoding.delta == 0)
    {
        computeDeltaAndOffsetFromMinMax(bw, encoding, useSymmetricEncodings, useUnsignedSymmetric, useStrictSymmetric);
    }
    else
    {
        throw std::runtime_error("Cannot determine how to compute partial encoding");
    }
}


BlockTensorQuantizer::BlockTensorQuantizer(TensorDims shape, int bitwidth, QuantizationMode quantScheme) :
    bitwidth(bitwidth),
    isEncodingValid(false),
    _quantScheme(quantScheme),
    _useStrictSymmetric(false),
    _useUnsignedSymmetric(false),
    _symmetric(false),
    _validStats(false),
    _shape(shape),
    _zeroPointShift(0.0)
{
    _encodings.resize(getNumel(shape));
    _encodingAnalyzer = getBlockEncodingAnalyzerInstance<float>(quantScheme, shape);
}

void BlockTensorQuantizer::resetEncodingStats()
{
    _validStats     = false;
    isEncodingValid = false;
    _encodingAnalyzer->resetStats();
}

void BlockTensorQuantizer::updateStats(const float* tensor, const TensorDims& tensorShape, bool useCuda,
                                       IAllocator* alloc, void* stream)
{
    _validStats                = true;
    ComputationMode cpuGpuMode = useCuda ? COMP_MODE_GPU : COMP_MODE_CPU;
    _encodingAnalyzer->updateStats(tensor, tensorShape, cpuGpuMode, alloc, stream);
}


// TODO: Let BlockTensorQuantizer own the encodings vector, do not take as argument
void BlockTensorQuantizer::quantizeDequantize(const float* input, float* output, const TensorDims& tensorShape,
                                              bool useCuda, void* stream) const
{
    auto mode = useCuda ? COMP_MODE_GPU : COMP_MODE_CPU;
    if (not isEncodingValid)
    {
        throw std::runtime_error("Cannot perform quantization before computing encodings");
    }
    if (getNumel(_shape) == 1)
    {
        // More efficient per-tensor quantization impl which avoids separate cudaMemcpy for encodings
        DlQuantization::quantizeDequantize(input, getNumel(tensorShape), _encodings[0], output, mode, ROUND_NEAREST,
                                           stream);
    }
    else
    {
        quantizeDequantizeBroadcast(input, output, _encodings, tensorShape, this->_shape, mode, stream);
    }
}

void BlockTensorQuantizer::setQuantScheme(QuantizationMode quantScheme)
{
    // update quantScheme held by Tensor Quantizer
    _quantScheme = quantScheme;

    // create new encoding analyzer instance and reset associated flags
    // _validStats is tightly coupled with the encoding analyzer instance, needs reset
    resetEncodingStats();
}

QuantizationMode BlockTensorQuantizer::getQuantScheme() const
{
    return _quantScheme;
}

bool BlockTensorQuantizer::getStrictSymmetric() const
{
    return _useStrictSymmetric;
}

void BlockTensorQuantizer::setStrictSymmetric(bool useStrictSymmetric)
{
    isEncodingValid     = false;
    _useStrictSymmetric = useStrictSymmetric;
}

bool BlockTensorQuantizer::getUnsignedSymmetric() const
{
    return _useUnsignedSymmetric;
}

void BlockTensorQuantizer::setUnsignedSymmetric(bool useUnsignedsymmetric)
{
    isEncodingValid       = false;
    _useUnsignedSymmetric = useUnsignedsymmetric;
}

std::vector<std::vector<std::tuple<double, double>>> BlockTensorQuantizer::getStatsHistogram() const
{
    return _encodingAnalyzer->getStatsHistogram();
}

void BlockTensorQuantizer::setPercentileValue(float percentile)
{
    _encodingAnalyzer->setPercentileValue(percentile);
}

float BlockTensorQuantizer::getPercentileValue() const
{
    return _encodingAnalyzer->getPercentileValue();
}

Encodings BlockTensorQuantizer::computeEncodings(bool useSymmetricEncodings) const
{
    // TODO: Move this flag/check to encoding analyzer
    if (not _validStats)
    {
        throw std::runtime_error("Cannot compute encodings before updating stats");
    }
    return _encodingAnalyzer->computeEncoding(bitwidth, useSymmetricEncodings, _useStrictSymmetric,
                                              _useUnsignedSymmetric, _zeroPointShift);
}

void BlockTensorQuantizer::setEncodings(const Encodings& encodings)
{
    if (encodings.size() != getNumel(_shape))
    {
        throw std::runtime_error("Length of encoding vector did not match BlockTensorQuantizer shape");
    }
    isEncodingValid = true;
    _encodings      = encodings;
    bitwidth        = encodings[0].bw;
}

}   // namespace DlQuantization

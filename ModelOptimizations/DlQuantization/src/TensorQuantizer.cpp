// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "DlQuantization/TensorQuantizer.h"
#include "DlQuantization/QuantizerFactory.hpp"
#include "quantization_utils.hpp"
#include "tensor_utils.hpp"
#include "trim_functions.hpp"
#include <Eigen/Core>
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace DlQuantization
{

BlockTensorQuantizer::BlockTensorQuantizer(TensorDims shape, int bitwidth, QuantizationMode quantScheme) :
    BlockTensorQuantizer(shape, QuantizationType::Int(bitwidth), quantScheme)
{
}

BlockTensorQuantizer::BlockTensorQuantizer(TensorDims shape, QuantizationType qtype, QuantizationMode quantScheme) :
    bitwidth(qtype.bitwidth()),
    isEncodingValid(false),
    _quantScheme(quantScheme),
    _qtype(qtype),
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
    updateStats<float>(tensor, tensorShape, useCuda, alloc, stream);
}

template <typename T>
void BlockTensorQuantizer::updateStats(const T* tensor, const TensorDims& tensorShape, bool useCuda, IAllocator* alloc,
                                       void* stream)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, Eigen::half> || std::is_same_v<T, Eigen::bfloat16>,
                  "BlockTensorQuantizer::updateStats only supports float, Eigen::half, and Eigen::bfloat16");
    _validStats                = true;
    ComputationMode cpuGpuMode = useCuda ? COMP_MODE_GPU : COMP_MODE_CPU;
    _encodingAnalyzer->updateStats(tensor, tensorShape, cpuGpuMode, alloc, stream);
}

template void BlockTensorQuantizer::updateStats<float>(const float*, const TensorDims&, bool, IAllocator*, void*);
template void BlockTensorQuantizer::updateStats<Eigen::half>(const Eigen::half*, const TensorDims&, bool, IAllocator*,
                                                             void*);
template void BlockTensorQuantizer::updateStats<Eigen::bfloat16>(const Eigen::bfloat16*, const TensorDims&, bool,
                                                                 IAllocator*, void*);

// TODO: Let BlockTensorQuantizer own the encodings vector, do not take as argument
template <typename T>
void BlockTensorQuantizer::quantizeDequantize(const T* input, T* output, const TensorDims& tensorShape, bool useCuda,
                                              void* stream, IForLoopRunner* runner) const
{
    auto mode = useCuda ? COMP_MODE_GPU : COMP_MODE_CPU;
    if (not isEncodingValid)
    {
        throw std::runtime_error("Cannot perform quantization before computing encodings");
    }

    if (_qtype.isFloat())
    {
        if (_qtype.bitwidth() != 8)
        {
            throw std::runtime_error("BlockTensorQuantizer floating-point QDQ currently supports FP8 qtypes only");
        }

        if constexpr (std::is_same_v<T, float>)
        {
            if (getNumel(_shape) == 1)
            {
                quantizeDequantizeFp8(input, getNumel(tensorShape), _encodings[0], output, _qtype.floatSpec(), mode,
                                      stream, runner);
            }
            else
            {
                quantizeDequantizeFp8Broadcast(input, output, _encodings, _qtype.floatSpec(), tensorShape, _shape,
                                                mode, stream, runner);
            }
        }
        else
        {
            throw std::runtime_error("BlockTensorQuantizer FP8 QDQ currently supports float tensors only");
        }
        return;
    }

    if (getNumel(_shape) == 1)
    {
        // More efficient per-tensor quantization impl which avoids separate cudaMemcpy for encodings
        DlQuantization::quantizeDequantize(input, getNumel(tensorShape), _encodings[0], output, mode, ROUND_NEAREST,
                                           stream, runner);
    }
    else
    {
        quantizeDequantizeBroadcast(input, output, _encodings, tensorShape, this->_shape, mode, stream, runner);
    }
}

template void BlockTensorQuantizer::quantizeDequantize<float>(const float*, float*, const TensorDims&, bool, void*,
                                                              IForLoopRunner*) const;
template void BlockTensorQuantizer::quantizeDequantize<Eigen::half>(const Eigen::half*, Eigen::half*, const TensorDims&,
                                                                    bool, void*, IForLoopRunner*) const;
template void BlockTensorQuantizer::quantizeDequantize<Eigen::bfloat16>(const Eigen::bfloat16*, Eigen::bfloat16*,
                                                                        const TensorDims&, bool, void*,
                                                                        IForLoopRunner*) const;

void BlockTensorQuantizer::setQuantScheme(QuantizationMode quantScheme)
{
    _quantScheme = quantScheme;
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

    if (_qtype.isFloat())
    {
        if (_qtype.bitwidth() != 8)
        {
            throw std::runtime_error("BlockTensorQuantizer floating-point encodings currently support FP8 qtypes only");
        }

        Encodings encodings = _encodingAnalyzer->computeEncoding(bitwidth, useSymmetricEncodings, _useStrictSymmetric,
                                                                 _useUnsignedSymmetric, _zeroPointShift);

        for (TfEncoding& encoding: encodings)
        {
            // Interim FP8 path: derive scale from the analyzer-produced range until raw amax stats are exposed.
            const double amax  = std::max(std::abs(encoding.min), std::abs(encoding.max));
            const double scale = computeFp8Scale(amax, _qtype.floatSpec());

            // For FP8, delta stores scale and offset is unused; min/max describe the scaled FP8 range.
            encoding.delta  = scale;
            encoding.offset = 0.0;
            encoding.bw     = _qtype.bitwidth();
            encoding.min    = -scale * _qtype.floatSpec().maxValue;
            encoding.max    = scale * _qtype.floatSpec().maxValue;
        }

        return encodings;
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
    // Keep qtype as construction-time configuration; encoding.bw only updates the legacy bitwidth field.
}

}   // namespace DlQuantization

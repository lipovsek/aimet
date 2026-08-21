// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "AimetOpUtils.h"
#include "QcQuantizeInfo.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

using namespace DlQuantization;

namespace
{

// Reference utilities used to validate FP8 QDQ values without reaching into production internals.
// The production fake-cast helper is private to trim_functions.cpp, so the tests keep an independent
// reference implementation here and compare observable runtime outputs against it.
float roundHalfToEvenReference(float value)
{
    const float lower      = std::floor(value);
    const float fractional = value - lower;

    if (fractional < 0.5f)
    {
        return lower;
    }
    if (fractional > 0.5f)
    {
        return lower + 1.0f;
    }

    return std::fmod(std::fabs(lower), 2.0f) == 0.0f ? lower : lower + 1.0f;
}

float fakeCastToFp8Reference(float value, const FloatQuantizationSpec& fp8Spec)
{
    const float maxValue     = static_cast<float>(fp8Spec.maxValue);
    const float exponentMin  = static_cast<float>(fp8Spec.exponentMin);
    const float mantissaBits = static_cast<float>(fp8Spec.mantissaBits);

    value = std::clamp(value, -maxValue, maxValue);

    float exponent = std::floor(std::log2(std::fabs(value)));
    if (exponent < exponentMin)
    {
        exponent = exponentMin;
    }

    const float step = std::exp2(exponent - mantissaBits);
    float       out  = roundHalfToEvenReference(value / step) * step;
    return std::clamp(out, -maxValue, maxValue);
}

float fp8QdqReference(float value, double scale, const FloatQuantizationSpec& fp8Spec)
{
    return fakeCastToFp8Reference(value / static_cast<float>(scale), fp8Spec) * static_cast<float>(scale);
}

float relativeError(float actual, float expected)
{
    return std::fabs(actual - expected) / std::max(std::fabs(expected), std::numeric_limits<float>::epsilon());
}

TfEncoding makeFp8Encoding(double scale, const FloatQuantizationSpec& fp8Spec)
{
    TfEncoding encoding;
    encoding.bw     = fp8Spec.bitwidth;
    encoding.delta  = scale;
    encoding.offset = 0.0;
    encoding.min    = -scale * fp8Spec.maxValue;
    encoding.max    = scale * fp8Spec.maxValue;
    return encoding;
}

// Runs chunks in reverse order on the calling thread. Reverse order proves each chunk
// recovers its own starting state rather than relying on the previous chunk having run,
// and recording numChunks proves the kernel actually dispatched through the runner.
class ReverseForLoopRunner : public IForLoopRunner
{
public:
    void run(std::function<void(size_t)> fn, size_t numChunks) const override
    {
        _numChunks = numChunks;
        for (size_t chunkId = numChunks; chunkId > 0; --chunkId)
        {
            fn(chunkId - 1);
        }
    }

    size_t numChunks() const
    {
        return _numChunks;
    }

private:
    mutable size_t _numChunks = 0;
};

size_t getEncodingIndexForShape(const TensorDims& inputShape, const TensorDims& encodingShape, size_t flatIndex)
{
    TensorDims inputStrides(inputShape.size());
    TensorDims encStrides(encodingShape.size());
    int64_t    inputStride = 1;
    int64_t    encStride   = 1;
    for (int dim = static_cast<int>(inputShape.size()) - 1; dim >= 0; --dim)
    {
        inputStrides[dim] = inputStride;
        encStrides[dim]   = encStride;
        inputStride *= inputShape[dim];
        encStride *= encodingShape[dim];
    }

    size_t encodingIdx = 0;
    size_t remainder   = flatIndex;
    for (size_t dim = 0; dim < inputShape.size(); ++dim)
    {
        const size_t coord = remainder / inputStrides[dim];
        remainder -= coord * inputStrides[dim];
        if (encodingShape[dim] != 1)
        {
            encodingIdx += coord * encStrides[dim];
        }
    }
    return encodingIdx;
}

}   // namespace

// Runtime path selection
//
// These tests focus on the C++ dispatch decision in QcQuantizeOp. FP8 is represented as a
// floating-point qtype, but unlike float16 it should still use BlockTensorQuantizer because the
// qtype carries FP8 format metadata needed by computeEncodings and QDQ.
TEST(TestFp8QcQuantizeRuntime, Fp8QtypeUsesTensorQuantizerPath)
{
    QcQuantizeInfo quantInfo;
    quantInfo.isIntDataType  = false;
    quantInfo.tensorQuantizer = std::make_shared<BlockTensorQuantizer>(TensorDims {}, QuantizationType::Fp8E4M3FN(),
                                                                       QUANTIZATION_TF);

    EXPECT_TRUE(usesTensorQuantizerPath(&quantInfo));
}

// This protects the current Python/Cython behavior, where some quantizers are still constructed
// from bitwidth only. If isIntDataType is false and the qtype is not a true floating-point qtype,
// the op should continue using the existing float path.
TEST(TestFp8QcQuantizeRuntime, Float16StyleQuantInfoDoesNotUseTensorQuantizerPath)
{
    QcQuantizeInfo quantInfo;
    quantInfo.isIntDataType  = false;
    quantInfo.tensorQuantizer = std::make_shared<BlockTensorQuantizer>(TensorDims {}, 8, QUANTIZATION_TF);

    EXPECT_FALSE(usesTensorQuantizerPath(&quantInfo));
}

// Future qtype-aware float16 should also stay on the existing float16 path. The tensor quantizer
// path is reserved for lower-precision float qtypes such as FP8.
TEST(TestFp8QcQuantizeRuntime, Float16QtypeDoesNotUseTensorQuantizerPath)
{
    QcQuantizeInfo quantInfo;
    quantInfo.isIntDataType = false;
    quantInfo.tensorQuantizer =
        std::make_shared<BlockTensorQuantizer>(TensorDims {},
                                               QuantizationType::Float(/*bitwidth=*/16, /*exponentBits=*/5,
                                                                       /*mantissaBits=*/10, /*exponentMin=*/-14,
                                                                       /*maxValue=*/65504.0, /*finite=*/false,
                                                                       /*unsignedZero=*/false),
                                               QUANTIZATION_TF);

    EXPECT_FALSE(usesTensorQuantizerPath(&quantInfo));
}

// Per-tensor FP8 QDQ
//
// This test installs an explicit encoding so it isolates the FP8 QDQ backend from calibration.
// It validates the clamp, round, and dequantize behavior for representative values.
TEST(TestFp8QcQuantizeRuntime, Fp8QuantizeDequantizeMatchesReferenceForProvidedEncoding)
{
    const QuantizationType qtype = QuantizationType::Fp8E4M3FN();
    BlockTensorQuantizer tensorQuantizer(TensorDims {}, qtype, QUANTIZATION_TF);
    tensorQuantizer.setEncodings({makeFp8Encoding(/*scale=*/1.0, qtype.floatSpec())});

    std::vector<float> input {
        -500.0f, -3.2f, -3.1f, -1.07f, -1.06f, 0.0f, 1.06f, 1.07f, 3.1f, 3.2f, 500.0f};
    std::vector<float> output(input.size());

    tensorQuantizer.quantizeDequantize(input.data(), output.data(), TensorDims {static_cast<int64_t>(input.size())},
                                       false);

    for (size_t idx = 0; idx < input.size(); ++idx)
    {
        EXPECT_FLOAT_EQ(output[idx], fp8QdqReference(input[idx], /*scale=*/1.0, qtype.floatSpec()));
    }
}

// Per-tensor FP8 QDQ for the e5m2 grid
//
// e5m2 has a wider exponent range and fewer mantissa bits than e4m3, so it pins the
// exponent/step derivation in the fake-cast against the reference implementation.
TEST(TestFp8QcQuantizeRuntime, Fp8E5M2QuantizeDequantizeMatchesReference)
{
    const QuantizationType qtype = QuantizationType::Fp8E5M2();
    BlockTensorQuantizer tensorQuantizer(TensorDims {}, qtype, QUANTIZATION_TF);
    tensorQuantizer.setEncodings({makeFp8Encoding(/*scale=*/0.5, qtype.floatSpec())});

    std::vector<float> input {-60000.0f, -3.2f, -0.1f, 0.0f, 0.1f, 3.2f, 60000.0f};
    std::vector<float> output(input.size());

    tensorQuantizer.quantizeDequantize(input.data(), output.data(), TensorDims {static_cast<int64_t>(input.size())},
                                       false);

    for (size_t idx = 0; idx < input.size(); ++idx)
    {
        EXPECT_FLOAT_EQ(output[idx], fp8QdqReference(input[idx], /*scale=*/0.5, qtype.floatSpec()));
    }
}

// Per-tensor one-shot runtime flow
//
// This exercises the same high-level state machine used by QcQuantizeOp in one-shot mode:
// reset stats, update stats, compute encodings, set encodings, then run QDQ.
TEST(TestFp8QcQuantizeRuntime, OneShotFp8RunsThroughTensorQuantizerStateMachine)
{
    QcQuantizeInfo quantInfo;
    quantInfo.opMode               = TensorQuantizerOpMode::oneShotQuantizeDequantize;
    quantInfo.useSymmetricEncoding = false;
    quantInfo.enabled              = true;
    quantInfo.isIntDataType        = false;
    quantInfo.tensorQuantizer      = std::make_shared<BlockTensorQuantizer>(TensorDims {}, QuantizationType::Fp8E4M3FN(),
                                                                            QUANTIZATION_TF);

    std::vector<float> input {-3.0f, 1.5f, 7.25f, -7.25f, 0.0f};
    std::vector<float> output(input.size());
    std::vector<int64_t> inputShape {static_cast<int64_t>(input.size())};

    modeSpecificActionTensorQuantizer(input.data(), output.data(), inputShape, quantInfo.tensorQuantizer.get(),
                                      quantInfo.opMode, quantInfo.useSymmetricEncoding, nullptr, false, nullptr);

    EXPECT_TRUE(quantInfo.tensorQuantizer->isEncodingValid);

    const Encodings encodings = quantInfo.tensorQuantizer->getEncodings();
    ASSERT_EQ(encodings.size(), 1);
    EXPECT_EQ(encodings[0].bw, 8);
    EXPECT_GT(encodings[0].delta, 0.0);
    EXPECT_DOUBLE_EQ(encodings[0].offset, 0.0);

    EXPECT_NE(output[0], input[0]);
    EXPECT_NE(output[1], input[1]);
    EXPECT_FLOAT_EQ(output[2], static_cast<float>(encodings[0].max));
    EXPECT_FLOAT_EQ(output[3], static_cast<float>(encodings[0].min));
    EXPECT_FLOAT_EQ(output[4], 0.0f);

    const double rawAmax  = 7.25;
    const double rawScale = rawAmax / QuantizationType::Fp8E4M3FN().floatSpec().maxValue;
    EXPECT_DOUBLE_EQ(encodings[0].delta, rawScale);
    float maxAbsError = 0.0f;
    float maxRelError = 0.0f;
    for (size_t idx = 0; idx < input.size(); ++idx)
    {
        const float expected = fp8QdqReference(input[idx], rawScale, QuantizationType::Fp8E4M3FN().floatSpec());
        maxAbsError          = std::max(maxAbsError, std::fabs(output[idx] - expected));
        maxRelError          = std::max(maxRelError, relativeError(output[idx], expected));
    }
    EXPECT_LT(maxAbsError, 0.05f);
    EXPECT_LT(maxRelError, 0.05f);
}

// Broadcast FP8 QDQ with explicit encodings
//
// Multiple encodings are installed directly to validate broadcast indexing and per-encoding FP8
// scales. This bypasses computeEncodings, so failures here point at broadcast QDQ rather than
// calibration behavior.
TEST(TestFp8QcQuantizeRuntime, BroadcastFp8QuantizeDequantizeMatchesReferenceForProvidedEncodings)
{
    const QuantizationType qtype         = QuantizationType::Fp8E4M3FN();
    const TensorDims      inputShape     = {2, 2, 2, 2};
    const TensorDims      quantizerShape = {2, 1, 1, 2};

    BlockTensorQuantizer tensorQuantizer(quantizerShape, qtype, QUANTIZATION_TF);

    const std::vector<double> scales {1.0, 0.5, 2.0, 4.0};
    Encodings                 encodings;
    for (double scale: scales)
    {
        encodings.push_back(makeFp8Encoding(scale, qtype.floatSpec()));
    }
    tensorQuantizer.setEncodings(encodings);

    std::vector<float> input {
        -500.0f, -250.0f, -3.2f, -1.07f,
        -1.06f, 0.0f,    1.06f, 1.07f,
        3.1f,   3.2f,    15.5f, 16.5f,
        200.0f, 300.0f,  700.0f, 2000.0f,
    };
    std::vector<float> output(input.size());

    modeSpecificActionTensorQuantizer(input.data(), output.data(), inputShape, &tensorQuantizer,
                                      TensorQuantizerOpMode::quantizeDequantize, false, nullptr, false, nullptr);

    for (size_t idx = 0; idx < input.size(); ++idx)
    {
        const size_t encodingIdx = getEncodingIndexForShape(inputShape, quantizerShape, idx);
        EXPECT_FLOAT_EQ(output[idx], fp8QdqReference(input[idx], scales[encodingIdx], qtype.floatSpec()));
    }
}

// Per-tensor FP8 QDQ dispatched through a for-loop runner
//
// Uses a tensor larger than MIN_ELEMENTS_PER_CHUNK with a deliberately non-chunk-aligned
// size, so results must match the reference regardless of how the range is split.
TEST(TestFp8QcQuantizeRuntime, Fp8PerTensorQdqUsesForLoopRunner)
{
    const QuantizationType qtype = QuantizationType::Fp8E4M3FN();
    BlockTensorQuantizer tensorQuantizer(TensorDims {}, qtype, QUANTIZATION_TF);
    tensorQuantizer.setEncodings({makeFp8Encoding(/*scale=*/0.25, qtype.floatSpec())});

    std::vector<float> input(4097);
    std::vector<float> output(input.size());
    for (size_t idx = 0; idx < input.size(); ++idx)
    {
        input[idx] = static_cast<float>(static_cast<int>(idx % 257) - 128) / 7.0f;
    }

    ReverseForLoopRunner runner;
    tensorQuantizer.quantizeDequantize(input.data(), output.data(), TensorDims {static_cast<int64_t>(input.size())},
                                       false, nullptr, &runner);

    EXPECT_GT(runner.numChunks(), 1);
    for (size_t idx = 0; idx < input.size(); ++idx)
    {
        EXPECT_FLOAT_EQ(output[idx], fp8QdqReference(input[idx], /*scale=*/0.25, qtype.floatSpec()));
    }
}

// Broadcast FP8 QDQ dispatched through a for-loop runner
//
// The broadcast kernel walks a coordinate counter to track which encoding applies to each
// element. Under chunking each chunk must reconstruct that counter from its own starting
// offset, so this checks per-element encoding selection against the reference while chunks
// run out of order.
TEST(TestFp8QcQuantizeRuntime, BroadcastFp8QdqUsesForLoopRunner)
{
    const QuantizationType qtype         = QuantizationType::Fp8E4M3FN();
    const TensorDims      inputShape     = {4, 3, 257};
    const TensorDims      quantizerShape = {4, 1, 1};

    BlockTensorQuantizer tensorQuantizer(quantizerShape, qtype, QUANTIZATION_TF);
    // Deliberately not powers of two. The FP8 grid is self-similar under power-of-two
    // scaling, so scales like {0.125, 0.25, 0.5} produce identical output and would hide
    // a chunk that selected the wrong encoding.
    const std::vector<double> scales {0.1, 0.35, 0.7, 1.3};
    Encodings encodings;
    for (double scale: scales)
    {
        encodings.push_back(makeFp8Encoding(scale, qtype.floatSpec()));
    }
    tensorQuantizer.setEncodings(encodings);

    const size_t numElements = 4 * 3 * 257;
    std::vector<float> input(numElements);
    std::vector<float> output(numElements);
    for (size_t idx = 0; idx < input.size(); ++idx)
    {
        input[idx] = static_cast<float>(static_cast<int>(idx % 389) - 194) / 9.0f;
    }

    ReverseForLoopRunner runner;
    tensorQuantizer.quantizeDequantize(input.data(), output.data(), inputShape, false, nullptr, &runner);

    EXPECT_GT(runner.numChunks(), 1);
    for (size_t idx = 0; idx < input.size(); ++idx)
    {
        const size_t encodingIdx = getEncodingIndexForShape(inputShape, quantizerShape, idx);
        EXPECT_FLOAT_EQ(output[idx], fp8QdqReference(input[idx], scales[encodingIdx], qtype.floatSpec()));
    }
}

// Broadcast one-shot runtime flow
//
// This covers the full broadcast path, including computeEncodings producing one FP8 scale per
// quantizer block and QDQ selecting the matching encoding for each input element.
TEST(TestFp8QcQuantizeRuntime, OneShotBroadcastFp8RunsThroughComputeEncodingsAndQdq)
{
    const QuantizationType qtype         = QuantizationType::Fp8E4M3FN();
    const TensorDims      inputShape     = {2, 2, 2, 2};
    const TensorDims      quantizerShape = {2, 1, 1, 2};

    BlockTensorQuantizer tensorQuantizer(quantizerShape, qtype, QUANTIZATION_TF);

    std::vector<float> input {
        -7.0f, -2.0f, -1.0f, -4.0f,
        -6.0f, 2.0f,  1.0f,  4.0f,
        -3.0f, 8.0f,  -5.0f, 16.0f,
        3.0f,  7.5f,  5.0f,  15.0f,
    };
    std::vector<float> output(input.size());

    modeSpecificActionTensorQuantizer(input.data(), output.data(), inputShape, &tensorQuantizer,
                                      TensorQuantizerOpMode::oneShotQuantizeDequantize, false, nullptr, false,
                                      nullptr);

    EXPECT_TRUE(tensorQuantizer.isEncodingValid);

    const Encodings encodings = tensorQuantizer.getEncodings();
    ASSERT_EQ(encodings.size(), 4);

    std::vector<double> rawAmaxByEncoding(encodings.size(), 0.0);
    for (size_t idx = 0; idx < input.size(); ++idx)
    {
        const size_t encodingIdx = getEncodingIndexForShape(inputShape, quantizerShape, idx);
        rawAmaxByEncoding[encodingIdx] =
            std::max(rawAmaxByEncoding[encodingIdx], static_cast<double>(std::fabs(input[idx])));
    }

    for (const TfEncoding& encoding: encodings)
    {
        EXPECT_EQ(encoding.bw, 8);
        EXPECT_GT(encoding.delta, 0.0);
        EXPECT_DOUBLE_EQ(encoding.offset, 0.0);
    }

    for (size_t encodingIdx = 0; encodingIdx < encodings.size(); ++encodingIdx)
    {
        const double rawScale = rawAmaxByEncoding[encodingIdx] / qtype.floatSpec().maxValue;
        EXPECT_DOUBLE_EQ(encodings[encodingIdx].delta, rawScale);
    }

    float maxAbsError = 0.0f;
    float maxRelError = 0.0f;
    for (size_t idx = 0; idx < input.size(); ++idx)
    {
        const size_t encodingIdx = getEncodingIndexForShape(inputShape, quantizerShape, idx);
        const double rawScale   = rawAmaxByEncoding[encodingIdx] / qtype.floatSpec().maxValue;
        const float  expected   = fp8QdqReference(input[idx], rawScale, qtype.floatSpec());
        maxAbsError             = std::max(maxAbsError, std::fabs(output[idx] - expected));
        maxRelError             = std::max(maxRelError, relativeError(output[idx], expected));
    }

    EXPECT_LT(maxAbsError, 0.1f);
    EXPECT_LT(maxRelError, 0.1f);

    // Also verify exact consistency with the encodings that were actually stored on the quantizer.
    for (size_t idx = 0; idx < input.size(); ++idx)
    {
        const size_t encodingIdx = getEncodingIndexForShape(inputShape, quantizerShape, idx);
        const float  expected   = fp8QdqReference(input[idx], encodings[encodingIdx].delta, qtype.floatSpec());
        EXPECT_FLOAT_EQ(output[idx], expected);
    }
}

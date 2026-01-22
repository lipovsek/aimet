// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include <random>

#include "DlQuantization/TensorQuantizer.h"
#include "test_quantization_lib.hpp"

using namespace DlQuantization;

class TestTensorQuantizer : public ::testing::Test
{
protected:
    std::vector<float> data1, data2, data3, data4;
    std::vector<uint32_t> shape1, shape2, shape3;
    std::unique_ptr<TensorQuantizer> enhancedTensorQuant;
    std::unique_ptr<TensorQuantizer> tfTensorQuant;

    void SetUp()
    {
        if (data1.size() == 0)
        {
            data1.resize(24);
            std::iota(std::begin(data1), std::end(data1), 0);
            shape1 = {2, 3, 2, 2};
        }

        if (data2.size() == 0)
        {
            data2.resize(60);
            float t = -15;
            for (uint32_t i = 0; i < data2.size(); ++i)
            {
                data2[i] = t;
                t += 0.5;
            }
            shape2 = {1, 4, 5, 3};
        }

        if (data3.size() == 0)
        {
            shape3 = {2, 5, 4, 1};
            data3.resize(40);
            std::mt19937 eng;
            std::normal_distribution<float> dist;
            for (auto& d: data3)
            {
                d = dist(eng);
            }
            std::iota(std::begin(data1), std::end(data1), 0);
        }

        if (data4.size() == 0)
        {
            float mean   = 2;
            float stddev = 2;
            std::normal_distribution<float> distribution(mean, stddev);
            std::mt19937 generator(1);

            unsigned int tensorCount = 6000;
            data4.resize(tensorCount);

            for (unsigned int i = 0; i < tensorCount; i++)
            {
                data4[i] = distribution(generator);
            }
        }

        enhancedTensorQuant.reset(new TensorQuantizer(QuantizationMode::QUANTIZATION_TF_ENHANCED, ROUND_NEAREST));
        tfTensorQuant.reset(new TensorQuantizer(QuantizationMode::QUANTIZATION_TF, ROUND_NEAREST));
    }
};

TEST_F(TestTensorQuantizer, SanityTestCpu)
{
    enhancedTensorQuant->setStrictSymmetric(false);
    enhancedTensorQuant->setUnsignedSymmetric(false);
    enhancedTensorQuant->updateStats(data4.data(), data4.size(), false);
    EXPECT_FALSE(enhancedTensorQuant->isEncodingValid);
    TfEncoding encoding = enhancedTensorQuant->computeEncoding(8, false);
    EXPECT_TRUE(enhancedTensorQuant->isEncodingValid);

    std::vector<float> inputTensor(data4.size(), 5);
    std::vector<float> quantizedTensor(data4.size());

    enhancedTensorQuant->quantizeDequantize(inputTensor.data(), inputTensor.size(), quantizedTensor.data(),
                                            encoding.min, encoding.max, 8, false);

    double minVal = 0;
    double maxVal = 0;
    for (auto x : data4)
    {
        minVal = std::min(minVal, static_cast<double>(x));
        maxVal = std::max(maxVal, static_cast<double>(x));
    }

    size_t PDF_SIZE = 512 * 3;
    double HISTOGRAM_BUCKET_SIZE = 3 * (maxVal - minVal) / PDF_SIZE;
    // Allow for worst case 2 x HISTOGRAM_BUCKET_SIZE error on (maxVal - minVal)
    EXPECT_NEAR(encoding.delta, (maxVal - minVal) / 255, 2 * HISTOGRAM_BUCKET_SIZE / 255);
    // Allow for worst case 1 x HISTOGRAM_BUCKET_SIZE error on minValue and 0.5 x delta from rounding offset
    EXPECT_NEAR(encoding.min, minVal, HISTOGRAM_BUCKET_SIZE + 0.5 * encoding.delta);
    EXPECT_NEAR(encoding.max, maxVal, HISTOGRAM_BUCKET_SIZE + 0.5 * encoding.delta);

    EXPECT_NE(inputTensor.data()[0], quantizedTensor.data()[0]);
    EXPECT_NEAR(quantizedTensor.data()[0], inputTensor.data()[0], 0.5 * encoding.delta);
}

TEST_F(TestTensorQuantizer, SanityTestComputeEncodingFromDataAsymmetricTFEnhanced)
{
    auto paramTensor = this->data4;
    auto tensorCount = paramTensor.size();
    TfEncoding encoding {};
    enhancedTensorQuant->computeEncodingFromData(8, paramTensor.data(), tensorCount, encoding,
                                                 ComputationMode::COMP_MODE_CPU, false, false, false);
    double MAX = *std::max_element(data4.begin(), data4.end());
    double MIN = *std::min_element(data4.begin(), data4.end());
    size_t PDF_SIZE = 512;
    double HISTOGRAM_BUCKET_SIZE = 3 * (MAX - MIN) / PDF_SIZE;
    EXPECT_NEAR(encoding.min, MIN, HISTOGRAM_BUCKET_SIZE + 0.5 * encoding.delta);
    EXPECT_NEAR(encoding.max, MAX, HISTOGRAM_BUCKET_SIZE + 0.5 * encoding.delta);
}

TEST_F(TestTensorQuantizer, SanityTestComputeEncodingFromDataSymmetricTF)
{
    auto paramTensor = this->data4;
    auto tensorCount = paramTensor.size();
    TfEncoding encoding {};
    tfTensorQuant->computeEncodingFromData(8, paramTensor.data(), tensorCount, encoding, ComputationMode::COMP_MODE_CPU,
                                           true, false, true);

    float expected_max = std::max(std::abs(*std::min_element(paramTensor.begin(), paramTensor.end())),
                                  std::abs(*std::max_element(paramTensor.begin(), paramTensor.end())));

    // Min and Max will get adjusted slightly to represent an exact zero with one of the quantized values
    // Adjustment is expected to be less than half a delta worth
    EXPECT_NEAR(encoding.max, expected_max, encoding.delta / 2 + 1e-4);
    EXPECT_EQ(encoding.max, -encoding.min);

    // Check that the center value is absolute 0
    EXPECT_NEAR(encoding.min + encoding.delta * (-encoding.offset), 0, 1e-7);

    EXPECT_FLOAT_EQ(encoding.delta, (encoding.max - encoding.min) / 254);

    // Check that offset is -127 - another check for strict symmetric encodings
    EXPECT_NEAR(encoding.offset, -127, 0);
    EXPECT_EQ(encoding.bw, 8);
}

// Check that existing tests pass with BlockTensorQuantizer
TEST_F(TestTensorQuantizer, SanityTestComputeEncodingFromDataSymmetricTFBlocked)
{
    auto paramTensor = this->data4;
    BlockTensorQuantizer tensorQuantizer({}, 8, QUANTIZATION_TF);
    tensorQuantizer.setStrictSymmetric(true);
    TensorDims tensorShape = {(TensorDim) paramTensor.size()};
    ASSERT_FALSE(tensorQuantizer.hasValidStats());
    tensorQuantizer.updateStats(paramTensor.data(), tensorShape, false);\
    ASSERT_TRUE(tensorQuantizer.hasValidStats());
    ASSERT_FALSE(tensorQuantizer.isEncodingValid);
    auto encoding = tensorQuantizer.computeEncodings(true)[0];

    float expected_max = std::max(std::abs(*std::min_element(paramTensor.begin(), paramTensor.end())),
                                  std::abs(*std::max_element(paramTensor.begin(), paramTensor.end())));

    // Min and Max will get adjusted slightly to represent an exact zero with one of the quantized values
    // Adjustment is expected to be less than half a delta worth
    EXPECT_NEAR(encoding.max, expected_max, encoding.delta / 2 + 1e-4);
    EXPECT_EQ(encoding.max, -encoding.min);

    // Check that the center value is absolute 0
    EXPECT_NEAR(encoding.min + encoding.delta * (-encoding.offset), 0, 1e-7);

    EXPECT_FLOAT_EQ(encoding.delta, (encoding.max - encoding.min) / 254);

    // Check that offset is -127 - another check for strict symmetric encodings
    EXPECT_NEAR(encoding.offset, -127, 0);
    EXPECT_EQ(encoding.bw, 8);
}

TEST_F(TestTensorQuantizer, SanityTestComputePartialEncodingDeltaOffsetSymmetricTF)
{
    auto paramTensor = this->data2;


    TfEncoding encoding {0, 0, 0, 0, 8};
    auto expected_encoding = getTfSymmetricEncoding(15.0, 8);

    // set up partial to mimic user provided scale and offset
    encoding.delta  = expected_encoding.delta;
    encoding.offset = expected_encoding.offset;

    tfTensorQuant->computePartialEncoding(8, encoding, true, false, true);


    // Expect the min and max values to be unchanged
    EXPECT_NEAR(encoding.max, expected_encoding.max, 0.001);
    EXPECT_NEAR(encoding.min, expected_encoding.min, 0.001);

    EXPECT_FLOAT_EQ(encoding.delta, expected_encoding.delta);
    EXPECT_FLOAT_EQ(encoding.offset, expected_encoding.offset);

    EXPECT_EQ(encoding.bw, 8);
}

TEST_F(TestTensorQuantizer, SanityTestComputePartialEncodingDeltaOffsetAsymmetricTF)
{
    auto paramTensor = this->data1;


    TfEncoding encoding {0, 0, 0, 0, 8};
    auto expected_encoding = getTfEncoding(0.0, 24.0, 8);

    // set up partial to mimic user provided scale and offset
    encoding.delta  = expected_encoding.delta;
    encoding.offset = expected_encoding.offset;

    tfTensorQuant->computePartialEncoding(8, encoding, false, false, false);


    // Expect the min and max values to be unchanged
    EXPECT_NEAR(encoding.max, expected_encoding.max, 0.001);
    EXPECT_NEAR(encoding.min, expected_encoding.min, 0.001);

    EXPECT_FLOAT_EQ(encoding.delta, expected_encoding.delta);
    EXPECT_FLOAT_EQ(encoding.offset, expected_encoding.offset);

    EXPECT_EQ(encoding.bw, 8);
}

TEST_F(TestTensorQuantizer, SanityTestComputePartialEncodingMinMaxSymmetricTF)
{
    auto paramTensor = this->data2;


    float expected_max = std::max(std::abs(*std::min_element(paramTensor.begin(), paramTensor.end())),
                                  std::abs(*std::max_element(paramTensor.begin(), paramTensor.end())));

    TfEncoding encoding {-expected_max, expected_max, 0, 0, 8};
    tfTensorQuant->computePartialEncoding(8, encoding, true, false, true);


    // Expect the min and max values to be unchanged
    EXPECT_EQ(encoding.max, expected_max);
    EXPECT_EQ(encoding.max, -encoding.min);

    // Check that the center value is absolute 0
    EXPECT_NEAR(encoding.min + encoding.delta * (-encoding.offset), 0, 1e-7);

    EXPECT_FLOAT_EQ(encoding.delta, (encoding.max - encoding.min) / 254);

    // Check that offset is -127 - another check for strict symmetric encodings
    EXPECT_NEAR(encoding.offset, -127, 0);
    EXPECT_EQ(encoding.bw, 8);
}

TEST_F(TestTensorQuantizer, SANITY_GeneratePerChannelParams)
{
    int bw       = 8;
    int32_t axis = 1;

    std::vector<TfEncoding> encodings;
    std::vector<uint32_t> splitShape;
    std::vector<std::vector<float>> splitParams;

    TensorQuantizer tensorQuantizer(QuantizationMode::QUANTIZATION_TF, ROUND_NEAREST);
    tensorQuantizer.generatePerChannelEncodings(data1.data(), shape1, axis, encodings, bw, splitParams, splitShape,
                                                false);

    ASSERT_EQ(encodings.size(), shape1[axis]);
    ASSERT_EQ(splitParams.size(), shape1[axis]);

    std::vector<uint32_t> expectedOutputShape = {2, 1, 2, 2};
    ASSERT_EQ(splitShape, expectedOutputShape);

    std::vector<std::vector<float>> expectedOutputData(3);
    expectedOutputData[0] = {0, 1, 2, 3, 12, 13, 14, 15};
    expectedOutputData[1] = {4, 5, 6, 7, 16, 17, 18, 19};
    expectedOutputData[2] = {8, 9, 10, 11, 20, 21, 22, 23};

    std::vector<TfEncoding> expectedEncodings(3);
    expectedEncodings[0] = getTfEncoding(0, 15, 8);
    expectedEncodings[1] = getTfEncoding(0, 19, 8);
    expectedEncodings[2] = getTfEncoding(0, 23, 8);

    ASSERT_EQ(splitParams.size(), expectedOutputData.size());
    ASSERT_EQ(splitParams[0], expectedOutputData[0]);
    ASSERT_EQ(splitParams[1], expectedOutputData[1]);
    ASSERT_EQ(splitParams[2], expectedOutputData[2]);

    for (uint32_t i = 0; i < encodings.size(); ++i)
    {
        EXPECT_TRUE(compareEncodings(encodings[i], expectedEncodings[i]));
    }
}


TEST_F(TestTensorQuantizer, SANITY_GeneratePerChannelParamsBlocked)
{
    BlockTensorQuantizer tensorQuantizer({3, 1, 1}, 8, QUANTIZATION_TF);
    tensorQuantizer.updateStats(data1.data(), {2, 3, 2, 2}, false);
    auto encodings = tensorQuantizer.computeEncodings(false);

    std::vector<TfEncoding> expectedEncodings(3);
    expectedEncodings[0] = getTfEncoding(0, 15, 8);
    expectedEncodings[1] = getTfEncoding(0, 19, 8);
    expectedEncodings[2] = getTfEncoding(0, 23, 8);

    for (uint32_t i = 0; i < encodings.size(); ++i)
    {
        EXPECT_TRUE(compareEncodings(encodings[i], expectedEncodings[i]));
    }
}


// 1. Quantize some per channel data using QuantizePerChannelParamsPacked()
TEST_F(TestTensorQuantizer, SANITY_QuantizePerChannelTensorPackedAsymmetric)
{
    int bw       = 8;
    int32_t axis = 1;

    std::vector<TfEncoding> expectedEncodings(4);
    expectedEncodings[0]                = getTfEncoding(-15, 0, 8);
    expectedEncodings[1]                = getTfEncoding(-7.5, 0, 8);
    expectedEncodings[2]                = getTfEncoding(0, 7, 8);
    expectedEncodings[3]                = getTfEncoding(0, 14.5, 8);
    std::vector<uint8_t> expectedParams = {0,   9,   17,  26,  34,  43,  51,  60,  68,  77,  85,  94,  102, 111, 119,
                                           0,   17,  34,  51,  68,  85,  102, 119, 136, 153, 170, 187, 204, 221, 238,
                                           0,   18,  36,  55,  73,  91,  109, 128, 146, 164, 182, 200, 219, 237, 255,
                                           132, 141, 149, 158, 167, 176, 185, 193, 202, 211, 220, 229, 237, 246, 255};


    std::vector<TfEncoding> encodings;
    std::vector<uint8_t> params_quantized(this->data2.size());
    tfTensorQuant->setUnsignedSymmetric(false);
    tfTensorQuant->quantizePerChannelTensorPacked(this->data2.data(), this->shape2, axis, params_quantized, encodings,
                                                  bw, DlQuantization::RoundingMode::ROUND_NEAREST, false, false);

    ASSERT_EQ(encodings.size(), this->shape2[axis]);
    ASSERT_EQ(encodings.size(), expectedEncodings.size());
    for (uint32_t i = 0; i < encodings.size(); ++i)
    {
        EXPECT_TRUE(compareEncodings(encodings[i], expectedEncodings[i]));
    }

    EXPECT_TRUE(compareTensors(params_quantized.data(), expectedParams.data(), this->data2.size()));
}

// 1. Quantize some per channel data using QuantizePerChannelParamsPacked()
TEST_F(TestTensorQuantizer, SANITY_QuantizePerChannelTensorPackedSymmetric)
{
    int bw       = 8;
    int32_t axis = 1;

    std::vector<TfEncoding> expectedEncodings;
    expectedEncodings.emplace_back(getTfSymmetricEncoding(15, 8));
    expectedEncodings.emplace_back(getTfSymmetricEncoding(7.5, 8));
    expectedEncodings.emplace_back(getTfSymmetricEncoding(7, 8));
    expectedEncodings.emplace_back(getTfSymmetricEncoding(14.5, 8));
    std::vector<int8_t> expectedParams = {
        -127, -123, -119, -114, -110, -106, -102, -97, -93, -89, -85, -80, -76, -72, -68, -127, -119, -110, -102, -93,
        -85,  -76,  -68,  -59,  -51,  -42,  -34,  -25, -17, -8,  0,   9,   18,  27,  36,  45,   54,   64,   73,   82,
        91,   100,  109,  118,  127,  66,   70,   74,  79,  83,  88,  92,  96,  101, 105, 109,  114,  118,  123,  127};


    std::vector<TfEncoding> encodings;
    std::vector<uint8_t> params_quantized(this->data2.size());
    tfTensorQuant->setUnsignedSymmetric(false);
    tfTensorQuant->quantizePerChannelTensorPacked(this->data2.data(), this->shape2, axis, params_quantized, encodings,
                                                  bw, DlQuantization::RoundingMode::ROUND_NEAREST, false, true);

    ASSERT_EQ(encodings.size(), this->shape2[axis]);
    ASSERT_EQ(encodings.size(), expectedEncodings.size());
    for (uint32_t i = 0; i < encodings.size(); ++i)
    {
        EXPECT_TRUE(compareEncodings(encodings[i], expectedEncodings[i]));
    }

    EXPECT_TRUE(compareTensors((int8_t*) params_quantized.data(), expectedParams.data(), this->data2.size()));
}

// 1. QuantizeDequantize per channel using Asymmetric mode
TEST_F(TestTensorQuantizer, SANITY_QuantizeDequantizePerChannelTensor)
{
    int bw       = 8;
    int32_t axis = 3;


    std::vector<TfEncoding> expectedEncodings(3);
    expectedEncodings[0] = getTfEncoding(-15, 13.5, 8);
    expectedEncodings[1] = getTfEncoding(-14.5, 14, 8);
    expectedEncodings[2] = getTfEncoding(-14, 14.5, 8);

    std::vector<float> expectedParams = {
        -14.9765, -14.5294, -13.9706, -13.5235, -12.9647, -12.5176, -11.9588, -11.5118, -10.9529, -10.5059,
        -9.94706, -9.5,     -9.05294, -8.49412, -8.04706, -7.48824, -7.04118, -6.48235, -6.03529, -5.47647,
        -5.02941, -4.47059, -4.02353, -3.46471, -3.01765, -2.45882, -2.01176, -1.45294, -1.00588, -0.447059,
        0,        0.447059, 1.00588,  1.45294,  2.01176,  2.45882,  3.01765,  3.46471,  4.02353,  4.47059,
        5.02941,  5.47647,  6.03529,  6.48235,  7.04118,  7.48824,  8.04706,  8.49412,  9.05294,  9.5,
        9.94706,  10.5059,  10.9529,  11.5118,  11.9588,  12.5176,  12.9647,  13.5235,  13.9706,  14.5294};


    std::vector<TfEncoding> encodings;
    std::vector<float> params_quantized(this->data2.size());
    tfTensorQuant->setUnsignedSymmetric(false);
    tfTensorQuant->quantizeDequantizePerChannelTensor(this->data2.data(), this->shape2, axis, params_quantized.data(),
                                                      encodings, bw, DlQuantization::RoundingMode::ROUND_NEAREST, false,
                                                      false);

    ASSERT_EQ(encodings.size(), this->shape2[axis]);
    ASSERT_EQ(encodings.size(), expectedEncodings.size());
    for (uint32_t i = 0; i < encodings.size(); ++i)
    {
        EXPECT_TRUE(compareEncodings(encodings[i], expectedEncodings[i]));
    }

    ASSERT_EQ(params_quantized.size(), expectedParams.size());
    for (uint32_t i = 0; i < expectedParams.size(); ++i)
    {
        EXPECT_NEAR(params_quantized[i], expectedParams[i], 0.001);
        EXPECT_NEAR(params_quantized[i], this->data2[i], 0.06);
    }
}

// 1. QuantizeDequantize per channel using Asymmetric mode
TEST_F(TestTensorQuantizer, SANITY_QuantizeDequantizePerChannelTensorBlocked)
{
    TensorDims tensorShape = {1, 4, 5, 3};
    BlockTensorQuantizer tensorQuantizer({3}, 8, QUANTIZATION_TF);

    std::vector<TfEncoding> expectedEncodings(3);
    expectedEncodings[0] = getTfEncoding(-15, 13.5, 8);
    expectedEncodings[1] = getTfEncoding(-14.5, 14, 8);
    expectedEncodings[2] = getTfEncoding(-14, 14.5, 8);

    std::vector<float> expectedParams = {
        -14.9765, -14.5294, -13.9706, -13.5235, -12.9647, -12.5176, -11.9588, -11.5118, -10.9529, -10.5059,
        -9.94706, -9.5,     -9.05294, -8.49412, -8.04706, -7.48824, -7.04118, -6.48235, -6.03529, -5.47647,
        -5.02941, -4.47059, -4.02353, -3.46471, -3.01765, -2.45882, -2.01176, -1.45294, -1.00588, -0.447059,
        0,        0.447059, 1.00588,  1.45294,  2.01176,  2.45882,  3.01765,  3.46471,  4.02353,  4.47059,
        5.02941,  5.47647,  6.03529,  6.48235,  7.04118,  7.48824,  8.04706,  8.49412,  9.05294,  9.5,
        9.94706,  10.5059,  10.9529,  11.5118,  11.9588,  12.5176,  12.9647,  13.5235,  13.9706,  14.5294};

    tensorQuantizer.updateStats(this->data2.data(), tensorShape, false);
    auto encodings = tensorQuantizer.computeEncodings(false);
    tensorQuantizer.setEncodings(encodings);

    std::vector<float> params_quantized(this->data2.size());
    tensorQuantizer.quantizeDequantize(this->data2.data(), params_quantized.data(), tensorShape, false);

    ASSERT_EQ(encodings.size(), expectedEncodings.size());
    for (uint32_t i = 0; i < encodings.size(); ++i)
    {
        EXPECT_TRUE(compareEncodings(encodings[i], expectedEncodings[i]));
    }

    ASSERT_EQ(params_quantized.size(), expectedParams.size());
    for (uint32_t i = 0; i < expectedParams.size(); ++i)
    {
        EXPECT_NEAR(params_quantized[i], expectedParams[i], 0.001);
        EXPECT_NEAR(params_quantized[i], this->data2[i], 0.06);
    }
}

// 1. QuantizeDequantize per channel using Asymmetric mode
TEST_F(TestTensorQuantizer, SANITY_QuantizeDequantizePerChannelTensorSymmetric)
{
    int bw       = 8;
    int32_t axis = 2;


    std::vector<TfEncoding> expectedEncodings(5);
    expectedEncodings[0] = getTfSymmetricEncoding(15, 8);
    expectedEncodings[1] = getTfSymmetricEncoding(13.5, 8);
    expectedEncodings[2] = getTfSymmetricEncoding(12, 8);
    expectedEncodings[3] = getTfSymmetricEncoding(13, 8);
    expectedEncodings[4] = getTfSymmetricEncoding(14.5, 8);

    std::vector<float> expectedParams = {
        -15,      -14.5276, -14.0551, -13.5,    -12.9685, -12.5433, -12,      -11.5276, -10.9606, -10.5433,
        -10.0315, -9.51968, -9.01968, -8.44882, -7.99213, -7.44094, -6.9685,  -6.49606, -5.95276, -5.52756,
        -4.99606, -4.53543, -3.9685,  -3.49606, -2.9685,  -2.45669, -2.04724, -1.48425, -1.02756, -0.456693,
        0,        0.472441, 0.944882, 1.48819,  2.01969,  2.55118,  3.02362,  3.49606,  3.9685,   4.50394,
        5.01575,  5.52756,  6.05118,  6.50787,  6.96457,  7.55906,  8.0315,   8.50394,  9.03543,  9.46063,
        9.99213,  10.4882,  10.9606,  11.5276,  11.9764,  12.4882,  13,       13.4724,  14.0433,  14.5};


    std::vector<TfEncoding> encodings;
    std::vector<float> params_quantized(this->data2.size());
    tfTensorQuant->setUnsignedSymmetric(false);
    tfTensorQuant->quantizeDequantizePerChannelTensor(this->data2.data(), this->shape2, axis, params_quantized.data(),
                                                      encodings, bw, DlQuantization::RoundingMode::ROUND_NEAREST, false,
                                                      true);

    ASSERT_EQ(encodings.size(), this->shape2[axis]);
    ASSERT_EQ(encodings.size(), expectedEncodings.size());
    for (uint32_t i = 0; i < encodings.size(); ++i)
    {
        EXPECT_TRUE(compareEncodings(encodings[i], expectedEncodings[i]));
    }

    ASSERT_EQ(params_quantized.size(), expectedParams.size());
    for (uint32_t i = 0; i < expectedParams.size(); ++i)
    {
        EXPECT_NEAR(params_quantized[i], expectedParams[i], 0.0001);
        EXPECT_NEAR(params_quantized[i], this->data2[i], 0.06);
    }
}


// test parameter quantization for bw=8, number range: -50...80
// also test encoding
TEST_F(TestTensorQuantizer, SANITY_QuantizeTensorPackedAsymmetric)
{
    // Test data
    float data[]            = {-40, -1, 0, 1, 2, -50, 80};
    uint8_t data_expected[] = {20, 96, 98, 100, 102, 0, 255};
    int cnt                 = 7;
    int bw                  = 8;

    // Do quantization
    TfEncoding encoding;
    std::vector<uint8_t> output(cnt);
    tfTensorQuant->setUnsignedSymmetric(false);
    tfTensorQuant->updateStats(data, cnt, false);
    encoding = tfTensorQuant->computeEncoding(bw, false);
    tfTensorQuant->quantizeTensorPacked(data, cnt, output, encoding.min, encoding.max, bw,
                                        DlQuantization::ROUND_NEAREST, false, false);
    // Check quantized values
    for (int i = 0; i < cnt; ++i)
    {
        EXPECT_EQ(output[i], data_expected[i]);
    }
    // Check encoding
    TfEncoding encoding_expected = getTfEncoding(-50, 80, bw);
    EXPECT_TRUE(compareEncodings(encoding_expected, encoding));
}


// 1. Quantize some data using QuantizeParamsPacked()
// 2. Dequantize the result using DeQuantize()
// The final result will be close to the original data
// Test both the vector and pointer API versions of QuantizeParamsPacked() and
// DeQuantize().
TEST_F(TestTensorQuantizer, SANITY_Dequantize)
{
    // Unquantized data
    float data_unquantized[] = {-40, -1, 0, 1, 2, -50, 80};
    // Expected quantized data
    uint8_t data_quantized_expected[] = {20, 96, 98, 100, 102, 0, 255};
    int cnt                           = 7;
    int bw                            = 8;
    // Do quantization
    TfEncoding encoding;
    std::vector<uint8_t> data_quantized(cnt);

    tfTensorQuant->setUnsignedSymmetric(false);
    tfTensorQuant->updateStats(data_unquantized, cnt, false);
    encoding = tfTensorQuant->computeEncoding(bw, false);
    tfTensorQuant->quantizeTensorPacked(data_unquantized, cnt, data_quantized, encoding.min, encoding.max, bw,
                                        DlQuantization::ROUND_NEAREST, false, false);

    // Check quantized values
    for (int i = 0; i < cnt; ++i)
    {
        EXPECT_EQ(data_quantized[i], data_quantized_expected[i]);
    }
    // De-quantize values
    std::vector<float> data_dequantized(cnt);
    // Test the pointer API.
    tfTensorQuant->dequantize(data_quantized.data(), cnt, encoding.min, encoding.max, bw, data_dequantized.data(),
                              false);

    // Check de-quantized values
    float data_dequantized_expected[] = {-39.7647f, -1.01961f, 0, 1.01961f, 2.03922f, -49.9608f, 80.0392f};
    for (int i = 0; i < cnt; ++i)
    {
        EXPECT_NEAR(data_dequantized[i], data_dequantized_expected[i], 1e-4);
    }
}


// 1. Quantize some data using QuantizeParamsPacked()
// 2. Dequantize the result using DeQuantize()
// The final result will be close to the original data
// Test both the vector and pointer API versions of QuantizeParamsPacked() and
// DeQuantize().
TEST_F(TestTensorQuantizer, SANITY_DequantizePerChannel)
{
    // Unquantized data
    float data_unquantized[] = {-40, -1, 0, 1, 2, -50, 80, 1, 30, 10, 25, 3, 2, -50, 70, 1};
    // Expected quantized data
    uint8_t data_quantized_expected[] = {20, 96, 98, 100, 102, 0, 255, 100, 170, 127, 159, 112, 110, 0, 255, 108};
    int cnt                           = 16;
    int bw                            = 8;
    size_t axis                       = 2;

    // Do quantization
    TfEncoding encoding;
    std::vector<uint8_t> data_quantized(cnt);
    std::vector<uint32_t> inputShape {1, 1, 2, 8};

    std::vector<TfEncoding> encodings;
    tfTensorQuant->setUnsignedSymmetric(false);
    tfTensorQuant->quantizePerChannelTensorPacked(data_unquantized, inputShape, axis, data_quantized, encodings, bw,
                                                  DlQuantization::RoundingMode::ROUND_NEAREST, false, false);

    // Check quantized values
    for (int i = 0; i < cnt; ++i)
    {
        EXPECT_EQ(data_quantized[i], data_quantized_expected[i]);
    }
    // De-quantize values
    std::vector<float> data_dequantized(cnt);
    // Test the pointer API.
    tfTensorQuant->dequantizePerChannelTensor(data_quantized.data(), inputShape, axis, encodings, bw,
                                              data_dequantized.data(), false);

    // Check de-quantized values
    float data_dequantized_expected[] = {-39.7647f, -1.01961f, 0,        1.01961f, 2.03922f, -49.9608f,
                                         80.0392f,  1.01961f,  30.1176f, 9.88235f, 24.9412f, 2.82353f,
                                         1.88235f,  -49.8824f, 70.1176f, 0.941176f};

    for (int i = 0; i < cnt; ++i)
    {
        EXPECT_NEAR(data_dequantized[i], data_dequantized_expected[i], 1e-4);
    }
}

#ifdef GPU_QUANTIZATION_ENABLED

TEST_F(TestTensorQuantizer, SanityTestGpu)
{
    TensorQuantizer tensorQuantizer(QuantizationMode::QUANTIZATION_TF_ENHANCED, ROUND_NEAREST);

    float mean   = 2;
    float stddev = 2;
    std::normal_distribution<float> distribution(mean, stddev);
    std::mt19937 generator(1);

    int tensorCount = 6000;
    std::vector<float> statsTensor(tensorCount);

    for (unsigned int i = 0; i < tensorCount; i++)
    {
        statsTensor[i] = distribution(generator);
    }
    Blob<GpuDevice<float>> statsTensorBlob(statsTensor.data(), tensorCount);

    tensorQuantizer.setStrictSymmetric(false);
    tensorQuantizer.setUnsignedSymmetric(false);
    tensorQuantizer.updateStats(statsTensorBlob.getDataPtrOnDevice(), statsTensor.size(), true);
    EXPECT_FALSE(tensorQuantizer.isEncodingValid);
    TfEncoding encoding = tensorQuantizer.computeEncoding(8, false);
    EXPECT_TRUE(tensorQuantizer.isEncodingValid);

    std::vector<float> inputTensor(tensorCount, 5);
    Blob<GpuDevice<float>> inputTensorBlob(inputTensor.data(), tensorCount);

    std::vector<float> quantizedTensor(tensorCount, 0);
    Blob<GpuDevice<float>> quantTensorBlob(quantizedTensor.data(), tensorCount);

    tensorQuantizer.quantizeDequantize(inputTensorBlob.getDataPtrOnDevice(), inputTensor.size(),
                                       quantTensorBlob.getDataPtrOnDevice(), encoding.min, encoding.max, 8, true);

    double MAX = *std::max_element(statsTensor.begin(), statsTensor.end());
    double MIN = *std::min_element(statsTensor.begin(), statsTensor.end());
    size_t PDF_SIZE = 512;
    double HISTOGRAM_BUCKET_SIZE = 3 * (MAX - MIN) / PDF_SIZE;
    EXPECT_NEAR(encoding.min, MIN,  HISTOGRAM_BUCKET_SIZE + 0.5 * encoding.delta);
    EXPECT_NEAR(encoding.max, MAX,  HISTOGRAM_BUCKET_SIZE + 0.5 * encoding.delta);

    EXPECT_NE(inputTensorBlob.getDataPtrOnCpu()[0], quantTensorBlob.getDataPtrOnCpu()[0]);
    EXPECT_NEAR(quantTensorBlob.getDataPtrOnCpu()[0], 5.0162, HISTOGRAM_BUCKET_SIZE);
}

TEST_F(TestTensorQuantizer, SanityTestGpuBlocked)
{
    BlockTensorQuantizer tensorQuantizer({}, 8, QuantizationMode::QUANTIZATION_TF_ENHANCED);

    float mean   = 2;
    float stddev = 2;
    std::normal_distribution<float> distribution(mean, stddev);
    std::mt19937 generator(1);

    int tensorCount = 6000;
    std::vector<float> statsTensor(tensorCount);
    TensorDims tensorShape = {tensorCount};

    for (unsigned int i = 0; i < tensorCount; i++)
    {
        statsTensor[i] = distribution(generator);
    }
    Blob<GpuDevice<float>> statsTensorBlob(statsTensor.data(), tensorCount);

    tensorQuantizer.setStrictSymmetric(false);
    tensorQuantizer.setUnsignedSymmetric(false);
    tensorQuantizer.updateStats(statsTensorBlob.getDataPtrOnDevice(), tensorShape, true);
    EXPECT_FALSE(tensorQuantizer.isEncodingValid);
    auto encodings = tensorQuantizer.computeEncodings(false);
    tensorQuantizer.setEncodings(encodings);
    TfEncoding encoding = encodings[0];
    EXPECT_TRUE(tensorQuantizer.isEncodingValid);

    std::vector<float> inputTensor(tensorCount, 5);
    Blob<GpuDevice<float>> inputTensorBlob(inputTensor.data(), tensorCount);

    std::vector<float> quantizedTensor(tensorCount, 0);
    Blob<GpuDevice<float>> quantTensorBlob(quantizedTensor.data(), tensorCount);

    tensorQuantizer.quantizeDequantize(inputTensorBlob.getDataPtrOnDevice(), quantTensorBlob.getDataPtrOnDevice(),
                                       tensorShape, true);

    double MAX = *std::max_element(statsTensor.begin(), statsTensor.end());
    double MIN = *std::min_element(statsTensor.begin(), statsTensor.end());
    size_t PDF_SIZE = 512 * 3;
    double HISTOGRAM_BUCKET_SIZE = 3 * (MAX - MIN) / PDF_SIZE;
    EXPECT_NEAR(encoding.min, MIN, HISTOGRAM_BUCKET_SIZE + 0.5 * encoding.delta);
    EXPECT_NEAR(encoding.max, MAX, HISTOGRAM_BUCKET_SIZE + 0.5 * encoding.delta);

    EXPECT_NE(inputTensorBlob.getDataPtrOnCpu()[0], quantTensorBlob.getDataPtrOnCpu()[0]);
    EXPECT_NEAR(quantTensorBlob.getDataPtrOnCpu()[0], 5.0162, HISTOGRAM_BUCKET_SIZE);
}

#endif

template <typename TypeParam>
class TestBlockQuantizerCpuGpu : public ::testing::Test
{};

TYPED_TEST_CASE(TestBlockQuantizerCpuGpu, TestDeviceTypes);

TYPED_TEST(TestBlockQuantizerCpuGpu, TestBlockQuantizerPerTensorQdq)
{
    if (!CheckRunTest<TypeParam>())
        return;

    typedef typename TypeParam::dataType DataType;

    BlockTensorQuantizer tensorQuantizer({}, 8, QUANTIZATION_TF);
    Encodings encodings(1);
    float min = -5;
    float max = 1;
    float delta = 6 / 255.;
    float offset = min / delta;
    int bw = 8;
    encodings[0] = {min, max, delta, offset, bw};
    tensorQuantizer.setEncodings(encodings);

    DataType input[6] = {-7, -5, -3, 0, .1, 2.5};
    DataType output[6];

    Blob<TypeParam> inputBlob(input, 6);
    Blob<TypeParam> outputBlob(output, 6);
    bool useCuda = TypeParam::modeCpuGpu == COMP_MODE_GPU;
    tensorQuantizer.quantizeDequantize(inputBlob.getDataPtrOnDevice(), outputBlob.getDataPtrOnDevice(),
                                       {6, 1}, useCuda);

    for (int i = 0; i < 6; i++)
    {
        DataType clipped = std::max(std::min(input[i], max), min);
        DataType expected = (std::round(clipped / delta - offset) + offset) * delta;
        EXPECT_NEAR(outputBlob.getDataPtrOnCpu()[i], expected, 0.001);
    }
}

TYPED_TEST(TestBlockQuantizerCpuGpu, TestBlockQuantizationEndToEnd)
{
    if (!CheckRunTest<TypeParam>())
        return;

    typedef typename TypeParam::dataType DataType;

    bool symmetric = true;
    int numElements = 12;
    TensorDims inputShape = {2, 6};
    TensorDims quantizerShape = {2, 2};
    BlockTensorQuantizer tensorQuantizer(quantizerShape, 8, QUANTIZATION_TF);

    DataType in[numElements] = {
        -5.4f, 10.f, -2.f,
        3.5f, 23.1f, 2.f,
        -10.f, -2.f, -1.f,
        -.1f, 0.3f, 0.1f
    };
    DataType out[numElements];

    Blob<TypeParam> inputBlob(in, numElements);
    Blob<TypeParam> outputBlob(out, numElements);
    bool useCuda = TypeParam::modeCpuGpu == COMP_MODE_GPU;
    tensorQuantizer.updateStats(inputBlob.getDataPtrOnDevice(), inputShape, useCuda);
    auto encodings = tensorQuantizer.computeEncodings(symmetric);
    tensorQuantizer.setEncodings(encodings);

    DataType expectedMax[4] = {10.f, 23.1f, 10.f * 127./128., .3f};
    for (size_t i = 0; i < 4; i++)
    {
        auto enc = encodings[i];
        EXPECT_NEAR(enc.max, expectedMax[i], 0.001);
        EXPECT_NEAR(enc.min + encodings[i].max, -1 * encodings[i].delta, 0.001);
        EXPECT_EQ(enc.offset, -128);
        EXPECT_NEAR(enc.delta, enc.max / 127, 0.001);
    }

    tensorQuantizer.quantizeDequantize(inputBlob.getDataPtrOnDevice(), outputBlob.getDataPtrOnDevice(),
                                       inputShape, useCuda);

    for (int i = 0; i < numElements; i++)
    {
        auto enc = encodings[i / 3];
        float min = enc.min; float max = enc.max; float delta = enc.delta; float offset = enc.offset;
        DataType clipped = std::max(std::min(in[i], max), min);
        DataType expected = (std::round(clipped / delta - offset) + offset) * delta;
        EXPECT_NEAR(outputBlob.getDataPtrOnCpu()[i], expected, 0.001);

    }

    EXPECT_THROW(tensorQuantizer.setPercentileValue(90.), std::runtime_error);
    EXPECT_THROW(tensorQuantizer.getPercentileValue(), std::runtime_error);
    EXPECT_THROW(tensorQuantizer.getStatsHistogram(), std::runtime_error);

}


TYPED_TEST(TestBlockQuantizerCpuGpu, TestQuantizerZeroPointShift)
{
    if (!CheckRunTest<TypeParam>())
        return;

    typedef typename TypeParam::dataType DataType;

    bool symmetric = true;
    int numElements = 12;
    TensorDims inputShape = {2, 6};
    TensorDims quantizerShape = {2, 1};
    BlockTensorQuantizer tensorQuantizer(quantizerShape, 2, QUANTIZATION_TF);
    tensorQuantizer.setZeroPointShift(0.5);

    DataType in[numElements] = {
        -3.f, 0.1f, -2.1f, 1.3f, 1.8f, 2.5f,
        -5.1f, -4.1f, -0.1f, 1.3f, 1.8f, 6.f
    };
    DataType out[numElements];
    DataType expectedOut[numElements] = {
        -3.f, 1.f, -3.f, 1.f, 1.f, 3.f,
        -6.f, -6.f, -2.f, 2.f, 2.f, 6.f
    };

    Blob<TypeParam> inputBlob(in, numElements);
    Blob<TypeParam> outputBlob(out, numElements);
    bool useCuda = TypeParam::modeCpuGpu == COMP_MODE_GPU;
    tensorQuantizer.updateStats(inputBlob.getDataPtrOnDevice(), inputShape, useCuda);
    auto encodings = tensorQuantizer.computeEncodings(symmetric);
    tensorQuantizer.setEncodings(encodings);

    DataType expectedMax[2] = {3.0f, 6.0f};
    for (size_t i = 0; i < 2; i++)
    {
        auto enc = encodings[i];
        EXPECT_NEAR(enc.max, expectedMax[i], 0.001);
        EXPECT_NEAR(enc.min, -encodings[i].max, 0.001);
        EXPECT_EQ(enc.offset, -1.5);
        EXPECT_NEAR(enc.delta, (enc.max - enc.min) / 3, 0.001);
    }

    tensorQuantizer.quantizeDequantize(inputBlob.getDataPtrOnDevice(), outputBlob.getDataPtrOnDevice(),
                                       inputShape, useCuda);

    for (int i = 0; i < numElements; i++)
    {
        EXPECT_NEAR(outputBlob.getDataPtrOnCpu()[i], expectedOut[i], 0.0001);
    }

    EXPECT_THROW(tensorQuantizer.computeEncodings(false), std::runtime_error);
}


TYPED_TEST(TestBlockQuantizerCpuGpu, TestZeroPointShiftTFEError)
{
    if (!CheckRunTest<TypeParam>())
        return;

    typedef typename TypeParam::dataType DataType;

    bool symmetric = true;
    int numElements = 12;
    TensorDims inputShape = {2, 6};
    TensorDims quantizerShape = {2, 1};
    BlockTensorQuantizer tensorQuantizer(quantizerShape, 2, QUANTIZATION_TF_ENHANCED);
    tensorQuantizer.setZeroPointShift(0.5);

    DataType in[numElements];
    DataType out[numElements];

    Blob<TypeParam> inputBlob(in, numElements);
    bool useCuda = TypeParam::modeCpuGpu == COMP_MODE_GPU;
    tensorQuantizer.updateStats(inputBlob.getDataPtrOnDevice(), inputShape, useCuda);
    EXPECT_THROW(tensorQuantizer.computeEncodings(symmetric), std::runtime_error);
}
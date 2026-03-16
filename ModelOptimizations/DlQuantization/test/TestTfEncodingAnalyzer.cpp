// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "test_quantization_lib.hpp"
#include <TfEncodingAnalyzer.h>

template <typename TypeParam>
class TestTfEncodingAnalyzer : public ::testing::Test
{
};

// Test on CPU and GPU with float and double
TYPED_TEST_SUITE(TestTfEncodingAnalyzer, TestDataTypesAndDevices);

TYPED_TEST(TestTfEncodingAnalyzer, Asymmetric)
{
    typedef typename TypeParam::dataType dataType;

    // Instantiate TfEncodingAnalyzer
    DlQuantization::TfEncodingAnalyzer<dataType> analyzer;

    float mean   = 2;
    float stddev = 2;
    std::normal_distribution<dataType> distribution(mean, stddev);
    std::mt19937 generator(10);

    unsigned int tensorCount = 6000;
    std::vector<dataType> tensor(tensorCount);

    for (int i = 0; i < tensorCount; i++)
    {
        tensor[i] = distribution(generator);
    }
    Blob<TypeParam> tensorBlob(tensor.data(), tensorCount);

    // Update the stats using these tensor values
    analyzer.updateStats(tensorBlob.getDataPtrOnDevice(), tensorCount, TypeParam::modeCpuGpu);

    // Get the encodings
    DlQuantization::TfEncoding encoding = analyzer.computeEncoding(8, false, false, false);

    EXPECT_NEAR(encoding.min, *std::min_element(tensor.begin(), tensor.end()), 0.03);
    EXPECT_NEAR(encoding.max, *std::max_element(tensor.begin(), tensor.end()), 0.03);
    EXPECT_FLOAT_EQ(encoding.delta, (encoding.max - encoding.min) / 255);
    EXPECT_FLOAT_EQ(encoding.offset, encoding.min / encoding.delta);
    EXPECT_EQ(encoding.bw, 8);
}

TYPED_TEST(TestTfEncodingAnalyzer, Symmetric)
{
    typedef typename TypeParam::dataType dataType;

    // Instantiate TfEncodingAnalyzer
    DlQuantization::TfEncodingAnalyzer<dataType> analyzer;

    float mean   = 2;
    float stddev = 2;
    std::normal_distribution<dataType> distribution(mean, stddev);
    std::mt19937 generator(100);

    unsigned int tensorCount = 6000;
    std::vector<dataType> tensor(tensorCount);

    for (int i = 0; i < tensorCount; i++)
    {
        tensor[i] = distribution(generator);
    }
    Blob<TypeParam> tensorBlob(tensor.data(), tensorCount);

    // Update the stats using these tensor values
    analyzer.updateStats(tensorBlob.getDataPtrOnDevice(), tensorCount, TypeParam::modeCpuGpu);

    // Get the encodings
    DlQuantization::TfEncoding encoding = analyzer.computeEncoding(8, true, false, false);

    float expected_max = std::max(std::abs(*std::min_element(tensor.begin(), tensor.end())),
                                  std::abs(*std::max_element(tensor.begin(), tensor.end())));

    // Min and Max will get adjusted slightly to represent an exact zero with one of the quantized values
    // Adjustment is expected to be less than half a delta worth
    EXPECT_NEAR(encoding.min, -expected_max, encoding.delta * 1.5 + 1e-4);
    EXPECT_NEAR(encoding.max, expected_max, encoding.delta / 2 + 1e-4);

    // Check that the center value is absolute 0
    EXPECT_NEAR(encoding.min + encoding.delta * (-encoding.offset), 0, 1e-7);

    EXPECT_FLOAT_EQ(encoding.delta, (encoding.max - encoding.min) / 255);

    // Check that offset is -128 - another check for symmetric encodings
    EXPECT_NEAR(encoding.offset, -128, 0);
    EXPECT_EQ(encoding.bw, 8);
}


TYPED_TEST(TestTfEncodingAnalyzer, StrictSymmetric)
{
    typedef typename TypeParam::dataType dataType;

    // Instantiate TfEncodingAnalyzer
    DlQuantization::TfEncodingAnalyzer<dataType> analyzer;

    float mean   = 2;
    float stddev = 2;
    std::normal_distribution<dataType> distribution(mean, stddev);
    std::mt19937 generator(100);

    unsigned int tensorCount = 6000;
    std::vector<dataType> tensor(tensorCount);

    for (int i = 0; i < tensorCount; i++)
    {
        tensor[i] = distribution(generator);
    }
    Blob<TypeParam> tensorBlob(tensor.data(), tensorCount);

    // Update the stats using these tensor values
    analyzer.updateStats(tensorBlob.getDataPtrOnDevice(), tensorCount, TypeParam::modeCpuGpu);

    // Get the encodings
    DlQuantization::TfEncoding encoding = analyzer.computeEncoding(8, true, true, false);

    float expected_max = std::max(std::abs(*std::min_element(tensor.begin(), tensor.end())),
                                  std::abs(*std::max_element(tensor.begin(), tensor.end())));

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

TYPED_TEST(TestTfEncodingAnalyzer, SymmetricUnsigned)
{
    typedef typename TypeParam::dataType dataType;

    // Instantiate TfEncodingAnalyzer
    DlQuantization::TfEncodingAnalyzer<dataType> analyzer;

    float mean   = 2;
    float stddev = 2;
    std::normal_distribution<dataType> distribution(mean, stddev);
    std::mt19937 generator(100);

    unsigned int tensorCount = 6000;
    std::vector<dataType> tensor(tensorCount);

    for (int i = 0; i < tensorCount; i++)
    {
        tensor[i] = distribution(generator);
        tensor[i] = std::max(tensor[i], (dataType) 0.0);
    }
    Blob<TypeParam> tensorBlob(tensor.data(), tensorCount);

    // Update the stats using these tensor values
    analyzer.updateStats(tensorBlob.getDataPtrOnDevice(), tensorCount, TypeParam::modeCpuGpu);

    // Get the encodings
    DlQuantization::TfEncoding encoding = analyzer.computeEncoding(8, true, false, true);

    float expected_max = std::max(std::abs(*std::min_element(tensor.begin(), tensor.end())),
                                  std::abs(*std::max_element(tensor.begin(), tensor.end())));

    // Min and Max will get adjusted slightly to represent an exact zero with one of the quantized values
    // Adjustment is expected to be less than half a delta worth
    EXPECT_NEAR(encoding.min, 0, 1e-7);
    EXPECT_NEAR(encoding.max, expected_max, encoding.delta / 2 + 1e-4);

    // Check that the center value is absolute 0
    EXPECT_NEAR(encoding.min + encoding.delta * (-encoding.offset), 0, 1e-7);

    EXPECT_FLOAT_EQ(encoding.delta, (encoding.max - encoding.min) / 255);

    // Check that offset is -128 - another check for symmetric encodings
    EXPECT_NEAR(encoding.offset, 0, 0);
    EXPECT_EQ(encoding.bw, 8);
}


TYPED_TEST(TestTfEncodingAnalyzer, SymmetricForcedSigned)
{
    typedef typename TypeParam::dataType dataType;

    // Instantiate TfEncodingAnalyzer
    DlQuantization::TfEncodingAnalyzer<dataType> analyzer;

    float mean   = 2;
    float stddev = 2;
    std::normal_distribution<dataType> distribution(mean, stddev);
    std::mt19937 generator(100);

    unsigned int tensorCount = 6000;
    std::vector<dataType> tensor(tensorCount);

    for (int i = 0; i < tensorCount; i++)
    {
        tensor[i] = distribution(generator);
        tensor[i] = std::max(tensor[i], (dataType) 0.0);
    }
    Blob<TypeParam> tensorBlob(tensor.data(), tensorCount);

    // Update the stats using these tensor values
    analyzer.updateStats(tensorBlob.getDataPtrOnDevice(), tensorCount, TypeParam::modeCpuGpu);

    // Get the encodings
    DlQuantization::TfEncoding encoding = analyzer.computeEncoding(8, true, false, false);

    float expected_max = std::max(std::abs(*std::min_element(tensor.begin(), tensor.end())),
                                  std::abs(*std::max_element(tensor.begin(), tensor.end())));

    // Min and Max will get adjusted slightly to represent an exact zero with one of the quantized values
    // Adjustment is expected to be less than half a delta worth
    EXPECT_NEAR(encoding.min, -expected_max, encoding.delta * 1.5 + 1e-4);
    EXPECT_NEAR(encoding.max, expected_max, encoding.delta / 2 + 1e-4);

    // Check that the center value is absolute 0
    EXPECT_NEAR(encoding.min + encoding.delta * (-encoding.offset), 0, 1e-7);

    EXPECT_FLOAT_EQ(encoding.delta, (encoding.max - encoding.min) / 255);

    // Check that offset is -128 - another check for symmetric encodings
    EXPECT_NEAR(encoding.offset, -128, 0);
    EXPECT_EQ(encoding.bw, 8);
}

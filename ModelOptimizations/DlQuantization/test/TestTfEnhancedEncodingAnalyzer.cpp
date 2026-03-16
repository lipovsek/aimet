// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "test_quantization_lib.hpp"
#include <TfEnhancedEncodingAnalyzer.h>

template <typename TypeParam>
class TestTfEnhancedEncodingAnalyzer : public ::testing::Test
{
};

// Test on CPU and GPU with float and double
TYPED_TEST_SUITE(TestTfEnhancedEncodingAnalyzer, TestDataTypesAndDevices);

TYPED_TEST(TestTfEnhancedEncodingAnalyzer, GetStats)
{
    typedef typename TypeParam::dataType dataType;

    // Instantiate TfEnhancedEncodingAnalyzer
    DlQuantization::TfEnhancedEncodingAnalyzer<dataType> analyzer;

    float mean   = 2;
    float stddev = 2;
    std::normal_distribution<dataType> distribution(mean, stddev);
    std::mt19937 generator(1);

    double min               = std::numeric_limits<double>::max();
    double max               = std::numeric_limits<double>::min();
    unsigned int tensorCount = 6000;
    std::vector<dataType> tensor(tensorCount);

    for (unsigned int i = 0; i < tensorCount; i++)
    {
        tensor[i] = distribution(generator);
        min       = std::min(min, double(tensor[i]));
        max       = std::max(max, double(tensor[i]));
    }
    Blob<TypeParam> tensorBlob(tensor.data(), tensorCount);

    // Update the stats using these tensor values
    analyzer.updateStats(tensorBlob.getDataPtrOnDevice(), tensorCount, TypeParam::modeCpuGpu);

    auto histogram = analyzer.getStatsHistogram();
    for (auto entry: histogram)
    {
        double leftEdge;
        double pdf;

        std::tie(leftEdge, pdf) = entry;
    }
}

TYPED_TEST(TestTfEnhancedEncodingAnalyzer, Asymmetric)
{
    typedef typename TypeParam::dataType dataType;

    // Instantiate TfEnhancedEncodingAnalyzer
    DlQuantization::TfEnhancedEncodingAnalyzer<dataType> analyzer;

    float mean   = 2;
    float stddev = 2;
    std::normal_distribution<dataType> distribution(mean, stddev);
    std::mt19937 generator(1);

    double min               = std::numeric_limits<double>::max();
    double max               = std::numeric_limits<double>::min();
    unsigned int tensorCount = 6000;
    std::vector<dataType> tensor(tensorCount);

    for (unsigned int i = 0; i < tensorCount; i++)
    {
        tensor[i] = distribution(generator);
        min       = std::min(min, double(tensor[i]));
        max       = std::max(max, double(tensor[i]));
    }
    Blob<TypeParam> tensorBlob(tensor.data(), tensorCount);

    // Update the stats using these tensor values
    analyzer.updateStats(tensorBlob.getDataPtrOnDevice(), tensorCount, TypeParam::modeCpuGpu);

    // Get the encodings
    DlQuantization::TfEncoding encoding = analyzer.computeEncoding(8, false, false, false);

    // We know we have a normal distribution. We expect the encoding to cover
    // at least 2 standard deviations, and at most 6.
    EXPECT_GT(encoding.min, mean - 6 * stddev);
    EXPECT_LT(encoding.min, mean - 2 * stddev);
    EXPECT_GT(encoding.max, mean + 2 * stddev);
    EXPECT_LT(encoding.max, mean + 6 * stddev);
}

TYPED_TEST(TestTfEnhancedEncodingAnalyzer, AllSameValuesAsymmetric)
{
    typedef typename TypeParam::dataType dataType;

    // Instantiate TfEnhancedEncodingAnalyzer
    DlQuantization::TfEnhancedEncodingAnalyzer<dataType> analyzer1;

    unsigned int tensorCount = 6000;
    std::vector<dataType> tensor1(tensorCount, 4);
    Blob<TypeParam> tensorBlob1(tensor1.data(), tensorCount);

    // Update the stats using these tensor values
    analyzer1.updateStats(tensorBlob1.getDataPtrOnDevice(), tensorCount, TypeParam::modeCpuGpu);

    // Get the encodings
    DlQuantization::TfEncoding encoding1 = analyzer1.computeEncoding(8, false, false, false);
    EXPECT_LE(encoding1.min, 0);   // 0 is included
    EXPECT_GE(encoding1.max, 3.5);


    // Instantiate TfEnhancedEncodingAnalyzer
    DlQuantization::TfEnhancedEncodingAnalyzer<dataType> analyzer2;

    std::vector<dataType> tensor2(tensorCount, -5);
    Blob<TypeParam> tensorBlob2(tensor2.data(), tensorCount);

    // Update the stats using these tensor values
    analyzer2.updateStats(tensorBlob2.getDataPtrOnDevice(), tensorCount, TypeParam::modeCpuGpu);

    // Get the encodings
    DlQuantization::TfEncoding encoding2 = analyzer2.computeEncoding(8, false, false, false);
    EXPECT_LE(encoding2.min, -4.5);
    EXPECT_GE(encoding2.max, 0);   // 0 is included
}

TYPED_TEST(TestTfEnhancedEncodingAnalyzer, AllZeroesAsymmetric)
{
    typedef typename TypeParam::dataType dataType;

    // Instantiate TfEnhancedEncodingAnalyzer
    DlQuantization::TfEnhancedEncodingAnalyzer<dataType> analyzer1;

    unsigned int tensorCount = 6000;
    std::vector<dataType> tensor1(tensorCount, 0);
    Blob<TypeParam> tensorBlob1(tensor1.data(), tensorCount);

    // Update the stats using these tensor values
    analyzer1.updateStats(tensorBlob1.getDataPtrOnDevice(), tensorCount, TypeParam::modeCpuGpu);

    // Get the encodings
    DlQuantization::TfEncoding encoding1 = analyzer1.computeEncoding(8, false, false, false);

    EXPECT_NEAR(encoding1.min, -1.00392, 0.0001);
    EXPECT_NEAR(encoding1.max, 0.996078, 0.0001);
    EXPECT_EQ(encoding1.offset, -128);
    EXPECT_EQ(encoding1.bw, 8);
}

TYPED_TEST(TestTfEnhancedEncodingAnalyzer, Symmetric)
{
    typedef typename TypeParam::dataType dataType;

    // Instantiate TfEnhancedEncodingAnalyzer
    DlQuantization::TfEnhancedEncodingAnalyzer<dataType> analyzer;

    float mean   = -2;
    float stddev = 1;
    std::normal_distribution<dataType> distribution(mean, stddev);
    std::mt19937 generator(1);

    unsigned int tensorCount = 6000;
    std::vector<dataType> tensor(tensorCount);

    for (unsigned int i = 0; i < tensorCount; i++)
    {
        tensor[i] = distribution(generator);
    }
    Blob<TypeParam> tensorBlob(tensor.data(), tensorCount);

    // Update the stats using these tensor values
    analyzer.updateStats(tensorBlob.getDataPtrOnDevice(), tensorCount, TypeParam::modeCpuGpu);

    // Get the encodings
    DlQuantization::TfEncoding encoding = analyzer.computeEncoding(8, true, false, false);

    dataType absoluteMin = *std::min_element(tensor.begin(), tensor.end());
    dataType absoluteMax = *std::max_element(tensor.begin(), tensor.end());

    absoluteMax = std::max(std::abs(absoluteMax), std::abs(absoluteMin));
    absoluteMin = -absoluteMax;

    EXPECT_GT(encoding.min, absoluteMin);
    EXPECT_LT(encoding.max, absoluteMax);

    EXPECT_FLOAT_EQ(encoding.delta, (encoding.max - encoding.min) / 255);
    EXPECT_FLOAT_EQ(encoding.offset, encoding.min / encoding.delta);
    EXPECT_EQ(encoding.offset, -128);
    EXPECT_EQ(encoding.bw, 8);
}

TYPED_TEST(TestTfEnhancedEncodingAnalyzer, StrictSymmetric)
{
    typedef typename TypeParam::dataType dataType;

    // Instantiate TfEnhancedEncodingAnalyzer
    DlQuantization::TfEnhancedEncodingAnalyzer<dataType> analyzer;

    float mean   = -2;
    float stddev = 1;
    std::normal_distribution<dataType> distribution(mean, stddev);
    std::mt19937 generator(1);

    unsigned int tensorCount = 6000;
    std::vector<dataType> tensor(tensorCount);

    for (unsigned int i = 0; i < tensorCount; i++)
    {
        tensor[i] = distribution(generator);
    }
    Blob<TypeParam> tensorBlob(tensor.data(), tensorCount);

    // Update the stats using these tensor values
    analyzer.updateStats(tensorBlob.getDataPtrOnDevice(), tensorCount, TypeParam::modeCpuGpu);

    // Get the encodings
    DlQuantization::TfEncoding encoding = analyzer.computeEncoding(8, true, true, false);

    dataType absoluteMin = *std::min_element(tensor.begin(), tensor.end());
    dataType absoluteMax = *std::max_element(tensor.begin(), tensor.end());

    absoluteMax = std::max(std::abs(absoluteMax), std::abs(absoluteMin));
    absoluteMin = -absoluteMax;

    EXPECT_GT(encoding.min, absoluteMin);
    EXPECT_LT(encoding.max, absoluteMax);

    EXPECT_FLOAT_EQ(encoding.delta, (encoding.max - encoding.min) / 254);
    EXPECT_FLOAT_EQ(encoding.offset, encoding.min / encoding.delta);
    EXPECT_EQ(encoding.offset, -127);
    EXPECT_EQ(encoding.bw, 8);
    EXPECT_EQ(encoding.min, -encoding.max);
}

TYPED_TEST(TestTfEnhancedEncodingAnalyzer, SymmetricUnsigned)
{
    typedef typename TypeParam::dataType dataType;

    // Instantiate TfEnhancedEncodingAnalyzer
    DlQuantization::TfEnhancedEncodingAnalyzer<dataType> analyzer;

    float mean   = -2;
    float stddev = 1;
    std::normal_distribution<dataType> distribution(mean, stddev);
    std::mt19937 generator(1);

    unsigned int tensorCount = 6000;
    std::vector<dataType> tensor(tensorCount);

    for (unsigned int i = 0; i < tensorCount; i++)
    {
        tensor[i] = distribution(generator);
        tensor[i] = std::max(tensor[i], (dataType) 0.0);
    }
    Blob<TypeParam> tensorBlob(tensor.data(), tensorCount);

    // Update the stats using these tensor values
    analyzer.updateStats(tensorBlob.getDataPtrOnDevice(), tensorCount, TypeParam::modeCpuGpu);

    // Get the encodings
    DlQuantization::TfEncoding encoding = analyzer.computeEncoding(8, true, false, true);

    dataType absoluteMin = *std::min_element(tensor.begin(), tensor.end());
    dataType absoluteMax = *std::max_element(tensor.begin(), tensor.end());

    absoluteMax = std::max(std::abs(absoluteMax), std::abs(absoluteMin));
    absoluteMin = -absoluteMax;

    size_t PDF_SIZE = 512 * 3;
    double HISTOGRAM_BUCKET_SIZE = 3 * (absoluteMax - absoluteMin) / PDF_SIZE;
    EXPECT_EQ(encoding.min, 0);
    EXPECT_NEAR(encoding.max, absoluteMax, 0.015 + HISTOGRAM_BUCKET_SIZE);

    EXPECT_FLOAT_EQ(encoding.delta, (encoding.max - encoding.min) / 255);
    EXPECT_FLOAT_EQ(encoding.offset, encoding.min / encoding.delta);
    EXPECT_EQ(encoding.bw, 8);
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

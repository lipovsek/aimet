//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//  3. Neither the name of the copyright holder nor the names of its contributors
//     may be used to endorse or promote products derived from this software
//     without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//  SPDX-License-Identifier: BSD-3-Clause
//
//  @@-COPYRIGHT-END-@@
//
//==============================================================================

#include <gtest/gtest.h>
#include <random>

#include <MinMaxEncodingAnalyzer.h>
#include "test_quantization_lib.hpp"

using namespace DlQuantization;

template <typename TypeParam>
class TestMinMaxEncodingAnalyzer : public ::testing::Test
{};

TYPED_TEST_SUITE(TestMinMaxEncodingAnalyzer, TestDataTypesAndDevices);

TYPED_TEST(TestMinMaxEncodingAnalyzer, UpdateBlockStatsSymmetric)
{
    if (!CheckRunTest<TypeParam>())
        return;

    typedef typename TypeParam::dataType DataType;


    TensorDims inputShape = {2, 6};
    MinMaxEncodingAnalyzer<DataType> analyzer({2, 2});

    int bitwidth = 8;
    bool symmetric = true;
    int numElements = 12;

    DataType in[numElements] = {
        -5.4f, 10.f, -2.f,
        3.5f, 23.1f, 2.f,
        -10.f * 128. / 127., -2.f, -1.f,
        -.1f, 0.3f, 0.1f
    };

    Blob<TypeParam> inputBlob(in, numElements);
    analyzer.updateStats(inputBlob.getDataPtrOnDevice(), inputShape, TypeParam::modeCpuGpu);
    auto encodings = analyzer.computeEncoding(bitwidth, symmetric, false, false);

    DataType expectedMax[4] = {10.f, 23.1f, 10.f, .3f};
    for (size_t i = 0; i < 4; i++)
    {
        auto enc = encodings[i];
        EXPECT_NEAR(enc.max, expectedMax[i], 0.001);
        EXPECT_NEAR(enc.min + encodings[i].max, -1 * encodings[i].delta, 0.001);
        EXPECT_EQ(enc.offset, -128);
        EXPECT_NEAR(enc.delta, enc.max / 127, 0.001);
    }

    EXPECT_THROW(analyzer.setPercentileValue(90.), std::runtime_error);
    EXPECT_THROW(analyzer.getPercentileValue(), std::runtime_error);
    EXPECT_THROW(analyzer.getStatsHistogram(), std::runtime_error);
}

TYPED_TEST(TestMinMaxEncodingAnalyzer, UpdateBlockStatsAsymmetric)
{
    if (!CheckRunTest<TypeParam>())
        return;

    typedef typename TypeParam::dataType DataType;


    TensorDims inputShape = {6, 2};
    MinMaxEncodingAnalyzer<DataType> analyzer({3, 2});

    int bitwidth = 8;
    bool symmetric = false;
    int numElements = 12;

    DataType in[numElements] = {
        -5.4f, 10.f,   -2.f, 3.5f,
        23.1f, 2.f,    -10.f, -2.f,
        -1.f, -.1f,    0.3f, 0.1f
    };

    Blob<TypeParam> inputBlob(in, numElements);
    analyzer.updateStats(inputBlob.getDataPtrOnDevice(), inputShape, TypeParam::modeCpuGpu);
    auto encodings = analyzer.computeEncoding(bitwidth, symmetric, false, false);

    DataType expectedMax[6] = {0., 10., 23.1, 2., 0.3f, .1f};
    DataType expectedMin[6] = {-5.4, 0., -10., -2., -1., -0.1};
    for (size_t i = 0; i < 4; i++)
    {
        auto enc = encodings[i];
        EXPECT_NEAR(enc.max, expectedMax[i], enc.delta);
        EXPECT_NEAR(enc.min, expectedMin[i], enc.delta);
        EXPECT_NEAR(enc.delta, (enc.max - enc.min) / 255, 0.001);
        EXPECT_NEAR(enc.offset, enc.min / enc.delta, 0.001);
    }
}

TYPED_TEST(TestMinMaxEncodingAnalyzer, ComputeEncodingsWithZeroPointShift)
{
    if (!CheckRunTest<TypeParam>())
        return;

    typedef typename TypeParam::dataType DataType;

    TensorDims inputShape = {2, 2};
    MinMaxEncodingAnalyzer<DataType> analyzer({2, 1});

    int bitwidth = 2;
    bool symmetric = true;
    int numElements = 12;

    DataType in[numElements] = {
        -5.4f, 10.f,   -5.f, 3.5f,
    };

    Blob<TypeParam> inputBlob(in, numElements);
    analyzer.updateStats(inputBlob.getDataPtrOnDevice(), inputShape, TypeParam::modeCpuGpu);
    auto encodings = analyzer.computeEncoding(bitwidth, symmetric, false, false, 0.5);

    DataType expectedMax[2] = {10.f, 5.f};
    DataType expectedMin[2] = {-10.f, -5.f};
    for (size_t i = 0; i < 2; i++)
    {
        auto enc = encodings[i];
        EXPECT_NEAR(enc.max, expectedMax[i], enc.delta);
        EXPECT_NEAR(enc.min, expectedMin[i], enc.delta);
        EXPECT_NEAR(enc.delta, (enc.max - enc.min) / 3, 0.001);
        EXPECT_EQ(enc.offset, -1.5f);
    }

    EXPECT_THROW(analyzer.computeEncoding(bitwidth, false, false, false, 0.5), std::runtime_error);
    EXPECT_THROW(analyzer.computeEncoding(bitwidth, true, true, false, 0.5), std::runtime_error);
}
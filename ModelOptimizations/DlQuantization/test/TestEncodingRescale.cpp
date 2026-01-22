// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include <vector>

#include "DlQuantization/EncodingRescale.hpp"

using namespace DlQuantization;

class TestEncodingRescale : public ::testing::Test
{
protected:
    std::vector<float> perChannelWeightScale;
    std::vector<float> perTensorWeightScale;
    std::vector<float> bias;
    std::vector<float> requantScale;
    std::vector<float> biasSim;
    ConvSpecArgs<float> convArgs;

    void SetUp()
    {
        if (perChannelWeightScale.size() == 0)
        {
            perChannelWeightScale.insert(perChannelWeightScale.end(), {-0.5f, -0.25f, 0.25, 0.5, 0.75});
        }
        if (perTensorWeightScale.size() == 0)
        {
            perTensorWeightScale.insert(perTensorWeightScale.end(), {0.75});
        }
        if (bias.size() == 0)
        {
            bias.insert(bias.end(), {-0.1f, -0.05f, 0.0f, 0.05f, 0.1f});
        }
        if (biasSim.size() == 0)
        {
            biasSim.resize(bias.size());
        }
        convArgs = {.out_encoding_delta = 0.0002f,
                      .out_encoding_offset = -128,
                      .input_scale = 0.0001f};
    }
};

TEST_F(TestEncodingRescale, SanityTestAct8BwPerChannelQuantSimOffsetWrap)
{
    // Instantiate TensorQuantizationSim
    requantScale.resize(perChannelWeightScale.size());
    convArgs.bw = 8;
    convArgs.weight_scale = perChannelWeightScale;
    getRescaledOutputAndBias(bias.data(), bias.size(), convArgs, biasSim.data(),
        requantScale.data(), false, true);

    std::vector<float> expectedRequantScale = {-0.25f, -0.125f, 0.125f, 0.25f, 0.375f};
    std::vector<float> expectedBiasSim = {1488, 976, 1024, 1512, 1674};

    EXPECT_EQ(bias.size(), biasSim.size());
    EXPECT_EQ(perChannelWeightScale.size(), requantScale.size());

    for (int i = 0; i < requantScale.size(); i++)
    {
        EXPECT_FLOAT_EQ(requantScale[i], expectedRequantScale[i]);
    }
    for (int i = 0; i < biasSim.size(); i++)
    {
        EXPECT_FLOAT_EQ(biasSim[i], expectedBiasSim[i]);
    }
}

TEST_F(TestEncodingRescale, SanityTestAct8BwPerTensorQuantSimOffsetWrap)
{
    // Instantiate TensorQuantizationSim
    requantScale.resize(perTensorWeightScale.size());
    convArgs.bw = 8;
    convArgs.weight_scale = perTensorWeightScale;
    getRescaledOutputAndBias(bias.data(), bias.size(), convArgs, biasSim.data(),
        requantScale.data(), false, true);

    std::vector<float> expectedRequantScale = {0.375f};
    std::vector<float> expectedBiasSim = {-992, -325, 341, 1008, 1675};

    EXPECT_EQ(bias.size(), biasSim.size());
    EXPECT_EQ(perTensorWeightScale.size(), requantScale.size());

    for (int i = 0; i < requantScale.size(); i++)
    {
        EXPECT_FLOAT_EQ(requantScale[i], expectedRequantScale[i]);
    }
    for (int i = 0; i < biasSim.size(); i++)
    {
        EXPECT_FLOAT_EQ(biasSim[i], expectedBiasSim[i]);
    }
}

TEST_F(TestEncodingRescale, SanityTestAct16BwPerChannelQuantSimOffsetWrap)
{
    // Instantiate TensorQuantizationSim
    requantScale.resize(perChannelWeightScale.size());
    convArgs.bw = 16;
    convArgs.weight_scale = perChannelWeightScale;
    getRescaledOutputAndBias(bias.data(), bias.size(), convArgs, biasSim.data(),
        requantScale.data(), false, true);

    // The calculation for requantScale in 16 bits is exactly the same as 8 bits, hence skipping this comparison.
    std::vector<float> expectedBiasSim = {5, 3, 4, 5, 6};

    EXPECT_EQ(bias.size(), biasSim.size());

    for (int i = 0; i < biasSim.size(); i++)
    {
        EXPECT_FLOAT_EQ(biasSim[i], expectedBiasSim[i]);
    }
}

TEST_F(TestEncodingRescale, SanityTestAct16BwPerTensorQuantSimOffsetWrap)
{
    requantScale.resize(perTensorWeightScale.size());
    convArgs.bw = 16;
    convArgs.weight_scale = perTensorWeightScale;
    getRescaledOutputAndBias(bias.data(), bias.size(), convArgs, biasSim.data(),
        requantScale.data(), false, true);

    // The calculation for requantScale in 16 bits is exactly the same as 8 bits, hence skipping this comparison.
    std::vector<float> expectedBiasSim = {-4, -2, 1, 3, 6};

    EXPECT_EQ(bias.size(), biasSim.size());

    for (int i = 0; i < biasSim.size(); i++)
    {
        EXPECT_FLOAT_EQ(biasSim[i], expectedBiasSim[i]);
    }
}

TEST_F(TestEncodingRescale, SanityTestAct8BwPerChannelQuantSimNoOffsetWrap)
{
    requantScale.resize(perChannelWeightScale.size());
    convArgs.bw = 8;
    convArgs.weight_scale = perChannelWeightScale;
    getRescaledOutputAndBias(bias.data(), bias.size(), convArgs, biasSim.data(),
        requantScale.data(), false, false);

    std::vector<float> expectedRequantScale = {-0.25f, -0.125f, 0.125f, 0.25f, 0.375f};
    std::vector<float> expectedBiasSim = {2000, 2000, 0, 1000, 1333};

    EXPECT_EQ(bias.size(), biasSim.size());
    EXPECT_EQ(perChannelWeightScale.size(), requantScale.size());

    for (int i = 0; i < requantScale.size(); i++)
    {
        EXPECT_FLOAT_EQ(requantScale[i], expectedRequantScale[i]);
    }

    for (int i = 0; i < biasSim.size(); i++)
    {
        EXPECT_FLOAT_EQ(biasSim[i], expectedBiasSim[i]);
    }
}

TEST_F(TestEncodingRescale, SanityTestAct8BwPerTensorQuantSimNoOffsetWrap)
{
    requantScale.resize(perTensorWeightScale.size());
    convArgs.bw = 8;
    convArgs.weight_scale = perTensorWeightScale;
    getRescaledOutputAndBias(bias.data(), bias.size(), convArgs, biasSim.data(),
        requantScale.data(), false, false);

    std::vector<float> expectedRequantScale = {0.375f};
    std::vector<float> expectedBiasSim = {-1333, -667, 0, 667, 1333};

    EXPECT_EQ(bias.size(), biasSim.size());
    EXPECT_EQ(perTensorWeightScale.size(), requantScale.size());

    for (int i = 0; i < requantScale.size(); i++)
    {
        EXPECT_FLOAT_EQ(requantScale[i], expectedRequantScale[i]);
    }
    for (int i = 0; i < biasSim.size(); i++)
    {
        EXPECT_FLOAT_EQ(biasSim[i], expectedBiasSim[i]);
    }
}

TEST_F(TestEncodingRescale, SanityTestAct16BwPerChannelQuantSimNoOffsetWrap)
{
    requantScale.resize(perChannelWeightScale.size());
    convArgs.bw = 16;
    convArgs.weight_scale = perChannelWeightScale;
    getRescaledOutputAndBias(bias.data(), bias.size(), convArgs, biasSim.data(),
        requantScale.data(), false, false);

    // The calculation for requantScale in 16 bits is exactly the same as 8 bits, hence skipping this comparison.
    std::vector<float> expectedBiasSim = {7, 7, 0, 3, 5};

    EXPECT_EQ(bias.size(), biasSim.size());

    for (int i = 0; i < biasSim.size(); i++)
    {
        EXPECT_FLOAT_EQ(biasSim[i], expectedBiasSim[i]);
    }
}

TEST_F(TestEncodingRescale, SanityTestAct16BwPerTensorQuantSimNoOffsetWrap)
{
    requantScale.resize(perTensorWeightScale.size());
    convArgs.bw = 16;
    convArgs.weight_scale = perTensorWeightScale;
    getRescaledOutputAndBias(bias.data(), bias.size(), convArgs, biasSim.data(),
        requantScale.data(), false, false);

    // The calculation for requantScale in 16 bits is exactly the same as 8 bits, hence skipping this comparison.
    std::vector<float> expectedBiasSim = {-6, -3, 0, 2, 5};

    EXPECT_EQ(bias.size(), biasSim.size());

    for (int i = 0; i < biasSim.size(); i++)
    {
        EXPECT_FLOAT_EQ(biasSim[i], expectedBiasSim[i]);
    }
}


// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include <vector>

#include "trim_functions.hpp"
#include <TensorQuantizationSim.h>


class TestTensorQuantizationSim : public ::testing::Test
{
};


TEST(TestTensorQuantizationSim, SanityTest)
{
    // Instantiate TensorQuantizationSim
    DlQuantization::TensorQuantizationSim<float> sim;

    // Create a dummy tensor
    const std::vector<float> tensor = {-0.5f, -0.25f, 0, 0.25, 0.5, 0.75};
    std::vector<float> outputTensor(tensor.size());

    uint8_t bw = 8;
    double min = -0.46;
    double max = 0.72;

    sim.quantizeDequantizeTensor(tensor.data(), tensor.size(), outputTensor.data(), min, max, bw,
                                 DlQuantization::RoundingMode::ROUND_NEAREST, false);

    std::vector<float> expectedOutput = {-0.45811754, -0.2498823, 0.0, 0.2498823, 0.49976459, 0.72188222};

    EXPECT_EQ(outputTensor.size(), expectedOutput.size());

    for (int i = 0; i < outputTensor.size(); i++)
    {
        EXPECT_FLOAT_EQ(outputTensor[i], expectedOutput[i]);
    }
}

TEST(TestTensorQuantizationSim, SanityTestWithGatedMin)
{
    // Instantiate TensorQuantizationSim
    DlQuantization::TensorQuantizationSim<float> sim;

    // Create a dummy tensor
    const std::vector<float> tensor = {-0.5f, -0.25f, 0, 0.25, 0.5, 0.75};
    std::vector<float> outputTensor(tensor.size());

    uint8_t bw = 8;
    // min is greater than 0, this will be gated at 0.0
    // New range will be 0 to 1.0
    double min = 0.5;
    double max = 1.0;

    sim.quantizeDequantizeTensor(tensor.data(), tensor.size(), outputTensor.data(), min, max, bw,
                                 DlQuantization::RoundingMode::ROUND_NEAREST, false);

    std::vector<float> expectedOutput = {0.0, 0.0, 0.0, 0.25098041, 0.49803925, 0.74901962};

    EXPECT_EQ(outputTensor.size(), expectedOutput.size());

    for (int i = 0; i < outputTensor.size(); i++)
    {
        EXPECT_FLOAT_EQ(outputTensor[i], expectedOutput[i]);
    }
}

TEST(TestTensorQuantizationSim, SanityTestWithGatedMinMaxEqual)
{
    // Instantiate TensorQuantizationSim
    DlQuantization::TensorQuantizationSim<float> sim;

    // Create a dummy tensor
    const std::vector<float> tensor = {-0.5f, -0.25f, 0, 0.25, 0.5, 0.75};
    std::vector<float> outputTensor(tensor.size());

    uint8_t bw = 8;
    // min max are equal
    // New range will be 0.5 to 0.5+1e-5
    double min = 0.5;
    double max = 0.5;

    sim.quantizeDequantizeTensor(tensor.data(), tensor.size(), outputTensor.data(), min, max, bw,
                                 DlQuantization::RoundingMode::ROUND_NEAREST, false);

    std::vector<float> expectedOutput = {0.0, 0.0, 0.0, 0.24901962, 0.5, 0.5};

    EXPECT_EQ(outputTensor.size(), expectedOutput.size());

    for (int i = 0; i < outputTensor.size(); i++)
    {
        EXPECT_FLOAT_EQ(outputTensor[i], expectedOutput[i]);
    }
}

TEST(TestTensorQuantizationSim, SanityTestWithGatedMax)
{
    // Instantiate TensorQuantizationSim
    DlQuantization::TensorQuantizationSim<float> sim;

    // Create a dummy tensor
    const std::vector<float> tensor = {-0.5f, -0.25f, 0, 0.25, 0.5, 0.75};
    std::vector<float> outputTensor(tensor.size());

    uint8_t bw = 8;
    // min is greater than 0, this will be gated at 0.0
    // New range will be -0.5 to 0.0
    double min = -0.5;
    double max = -0.1;

    sim.quantizeDequantizeTensor(tensor.data(), tensor.size(), outputTensor.data(), min, max, bw,
                                 DlQuantization::RoundingMode::ROUND_NEAREST, false);

    std::vector<float> expectedOutput = {-0.5, -0.24901962, 0.0, 0.0, 0.0, 0.0};

    EXPECT_EQ(outputTensor.size(), expectedOutput.size());

    for (int i = 0; i < outputTensor.size(); i++)
    {
        EXPECT_FLOAT_EQ(outputTensor[i], expectedOutput[i]);
    }
}

TEST(TestTensorQuantizationSim, SanityTestWithQuantizeOnlyUnsigned)
{
    // Instantiate TensorQuantizationSim
    DlQuantization::TensorQuantizationSim<float> sim;

    // Create a dummy tensor
    const std::vector<float> tensor = {-0.5f, -0.25f, 0, 0.25, 0.5, 0.75};
    std::vector<float> outputTensor(tensor.size());

    uint8_t bw = 8;
    double min = -0.46;
    double max = 0.72;

    sim.quantizeTensor(tensor.data(), tensor.size(), outputTensor.data(), min, max, bw,
                       DlQuantization::RoundingMode::ROUND_NEAREST, false, false);

    std::vector<float> expectedOutput = {0, 45, 99, 153, 207, 255};

    EXPECT_EQ(outputTensor.size(), expectedOutput.size());

    for (int i = 0; i < outputTensor.size(); i++)
    {
        EXPECT_FLOAT_EQ(outputTensor[i], expectedOutput[i]);
    }
}

TEST(TestTensorQuantizationSim, SanityTestWithDeQuantizeOnlyUnsigned)
{
    // Instantiate TensorQuantizationSim
    DlQuantization::TensorQuantizationSim<float> sim;

    // Create a dummy tensor
    std::vector<uint8_t> tensor             = {0, 45, 99, 153, 207, 255};
    const std::vector<float> expectedOutput = {-0.5f, -0.25f, 0, 0.25, 0.5, 0.75};
    std::vector<float> outputTensor(tensor.size());

    uint8_t bw = 8;
    double min = -0.46;
    double max = 0.72;

    sim.dequantizeTensor(tensor.data(), tensor.size(), outputTensor.data(), min, max, bw, false);


    EXPECT_EQ(outputTensor.size(), expectedOutput.size());

    for (int i = 0; i < outputTensor.size(); i++)
    {
        EXPECT_NEAR(outputTensor[i], expectedOutput[i], 0.06);
    }
}

TEST(TestTensorQuantizationSim, SanityTestWithQuantizeOnlySigned)
{
    // Instantiate TensorQuantizationSim
    DlQuantization::TensorQuantizationSim<float> sim;

    // Create a dummy tensor
    const std::vector<float> tensor = {-0.5f, -0.25f, 0, 0.25, 0.5, 0.75};
    std::vector<float> outputTensor(tensor.size());

    uint8_t bw = 8;
    double min = -0.46;
    double max = 0.72;

    sim.quantizeTensor(tensor.data(), tensor.size(), outputTensor.data(), min, max, bw,
                       DlQuantization::RoundingMode::ROUND_NEAREST, false, true);

    std::vector<float> expectedOutput = {-128, -83, -29, 25, 79, 127};

    EXPECT_EQ(outputTensor.size(), expectedOutput.size());

    for (int i = 0; i < outputTensor.size(); i++)
    {
        EXPECT_FLOAT_EQ(outputTensor[i], expectedOutput[i]);
    }
}

TEST(TestTensorQuantizationSim, SanityTestWithQuantizePackedOnlyUnsigned)
{
    // Instantiate TensorQuantizationSim
    DlQuantization::TensorQuantizationSim<float> sim;

    // Create a dummy tensor
    const std::vector<float> tensor = {-0.5f, -0.25f, 0, 0.25, 0.5, 0.75};
    std::vector<uint8_t> outputTensor(tensor.size());

    uint8_t bw = 8;
    double min = -0.46;
    double max = 0.72;

    sim.quantizeTensorPacked(tensor.data(), tensor.size(), outputTensor, min, max, bw,
                             DlQuantization::RoundingMode::ROUND_NEAREST, false, false);

    std::vector<float> expectedOutput = {0, 45, 99, 153, 207, 255};

    EXPECT_EQ(outputTensor.size(), expectedOutput.size());

    for (int i = 0; i < outputTensor.size(); i++)
    {
        EXPECT_FLOAT_EQ(outputTensor[i], expectedOutput[i]);
    }
}


TEST(TestTensorQuantizationSim, SanityTestWithQuantizePerChannelUnsigned)
{
    // Instantiate TensorQuantizationSim
    DlQuantization::TensorQuantizationSim<float> sim;

    // Create a dummy tensor
    const std::vector<float> tensor = {-0.5f, -0.25f, 0, 0.25, 0.5, 0.75};
    std::vector<uint8_t> outputTensor(tensor.size());

    uint8_t bw = 8;
    double min = -0.46;
    double max = 0.72;


    uint32_t axis = 3;
    std::vector<uint32_t> inputShape {1, 1, 2, 3};

    std::vector<DlQuantization::TfEncoding> encodings;
    std::vector<uint32_t> splitShape {1, 1, 1, 3};
    std::vector<std::vector<float>> splitParams(2, std::vector<float>(3));

    splitParams[0] = {-0.5f, -0.25f, 0};
    splitParams[1] = {0.25, 0.5, 0.75};
    encodings.resize(splitParams.size());
    encodings[0] = {min, max, 0, 0};
    encodings[1] = {min, max, 0, 0};

    ASSERT_EQ(encodings.size(), 2);
    ASSERT_EQ(splitParams.size(), 2);

    std::vector<float> params_quantized(tensor.size());
    sim.quantizePerChannelTensorPacked(splitParams, splitShape, axis, outputTensor, encodings, bw,
                                       DlQuantization::RoundingMode::ROUND_NEAREST, false, false);

    std::vector<uint8_t> expectedOutput = {0, 45, 99, 153, 207, 255};

    EXPECT_EQ(outputTensor.size(), expectedOutput.size());

    for (int i = 0; i < outputTensor.size(); i++)
    {
        EXPECT_NEAR(outputTensor[i], expectedOutput[i], 1e-3);
    }
}

TEST(TestTensorQuantizationSim, SanityTestWithQuantizeDequantizePerChannel)
{
    // Instantiate TensorQuantizationSim
    DlQuantization::TensorQuantizationSim<float> sim;

    // Create a dummy tensor
    const std::vector<float> tensor = {-0.5f, -0.25f, 0, 0.25, 0.5, 0.75, 0, 0.25, 0.5, 0.75, -0.5f, -0.25f};
    std::vector<float> outputTensor(tensor.size());

    uint8_t bw = 8;
    // min is greater than 0, this will be gated at 0.0
    // New range will be -0.5 to 0.0
    double min    = -0.5;
    double max    = -0.1;
    uint32_t axis = 2;
    std::vector<uint32_t> inputShape {1, 1, 2, 6};

    std::vector<DlQuantization::TfEncoding> encodings;
    std::vector<uint32_t> splitShape {1, 6};
    std::vector<std::vector<float>> splitParams(2, std::vector<float>(6));

    splitParams[0] = {-0.5f, -0.25f, 0, 0.25, 0.5, 0.75};
    splitParams[1] = {0, 0.25, 0.5, 0.75, -0.5f, -0.25f};
    encodings.resize(splitParams.size());
    encodings[0] = {min, max, 0, 0};
    encodings[1] = {min, max, 0, 0};

    ASSERT_EQ(encodings.size(), 2);
    ASSERT_EQ(splitParams.size(), 2);

    std::vector<float> params_quantized(tensor.size());
    sim.quantizeDequantizePerChannelTensor(splitParams, splitShape, axis, outputTensor.data(), encodings, bw,
                                           DlQuantization::RoundingMode::ROUND_NEAREST, false);

    std::vector<float> expectedOutput = {-0.5f, 0, -0.24902f, 0, 0, 0, 0, 0, 0, -0.5f, 0, -0.24902f};

    EXPECT_EQ(outputTensor.size(), expectedOutput.size());

    for (int i = 0; i < outputTensor.size(); i++)
    {
        EXPECT_NEAR(outputTensor[i], expectedOutput[i], 1e-3);
    }
}


TEST(TestTensorQuantizationSim, SanityTestWithDequantizePerChannel)
{
    // Instantiate TensorQuantizationSim
    DlQuantization::TensorQuantizationSim<float> sim;

    // Create a dummy tensor
    std::vector<uint8_t> tensor = {0, 45, 99, 153, 207, 255};
    std::vector<float> outputTensor(tensor.size());

    uint8_t bw    = 8;
    double min    = -0.46;
    double max    = 0.72;
    uint32_t axis = 2;
    std::vector<uint32_t> inputShape {1, 1, 2, 3};

    std::vector<DlQuantization::TfEncoding> encodings;
    encodings.resize(2);
    encodings[0] = {min, max, 0, 0};
    encodings[1] = {min, max, 0, 0};

    ASSERT_EQ(encodings.size(), 2);

    std::vector<float> params_quantized(tensor.size());
    sim.dequantizePerChannelTensor(tensor.data(), inputShape, axis, outputTensor.data(), bw, encodings, false);

    std::vector<float> expectedOutput = {-0.5f, -0.25f, 0, 0.25, 0.5, 0.75};

    EXPECT_EQ(outputTensor.size(), expectedOutput.size());

    for (int i = 0; i < outputTensor.size(); i++)
    {
        EXPECT_NEAR(outputTensor[i], expectedOutput[i], 0.06);
    }
}

TEST(TestTensorQuantizationSim, SanityTestWith32BitQuantizeOnlySigned)
{
    // Instantiate TensorQuantizationSim
    DlQuantization::TensorQuantizationSim<float> sim;

    // Create a dummy tensor
    const std::vector<float> tensor = {-1.0};
    std::vector<float> outputTensor(tensor.size());

    uint8_t bw = 32;
    double min = -1.0;
    double max = 1.0;

    sim.quantizeTensor(tensor.data(), tensor.size(), outputTensor.data(), min, max, bw,
                       DlQuantization::RoundingMode::ROUND_NEAREST, false, true);

    std::vector<float> expectedOutput = {-2147483648};
    EXPECT_FLOAT_EQ(outputTensor[0], expectedOutput[0]);
}

TEST(TestTensorQuantizationSim, SanityTestFillEncodingInfo)
{
    DlQuantization::TfEncoding encoding = DlQuantization::TfEncoding();
    encoding.min                        = -5;
    encoding.max                        = 10;
    encoding.delta                      = DlQuantization::computeDelta(encoding.min, encoding.max, 7);
    encoding.offset                     = DlQuantization::computeOffset(encoding.min, encoding.delta);
    encoding.bw                         = 3;

    DlQuantization::TensorQuantizationSim<float> sim;

    sim.fillEncodingInfo(encoding, encoding.bw, encoding.min, encoding.max);

    double expectedEncodingMin = -4.2857142857142857142857142857142;
    double expectedEncodingMax = 10.714285714285714285714285714286;

    EXPECT_DOUBLE_EQ(encoding.min, expectedEncodingMin);
    EXPECT_DOUBLE_EQ(encoding.max, expectedEncodingMax);
}

TEST(TestTensorQuantizationSim, SanityTestFillEncodingInfoNumStepsChange)
{
    DlQuantization::TfEncoding encoding = DlQuantization::TfEncoding();
    encoding.min                        = -5;
    encoding.max                        = 5;
    encoding.delta                      = DlQuantization::computeDelta(encoding.min, encoding.max, 7);
    encoding.offset                     = DlQuantization::computeOffset(encoding.min, encoding.delta);
    encoding.bw                         = 3;

    DlQuantization::TensorQuantizationSim<float> sim;

    sim.fillEncodingInfo(encoding, encoding.bw, encoding.min, encoding.max);

    double expectedEncodingMin = -5;
    double expectedEncodingMax = 5;

    EXPECT_DOUBLE_EQ(encoding.min, expectedEncodingMin);
    EXPECT_DOUBLE_EQ(encoding.max, expectedEncodingMax);
}
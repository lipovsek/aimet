// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdint>
#include <vector>

#include "DlQuantization/Quantization.hpp"
#include "test_quantization_lib.hpp"
#include "trim_functions.hpp"
#include "gtest/gtest.h"

#include <Eigen/Core>

#ifdef GPU_QUANTIZATION_ENABLED
#include "cuda_runtime_api.h"
#endif


template <typename DTYPE>
void launchBlockQdqKernelT(DTYPE* in, DTYPE* out, std::vector<DlQuantization::TfEncoding>& encodings,
                           const DlQuantization::TensorDims& inputShape,
                           const DlQuantization::TensorDims& encodingShape,
                           int64_t numElements, bool useCuda)
{
    void* inputBuffer;
    void* outputBuffer;

    if (useCuda)
    {
#ifdef GPU_QUANTIZATION_ENABLED
        cudaMalloc(&inputBuffer, sizeof(DTYPE) * numElements);
        cudaMalloc(&outputBuffer, sizeof(DTYPE) * numElements);
        // copy input to gpu
        cudaMemcpy(inputBuffer, in, numElements * sizeof(DTYPE), cudaMemcpyHostToDevice);

        DlQuantization::quantizeDequantizeBroadcast((DTYPE*) inputBuffer, (DTYPE*) outputBuffer, encodings, inputShape, encodingShape, DlQuantization::COMP_MODE_GPU);

        // copy output to cpu
        cudaMemcpy(out, outputBuffer, numElements * sizeof(DTYPE), cudaMemcpyDeviceToHost);
        // free gpu memory
        cudaFree(outputBuffer);
        cudaFree(inputBuffer);
#endif
    }
    else
    {
        DlQuantization::quantizeDequantizeBroadcast(in, out, encodings, inputShape, encodingShape, DlQuantization::COMP_MODE_CPU);
    }

}


typedef ::testing::Types<float, Eigen::half> BroadcastQdqTypes;

template <typename T>
class TestBroadcastQdq : public ::testing::Test
{};

TYPED_TEST_SUITE(TestBroadcastQdq, BroadcastQdqTypes);


TYPED_TEST(TestBroadcastQdq, TestQuantizeDequantizeBroadcast)
{
    using DTYPE = TypeParam;
    constexpr int numel = 16;
    DlQuantization::TensorDims inputShape = {2, 2, 2, 2};
    DlQuantization::TensorDims encodingShape = {2, 1, 1, 2};
    const std::vector<int64_t> inputStrides = {8, 4, 2, 1};
    const std::vector<int64_t> encodingStrides = {2, 0, 0, 1};
    const std::vector<float> encodingMax = {63.5, 127.0, 254.0, 508.0};
    const std::vector<float> encodingMin = {-64.0, -128.0, -256.0, -512.0};
    const std::vector<float> encodingDelta = {0.5, 1.0, 2.0, 4.0};
    const std::vector<float> encodingOffset = {-128, -128, -128, -128};
    std::vector<DlQuantization::TfEncoding> encodings(encodingMax.size());

    for (int i = 0; i < encodings.size(); i++)
    {
        encodings[i].min = encodingMin[i];
        encodings[i].max = encodingMax[i];
        encodings[i].delta = encodingDelta[i];
        encodings[i].offset = encodingOffset[i];
    }

    float inputF32[numel] = {
        -125.1, -125.1,    48.3, 48.3,
        68.3, 68.3,       -3.1, -3.1,

        -125.1, -125.1,    48.3, 48.3,
        68.3, 68.3,        -3.1, -3.1,
    };

    DTYPE input[numel];
    DTYPE out[numel];
    for (int i = 0; i < numel; i++)
        input[i] = DTYPE(inputF32[i]);

    float expected[numel] = {
        -64.0, -125.0,     48.5, 48.0,
         63.5, 68.0,       -3.0, -3.0,

        -126.0, -124.0,    48.0, 48.0,
         68.0, 68.0,        -4.0, -4.0

    };



    std::vector<bool> useCuda = {false};
#ifdef GPU_QUANTIZATION_ENABLED
    useCuda.push_back(true);
#endif

    for (auto && c : useCuda)
    {
        // Launch the kernel
        launchBlockQdqKernelT(input, out, encodings, inputShape, encodingShape, numel, c);

        for (int i = 0; i < numel; i++)
        {
            EXPECT_EQ(out[i], DTYPE(expected[i]));
            out[i] = DTYPE(0.); // Clear output
        }
    }
}


TYPED_TEST(TestBroadcastQdq, TestQuantizeDequantizeBroadcast2)
{
    using DTYPE = TypeParam;
    constexpr int numel = 24;
    DlQuantization::TensorDims inputShape = {2, 3, 4};
    DlQuantization::TensorDims encodingShape = {2, 3, 1};
    const std::vector<int64_t> inputStrides = {12, 4, 1};
    const std::vector<int64_t> encodingStrides = {3, 1, 0};
    const std::vector<float> encodingDelta = {0.25, 1.0, 0.5, 2.0, 0.25, 10.0};
    const std::vector<float> encodingOffset = {0, 0, 0, -1, -10, 0};
    const std::vector<float> encodingMin = {0, 0, 0, -2, -2.5, 0};
    const std::vector<float> encodingMax = {255. * 0.25, 255.0, 127.5, 508., 245. * 0.25, 2550.};
    std::vector<DlQuantization::TfEncoding> encodings(encodingMax.size());
    for (int i = 0; i < encodings.size(); i++)
    {
        encodings[i].min = encodingMin[i];
        encodings[i].max = encodingMax[i];
        encodings[i].delta = encodingDelta[i];
        encodings[i].offset = encodingOffset[i];
    }

    float inputF32[numel] = {
        0.126, 10.4, -12.3, 10000,
        0.126, 10.4, -12.3, 10000,
        0.126, 10.4, -12.3, 10000,
        0.126, 10.4, -12.3, 10000,
        0.126, 10.4, -12.3, 10000,
        0.126, 10.4, -12.3, 10000,
    };

    DTYPE input[numel];
    DTYPE out[numel];
    for (int i = 0; i < numel; i++)
        input[i] = DTYPE(inputF32[i]);

    float expected[numel] = {
        0.25, 10.5, 0, 63.75,  // scale = .25
        0., 10., 0., 255.,  // scale = 1
        0., 10.5, 0., 127.5,  // scale = 0.5
        0., 10., -2., 508.,  // scale = 2. offset=-1
        0.25, 10.5, -2.5, 61.25,  // scale = .25
        0., 10., 0, 2550.,  // scale = 10
    };



    std::vector<bool> useCuda = {false};
#ifdef GPU_QUANTIZATION_ENABLED
    useCuda.push_back(true);
#endif


    for (auto && c : useCuda)
    {
        // Launch the kernel
        launchBlockQdqKernelT(input, out, encodings, inputShape, encodingShape, numel, c);

        for (int i = 0; i < numel; i++)
        {
            EXPECT_EQ(out[i], DTYPE(expected[i]));
            out[i] = DTYPE(0.); // Clear output
        }
    }
}


TYPED_TEST(TestBroadcastQdq, TestQuantizeDequantizeBroadcast3)
{
    using DTYPE = TypeParam;
    constexpr int numel = 24;
    DlQuantization::TensorDims inputShape = {4, 2, 3};
    DlQuantization::TensorDims encodingShape = {2, 3};
    const std::vector<int64_t> inputStrides = {6, 3, 1};
    const std::vector<int64_t> encodingStrides = {0, 3, 1};
    const std::vector<float> encodingDelta = {0.25, 1.0, 0.5, 2.0, 0.25, 10.0};
    const std::vector<float> encodingOffset = {0, 0, 0, -1, -10, 0};
    const std::vector<float> encodingMin = {0, 0, 0, -2, -2.5, 0};
    const std::vector<float> encodingMax = {255. * 0.25, 255.0, 127.5, 508., 245. * 0.25, 2550.};
    std::vector<DlQuantization::TfEncoding> encodings(encodingMax.size());

    for (int i = 0; i < encodings.size(); i++)
    {
        encodings[i].min = encodingMin[i];
        encodings[i].max = encodingMax[i];
        encodings[i].delta = encodingDelta[i];
        encodings[i].offset = encodingOffset[i];
    }

    float inputF32[numel] = {
        0.126, 0.126, 0.126, 0.126, 0.126, 0.126,
        10.4,  10.4,  10.4,  10.4,  10.4,  10.4,
        -12.3, -12.3, -12.3, -12.3, -12.3, -12.3,
        10000, 10000, 10000, 10000, 10000, 10000,
    };

    DTYPE input[numel];
    DTYPE out[numel];
    for (int i = 0; i < numel; i++)
        input[i] = DTYPE(inputF32[i]);

    float expected[numel] = {
    //  0.25     1.0      0.5      2.0      .25      10.0
        0.25,    0.,      0.,      0.,      0.25,    0.,
        10.5,    10.,     10.5,    10.,     10.5,    10.,
        -0.,     0.,      0.,      -2,      -2.5,    0.,
        63.75,   255.,    127.5,   508,     61.25,   2550,
    };



    std::vector<bool> useCuda = {false};
#ifdef GPU_QUANTIZATION_ENABLED
    useCuda.push_back(true);
#endif


    for (auto && c : useCuda)
    {
        // Launch the kernel
        launchBlockQdqKernelT(input, out, encodings, inputShape, encodingShape, numel, c);

        for (int i = 0; i < numel; i++)
        {
            EXPECT_EQ(out[i], DTYPE(expected[i]));
            out[i] = DTYPE(0.); // Clear output
        }
    }
}


TEST(TestOnnxTensorOps, TestQuantizeDequantizeFp16)
{
    constexpr int numel = 6;

    DlQuantization::TfEncoding encoding = getTfEncoding(-0.5, 0.775, 8);

    float inputF32[numel] = {-0.501, -0.2501, 0., 0.2501, 0.501, 0.8};
    Eigen::half input[numel];
    Eigen::half output[numel];
    for (int i = 0; i < numel; i++)
        input[i] = Eigen::half(inputF32[i]);

    // fp32 reference
    float refOutput[numel];
    DlQuantization::quantizeDequantize(inputF32, (uint64_t) numel, encoding, refOutput,
                                       DlQuantization::COMP_MODE_CPU,
                                       DlQuantization::ROUND_NEAREST, nullptr);

    // fp16
    DlQuantization::quantizeDequantize(input, (uint64_t) numel, encoding, output,
                                       DlQuantization::COMP_MODE_CPU,
                                       DlQuantization::ROUND_NEAREST, nullptr);

    for (int i = 0; i < numel; i++)
    {
        EXPECT_NEAR(float(output[i]), refOutput[i], 0.01);
    }
}


TEST(TestOnnxTensorOps, TestQuantizeDequantizeBFloat16)
{
    constexpr int numel = 6;

    DlQuantization::TfEncoding encoding = getTfEncoding(-0.5, 0.775, 8);

    float inputF32[numel] = {-0.501, -0.2501, 0., 0.2501, 0.501, 0.8};
    Eigen::bfloat16 input[numel];
    Eigen::bfloat16 output[numel];
    for (int i = 0; i < numel; i++)
        input[i] = Eigen::bfloat16(inputF32[i]);

    // fp32 reference
    float refOutput[numel];
    DlQuantization::quantizeDequantize(inputF32, (uint64_t) numel, encoding, refOutput, DlQuantization::COMP_MODE_CPU,
                                       DlQuantization::ROUND_NEAREST, nullptr);

    // bf16
    DlQuantization::quantizeDequantize(input, (uint64_t) numel, encoding, output, DlQuantization::COMP_MODE_CPU,
                                       DlQuantization::ROUND_NEAREST, nullptr);

    // bf16 has 7 mantissa bits (vs fp16's 10), so use a looser tolerance.
    for (int i = 0; i < numel; i++)
    {
        EXPECT_NEAR(float(output[i]), refOutput[i], 0.04);
    }
}


#ifdef GPU_QUANTIZATION_ENABLED
TEST(TestOnnxTensorOps, TestQuantizeDequantizeFp16Gpu)
{
    constexpr int numel = 6;

    DlQuantization::TfEncoding encoding = getTfEncoding(-0.5, 0.775, 8);

    float inputF32[numel] = {-0.501, -0.2501, 0., 0.2501, 0.501, 0.8};
    Eigen::half input[numel];
    for (int i = 0; i < numel; i++)
        input[i] = Eigen::half(inputF32[i]);

    // fp32 reference (CPU)
    float refOutput[numel];
    DlQuantization::quantizeDequantize(inputF32, (uint64_t) numel, encoding, refOutput,
                                       DlQuantization::COMP_MODE_CPU,
                                       DlQuantization::ROUND_NEAREST, nullptr);

    // fp16 on GPU
    Eigen::half* gpuIn;
    Eigen::half* gpuOut;
    cudaMalloc((void**) &gpuIn, sizeof(Eigen::half) * numel);
    cudaMalloc((void**) &gpuOut, sizeof(Eigen::half) * numel);
    cudaMemcpy(gpuIn, input, sizeof(Eigen::half) * numel, cudaMemcpyHostToDevice);

    DlQuantization::quantizeDequantize(gpuIn, (uint64_t) numel, encoding, gpuOut,
                                       DlQuantization::COMP_MODE_GPU,
                                       DlQuantization::ROUND_NEAREST, nullptr);

    Eigen::half output[numel];
    cudaMemcpy(output, gpuOut, sizeof(Eigen::half) * numel, cudaMemcpyDeviceToHost);
    cudaFree(gpuIn);
    cudaFree(gpuOut);

    for (int i = 0; i < numel; i++)
    {
        EXPECT_NEAR(float(output[i]), refOutput[i], 0.01);
    }
}


TEST(TestOnnxTensorOps, TestQuantizeDequantizeBFloat16Gpu)
{
    constexpr int numel = 6;

    DlQuantization::TfEncoding encoding = getTfEncoding(-0.5, 0.775, 8);

    float inputF32[numel] = {-0.501, -0.2501, 0., 0.2501, 0.501, 0.8};
    Eigen::bfloat16 input[numel];
    for (int i = 0; i < numel; i++)
        input[i] = Eigen::bfloat16(inputF32[i]);

    // fp32 reference (CPU)
    float refOutput[numel];
    DlQuantization::quantizeDequantize(inputF32, (uint64_t) numel, encoding, refOutput, DlQuantization::COMP_MODE_CPU,
                                       DlQuantization::ROUND_NEAREST, nullptr);

    // bf16 on GPU
    Eigen::bfloat16* gpuIn;
    Eigen::bfloat16* gpuOut;
    cudaMalloc((void**) &gpuIn, sizeof(Eigen::bfloat16) * numel);
    cudaMalloc((void**) &gpuOut, sizeof(Eigen::bfloat16) * numel);
    cudaMemcpy(gpuIn, input, sizeof(Eigen::bfloat16) * numel, cudaMemcpyHostToDevice);

    DlQuantization::quantizeDequantize(gpuIn, (uint64_t) numel, encoding, gpuOut, DlQuantization::COMP_MODE_GPU,
                                       DlQuantization::ROUND_NEAREST, nullptr);

    Eigen::bfloat16 output[numel];
    cudaMemcpy(output, gpuOut, sizeof(Eigen::bfloat16) * numel, cudaMemcpyDeviceToHost);
    cudaFree(gpuIn);
    cudaFree(gpuOut);

    // bf16 has 7 mantissa bits (vs fp16's 10), so use a looser tolerance.
    for (int i = 0; i < numel; i++)
    {
        EXPECT_NEAR(float(output[i]), refOutput[i], 0.04);
    }
}
#endif

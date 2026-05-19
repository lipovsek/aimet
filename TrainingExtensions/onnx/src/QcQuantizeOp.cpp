// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "QcQuantizeOp.h"
#include "AimetOpUtils.h"
#include "DlQuantization/IForLoopRunner.h"
#include "trim_functions.hpp"

#include <iostream>
#include <type_traits>
#include <vector>


// Map ORT 16-bit float wrapper types onto Eigen's native low-precision types for kernel dispatch.
template <typename T>
using OrtToNativeT = std::conditional_t<std::is_same_v<T, Ort::Float16_t>, Eigen::half,
                                        std::conditional_t<std::is_same_v<T, Ort::BFloat16_t>, Eigen::bfloat16, T>>;

template <typename T>
inline constexpr bool is_16bit_float_v = std::is_same_v<T, Ort::Float16_t> || std::is_same_v<T, Ort::BFloat16_t>;


// Concrete IForLoopRunner that delegates to ORT's intra-op thread pool.
class OrtForLoopRunner : public DlQuantization::IForLoopRunner
{
public:
    explicit OrtForLoopRunner(OrtKernelContext* ctx) : _ctx(ctx) {}

    void run(std::function<void(size_t)> fn, size_t numChunks) const override
    {
        Ort::KernelContext ctx(_ctx);
        // Bridge from std::function to the C function pointer that ORT expects.
        ctx.ParallelFor(
            [](void* userData, size_t i) {
                (*reinterpret_cast<std::function<void(size_t)>*>(userData))(i);
            },
            numChunks, 0, &fn);
    }

private:
    OrtKernelContext* _ctx;
};


#ifdef ONNX_CUDA
static DlQuantization::CudaAllocator cudaAllocator;
#endif
static DlQuantization::CpuAllocator cpuAllocator;

template<typename T>
QcQuantizeOp<T>::QcQuantizeOp(const OrtApi* api, const OrtKernelInfo* info) : api_(*api), info_(info)
{
    int64_t quantInfoPointer;
    (void) api->KernelInfoGetAttribute_int64(info_, "quant_info", &quantInfoPointer);
    quantInfo = reinterpret_cast<struct QcQuantizeInfo*>(quantInfoPointer);
}

template <typename T>
void QcQuantizeOp<T>::computeImpl(const Ort::Custom::Tensor<T>& input, Ort::Custom::Tensor<T>& output, void* stream,
                                  bool useCuda, DlQuantization::IAllocator* allocator, OrtKernelContext* ortCtx)
{
    // Setup inputs
    auto inputShape = input.Shape();
    auto size       = input.NumberOfElement();
    auto result     = output.Allocate(inputShape);

    DlQuantization::TensorQuantizerOpMode opMode = quantInfo->opMode;
    // Disable unused quantizers
    if (!quantInfo->enabled)
    {
        opMode = DlQuantization::TensorQuantizerOpMode::passThrough;
    }

    using DTYPE           = OrtToNativeT<T>;
    const DTYPE* inputPtr = reinterpret_cast<const DTYPE*>(input.Data());
    DTYPE* resultPtr      = reinterpret_cast<DTYPE*>(result);

    // Create parallel for-loop runner from ORT kernel context (CPU only)
    std::unique_ptr<OrtForLoopRunner> runner;
    if (ortCtx && !useCuda)
    {
        runner = std::make_unique<OrtForLoopRunner>(ortCtx);
    }

    if (quantInfo->isIntDataType)
    {
        modeSpecificActionBroadcastInt(inputPtr, resultPtr, inputShape, quantInfo->tensorQuantizer.get(), opMode,
            quantInfo->useSymmetricEncoding, allocator, useCuda, stream, runner.get());
    }
    else
    {
        if constexpr (is_16bit_float_v<T>)
        {
            // Data is already in native low-precision float: no quantization simulation needed, just copy
            copyInputTensorsToOutputTensors(inputPtr, size, resultPtr, useCuda, stream);
        }
        else
        {
            modeSpecificActionFloat(inputPtr, size, resultPtr, opMode, allocator, useCuda, stream);
        }
    }

    // We only ever need to run in oneShotQuantizeDequantize once, afterwards just use quantizeDequantize
    if (opMode == DlQuantization::TensorQuantizerOpMode::oneShotQuantizeDequantize)
    {
        quantInfo->opMode = DlQuantization::TensorQuantizerOpMode::quantizeDequantize;
    }
}

template<typename T>
struct QcQuantizeOpCpu : QcQuantizeOp<T>
{
    using QcQuantizeOp<T>::QcQuantizeOp;

    void Compute(OrtKernelContext* ctx, const Ort::Custom::Tensor<T>& input, Ort::Custom::Tensor<T>& output)
    {
        this->computeImpl(input, output, nullptr, false, &cpuAllocator, ctx);
    }
};


#ifdef ONNX_CUDA

template<typename T>
struct QcQuantizeOpCuda : QcQuantizeOp<T>
{
    using QcQuantizeOp<T>::QcQuantizeOp;

    void Compute(const Ort::Custom::CudaContext& cuda_ctx, const Ort::Custom::Tensor<T>& input,
                 Ort::Custom::Tensor<T>& output)
    {
        cudaStream_t stream = cuda_ctx.cuda_stream;
        this->computeImpl(input, output, stream, true, &cudaAllocator);
    }
};

#endif


void RegisterOps(Ort::CustomOpDomain& domain)
{
    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> qcQuantCpuOpPointer {
        Ort::Custom::CreateLiteCustomOp<QcQuantizeOpCpu<float>>("QcQuantizeOp", "CPUExecutionProvider")};
    domain.Add(qcQuantCpuOpPointer.get());

    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> qcQuantCpuOpFloat16Pointer {
        Ort::Custom::CreateLiteCustomOp<QcQuantizeOpCpu<Ort::Float16_t>>("QcQuantizeOp", "CPUExecutionProvider")};
    domain.Add(qcQuantCpuOpFloat16Pointer.get());

    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> qcQuantCpuOpBFloat16Pointer {
        Ort::Custom::CreateLiteCustomOp<QcQuantizeOpCpu<Ort::BFloat16_t>>("QcQuantizeOp", "CPUExecutionProvider")};
    domain.Add(qcQuantCpuOpBFloat16Pointer.get());
#ifdef ONNX_CUDA
    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> qcQuantCudaOpPointer {
        Ort::Custom::CreateLiteCustomOp<QcQuantizeOpCuda<float>>("QcQuantizeOp", "CUDAExecutionProvider")};
    domain.Add(qcQuantCudaOpPointer.get());

    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> qcQuantCudaOpFloat16Pointer {
        Ort::Custom::CreateLiteCustomOp<QcQuantizeOpCuda<Ort::Float16_t>>("QcQuantizeOp", "CUDAExecutionProvider")};
    domain.Add(qcQuantCudaOpFloat16Pointer.get());

    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> qcQuantCudaOpBFloat16Pointer {
        Ort::Custom::CreateLiteCustomOp<QcQuantizeOpCuda<Ort::BFloat16_t>>("QcQuantizeOp", "CUDAExecutionProvider")};
    domain.Add(qcQuantCudaOpBFloat16Pointer.get());
#endif
}

// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef AIMET_MAIN_QCQUANTIZEOP_H
#define AIMET_MAIN_QCQUANTIZEOP_H

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "QcQuantizeInfo.h"

#ifdef ONNX_CUDA
// IMPORTANT: cuda_context.h needs to be included before onnxruntime_lite_custom_op.h
#include "core/providers/cuda/cuda_context.h"
#endif

#include "onnxruntime_lite_custom_op.h"

template<typename T>
struct QcQuantizeOp
{
    QcQuantizeOp(const OrtApi* api, const OrtKernelInfo* info);

    void computeImpl(const Ort::Custom::Tensor<T>& input, Ort::Custom::Tensor<T>& output, void* stream,
                     bool useCuda, DlQuantization::IAllocator* allocator,
                     OrtKernelContext* ortCtx = nullptr);

protected:
    struct QcQuantizeInfo* quantInfo;

private:
    std::unique_ptr<DlQuantization::ITensorQuantizationSim<float>> tensorQuantizationSim;
    const OrtKernelInfo* info_;
    OrtApi api_;
};


void RegisterOps(Ort::CustomOpDomain& domain);

#endif   // AIMET_MAIN_QCQUANTIZEOP_H

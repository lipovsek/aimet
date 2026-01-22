// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "RegisterCustomOps.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <vector>
#include <cmath>
#include <mutex>
#include <system_error>

#include "QcQuantizeOp.h"
#include "AimetOpUtils.h"
#include "onnxruntime_lite_custom_op.h"

static const char* c_OpDomain    = "aimet.customop.cpu";
static const char* c_OpDomainGPU = "aimet.customop.cuda";


OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api)
{

    Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);
    OrtStatus* result = nullptr;
    try
    {
        Ort::CustomOpDomain domain {c_OpDomain};
        RegisterOps(domain);

        Ort::UnownedSessionOptions session_options(options);
        session_options.Add(domain);
        AddOrtCustomOpDomainToContainer(std::move(domain));

#ifdef ONNX_CUDA
        // This is for backward compatibility, in the new custom OP API we do not need separate domains for cpu/gpu
        Ort::CustomOpDomain cuda_domain {c_OpDomainGPU};
        RegisterOps(cuda_domain);
        session_options.Add(cuda_domain);
        AddOrtCustomOpDomainToContainer(std::move(cuda_domain));
#endif
    }
    catch(const std::exception& e)
    {
        Ort::Status status{e};
        result = status.release();
    }

    return result;

}

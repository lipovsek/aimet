// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef AIMET_REGISTERCUSTOMOPS_H
#define AIMET_REGISTERCUSTOMOPS_H

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

extern "C" {

#ifdef _WIN32
__declspec(dllexport)
#endif
OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api);

}


#endif   // AIMET_REGISTERCUSTOMOPS_H

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

add_custom_target(AcceptanceTests)

if (ENABLE_TORCH AND NOT ENABLE_ONNX)
    add_dependencies(AcceptanceTests
            AcceptanceTests.Torch)
endif (ENABLE_TORCH AND NOT ENABLE_ONNX)

if (ENABLE_ONNX)
    add_dependencies(AcceptanceTests
            AcceptanceTests.ONNX)
endif(ENABLE_ONNX)

if (ENABLE_TORCH AND NOT ENABLE_ONNX)
    add_subdirectory(torch)
endif (ENABLE_TORCH AND NOT ENABLE_ONNX)

if (ENABLE_ONNX)
    add_subdirectory(onnx)
endif(ENABLE_ONNX)

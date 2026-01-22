# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

if (ENABLE_CUDA)
    set(CUDA_FLAG "not blah")
    set(USE_CUDA True)
else (ENABLE_CUDA)
    set(CUDA_FLAG "not cuda")
    set(USE_CUDA False)
endif (ENABLE_CUDA)

add_custom_target(AcceptanceTests.ONNX
      VERBATIM COMMAND ${CMAKE_COMMAND} -E env
      "${AIMET_PYTHONPATH}"
      ${Python3_EXECUTABLE} -m pytest -s ${CMAKE_CURRENT_SOURCE_DIR} -m ${CUDA_FLAG} --junitxml=${CMAKE_CURRENT_BINARY_DIR}/py_test_output.xml)
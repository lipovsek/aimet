# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

add_test(CommonTrainingExtensionTest
      ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR} --junitxml=${CMAKE_CURRENT_BINARY_DIR}/py_test_output.xml
      )
if(NOT DEFINED ENV{FAST_RELEASE})
    # When building PyPI wheel packages (aka FAST_RELEASE),
    # we only want to test the pure wheel package without relying on any other build artifacts
    set_property(TEST CommonTrainingExtensionTest
          APPEND PROPERTY ENVIRONMENT "${AIMET_PYTHONPATH}")
endif()

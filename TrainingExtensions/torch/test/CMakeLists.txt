# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

if (ENABLE_CUDA)
    add_test(TorchTrainingExtensionTest
            ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR} --junitxml=${CMAKE_CURRENT_BINARY_DIR}/py_test_output.xml
            )

else (ENABLE_CUDA)
    add_test(TorchTrainingExtensionTest
            ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR} -m "not cuda" --junitxml=${CMAKE_CURRENT_BINARY_DIR}/py_test_output.xml
            )

endif (ENABLE_CUDA)

if(NOT DEFINED ENV{FAST_RELEASE})
    # When building PyPI wheel packages (aka FAST_RELEASE),
    # we only want to test the pure wheel package without relying on any other build artifacts
    set_property(TEST TorchTrainingExtensionTest
          APPEND PROPERTY ENVIRONMENT "${AIMET_PYTHONPATH}")
endif()

if(DEFINED ENV{DEPENDENCY_DATA_PATH})
set_property(TEST TorchTrainingExtensionTest
      APPEND PROPERTY ENVIRONMENT "${DEPENDENCY_DATA_PATH}")
endif(DEFINED ENV{DEPENDENCY_DATA_PATH})


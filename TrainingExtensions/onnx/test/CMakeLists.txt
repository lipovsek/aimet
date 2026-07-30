# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

add_executable(OnnxCppOpsTest
        TestFp8QcQuantizeRuntime.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/AimetOpUtils.cpp
        )

target_include_directories(OnnxCppOpsTest PRIVATE
        ${ONNXRUNTIME_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../ModelOptimizations/DlQuantization/src
        )

target_link_libraries(OnnxCppOpsTest
        MoDlQuantization
        Eigen3::Eigen
        gtest_main
        )

if (ENABLE_CUDA)
    target_compile_options(OnnxCppOpsTest
            PRIVATE
                -DGPU_QUANTIZATION_ENABLED=1
            )

    target_link_libraries(OnnxCppOpsTest
            MoDlQuantizationCuda
            )
endif (ENABLE_CUDA)

add_test(NAME OnnxCppOpsTest
        COMMAND $<TARGET_FILE:OnnxCppOpsTest>
        )

if (ENABLE_CUDA)
    add_test(OnnxTrainingExtensionTest
            ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR} --junitxml=${CMAKE_CURRENT_BINARY_DIR}/py_test_output.xml
            )

else (ENABLE_CUDA)
    add_test(OnnxTrainingExtensionTest
            ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR} -m "not cuda" --junitxml=${CMAKE_CURRENT_BINARY_DIR}/py_test_output.xml
            )

endif (ENABLE_CUDA)

if(DEFINED ENV{DEPENDENCY_DATA_PATH})
    set_property(TEST OnnxTrainingExtensionTest
        APPEND PROPERTY ENVIRONMENT "${DEPENDENCY_DATA_PATH}")
endif(DEFINED ENV{DEPENDENCY_DATA_PATH})

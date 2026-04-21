# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

add_subdirectory(src/python)

if (ENABLE_TESTS)
    add_subdirectory(test)
endif()

if (ENABLE_ONNX)
    add_custom_target(whl_prep_cp_common DEPENDS
        whl_prep_cp_common_DlCompression
        whl_prep_cp_common_DlEqualization
        whl_prep_cp_common_DlQuantization
        whl_prep_cp_common_PyModelOptimizations
    )
    add_dependencies(whl_prep_cp whl_prep_cp_common)

    add_custom_target(whl_prep_ln_common DEPENDS
        whl_prep_ln_common_DlCompression
        whl_prep_ln_common_DlEqualization
        whl_prep_ln_common_DlQuantization
        whl_prep_ln_common_PyModelOptimizations
    )
endif()

message(STATUS "Package rename command: ${src_packaging_dir}/convert_wheel_format.sh ${CMAKE_BINARY_DIR}")
execute_process(
    COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/src/python/_aimet_common/_gen.py"
            --output-dir ${CMAKE_BINARY_DIR}/artifacts/_aimet_common
            --build-dir ${CMAKE_BINARY_DIR}
)
if(SKBUILD)
    if (ENABLE_TORCH)
        install(DIRECTORY ${CMAKE_BINARY_DIR}/artifacts/_aimet_common/
                DESTINATION aimet_torch/common
                FILES_MATCHING
                PATTERN "_version.py")
    endif()
    if (ENABLE_ONNX)
        install(DIRECTORY ${CMAKE_BINARY_DIR}/artifacts/_aimet_common/
                DESTINATION aimet_onnx/common
                FILES_MATCHING
                PATTERN "_version.py")
    endif()
else()
install(DIRECTORY ${CMAKE_BINARY_DIR}/artifacts/_aimet_common/
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/_aimet_common
        FILES_MATCHING
        PATTERN "_version.py"
      )
endif()

if (ENABLE_ONNX)
    add_dependencies(whl_prep_ln whl_prep_ln_common)
endif()
whl_add_whl_action_target(common)

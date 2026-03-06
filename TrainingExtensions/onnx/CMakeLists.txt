# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

add_subdirectory(src/python)

if (ENABLE_TESTS)
    add_subdirectory(test)
endif()

add_library(OnnxCppOps SHARED
        src/QcQuantizeOp.h
        src/AimetOpUtils.h
        src/AimetOpUtils.cpp
        src/RegisterCustomOps.h
        src/RegisterCustomOps.cpp
        src/QcQuantizeInfo.cpp
        src/QcQuantizeInfo.h
        src/QcQuantizeOp.cpp
        src/OnnxOpUtils.h
)

target_include_directories(OnnxCppOps PRIVATE
        ${ONNXRUNTIME_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../ModelOptimizations/DlQuantization/src
        )

target_link_libraries(OnnxCppOps PUBLIC
        MoDlQuantization
        Eigen3::Eigen
        )

if (ENABLE_CUDA)
    target_compile_options(OnnxCppOps PRIVATE
            -DONNX_CUDA=1
            )

    target_link_libraries(OnnxCppOps PUBLIC
            MoDlQuantizationCuda
            )
endif (ENABLE_CUDA)

if (SKBUILD)
    set_target_properties(OnnxCppOps PROPERTIES
        PREFIX "lib"
        OUTPUT_NAME "aimet_onnxrt_ops"
        )

    # Set RPATH using platform-specific syntax
    if (APPLE)
        set_target_properties(OnnxCppOps PROPERTIES
            INSTALL_RPATH "@loader_path"
            )
    elseif (NOT WIN32)
        set_target_properties(OnnxCppOps PROPERTIES
            INSTALL_RPATH "\$ORIGIN"
            )
    endif()

    install(TARGETS OnnxCppOps
            DESTINATION aimet_onnx/common)

else (SKBUILD)
    set_target_properties(OnnxCppOps PROPERTIES
        PREFIX "lib"
        OUTPUT_NAME "aimet_onnxrt_ops"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/artifacts/_aimet_common"
        )

install(TARGETS OnnxCppOps
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/_aimet_common
        )

endif (SKBUILD)

# ----

Python3_add_library(quant_info SHARED
        src/QcQuantizeInfo.h
        src/QcQuantizeInfo.cpp)

    target_link_libraries(quant_info PUBLIC
            MoDlQuantization
            )

if (ENABLE_CUDA)
    target_link_libraries(quant_info PUBLIC
            MoDlQuantizationCuda
            )
endif (ENABLE_CUDA)

if (SKBUILD)
    set_target_properties(quant_info
        PROPERTIES
        PREFIX "lib"
        OUTPUT_NAME "quant_info"
        )

    # Set RPATH using platform-specific syntax
    if (APPLE)
        set_target_properties(quant_info PROPERTIES
            INSTALL_RPATH "@loader_path"
            )
    elseif (NOT WIN32)
        set_target_properties(quant_info PROPERTIES
            INSTALL_RPATH "\$ORIGIN"
            )
    endif()

    install(TARGETS quant_info
            DESTINATION aimet_onnx/common)
else (SKBUILD)

set_target_properties(quant_info
        PROPERTIES
        PREFIX "lib"
        OUTPUT_NAME "quant_info"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/artifacts/_aimet_common/"
        )

install(TARGETS quant_info
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/_aimet_common
        )

# ----

if (WIN32)
    add_custom_target(whl_prep_cp_onnx
        COMMAND ${CMAKE_COMMAND} -E make_directory ${WHL_PREP_AIMET_ONNX_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:OnnxCppOps> ${WHL_PREP_AIMET_ONNX_DIR}/$<TARGET_FILE_NAME:OnnxCppOps>
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:quant_info> ${WHL_PREP_AIMET_ONNX_DIR}/$<TARGET_FILE_NAME:quant_info>
        DEPENDS OnnxCppOps quant_info
    )
    add_custom_target(whl_prep_ln_onnx
        COMMAND ${CMAKE_COMMAND} -E make_directory ${WHL_PREP_AIMET_ONNX_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:OnnxCppOps> ${WHL_PREP_AIMET_ONNX_DIR}/$<TARGET_FILE_NAME:OnnxCppOps>
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:quant_info> ${WHL_PREP_AIMET_ONNX_DIR}/$<TARGET_FILE_NAME:quant_info>
        DEPENDS OnnxCppOps quant_info
    )
elseif (APPLE)
    # MacOS: Use install_name_tool to set RPATH
    add_custom_target(whl_prep_cp_onnx
        COMMAND ${CMAKE_COMMAND} -E make_directory ${WHL_PREP_AIMET_ONNX_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:OnnxCppOps> ${WHL_PREP_AIMET_ONNX_DIR}/$<TARGET_FILE_NAME:OnnxCppOps>
        COMMAND install_name_tool -add_rpath '@loader_path' ${WHL_PREP_AIMET_ONNX_DIR}/$<TARGET_FILE_NAME:OnnxCppOps>
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:quant_info> ${WHL_PREP_AIMET_ONNX_DIR}/$<TARGET_FILE_NAME:quant_info>
        COMMAND install_name_tool -add_rpath '@loader_path' ${WHL_PREP_AIMET_ONNX_DIR}/$<TARGET_FILE_NAME:quant_info>
        DEPENDS OnnxCppOps quant_info
    )
    add_custom_target(whl_prep_ln_onnx
        COMMAND ${CMAKE_COMMAND} -E make_directory ${WHL_PREP_AIMET_ONNX_DIR}
        COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:OnnxCppOps> ${WHL_PREP_AIMET_ONNX_DIR}/$<TARGET_FILE_NAME:OnnxCppOps>
        COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:quant_info> ${WHL_PREP_AIMET_ONNX_DIR}/$<TARGET_FILE_NAME:quant_info>
        DEPENDS OnnxCppOps quant_info
    )
else()
    # Linux: Use patchelf to set RPATH
    add_custom_target(whl_prep_cp_onnx
        COMMAND ${CMAKE_COMMAND} -E make_directory ${WHL_PREP_AIMET_ONNX_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:OnnxCppOps> ${WHL_PREP_AIMET_ONNX_DIR}/$<TARGET_FILE_NAME:OnnxCppOps>
        COMMAND ${PATCHELF_EXE} --set-rpath '$$ORIGIN' ${WHL_PREP_AIMET_ONNX_DIR}/$<TARGET_FILE_NAME:OnnxCppOps>
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:quant_info> ${WHL_PREP_AIMET_ONNX_DIR}/$<TARGET_FILE_NAME:quant_info>
        COMMAND ${PATCHELF_EXE} --set-rpath '$$ORIGIN' ${WHL_PREP_AIMET_ONNX_DIR}/$<TARGET_FILE_NAME:quant_info>
        DEPENDS OnnxCppOps quant_info
    )
    add_custom_target(whl_prep_ln_onnx
        COMMAND ${CMAKE_COMMAND} -E make_directory ${WHL_PREP_AIMET_ONNX_DIR}
        COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:OnnxCppOps> ${WHL_PREP_AIMET_ONNX_DIR}/$<TARGET_FILE_NAME:OnnxCppOps>
        COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:quant_info> ${WHL_PREP_AIMET_ONNX_DIR}/$<TARGET_FILE_NAME:quant_info>
        DEPENDS OnnxCppOps quant_info
    )
endif()

add_dependencies(whl_prep_cp whl_prep_cp_onnx)
add_dependencies(whl_prep_ln whl_prep_ln_onnx)

endif (SKBUILD)

if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set_target_properties(quant_info
        PROPERTIES
        SUFFIX ".${Python3_SOABI}.pyd"
        )
else()
    set_target_properties(quant_info
        PROPERTIES
        SUFFIX ".${Python3_SOABI}.so"
        )
endif()

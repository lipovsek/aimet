# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

project(aimet_common)

if(ENABLE_ONNX AND NOT TARGET PyModelOptimizations)
    add_subdirectory("../../../../" "${CMAKE_CURRENT_BINARY_DIR}/aimet" EXCLUDE_FROM_ALL)
    install(
        FILES $<TARGET_FILE:PyModelOptimizations>
        DESTINATION ${CMAKE_INSTALL_PREFIX}
    )
elseif(TARGET PyModelOptimizations OR NOT ENABLE_ONNX)
    if(SKBUILD)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/aimet_common
        DESTINATION .
        PATTERN "__pycache*" EXCLUDE
        PATTERN "CMakeLists.txt" EXCLUDE
    )
        if (ENABLE_TORCH)
            install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/_aimet_common/
                DESTINATION aimet_torch/common
                PATTERN "__pycache*" EXCLUDE
                PATTERN "CMakeLists.txt" EXCLUDE
                PATTERN "_version.pyi" EXCLUDE
                PATTERN "_gen.py" EXCLUDE
               )
        endif()
        if (ENABLE_ONNX)
            install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/_aimet_common/
                DESTINATION aimet_onnx/common
                PATTERN "__pycache*" EXCLUDE
                PATTERN "CMakeLists.txt" EXCLUDE
                PATTERN "_version.pyi" EXCLUDE
                PATTERN "_gen.py" EXCLUDE
               )
        endif()
    else()
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PATTERN "__pycache*" EXCLUDE
        PATTERN "CMakeLists.txt" EXCLUDE
        PATTERN "_version.pyi" EXCLUDE
        PATTERN "_gen.py" EXCLUDE
        )
    endif()
endif()

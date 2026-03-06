# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

Python3_add_library(PyModelOptimizations SHARED
        PyModelOptimizations.cpp
        PyTensorQuantizer.cpp)

target_link_libraries(PyModelOptimizations PUBLIC
        MoDlQuantization
        )

if (ENABLE_CUDA)
    target_link_libraries(PyModelOptimizations PUBLIC
            MoDlQuantizationCuda
            )
endif (ENABLE_CUDA)

set_target_properties(PyModelOptimizations
      PROPERTIES
         OUTPUT_NAME "pymo"
         PREFIX "_lib"
      )

add_dependencies(PyModelOptimizations generate_deps)

if (SKBUILD)
    if (APPLE)
        # MacOS: Use @loader_path instead of $ORIGIN
        set_target_properties(PyModelOptimizations PROPERTIES
            INSTALL_RPATH "@loader_path:@loader_path/../../.."
            )
    elseif (NOT WIN32)
        # Linux: Use $ORIGIN
        set_target_properties(PyModelOptimizations PROPERTIES
            INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../../.."
            )
    endif()
    if (ENABLE_TORCH)
        install(TARGETS PyModelOptimizations
                DESTINATION aimet_torch/common)
    endif()
    if (ENABLE_ONNX)
        install(TARGETS PyModelOptimizations
                DESTINATION aimet_onnx/common)
    endif()
else (SKBUILD)
    set_target_properties(PyModelOptimizations
          PROPERTIES
             LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/artifacts/_aimet_common/"
          )

    install(TARGETS PyModelOptimizations
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/_aimet_common
    )

    add_custom_target(whl_prep_cp_common_PyModelOptimizations
            COMMAND ${CMAKE_COMMAND} -E make_directory ${WHL_PREP_AIMET_COMMON_DIR}
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:PyModelOptimizations> ${WHL_PREP_AIMET_COMMON_DIR}/$<TARGET_FILE_NAME:PyModelOptimizations>
            DEPENDS PyModelOptimizations
    )
    add_custom_target(whl_prep_ln_common_PyModelOptimizations
            COMMAND ${CMAKE_COMMAND} -E make_directory ${WHL_PREP_AIMET_COMMON_DIR}
            COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:PyModelOptimizations> ${WHL_PREP_AIMET_COMMON_DIR}/$<TARGET_FILE_NAME:PyModelOptimizations>
            DEPENDS PyModelOptimizations
    )
endif (SKBUILD)

if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set_target_properties(PyModelOptimizations
          PROPERTIES
          SUFFIX ".${Python3_SOABI}.pyd"
          )
else()
    set_target_properties(PyModelOptimizations
          PROPERTIES
          SUFFIX ".${Python3_SOABI}.so"
          )
endif()

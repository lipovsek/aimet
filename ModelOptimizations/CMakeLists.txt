# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

if (WIN32)
    set(AIMET_PLATFORM_DIR "x86_64-windows")
else()
    set(AIMET_PLATFORM_DIR "x86_64-linux-gnu")
endif()

macro(whl_add_whl_prep_common_target name)
    add_custom_target(whl_prep_cp_common_${name}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${WHL_PREP_AIMET_COMMON_DIR}/${AIMET_PLATFORM_DIR}/include
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/include/${name} ${WHL_PREP_AIMET_COMMON_DIR}/${AIMET_PLATFORM_DIR}/include/${name}
        DEPENDS Mo${name}Test
    )
    add_custom_target(whl_prep_ln_common_${name}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${WHL_PREP_AIMET_COMMON_DIR}/${AIMET_PLATFORM_DIR}/include
        COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/include/${name} ${WHL_PREP_AIMET_COMMON_DIR}/${AIMET_PLATFORM_DIR}/include/${name}
        DEPENDS Mo${name}Test
    )
endmacro()

add_subdirectory(DlQuantization)
if (BUILD_PYMO_COMPRESSION)
    add_subdirectory(DlCompression)
endif()
if (BUILD_PYMO_EQUALIZATION)
    add_subdirectory(DlEqualization)
endif()
add_subdirectory(PyModelOptimizations)

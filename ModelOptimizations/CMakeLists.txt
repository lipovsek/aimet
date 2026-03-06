# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

if (WIN32)
    # Detect architecture for Windows builds (including cross-compilation)
    if (CMAKE_GENERATOR_PLATFORM MATCHES "[Aa][Rr][Mm]64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
        set(AIMET_PLATFORM_DIR "arm64-windows")
    elseif (CMAKE_GENERATOR_PLATFORM MATCHES "x64|Win64|x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(AIMET_PLATFORM_DIR "x86_64-windows")
    else()
        message(FATAL_ERROR "Unsupported arch: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
elseif (APPLE)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(AIMET_PLATFORM_DIR "arm64-macos")
    else()
        message(FATAL_ERROR "Unsupported arch: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
elseif (UNIX)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(AIMET_PLATFORM_DIR "aarch64-linux-gnu")
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(AIMET_PLATFORM_DIR "x86_64-linux-gnu")
    else()
        message(FATAL_ERROR "Unsupported arch: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}.")
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

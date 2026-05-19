# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# INPUTS:
#
# OUTPUTS:
# - TORCH_VERSION
# - TORCH_CMAKE_PREFIX_PATH
# - TORCH_DIR
# - CMAKE_CUDA_ARCHITECTURES
# - TORCH_CUDA_ARCH_LIST

function(set_torch_version)
    if (NOT ${Python3_FOUND})
        message(FATAL_ERROR "Need Python3 executable to determine PyTorch version.")
    endif()

    execute_process(COMMAND "${Python3_EXECUTABLE}" "-c" "import torch; print(torch.__version__)"
            OUTPUT_VARIABLE TORCH_VERSION_
            OUTPUT_STRIP_TRAILING_WHITESPACE)

    message(STATUS "Found Torch version: ${TORCH_VERSION_}")
    set(TORCH_VERSION ${TORCH_VERSION_} PARENT_SCOPE)
endfunction()

function(set_torch_cmake_prefix_path)
    if (NOT ${Python3_FOUND})
        message(FATAL_ERROR "Need Python3 executable to determine PyTorch CMake prefix path.")
    endif()

    execute_process(COMMAND ${Python3_EXECUTABLE} "-c" "import torch;print(torch.utils.cmake_prefix_path)"
                    RESULT_VARIABLE TORCH_NOT_FOUND
                    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    )

    # Append PyTorch CMake directory to CMAKE_PREFIX_PATH
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${TORCH_CMAKE_PREFIX_PATH}" PARENT_SCOPE)

    # Side-effect: set TORCH_DIR variable (to be used somewhere later)
    get_filename_component(TORCH_DIR_ ${TORCH_CMAKE_PREFIX_PATH}/../../ ABSOLUTE)
    set(TORCH_DIR ${TORCH_DIR_} PARENT_SCOPE)
    message(STATUS "Set TORCH_DIR = ${TORCH_DIR_}")
endfunction()

function(check_torch_cxx_abi_compatibility)
    if (NOT ${Python3_FOUND})
        message(FATAL_ERROR "Need Python3 executable to determine PyTorch version.")
    endif()

    # Use CMAKE_CXX_COMPILER instead of hardcoded "g++" to support clang on MacOS
    execute_process(COMMAND "${Python3_EXECUTABLE}" "-c" "import torch.utils.cpp_extension;print(torch.utils.cpp_extension.get_compiler_abi_compatibility_and_version(\"${CMAKE_CXX_COMPILER}\")[0])"
            OUTPUT_VARIABLE TORCH_IS_ABI_COMPATIBLE_
            OUTPUT_STRIP_TRAILING_WHITESPACE)

    message(STATUS "TORCH_IS_ABI_COMPATIBLE_ = ${TORCH_IS_ABI_COMPATIBLE_}")
    if (NOT TORCH_IS_ABI_COMPATIBLE_)
        message(FATAL_ERROR "Torch/C++ compiler ABI incompatibility detected.")
    endif()
endfunction()

# ----

macro(update_torch_cuda_arch_list)
    # Skip CUDA architecture detection on MacOS (CUDA not supported)
    if (APPLE)
        message(STATUS "** MacOS detected: CUDA not supported, skipping CUDA architecture detection **")
        set(CMAKE_CUDA_ARCHITECTURES "")
        set(TORCH_CUDA_ARCH_LIST "")
        # Note: Do NOT use return() here - it would exit the calling CMakeLists.txt, not just this macro
    else()
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 52 60 61 70 72)
        endif()
        message(STATUS "** Initial CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES} **")

        execute_process(COMMAND nvcc --list-gpu-arch
                        RESULT_VARIABLE NVCC_NOT_FOUND
                        OUTPUT_VARIABLE CMAKE_CUDA_ARCHITECTURES_RAW
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        )

        if(NVCC_NOT_FOUND)
            message(WARNING "nvcc not found or failed to execute. Please ensure CUDA is installed and nvcc is in your PATH.")
            set(CMAKE_CUDA_ARCHITECTURES "")
        else()
            string(REPLACE "compute_" "" CMAKE_CUDA_ARCHITECTURES_RAW "${CMAKE_CUDA_ARCHITECTURES_RAW}")
            # Replace newline with semicolon to form proper list
            string(REPLACE "\n" ";" CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES_RAW}")
        endif()

        message(STATUS "** Updated CMAKE_CUDA_ARCHITECTURES to ${CMAKE_CUDA_ARCHITECTURES} **")

        # Set torch cuda architecture list variable
        # Convert to the proper format (Reference: https://stackoverflow.com/a/74962874)
        #   - Insert "." before the last digit of the architecture version (ex. 50 --> 5.0, 120 --> 12.0)
        #   - Replace semi-colons in list with spaces
        set(TORCH_CUDA_ARCH_LIST ${CMAKE_CUDA_ARCHITECTURES})
        list(TRANSFORM TORCH_CUDA_ARCH_LIST REPLACE "([0-9]+)([0-9])" "\\1.\\2")
        string(REPLACE ";" " " TORCH_CUDA_ARCH_LIST "${TORCH_CUDA_ARCH_LIST}")
        message(STATUS "** Updated TORCH_CUDA_ARCH_LIST to ${TORCH_CUDA_ARCH_LIST} **")
    endif()
endmacro()

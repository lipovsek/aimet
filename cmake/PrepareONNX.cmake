# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

function(set_onnx_version)
    if (NOT ${Python3_FOUND})
        message(FATAL_ERROR "Need Python3 executable to determine ONNX version.")
    endif()

    execute_process(COMMAND "${Python3_EXECUTABLE}" "-c" "import onnx; print(onnx.__version__)"
            OUTPUT_VARIABLE ONNX_VERSION_
            OUTPUT_STRIP_TRAILING_WHITESPACE)

    message(STATUS "Found ONNX version: ${ONNX_VERSION_}")
    set(ONNX_VERSION ${ONNX_VERSION_} PARENT_SCOPE)
endfunction()

function(set_onnxruntime_variables)
    find_path(ONNXRUNTIME_INCLUDE_DIR_ "onnxruntime_cxx_api.h"
            PATHS ${onnxruntime_headers_SOURCE_DIR}/include
            REQUIRED
            NO_CMAKE_FIND_ROOT_PATH)
    find_library(ONNXRUNTIME_LIBRARIES_
            NAMES libonnxruntime.so onnxruntime.lib
            PATHS ${onnxruntime_headers_SOURCE_DIR}/lib
            REQUIRED
            NO_CMAKE_FIND_ROOT_PATH)

    message(STATUS "** ONNXRUNTIME_INCLUDE_DIR = ${ONNXRUNTIME_INCLUDE_DIR_}")
    set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_INCLUDE_DIR_} PARENT_SCOPE)

    message(STATUS "** ONNXRUNTIME_LIBRARIES = ${ONNXRUNTIME_LIBRARIES_}")
    set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIBRARIES_} PARENT_SCOPE)
endfunction()

macro(update_onnx_cuda_arch_list)
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
endmacro()

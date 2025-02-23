#=============================================================================
#
#  @@-COPYRIGHT-START-@@
#  
#  Copyright (c) 2018-2024, Qualcomm Innovation Center, Inc. All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without 
#  modification, are permitted provided that the following conditions are met:
#  
#  1. Redistributions of source code must retain the above copyright notice, 
#     this list of conditions and the following disclaimer.
#  
#  2. Redistributions in binary form must reproduce the above copyright notice, 
#     this list of conditions and the following disclaimer in the documentation 
#     and/or other materials provided with the distribution.
#  
#  3. Neither the name of the copyright holder nor the names of its contributors 
#     may be used to endorse or promote products derived from this software 
#     without specific prior written permission.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#  
#  SPDX-License-Identifier: BSD-3-Clause
#  
#  @@-COPYRIGHT-END-@@
#
#=============================================================================

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
            REQUIRED)
    find_library(ONNXRUNTIME_LIBRARIES_
            NAMES libonnxruntime.so
            PATHS ${onnxruntime_headers_SOURCE_DIR}/lib
            REQUIRED)

    message(STATUS "** ONNXRUNTIME_INCLUDE_DIR = ${ONNXRUNTIME_INCLUDE_DIR_}")
    set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_INCLUDE_DIR_} PARENT_SCOPE)

    message(STATUS "** ONNXRUNTIME_LIBRARIES = ${ONNXRUNTIME_LIBRARIES_}")
    set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIBRARIES_} PARENT_SCOPE)
endfunction()

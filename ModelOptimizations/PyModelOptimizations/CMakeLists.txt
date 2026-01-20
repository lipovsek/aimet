#=============================================================================
#
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018 - 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
    if (NOT WIN32)
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

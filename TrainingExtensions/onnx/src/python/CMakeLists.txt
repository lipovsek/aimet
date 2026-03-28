# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

if(SKBUILD)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/aimet_onnx
        DESTINATION .
        PATTERN "__pycache*" EXCLUDE
        PATTERN "common" EXCLUDE
        PATTERN "*.pyx" EXCLUDE
        PATTERN "*.pxd" EXCLUDE
       )
else()
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PATTERN "__pycache*" EXCLUDE
        PATTERN "CMakeLists.txt" EXCLUDE
        PATTERN "*.pyx" EXCLUDE
        PATTERN "*.pxd" EXCLUDE
        )
endif()

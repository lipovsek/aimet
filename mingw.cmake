# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# the name of the target operating system
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# which compilers to use for C and C++
set(CMAKE_C_COMPILER   x86_64-w64-mingw32-gcc-posix)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++-posix)

# Win64 python library location
set(Python3_ROOT_DIR      /opt/conda/envs/win64)
set(Python3_INCLUDE_DIR   ${Python3_ROOT_DIR}/include)
file(GLOB Python3_LIBRARY ${Python3_ROOT_DIR}/libs/python3[0-9]*.lib)

# where is the target environment located
set(CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32 ${Python3_ROOT_DIR})

# adjust the default behavior of the FIND_XXX() commands:
# search programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# search headers and libraries in the target environment
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Standard libraries should be linked statically for Windows
add_link_options(-static -static-libgcc -static-libstdc++)

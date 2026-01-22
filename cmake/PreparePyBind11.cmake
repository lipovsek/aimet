# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

macro(add_library_pybind11)
    if (NOT ${Python3_FOUND})
        message(FATAL_ERROR "Need Python3 executable to determine pybind11 include path.")
    endif()

    execute_process(COMMAND ${Python3_EXECUTABLE} "-c" "import pybind11;print(pybind11.get_include())"
                    RESULT_VARIABLE PYBIND11_NOT_FOUND
                    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR_
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    )

    # If we enable PyTorch builds then use the pybind11 headers that are part of the torch pip install
    # So we don't have a version mismatch - between PyTorch custom C++ op code and PyMO
    find_path(PYBIND11_HEADER "pybind11.h"
            PATHS ${TORCH_INCLUDE_DIRS} ${PYBIND11_INCLUDE_DIR_}
            PATH_SUFFIXES "pybind11"
            REQUIRED
            NO_DEFAULT_PATH
            NO_CMAKE_FIND_ROOT_PATH
            )

    get_filename_component(PYBIND11_INCLUDE_DIR ${PYBIND11_HEADER} DIRECTORY)

    if (NOT PYBIND11_INCLUDE_DIR)
        message(FATAL_ERROR "Could not find pybind11.")
    endif()

    add_library(pybind11 SHARED IMPORTED)

    set_target_properties(pybind11 PROPERTIES
            IMPORTED_IMPLIB ${Python3_LIBRARIES}
            IMPORTED_LOCATION ${Python3_LIBRARIES}
            INTERFACE_INCLUDE_DIRECTORIES "${PYBIND11_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES Python3::Module
            )
endmacro()

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

###############################################################################
# A graph of whl_* targets where action could be "bdist_wheel", "install"
# or "develop". In case of "develop", we do not copy files, we create symbolic
# links instead. Thus, there are two types of targets: "_cp_" and "_ln_" (copy
# or link).
# You could also generate a DOT file with full graph by adding
# '--graphviz=/tmp/build/graph.dot' to cmake configure:
#       cmake -S /aimet-src/ -B /tmp/build --graphviz=/tmp/build/graph.dot \
#           -DENABLE_CUDA=1 -DENABLE_TORCH=1
# A file 'CMakeGraphVizOptions.cmake' with the following content
#       set(GRAPHVIZ_CUSTOM_TARGETS TRUE)
#       set(GRAPHVIZ_IGNORE_TARGETS "^[^w][^h][^l].*")
# will leave onle whl_* targers on a diagram. Other oprions you could find
# in the official documentation:
# https://cmake.org/cmake/help/v3.17/module/CMakeGraphVizOptions.html
###############################################################################
#
#          +------- whl_{action}-------+
#          |                           |
#          |                           |
#          v                           v
#  whl_{action}_torch          whl_{action}_common
#          |                           |
#          |                           v
#          +-----> whl_prep_(cp/ln) <--+
#                        |
#          +-------------+-------------+
#          v                           v
#  whl_prep_(cp/ln)_torch    whl_prep_(cp/ln)_common
#                                      |
#                                      v
#                       whl_prep_(cp/ln)_common_DlQuantization -------+
#                       whl_prep_(cp/ln)_common_PyModelOptimizations -|
#                                        |                            |
#                                        v                            v
#                                  build libpymo                 install headers


# Define targets:
#     - whl_build to build wheels
#     - whl_install to install wheels
#     - whl_develop to install wheels in editable mode
set(WHL_ACTIONS "bdist_wheel" "develop" "install")
foreach(whl_action ${WHL_ACTIONS})
    add_custom_target(whl_${whl_action})
endforeach()
# A human friemdly alias
add_custom_target(whl_build DEPENDS whl_bdist_wheel)

# A target to make a copy of C++ part to use as a dependency for bdist_wheel and install targets
add_custom_target(whl_prep_cp)
# A target to make a symlink of C++ part to use as a dependency for develop target
add_custom_target(whl_prep_ln)


set(whl_action_args_bdist_wheel --dist-dir ${WHL_PREP_DIR})
set(whl_action_args_develop "--no-deps")
set(whl_action_args_install "")

macro(whl_add_whl_action_target whl_name)
    foreach(whl_action ${WHL_ACTIONS})
        add_custom_target(whl_${whl_action}_${whl_name}
            ${CMAKE_COMMAND} -E env
                WHL_PREP_DIR=${WHL_PREP_DIR}
                AIMET_CU_VER=$<IF:$<BOOL:${ENABLE_CUDA}>,$ENV{CUDA_VERSION},"cpu">
                AIMET_PT_VER=$<IF:$<BOOL:${ENABLE_TORCH}>,${Torch_VERSION},"">
                AIMET_ONNX_VER=$<IF:$<BOOL:${ENABLE_ONNX}>,${Onnx_VERSION},"">
            ${Python3_EXECUTABLE} setup.py build --build-base ${WHL_PREP_DIR}/build_${whl_name} ${whl_action} ${whl_action_args_${whl_action}}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/python
        )
        if (${whl_action} STREQUAL "develop")
            add_dependencies(whl_${whl_action}_${whl_name} whl_prep_ln)
        else ()
            add_dependencies(whl_${whl_action}_${whl_name} whl_prep_cp)
        endif ()
        add_dependencies(whl_${whl_action} whl_${whl_action}_${whl_name})
    endforeach()
endmacro()

add_subdirectory(common)

if (ENABLE_TORCH)
    add_subdirectory(torch)
endif(ENABLE_TORCH)

if (ENABLE_ONNX)
    add_subdirectory(onnx)
endif(ENABLE_ONNX)

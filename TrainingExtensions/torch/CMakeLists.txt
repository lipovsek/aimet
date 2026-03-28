# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

if (SKBUILD)
    macro(torch_extension extension_name extension_source)
        Python3_add_library(${extension_name} MODULE ${extension_source})

        if (APPLE)
            set(TORCH_PYTHON_LIB "${TORCH_INSTALL_PREFIX}/lib/libtorch_python.dylib")
        else()
            set(TORCH_PYTHON_LIB "${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so")
        endif()

        target_link_libraries(${extension_name}
            PUBLIC
                torch
                ${TORCH_PYTHON_LIB}
            PRIVATE
                MoDlQuantization
            )

        if (ENABLE_CUDA)
            target_link_libraries(${extension_name}
                PRIVATE
                    MoDlQuantizationCuda
                )
        endif (ENABLE_CUDA)

        target_compile_definitions(${extension_name}
            PRIVATE
                TORCH_EXTENSION_NAME=${extension_name}
                ENABLE_CUDA_PYTORCH=$<BOOL:${ENABLE_CUDA}>
            )

        if (APPLE)
            set(EXTENSION_RPATH "@loader_path:@loader_path/../torch/lib:@loader_path/../../..")
        else()
            set(EXTENSION_RPATH "\$ORIGIN:\$ORIGIN/../torch/lib:\$ORIGIN/../../..")
        endif()

        set_target_properties(${extension_name} PROPERTIES
            CXX_STANDARD 17
            CUDA_SEPARABLE_COMPILATION ON
            INTERPROCEDURAL_OPTIMIZATION ON
            CXX_VISIBILITY_PRESET "hidden"
            VISIBILITY_INLINES_HIDDEN ON
            PREFIX ""
            INSTALL_RPATH "${EXTENSION_RPATH}"
            )

        install(TARGETS ${extension_name}
            LIBRARY DESTINATION aimet_torch/common
            )
    endmacro()

    torch_extension(AimetTensorQuantizer src/AimetTensorQuantizer.cpp)
    torch_extension(AimetEncodingRescaler src/AimetEncodingRescaler.cpp)
    add_subdirectory(src/python)
else (SKBUILD)
    set(SETUP_PY "${CMAKE_CURRENT_BINARY_DIR}/src/setup.py")
    set(OUTPUT  "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp"
          "${CMAKE_CURRENT_BINARY_DIR}/build"
          )
    set(DEPS
        "src/AimetTensorQuantizer.cpp"
        "src/AimetEncodingRescaler.cpp"
    )


    get_target_property(DlQuantizationIncludes MoDlQuantization INCLUDE_DIRECTORIES)
    string(REPLACE ";" "','" DlQuantizationIncludePaths "${DlQuantizationIncludes}")

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/setup.py.in ${SETUP_PY})

    if (ENABLE_CUDA)
        set(ENABLE_CUDA_PYTORCH 1)
    else()
        set(ENABLE_CUDA_PYTORCH 0)
    endif()

    # Set EXTRA_LINK_ARGS to specify RPATH to not use LD_LIBRARY_PATH at runtime.
    # Use an absolute path RPATH because the SO file is located in the build directory. It is not
    # possible to use $ORIGIN to point to folder with torch libs.
    # Later, when a SO is copied into a wheel, RPATH will be changed to use relative paths ($ORIGIN)
    if (APPLE)
        # MacOS: Use @loader_path instead of $ORIGIN, no --enable-new-dtags
        set(EXTRA_LINK_ARGS "-Wl,-rpath;-Wl,$<TARGET_FILE_DIR:PyModelOptimizations>;-Wl,-rpath;-Wl,${TORCH_DIR}/lib")
    else()
        # Linux: Use $ORIGIN and --enable-new-dtags
        set(EXTRA_LINK_ARGS "-Wl,-rpath;-Wl,$<TARGET_FILE_DIR:PyModelOptimizations>;-Wl,-rpath;-Wl,${TORCH_DIR}/lib;-Wl,--enable-new-dtags")
    endif()

    add_custom_command(OUTPUT ${OUTPUT}
          COMMAND ${CMAKE_COMMAND}
                  -E env
                      PYTHONPATH="${CMAKE_BINARY_DIR}/artifacts"
                      ENABLE_CUDA_PYTORCH=${ENABLE_CUDA_PYTORCH}
                      TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
                      EXTRA_LINK_ARGS="${EXTRA_LINK_ARGS}"
                  ${Python3_EXECUTABLE} ${SETUP_PY} install  --install-base=${CMAKE_BINARY_DIR}/artifacts
                                                 --install-purelib=${CMAKE_BINARY_DIR}/artifacts
                                                 --install-platlib=${CMAKE_BINARY_DIR}/artifacts
                                                 --install-scripts=${CMAKE_BINARY_DIR}/artifacts
                                                 --install-data=${CMAKE_BINARY_DIR}/artifacts
                                                 --install-headers=${CMAKE_BINARY_DIR}/artifacts
                                                 --single-version-externally-managed
                                                 --record=${CMAKE_BINARY_DIR}/artifacts/install_record.txt
          COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
          DEPENDS ${DEPS}
          )

    add_custom_target(TorchCppOps ALL
          DEPENDS ${OUTPUT}
          SOURCES
          src/AimetTensorQuantizer.cpp
          src/AimetEncodingRescaler.cpp
          )

    # With --single-version-externally-managed, .so files are placed directly in artifacts/
    set(FILE_PATH_TENSOR_QUANT "${CMAKE_BINARY_DIR}/artifacts/AimetTensorQuantizer*.so")
    set(FILE_PATH_RESCALER "${CMAKE_BINARY_DIR}/artifacts/AimetEncodingRescaler*.so")

    # Use shell to copy files since CMake's -E copy doesn't support glob patterns
    add_custom_target(BuildTorchExtensions ALL
        COMMAND bash -c "cp ${CMAKE_BINARY_DIR}/artifacts/AimetTensorQuantizer*.so ${CMAKE_BINARY_DIR}/artifacts/_aimet_common/"
        COMMAND bash -c "cp ${CMAKE_BINARY_DIR}/artifacts/AimetEncodingRescaler*.so ${CMAKE_BINARY_DIR}/artifacts/_aimet_common/")

    add_dependencies(TorchCppOps PyModelOptimizations)
    add_dependencies(BuildTorchExtensions TorchCppOps)

    add_subdirectory(src/python)

    if (ENABLE_TESTS)
        if (ENABLE_ONNX)
            # Skip PyTorch tests when building for ONNX.
            # NB, some of the PyTorch tests fail when executed in ONNX build.
            # michof: We should consider enabling these tests and making sure they succeed, as long as
            # we depend on PyTorch for ONNX build and as long as test Docker images are materially
            # different.
            message(NOTICE "**********************************************")
            message(NOTICE "ONNX build is enabled, skipping PyTorch tests.")
            message(NOTICE "**********************************************")
        else()
            add_subdirectory(test)
        endif()
    endif()

    install(DIRECTORY ${CMAKE_BINARY_DIR}/artifacts/_aimet_common/
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/_aimet_common
            FILES_MATCHING
            PATTERN "Aimet*.so"
          )

    if (APPLE)
        # MacOS: Use install_name_tool to set RPATH
        add_custom_target(whl_prep_cp_torch
              COMMAND ${CMAKE_COMMAND} -E make_directory ${WHL_PREP_AIMET_TORCH_DIR}
              COMMAND find ${CMAKE_BINARY_DIR}/artifacts -path \"*/aimet_tensor_quantizer*\" -name \"Aimet*.so\"
                      -exec sh -c '$$0 -f $$1 ${WHL_PREP_AIMET_TORCH_DIR}/$$\(basename $$1\)' cp {} \\\;
              COMMAND find ${WHL_PREP_AIMET_TORCH_DIR} -name 'Aimet*.so'
                      -exec install_name_tool -add_rpath '@loader_path' {} \\\;
              COMMAND find ${WHL_PREP_AIMET_TORCH_DIR} -name 'Aimet*.so'
                      -exec install_name_tool -add_rpath '@loader_path/../torch/lib' {} \\\;
              DEPENDS
                  TorchCppOps
        )
    else()
        # Linux: Use patchelf to set RPATH
        add_custom_target(whl_prep_cp_torch
              COMMAND ${CMAKE_COMMAND} -E make_directory ${WHL_PREP_AIMET_TORCH_DIR}
              COMMAND find ${CMAKE_BINARY_DIR}/artifacts -path \"*/aimet_tensor_quantizer*\" -name \"Aimet*.so\"
                      -exec sh -c '$$0 -f $$1 ${WHL_PREP_AIMET_TORCH_DIR}/$$\(basename $$1\)' cp {} \\\;
              COMMAND find ${WHL_PREP_AIMET_TORCH_DIR} -name 'Aimet*.so'
                      -exec ${PATCHELF_EXE} --set-rpath '$$ORIGIN:$$ORIGIN/../torch/lib' {} \\\;
              DEPENDS
                  TorchCppOps
        )
    endif()
    add_custom_target(whl_prep_ln_torch
          COMMAND ${CMAKE_COMMAND} -E make_directory ${WHL_PREP_AIMET_TORCH_DIR}
          COMMAND find ${CMAKE_BINARY_DIR}/artifacts -path \"*/aimet_tensor_quantizer*\" -name \"Aimet*.so\"
                  -exec sh -c '$$0 -f $$1 ${WHL_PREP_AIMET_TORCH_DIR}/$$\(basename $$1\)' "ln -s" {} \\\;
          DEPENDS
              TorchCppOps
    )

    add_dependencies(whl_prep_cp whl_prep_cp_torch)
    add_dependencies(whl_prep_ln whl_prep_ln_torch)
    whl_add_whl_action_target(torch)
endif (SKBUILD)

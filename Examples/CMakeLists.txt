# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

add_custom_target(copy_examples
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/Docs_SOURCE/tutorials/notebooks)

if(SKBUILD)
else()
#TODO Remove the OPTIONAL flag after Common Examples code gets added
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/common
    DESTINATION Examples
    OPTIONAL
    PATTERN "__pycache*" EXCLUDE
    PATTERN "CMakeLists.txt" EXCLUDE
    )

if (ENABLE_TORCH)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/torch
    DESTINATION Examples
    PATTERN "__pycache*" EXCLUDE
    PATTERN "CMakeLists.txt" EXCLUDE
    )

endif(ENABLE_TORCH)

# Create a tarball archive of the Examples code
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E tar cvzf ${CMAKE_INSTALL_PREFIX}/Examples.tar.gz .
                              WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})"
        )
endif()

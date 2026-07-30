# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

include_directories(BEFORE
      ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

add_executable(MoDlQuantizationTest
      test_quantization_lib.hpp
      TestTfEnhancedEncodingAnalyzer.cpp
      TestTfEncodingAnalyzer.cpp
      TestTensorQuantizer.cpp
      TestUtilities.cpp
      TestPercentileEncodingAnalyzer.cpp
      TestMinMaxEncodingAnalyzer.cpp
      TestQuantizationFunctions.cpp
      TestEncodingAnalyzerWrapper.cpp
      )

if (ENABLE_CUDA)
    target_compile_options(MoDlQuantizationTest
            PRIVATE
                -DGPU_QUANTIZATION_ENABLED=1
            )

    target_link_libraries(MoDlQuantizationTest
            MoDlQuantization
            MoDlQuantizationCuda
            gtest_main
            )

else (ENABLE_CUDA)
    target_link_libraries(MoDlQuantizationTest
            MoDlQuantization
            gtest_main
            )

endif (ENABLE_CUDA)


add_test(NAME MoDlQuantizationTest
      COMMAND $<TARGET_FILE:MoDlQuantizationTest> --gtest_output=xml:cpp_test_output.xml)


#GTEST_ADD_TESTS(MoDlQuantizationTest "--gtest_output=xml:cpp_test_output.xml" AUTO)

#!/usr/bin/python

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# move xml files generated from unit tests to target directory
# so that XUnitBuilder plugin from Jenkins can find them


import sys
import os
import shutil

search_root = os.path.join(sys.argv[1], "build")
unit_test_dir = os.path.join(sys.argv[1], "unit_test_results")
shutil.rmtree(unit_test_dir, ignore_errors=True)

if not os.path.exists(unit_test_dir):
    os.makedirs(unit_test_dir)

for dirpath, dirs, files in os.walk(search_root, onerror=None, followlinks=False):
    output_file = None

    if "py_test_output.xml" in files:
        output_file = "py_test_output.xml"
    elif "cpp_test_output.xml" in files:
        output_file = "cpp_test_output.xml"

    if output_file is not None:
        src_file = os.path.abspath(os.path.join(dirpath, output_file))
        dst_file = os.path.join(unit_test_dir, src_file.replace("/", "_"))
        shutil.copy2(src_file, dst_file)

        if not os.path.exists(dst_file):
            print(
                "Copying Unit test results to report directory Failed. Destination path %s does not exist."
                % dst_file
            )

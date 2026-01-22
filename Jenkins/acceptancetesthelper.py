#!/usr/bin/python3

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# move xml files generated from acceptance tests to target directory
# so that XUnitBuilder plugin from Jenkins can find them

import sys
import os
import shutil

search_root = os.path.join(sys.argv[1], "build", "NightlyTests")
acceptance_test_dir = os.path.join(sys.argv[1], "acceptance_test_results")
shutil.rmtree(acceptance_test_dir, ignore_errors=True)

if not os.path.exists(acceptance_test_dir):
    os.makedirs(acceptance_test_dir)

for dirpath, dirs, files in os.walk(search_root, onerror=None, followlinks=False):
    output_files = []
    output_files = [f for f in files if os.path.splitext(f)[1] == ".xml"]

    for output_file in output_files:
        src_file = os.path.abspath(os.path.join(dirpath, output_file))
        dst_file = os.path.join(acceptance_test_dir, src_file.replace("/", "_"))
        shutil.copy2(src_file, dst_file)

        if not os.path.exists(dst_file):
            print(
                "Copying Acceptance test results to report directory Failed. Destination path %s does not exist."
                % dst_file
            )

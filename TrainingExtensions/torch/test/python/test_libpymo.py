# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# /usr/bin/env python3.8

import aimet_common._libpymo as mo_libpymo
import aimet_common.py_libpymo as py_libpymo


class Testlibpymo:
    def test_libpymo_api(self):
        """
        test public APIs from mo_libpymo are part of py_libpymo
        """
        for attribute in dir(mo_libpymo):
            if not attribute.startswith("_"):
                assert hasattr(py_libpymo, attribute)

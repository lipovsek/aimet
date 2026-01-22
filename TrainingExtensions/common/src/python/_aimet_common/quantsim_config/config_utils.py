# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Utilities for quantsim configurations"""

import os


def get_path_for_target_config(target_config: str) -> str:
    """
    Returns path for target config such as htp_quantsim_config_v73, aic100_config, eai_quantsim_config

    :return: path for target config file
    """
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"{target_config}.json"
    )

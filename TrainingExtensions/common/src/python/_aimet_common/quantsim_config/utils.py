# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Utilities for quantsim configurations"""

import os


def get_path_for_per_channel_config():
    """
    Returns path for default per channel config file

    :return: path for default per channel config file
    """
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "default_config_per_channel.json"
    )


def get_path_for_per_tensor_config():
    """
    Returns path for default per tensor config file

    :return: path for default per tensor config file
    """
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "default_config.json"
    )

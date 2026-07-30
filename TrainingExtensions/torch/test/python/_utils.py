# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


from pathlib import Path


def per_tensor_config():
    from aimet_torch.common import quantsim_config

    for path in quantsim_config.__path__:
        return Path(path) / "default_config.json"

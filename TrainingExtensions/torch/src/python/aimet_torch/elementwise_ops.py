# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""This module is deprecated. Use aimet_torch._base.nn.modules.custom instead."""

from aimet_torch._base.nn.modules.custom import *  # pylint: disable=wildcard-import, unused-wildcard-import


if __name__ != "__main__":
    import aimet_torch._base.nn.modules.custom as modules
    from aimet_torch.utils import _red
    import warnings

    warnings.warn(
        _red(
            f'"{__name__}" is renamed to "{modules.__name__}" and will be deprecated soon in the later versions.'
        ),
        stacklevel=2,
    )
    del warnings
    del _red
    del modules

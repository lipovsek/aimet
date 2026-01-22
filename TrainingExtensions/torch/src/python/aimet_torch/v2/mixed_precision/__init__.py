# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=all

from .manual_mixed_precision import MixedPrecisionConfigurator
from .utils import Precision, SupportedDType

from aimet_torch._base.mixed_precision import (
    choose_mixed_precision as _choose_mixed_precision,
)
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.amp.utils import _mock_v1_quantizers


def choose_mixed_precision(sim: QuantizationSimModel, *args, **kwargs):
    __doc__ = _choose_mixed_precision.__doc__  # pylint: disable=redefined-builtin, unused-variable

    with _mock_v1_quantizers(sim):
        return _choose_mixed_precision(sim, *args, **kwargs)

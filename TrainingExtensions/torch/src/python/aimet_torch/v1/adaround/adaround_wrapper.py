# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Custom Wrapper for quantizing weights using Adaround"""

import contextlib
from typing import Tuple
import torch
import torch.nn

# Import AIMET specific modules
from aimet_torch.common import aimet_tensor_quantizer as AimetTensorQuantizer
from aimet_torch.common.defs import MAP_QUANT_SCHEME_TO_PYMO
from aimet_torch._base.adaround.adaround_wrapper import AdaroundWrapperBase
from aimet_torch.v1.tensor_quantizer import StaticGridPerChannelQuantizer
from aimet_torch.v1.quantsim_straight_through_grad import broadcast_to_tensor


class AdaroundWrapper(AdaroundWrapperBase):
    """
    Adaround wrapper class for AIMET v1
    """

    def get_original_module(self) -> torch.nn.Module:
        """
        Returns original module so that we can check its
        module type or access its weight
        """
        # pylint: disable=protected-access
        return self.module_to_wrap._module_to_wrap

    @contextlib.contextmanager
    def _disable_weight_quantizer(self):
        """
        Temporarily disable weight quantizer
        """
        weight_quantizer = self.module_to_wrap.param_quantizers[self.weight_name]
        is_enabled = weight_quantizer.enabled
        weight_quantizer.enabled = False
        yield
        weight_quantizer.enabled = is_enabled

    def _is_weight_quantizer_enabled(self) -> bool:
        """
        Returns true if the weight quantizer is enabled
        """
        quantizer = self.module_to_wrap.param_quantizers[self.weight_name]
        return quantizer.enabled

    def _get_weight_quantizer_channel_axis(self) -> int:
        """
        Returns channel axis of the current weight quantizer
        """
        # pylint: disable = protected-access
        quantizer = self.module_to_wrap.param_quantizers[self.weight_name]
        if isinstance(quantizer, StaticGridPerChannelQuantizer):
            return quantizer._ch_axis
        return 0

    def _get_weight_quantizer_delta_and_offset(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns delta and offset of the weight quantizer
        """
        quantizer = self.module_to_wrap.param_quantizers[self.weight_name]
        if isinstance(quantizer.encoding, list):
            # pylint: disable = protected-access
            cpp_op = AimetTensorQuantizer.AimetTensorQuantizer(
                MAP_QUANT_SCHEME_TO_PYMO[quantizer.quant_scheme]
            )
            delta, offset = cpp_op.makeDeltaOffsetTensor(
                self.weight.device, quantizer.encoding
            )
        else:
            delta, offset = quantizer.encoding.delta, quantizer.encoding.offset

        ch_axis = self._get_weight_quantizer_channel_axis()
        return broadcast_to_tensor(self.weight, delta, ch_axis), broadcast_to_tensor(
            self.weight, offset, ch_axis
        )

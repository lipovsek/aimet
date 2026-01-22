# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""v1 lazy quant wrapper / quantizer"""

from typing import Tuple, Optional
from aimet_torch.quantsim_config.builder import LazyQuantizeWrapper, LazyQuantizer
from aimet_torch.v1.utils import get_v1_quant_scheme_for_initialization
from aimet_torch.v1.qc_quantize_op import (
    QcQuantizeWrapper,
    StaticGridQuantWrapper,
    tensor_quantizer_factory,
)
from aimet_torch.v1.tensor_quantizer import (
    TensorQuantizer,
    StaticGridPerChannelQuantizer,
)


class _V1LazyQuantizer(LazyQuantizer):
    def realize(self) -> TensorQuantizer:
        """Returns v1 quantizer using collected information."""
        quant_scheme_for_initialization = get_v1_quant_scheme_for_initialization(
            self.quant_scheme
        )

        if self.channel_axis is not None:
            assert self.input_tensor_shape
            num_channels = self.input_tensor_shape[self.channel_axis]

            quantizer = StaticGridPerChannelQuantizer(
                self.bitwidth,
                self.round_mode,
                quant_scheme_for_initialization,
                self.use_symmetric_encodings,
                num_channels,
                self.enabled,
                self.channel_axis,
                self.data_type,
            )
        else:
            quantizer = tensor_quantizer_factory(
                self.bitwidth,
                self.round_mode,
                quant_scheme_for_initialization,
                self.use_symmetric_encodings,
                self.enabled,
                self.data_type,
            )

        self._set_internal_quantizer_properties(quantizer)

        return quantizer

    def _set_internal_quantizer_properties(self, quantizer: TensorQuantizer):
        """
        Sets internal quantizer properties of v1 quantizer
        using collected information.

        :param quantizer: quantizer to update its internal properties
        """
        if self.encoding_min_max_fixed_vals is not None:
            quantizer.encoding_min_max_fixed_vals = self.encoding_min_max_fixed_vals
        quantizer.is_unsigned_symmetric = self.is_unsigned_symmetric
        quantizer.use_unsigned_symmetric = self.use_unsigned_symmetric
        quantizer.use_strict_symmetric = self.use_strict_symmetric

    @LazyQuantizer.encoding_min_max_fixed_vals.setter
    def encoding_min_max_fixed_vals(
        self, min_max_vals: Tuple[Optional[float], Optional[float]]
    ):
        # pylint: disable=redefined-builtin
        min, max = min_max_vals

        # NOTE: aimet_torch v1 does NOT support partial encoding freeze
        # not to break legacy code.
        if min is None or max is None:
            return

        LazyQuantizer.encoding_min_max_fixed_vals.fset(self, (min, max))


class _V1LazyQuantizeWrapper(LazyQuantizeWrapper):
    @property
    def _lazy_qtzr_cls(self):
        return _V1LazyQuantizer

    def realize(self) -> QcQuantizeWrapper:
        """
        Realizes v1 quant wrapper using collected information

        :return: v1 quant wrapper with specified properties
        """
        quant_scheme_for_initialization = get_v1_quant_scheme_for_initialization(
            self._quant_scheme
        )

        quantized_module = StaticGridQuantWrapper(
            self._module_to_wrap,
            self._weight_bw,
            self._activation_bw,
            self._rounding_mode,
            quant_scheme_for_initialization,
            self._is_output_quantized,
            self._is_symmetric,
            self._num_inputs,
            self._num_outputs,
            self._data_type,
        )

        quantized_module.input_quantizers = [
            quant_builder.realize() for quant_builder in self.input_quantizers
        ]
        quantized_module.output_quantizers = [
            quant_builder.realize() for quant_builder in self.output_quantizers
        ]
        quantized_module.param_quantizers = {
            param_name: quant_builder.realize()
            for (param_name, quant_builder) in self.param_quantizers.items()
        }
        quantized_module.supported_kernels = self.supported_kernels

        return quantized_module

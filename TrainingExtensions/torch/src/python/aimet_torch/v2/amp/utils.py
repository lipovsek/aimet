# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=missing-function-docstring, too-many-ancestors
"""Utilities for mixed precision feature in aimet_torch.v2"""

from contextlib import contextmanager
from typing import Union, Optional

import torch

from aimet_torch.common.defs import QuantizationDataType
from aimet_torch.v2.quantization.affine import QuantizeDequantize
from aimet_torch.v2.quantization.float import FloatQuantizeDequantize
from aimet_torch.v2.quantsim import QuantizationSimModel


@contextmanager
def _mock_v1_quantizers(sim: QuantizationSimModel):
    # pylint: disable=too-many-branches
    assert isinstance(sim, QuantizationSimModel)

    try:
        for _, qmodule in sim.named_qmodules():
            for i, qtzr in enumerate(qmodule.input_quantizers):
                if not isinstance(qtzr, _V1QuantizerMixin):
                    qmodule.input_quantizers[i] = _V1QuantizerMixin.from_v2_quantizer(
                        qtzr
                    )

            for i, qtzr in enumerate(qmodule.output_quantizers):
                if not isinstance(qtzr, _V1QuantizerMixin):
                    qmodule.output_quantizers[i] = _V1QuantizerMixin.from_v2_quantizer(
                        qtzr
                    )

            for name, qtzr in list(qmodule.param_quantizers.items()):
                if not isinstance(qtzr, _V1QuantizerMixin):
                    qmodule.param_quantizers[name] = (
                        _V1QuantizerMixin.from_v2_quantizer(qtzr)
                    )

        yield
    finally:
        for _, qmodule in sim.named_qmodules():
            for i, qtzr in enumerate(qmodule.input_quantizers):
                if isinstance(qtzr, _V1QuantizerMixin):
                    qmodule.input_quantizers[i] = qtzr.to_v2_quantizer()

            for i, qtzr in enumerate(qmodule.output_quantizers):
                if isinstance(qtzr, _V1QuantizerMixin):
                    qmodule.output_quantizers[i] = qtzr.to_v2_quantizer()

            for name, qtzr in list(qmodule.param_quantizers.items()):
                if isinstance(qtzr, _V1QuantizerMixin):
                    qmodule.param_quantizers[name] = qtzr.to_v2_quantizer()


class _V1QuantizerMixin:
    """
    Mixin that implements some v1 quantizer APIs to let v2 quantizers mimic v1 quantizers.

    This class was devised only for internal purposes to reuse the current MixedPrecisionAlgo code
    that heavily relies on v1 quantizer APIs.
    """

    bitwidth: int
    enabled: bool
    data_type: QuantizationDataType

    def forward(self, x):
        if not self.enabled:
            return x

        if self.data_type == QuantizationDataType.float:
            if self.bitwidth == 32:
                return x
            if self.bitwidth == 16:
                return x.to(torch.float16).to(x.dtype)
            raise RuntimeError

        return super().forward(x)

    @contextmanager
    def compute_encodings(self):
        if not self.enabled:
            yield
            return

        with super().compute_encodings():
            yield

    @classmethod
    def from_v2_quantizer(
        cls, qtzr: Optional[QuantizeDequantize]
    ) -> Union["_V1DisabledQuantizer", "_V1QuantizeDequantize"]:
        """
        Creates a mock that mimics v1 quantizer APIs from v2 quantizer

        Args:
            qtzr: v2 quantizer
        """
        mock_v1_qtzr = None

        if qtzr is None:
            return _V1DisabledQuantizer(shape=(), bitwidth=16, symmetric=False)
        if isinstance(qtzr, QuantizeDequantize):
            mock_v1_qtzr = cls.__new__(_V1QuantizeDequantize)
        else:
            raise RuntimeError

        # NOTE: Mock v1 quantizer shares the same storage as the original quantizer.
        #       Any attribute changes made to the mock quantizer will be
        #       also applied to the original quantizer
        mock_v1_qtzr.__dict__ = qtzr.__dict__
        mock_v1_qtzr.enabled = True
        mock_v1_qtzr.data_type = QuantizationDataType.int
        return mock_v1_qtzr

    def to_v2_quantizer(
        self,
    ) -> Union[QuantizeDequantize, FloatQuantizeDequantize, None]:
        """Revert v1 quantizer mock to v2 quantizer"""
        if isinstance(self, _V1DisabledQuantizer):
            return None

        if hasattr(self, "_cached_v2_quantizer"):
            return self._cached_v2_quantizer[0]

        if self.data_type == QuantizationDataType.float:
            if self.bitwidth == 32:
                v2_qtzr = None
            elif self.bitwidth == 16:
                v2_qtzr = FloatQuantizeDequantize(dtype=torch.float16)
            else:
                raise RuntimeError
        else:
            v2_qtzr = QuantizeDequantize.__new__(QuantizeDequantize)
            v2_qtzr.__dict__ = {**self.__dict__}
            delattr(v2_qtzr, "enabled")
            delattr(v2_qtzr, "data_type")

        # Note: assign as tuple to prevent adding to self._modules, creating recursive reference
        self._cached_v2_quantizer = (v2_qtzr,)
        return v2_qtzr


class _V1DisabledQuantizer(_V1QuantizerMixin, QuantizeDequantize):
    @property
    def enabled(self):
        return False

    @enabled.setter
    def enabled(self, val):
        if val:
            raise RuntimeError

    def __bool__(self):
        return False


class _V1QuantizeDequantize(_V1QuantizerMixin, QuantizeDequantize): ...

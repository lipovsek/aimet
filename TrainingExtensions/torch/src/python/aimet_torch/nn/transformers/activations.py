# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Quantized activation functions"""

import torch
from aimet_torch.nn.true_quant import QuantizationMixin

try:
    from transformers.activations import SiLUActivation

    @QuantizationMixin.implements(SiLUActivation)
    class QuantizedSiLU(QuantizationMixin, SiLUActivation):
        """Quantized CustomSiLU"""

        __quant_init__ = QuantizationMixin.__unary__

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
            (input_qtzr,) = self.input_quantizers
            (output_qtzr,) = self.output_quantizers

            if input_qtzr:
                x = input_qtzr(x)

            out = super().forward(x)

            if output_qtzr:
                out = output_qtzr(out)

            return out
except ImportError:
    # Older version of transformers use torch.nn.SiLu directly
    pass

try:
    from transformers.activations import GELUActivation

    @QuantizationMixin.implements(GELUActivation)
    class QuantizedGELU(QuantizationMixin, GELUActivation):
        """Quantized CustomGELU"""

        __quant_init__ = QuantizationMixin.__unary__

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
            (input_qtzr,) = self.input_quantizers
            (output_qtzr,) = self.output_quantizers

            if input_qtzr:
                x = input_qtzr(x)

            out = super().forward(x)

            if output_qtzr:
                out = output_qtzr(out)

            return out
except ImportError:
    # Older version of transformers use torch.nn.GELU directly
    pass

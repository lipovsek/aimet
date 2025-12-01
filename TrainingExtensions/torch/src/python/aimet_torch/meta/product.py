#  =============================================================================
#
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
#
#  =============================================================================

from typing import Optional

from aimet_torch.common.connected_graph.product import Product as _Product
from ..nn import QuantizationMixin
from .operation import Op

_V2 = True


class Product(_Product):
    def is_quantized(self):
        producer: Optional[Op] = self._producer

        if self.is_parm:
            # Parameters are always quantized with param quantizers
            return True

        if self.is_model_input or self.is_const or not producer:
            # Model inputs and non-param constants are not quantized yet
            return False

        producer_module = producer.get_module()
        if producer_module:
            if _V2:
                # If the producer is a quantized layer, assume the output
                # must have been quantized by the previous layer.
                # This doesn't cover all cases, but it's a reasonable assumption
                # for a short-term fix.
                return isinstance(producer_module, QuantizationMixin) or (
                    producer.is_grid_preserving_op()
                    and (
                        producer.inputs[0].is_quantized()
                        or producer.inputs[0].is_model_input
                        or producer.inputs[0].is_const
                    )
                )

            # Producer is nn.Module. This product will have been quantized by
            # the output quantizer of producer
            return True

        if producer.is_grid_preserving_op():
            # Producer is a functional data movement op, such as torch.reshape.
            # Check if the producer's input were already quantized
            return producer.inputs[0].is_quantized()

        return True

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Torch quantsim utils"""

from aimet_torch.v2.nn.true_quant import QuantizedConv2d, QuantizedLinear
from aimet_torch.v2.quantsim.config_utils import (
    set_blockwise_quantization_for_weights,
    set_grouped_blockwise_quantization_for_weights,
)
from aimet_torch import QuantizationSimModel

from GenAILab.qai_hub_lm.precision import (
    Granularity,
    PrecisionConfig,
    WeightPrecision,
)


def _apply_block_granularity_to_decoder_stack(
    quantsim: QuantizationSimModel, precision: PrecisionConfig, lm_head=None
):
    """Apply block-level granularity (LPBQ/BQ) to weight quantizers if configured."""
    if lm_head is None:
        lm_head = quantsim.model.model.lm_head

    block_prec = precision.blocks["default"]
    arg = lambda module: (
        isinstance(module, (QuantizedConv2d, QuantizedLinear))
        and module.param_quantizers["weight"]
        and module.param_quantizers["weight"].bitwidth == block_prec.qtype.bits
        and module is not lm_head
    )
    if block_prec.granularity == Granularity.LPBQ:
        set_grouped_blockwise_quantization_for_weights(
            sim=quantsim,
            arg=arg,
            bitwidth=block_prec.qtype.bits,
            symmetric=True,
            decompressed_bw=8,
            block_size=block_prec.block_size,
            block_grouping=-1,
        )
    elif block_prec.granularity == Granularity.BQ:
        set_blockwise_quantization_for_weights(
            sim=quantsim,
            arg=arg,
            bitwidth=block_prec.qtype.bits,
            symmetric=True,
            block_size=block_prec.block_size,
        )


def _set_lm_head_precision(
    quantsim: QuantizationSimModel, precision: WeightPrecision, lm_head=None
):
    if lm_head is None:
        lm_head = quantsim.model.model.lm_head
    arg = lambda module: (module is lm_head)
    if precision.granularity == Granularity.LPBQ:
        set_grouped_blockwise_quantization_for_weights(
            sim=quantsim,
            arg=arg,
            bitwidth=precision.qtype.bits,
            symmetric=True,
            decompressed_bw=8,
            block_size=precision.block_size,
            block_grouping=-1,
        )
    elif precision.granularity == Granularity.BQ:
        set_blockwise_quantization_for_weights(
            sim=quantsim,
            arg=arg,
            bitwidth=precision.qtype.bits,
            symmetric=True,
            block_size=precision.block_size,
        )
    else:
        lm_head.param_quantizers["weight"].bitwidth = precision.qtype.bits

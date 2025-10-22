# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-docstring

from dataclasses import dataclass
from typing import Type
from transformers.models.llama.modeling_llama import LlamaModel, LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2DecoderLayer

try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3DecoderLayer
except ImportError:
    Qwen3Model = Qwen3DecoderLayer = None

from aimet_torch.experimental.transforms.transform_config import (
    LlamaBlockInterface,
    Qwen2BlockInterface,
    Qwen3BlockInterface,
)


@dataclass
class FPTQuantConfig:
    block_type: Type = None  # block types to use in a given model
    block_interface: Type = None  # interface class describing block layout


fptquant_model_config_dict = {
    LlamaModel: FPTQuantConfig(
        block_type=LlamaDecoderLayer, block_interface=LlamaBlockInterface
    ),
    Qwen2Model: FPTQuantConfig(
        block_type=Qwen2DecoderLayer, block_interface=Qwen2BlockInterface
    ),
}

if Qwen3Model is not None and Qwen3DecoderLayer is not None:
    fptquant_model_config_dict.update(
        {
            Qwen3Model: FPTQuantConfig(
                block_type=Qwen3DecoderLayer,
                block_interface=Qwen3BlockInterface,
            )
        }
    )

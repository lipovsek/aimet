# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Test aimet-torch transform config"""

import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.llama import LlamaConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm
from transformers.models.qwen2 import Qwen2Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm
from transformers.models.qwen3 import Qwen3Config
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer, Phi3RMSNorm
from transformers.models.phi3.modeling_phi3 import Phi3Config
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLDecoderLayer,
)

# Handle transformers version differences
# transformers >= 5.0 renamed Qwen2RMSNorm to Qwen2_5_VLRMSNorm
try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLRMSNorm as Qwen25RMSNorm,
    )
except AttributeError:
    # Fall back to old name for transformers < 5.0
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2RMSNorm as Qwen25RMSNorm,
    )

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLTextConfig,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextRMSNorm,
    Qwen3VLVisionPatchMerger,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)


from aimet_torch.experimental.transforms.transform_config import (
    LlamaBlockInterface,
    Qwen2BlockInterface,
    Qwen3BlockInterface,
    Phi3BlockInterface,
    Qwen2dot5VLBackboneBlockInterface,
    Qwen3VLBackboneBlockInterface,
    Qwen3VLMergerInterface,
)


@pytest.mark.parametrize(
    "decoder_cls, norm_cls, config_cls, block_interface_cls",
    [
        (LlamaDecoderLayer, LlamaRMSNorm, LlamaConfig, LlamaBlockInterface),
        (Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2Config, Qwen2BlockInterface),
        (Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3Config, Qwen3BlockInterface),
        (Phi3DecoderLayer, Phi3RMSNorm, Phi3Config, Phi3BlockInterface),
        (
            Qwen2_5_VLDecoderLayer,
            Qwen25RMSNorm,
            Qwen2_5_VLTextConfig,
            Qwen2dot5VLBackboneBlockInterface,
        ),
        (
            Qwen3VLTextDecoderLayer,
            Qwen3VLTextRMSNorm,
            Qwen3VLTextConfig,
            Qwen3VLBackboneBlockInterface,
        ),
    ],
)
def test_builtin_block_interface(
    decoder_cls, norm_cls, config_cls, block_interface_cls
):
    config = config_cls(
        hidden_size=32,
        intermediate_size=32,
        num_attention_heads=1,
        num_hidden_layers=1,
    )
    block = decoder_cls(config=config, layer_idx=0)
    block_interface = block_interface_cls(block)
    for layer in block_interface.qkv_layers():
        assert isinstance(layer, torch.nn.Linear)
    assert isinstance(block_interface.o_proj, torch.nn.Linear)
    for layer in block_interface.gate_up_layers():
        assert isinstance(layer, torch.nn.Linear)
    assert isinstance(block_interface.down_proj, torch.nn.Linear)
    assert isinstance(block_interface.input_norm, norm_cls)
    assert isinstance(block_interface.post_attention_norm, norm_cls)


@pytest.mark.parametrize(
    "merger_cls, norm_cls, config_cls, merger_interface_cls",
    [
        (
            Qwen3VLVisionPatchMerger,
            torch.nn.LayerNorm,
            Qwen3VLVisionConfig,
            Qwen3VLMergerInterface,
        ),
    ],
)
def test_builtin_merger_interface(
    merger_cls, norm_cls, config_cls, merger_interface_cls
):
    config = config_cls()
    merger = merger_cls(config=config)
    merger_interface = merger_interface_cls(merger)
    assert isinstance(merger_interface.linear1, torch.nn.Linear)
    assert isinstance(merger_interface.linear2, torch.nn.Linear)
    assert isinstance(merger_interface.norm, norm_cls)

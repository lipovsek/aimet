# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Utils for converting 2D attention mask to 4D attention mask."""

from packaging import version
import torch
import transformers

TRANSFORMERS_VERSION_5_1_OR_LATER = version.parse(
    transformers.__version__
) >= version.parse("5.1.0")

if TRANSFORMERS_VERSION_5_1_OR_LATER:
    from transformers.masking_utils import eager_mask, causal_mask_function
else:
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter

if TRANSFORMERS_VERSION_5_1_OR_LATER:

    def convert_2d_attention_mask_to_4d(
        padded_attention_mask: torch.Tensor, sequence_length: int, context_length: int
    ):
        # New implementation
        return eager_mask(
            batch_size=padded_attention_mask.shape[0],
            cache_position=torch.arange(
                context_length - sequence_length,
                context_length,
                dtype=torch.int32,
                device=padded_attention_mask.device,
            ),
            kv_length=context_length,
            mask_function=causal_mask_function,
            attention_mask=padded_attention_mask.to(dtype=torch.bool),
            dtype=torch.float32,
        )
else:

    def convert_2d_attention_mask_to_4d(
        padded_attention_mask: torch.Tensor, sequence_length: int, context_length: int
    ):
        # Old implementation
        attention_mask_converter = AttentionMaskConverter(True)
        return attention_mask_converter.to_4d(
            padded_attention_mask,
            query_length=sequence_length,
            key_value_length=context_length,
            dtype=torch.float32,
        )

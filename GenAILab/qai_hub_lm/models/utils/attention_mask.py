# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Utils for converting 2D attention mask to 4D attention mask.

Supports both full causal masks and sliding-window causal masks.
"""

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
        q_length = sequence_length
        q_offset = context_length - sequence_length
        return eager_mask(
            batch_size=padded_attention_mask.shape[0],
            q_length=q_length,
            q_offset=q_offset,
            cache_position=torch.arange(
                q_offset,
                q_offset + q_length,
                dtype=torch.int32,
                device=padded_attention_mask.device,
            ),
            kv_length=context_length,
            mask_function=causal_mask_function,
            attention_mask=padded_attention_mask.to(dtype=torch.bool),
            dtype=torch.float32,
        ).to(device=padded_attention_mask.device)
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


def convert_2d_attention_mask_to_4d_sliding_window(
    padded_attention_mask: torch.Tensor,
    sequence_length: int,
    context_length: int,
    sliding_window_size: int,
) -> torch.Tensor:
    """Convert a 2D attention mask to a 4D sliding-window causal mask.

    Starts from the standard causal mask and additionally masks out key/value
    positions that fall outside the sliding window for each query position.

    Args:
        padded_attention_mask: 2D mask ``(batch, context_length)``
        sequence_length: number of query tokens
        context_length: total KV + query length
        sliding_window_size: maximum look-back distance (in tokens)

    Returns:
        4D mask ``(batch, 1, sequence_length, context_length)``
    """
    causal_mask = convert_2d_attention_mask_to_4d(
        padded_attention_mask, sequence_length, context_length
    )

    device = padded_attention_mask.device
    # Query positions within the full context
    query_positions = torch.arange(
        context_length - sequence_length,
        context_length,
        device=device,
    ).unsqueeze(1)  # (seq_len, 1)
    kv_positions = torch.arange(context_length, device=device).unsqueeze(
        0
    )  # (1, context_len)

    # Mask positions whose distance exceeds the window
    outside_window = (query_positions - kv_positions) >= sliding_window_size
    outside_window = outside_window.unsqueeze(0).unsqueeze(0)  # (1, 1, S, C)

    min_val = torch.finfo(causal_mask.dtype).min
    causal_mask = causal_mask.masked_fill(outside_window, min_val)
    return causal_mask

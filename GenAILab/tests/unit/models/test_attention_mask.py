# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for 2D to 4D attention mask conversion."""

import torch

from GenAILab.qai_hub_lm.utils.attention_mask import (
    convert_2d_attention_mask_to_4d,
)


class TestConvert2dTo4d:
    def test_shape(self):
        batch_size, seq_len, ctx_len = 2, 8, 32
        mask_2d = torch.ones(batch_size, ctx_len, dtype=torch.int32)
        mask_4d = convert_2d_attention_mask_to_4d(mask_2d, seq_len, ctx_len)
        assert mask_4d.shape == (batch_size, 1, seq_len, ctx_len)

    def test_causal_structure(self):
        seq_len, ctx_len = 4, 4
        mask_2d = torch.ones(1, ctx_len, dtype=torch.int32)
        mask_4d = convert_2d_attention_mask_to_4d(mask_2d, seq_len, ctx_len)
        mask = mask_4d[0, 0]
        # Causal: position i can attend to positions <= i
        # Diagonal and below should be 0 (attend), above should be negative (masked)
        for q in range(seq_len):
            for k in range(ctx_len):
                if k <= q:
                    assert mask[q, k] >= 0, f"Position ({q},{k}) should be unmasked"
                else:
                    assert mask[q, k] < 0, f"Position ({q},{k}) should be masked"

    def test_padding_propagated(self):
        seq_len, ctx_len = 4, 8
        mask_2d = torch.ones(1, ctx_len, dtype=torch.int32)
        # Mark first 4 positions as padding
        mask_2d[0, :4] = 0
        mask_4d = convert_2d_attention_mask_to_4d(mask_2d, seq_len, ctx_len)
        mask = mask_4d[0, 0]
        # Padding columns should be all-negative (masked) across all query positions
        for q in range(seq_len):
            for k in range(4):
                assert mask[q, k] < 0, f"Padding position ({q},{k}) should be masked"

    def test_all_ones(self):
        seq_len, ctx_len = 4, 8
        mask_2d = torch.ones(1, ctx_len, dtype=torch.int32)
        mask_4d = convert_2d_attention_mask_to_4d(mask_2d, seq_len, ctx_len)
        # All non-future positions should be unmasked
        mask = mask_4d[0, 0]
        # With seq_len < ctx_len, query positions correspond to the last seq_len positions
        # of the context. So query 0 = context position (ctx_len - seq_len) = 4
        for q in range(seq_len):
            ctx_pos = ctx_len - seq_len + q
            for k in range(ctx_len):
                if k <= ctx_pos:
                    assert mask[q, k] >= 0

    def test_batch_independent(self):
        seq_len, ctx_len = 4, 8
        mask_2d = torch.ones(2, ctx_len, dtype=torch.int32)
        mask_2d[1, :4] = 0  # second batch has padding
        mask_4d = convert_2d_attention_mask_to_4d(mask_2d, seq_len, ctx_len)
        # First batch should have no padding effects
        assert (mask_4d[0, 0, :, 0] >= 0).any()
        # Second batch padding columns should be masked
        assert (mask_4d[1, 0, :, 0] < 0).all()

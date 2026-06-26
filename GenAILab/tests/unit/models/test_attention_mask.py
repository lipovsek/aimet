# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for 2D to 4D attention mask conversion."""

import torch

from GenAILab.qai_hub_lm.models.utils.attention_mask import (
    convert_2d_attention_mask_to_4d,
    convert_2d_attention_mask_to_4d_sliding_window,
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


class TestSlidingWindowMask:
    def test_shape(self):
        batch_size, seq_len, ctx_len, window = 2, 8, 32, 16
        mask_2d = torch.ones(batch_size, ctx_len, dtype=torch.int32)
        mask_4d = convert_2d_attention_mask_to_4d_sliding_window(
            mask_2d, seq_len, ctx_len, window
        )
        assert mask_4d.shape == (batch_size, 1, seq_len, ctx_len)

    def test_window_masks_distant_positions(self):
        """Positions outside the sliding window should be masked."""
        seq_len, ctx_len, window = 4, 16, 4
        mask_2d = torch.ones(1, ctx_len, dtype=torch.int32)
        mask_4d = convert_2d_attention_mask_to_4d_sliding_window(
            mask_2d, seq_len, ctx_len, window
        )
        mask = mask_4d[0, 0]
        # Last query (context position 15) with window=4 can attend to
        # positions 12..15 only. Position 11 is outside the window.
        assert mask[-1, 11] < 0
        assert mask[-1, 12] >= 0
        assert mask[-1, 15] >= 0

    def test_window_does_not_mask_within_range(self):
        """Positions within the window should remain unmasked."""
        seq_len, ctx_len, window = 4, 8, 8
        mask_2d = torch.ones(1, ctx_len, dtype=torch.int32)
        mask_4d = convert_2d_attention_mask_to_4d_sliding_window(
            mask_2d, seq_len, ctx_len, window
        )
        mask = mask_4d[0, 0]
        # Window == context length means nothing extra is masked by sliding window
        for q in range(seq_len):
            ctx_pos = ctx_len - seq_len + q
            for k in range(ctx_pos + 1):
                assert mask[q, k] >= 0

    def test_left_padded_kv_cache_decode(self):
        """Core regression test: decode with left-padded KV cache.

        When input_length=1 but sequence_length is large, there's a padding
        gap between valid KV entries and the query token. The sliding window
        must use rope positions (not cache indices) to determine distance.

        Layout: [kv_padding | valid_KV | input_padding | query]
        """
        seq_len, ctx_len, window = 128, 256, 64
        # Simulate: 50 valid KV tokens, then 127 input padding zeros, then 1 query
        mask_2d = torch.zeros(1, ctx_len, dtype=torch.int32)
        kv_padding = ctx_len - seq_len - 50  # = 78
        mask_2d[0, kv_padding : kv_padding + 50] = 1  # valid KV
        mask_2d[0, -1] = 1  # valid query token (input_length=1)

        mask_4d = convert_2d_attention_mask_to_4d_sliding_window(
            mask_2d, seq_len, ctx_len, window
        )
        mask = mask_4d[0, 0]

        # The query is at rope position 50. The most recent KV token is at
        # rope position 49 (distance=1). With window=64, ALL 50 valid KV
        # tokens (rope positions 0..49) are within the window and should be
        # unmasked.
        query_row = mask[-1]  # last query position
        for k_idx in range(kv_padding, kv_padding + 50):
            assert query_row[k_idx] >= 0, (
                f"Valid KV at cache index {k_idx} should be unmasked; "
                f"rope distance is only {50 - (k_idx - kv_padding)}"
            )

    def test_left_padded_kv_cache_window_boundary(self):
        """Verify the window boundary is exact with padding present."""
        seq_len, ctx_len, window = 64, 128, 10
        # 20 valid KV tokens, then 63 input padding zeros, then 1 query
        mask_2d = torch.zeros(1, ctx_len, dtype=torch.int32)
        kv_padding = ctx_len - seq_len - 20  # = 44
        mask_2d[0, kv_padding : kv_padding + 20] = 1  # valid KV (rope 0..19)
        mask_2d[0, -1] = 1  # query token (rope 20)

        mask_4d = convert_2d_attention_mask_to_4d_sliding_window(
            mask_2d, seq_len, ctx_len, window
        )
        query_row = mask_4d[0, 0, -1]

        # Query at rope 20, window=10 → can attend to rope 11..20
        # Rope 10 is at distance 10 → >= window → masked
        # Rope 11 is at distance 9 → < window → unmasked
        rope_10_idx = kv_padding + 10
        rope_11_idx = kv_padding + 11
        assert query_row[rope_10_idx] < 0, "Rope pos 10 (distance=10) should be masked"
        assert query_row[rope_11_idx] >= 0, (
            "Rope pos 11 (distance=9) should be unmasked"
        )

    def test_prefill_no_padding_gap(self):
        """Prefill (input_length == sequence_length): no padding gap, basic window behavior."""
        seq_len, ctx_len, window = 8, 8, 4
        mask_2d = torch.ones(1, ctx_len, dtype=torch.int32)
        mask_4d = convert_2d_attention_mask_to_4d_sliding_window(
            mask_2d, seq_len, ctx_len, window
        )
        mask = mask_4d[0, 0]
        # Query at position 7 can attend to positions 4,5,6,7 (distance < 4)
        assert mask[7, 3] < 0  # distance = 4, outside
        assert mask[7, 4] >= 0  # distance = 3, inside
        assert mask[7, 7] >= 0  # distance = 0, inside

    def test_batch_with_different_padding(self):
        """Different batch items can have different amounts of valid KV."""
        seq_len, ctx_len, window = 32, 64, 8
        mask_2d = torch.zeros(2, ctx_len, dtype=torch.int32)

        # Batch 0: 10 valid KV + 1 query
        mask_2d[0, 22:32] = 1  # valid KV at indices 22..31
        mask_2d[0, -1] = 1  # query

        # Batch 1: 20 valid KV + 1 query
        mask_2d[1, 12:32] = 1  # valid KV at indices 12..31
        mask_2d[1, -1] = 1  # query

        mask_4d = convert_2d_attention_mask_to_4d_sliding_window(
            mask_2d, seq_len, ctx_len, window
        )

        # Batch 0: query at rope 10, window=8 → rope 3..10 visible (indices 25..31)
        q0 = mask_4d[0, 0, -1]
        assert q0[24] < 0  # rope 2, distance=8, outside
        assert q0[25] >= 0  # rope 3, distance=7, inside

        # Batch 1: query at rope 20, window=8 → rope 13..20 visible (indices 25..31)
        q1 = mask_4d[1, 0, -1]
        assert q1[24] < 0  # rope 12, distance=8, outside
        assert q1[25] >= 0  # rope 13, distance=7, inside

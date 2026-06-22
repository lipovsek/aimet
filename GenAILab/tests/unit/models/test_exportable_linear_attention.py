# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Functional tests for the Qwen 3.5 ExportableLinearAttention kernel.

These exercise the two functions that make up the exportable gated delta
rule adaptation:

  - ``_solve_triangular``: the product-form Newton approximation of
    ``(I - A)^{-1}`` used for the intra-chunk solve.
  - ``exportable_gated_delta_rule``: the unified prefill/decode kernel,
    checked for parity against HuggingFace's reference
    ``torch_chunk_gated_delta_rule`` and for correct attention-mask handling
    when inputs are padded to a fixed export length.
"""

import pytest

torch = pytest.importorskip("torch")

# The reference kernel and the adaptation both require a transformers version
# that ships the qwen3_5 model.
modeling_qwen3_5 = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

from GenAILab.qai_hub_lm.transforms.exportable_linear_attention import (  # noqa: E402
    _solve_triangular,
    exportable_gated_delta_rule,
)

CHUNK = 64
# Small head config keeps the tests fast; parity is independent of size.
N_HEADS, K_DIM, V_DIM = 4, 16, 16


def _rand_inputs(batch, seq, *, seed=0):
    """Random (query, key, value, g, beta) in HF [B, S, H, D] layout."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, seq, N_HEADS, K_DIM, generator=gen)
    k = torch.randn(batch, seq, N_HEADS, K_DIM, generator=gen)
    v = torch.randn(batch, seq, N_HEADS, V_DIM, generator=gen)
    # g is a negative log-decay (softplus keeps it well-behaved); beta in (0, 1).
    g = -torch.nn.functional.softplus(torch.randn(batch, seq, N_HEADS, generator=gen))
    beta = torch.rand(batch, seq, N_HEADS, generator=gen)
    return q, k, v, g, beta


def _reference(q, k, v, g, beta):
    out, _ = modeling_qwen3_5.torch_chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, chunk_size=CHUNK, use_qk_l2norm_in_kernel=True
    )
    return out


def _pad_seq(x, pad, *, left):
    """Pad the sequence dim (dim=1 of [B, S, H, ...]) by ``pad`` positions."""
    # F.pad pads from the last dim backwards; sequence is dim 1.
    n_trailing = x.dim() - 2  # dims after the sequence axis
    spec = [0, 0] * n_trailing + ([pad, 0] if left else [0, pad])
    return torch.nn.functional.pad(x, spec)


def _max_err(a, b):
    return (a - b).abs().max().item()


# Parity tolerance: the kernel runs in fp32 and the Newton solve is
# approximate, so machine-precision-ish (not bit-exact) agreement is expected.
TOL = 1e-4


class TestSolveTriangular:
    def test_inverts_unit_lower_triangular(self):
        """``_solve_triangular(A)`` approximates ``(I - A)^{-1}`` for strict-LT A."""
        gen = torch.Generator().manual_seed(7)
        a = torch.randn(CHUNK, CHUNK, generator=gen)
        # Strictly lower-triangular, modest magnitude (matches kernel usage).
        a = torch.tril(a, diagonal=-1) * 0.1

        approx = _solve_triangular(a, CHUNK)
        exact = torch.linalg.inv(torch.eye(CHUNK) - a)
        assert _max_err(approx, exact) < 1e-4

    def test_zero_matrix_gives_identity(self):
        a = torch.zeros(CHUNK, CHUNK)
        approx = _solve_triangular(a, CHUNK)
        assert _max_err(approx, torch.eye(CHUNK)) < 1e-6


class TestExportableGatedDeltaRule:
    def test_parity_exact_chunk_no_padding(self):
        """Full chunk, no padding: must match the HF reference."""
        q, k, v, g, beta = _rand_inputs(1, CHUNK)
        out, _ = exportable_gated_delta_rule(
            q, k, v, g=g, beta=beta, chunk_size=CHUNK, use_qk_l2norm_in_kernel=True
        )
        assert _max_err(out, _reference(q, k, v, g, beta)) < TOL

    def test_parity_right_pad_without_mask(self):
        """Right-padding is causal-safe: real positions match even without a mask."""
        real_len = 40
        q, k, v, g, beta = _rand_inputs(1, real_len)
        ref = _reference(q, k, v, g, beta)

        pad = CHUNK - real_len
        qp, kp, vp = (_pad_seq(x, pad, left=False) for x in (q, k, v))
        gp, bp = (_pad_seq(x, pad, left=False) for x in (g, beta))
        out, _ = exportable_gated_delta_rule(
            qp, kp, vp, g=gp, beta=bp, chunk_size=CHUNK, use_qk_l2norm_in_kernel=True
        )
        assert _max_err(out[:, :real_len], ref) < TOL

    def test_left_pad_requires_mask(self):
        """Left-padding garbage pollutes the state unless a mask zeros it out."""
        real_len = 40
        q, k, v, g, beta = _rand_inputs(1, CHUNK, seed=3)  # full buffer of "garbage"
        # The "true" sequence is the last real_len positions.
        ref = _reference(
            q[:, CHUNK - real_len :],
            k[:, CHUNK - real_len :],
            v[:, CHUNK - real_len :],
            g[:, CHUNK - real_len :],
            beta[:, CHUNK - real_len :],
        )

        mask = torch.zeros(1, CHUNK)
        mask[:, CHUNK - real_len :] = 1

        out_masked, _ = exportable_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            chunk_size=CHUNK,
            use_qk_l2norm_in_kernel=True,
            attention_mask=mask,
        )
        out_unmasked, _ = exportable_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            chunk_size=CHUNK,
            use_qk_l2norm_in_kernel=True,
            attention_mask=None,
        )

        real = slice(CHUNK - real_len, None)
        # With the mask, real positions match the reference.
        assert _max_err(out_masked[:, real], ref) < TOL
        # Without it, the leading garbage corrupts the recurrent state.
        assert _max_err(out_unmasked[:, real], ref) > 1e-2

    def test_batch_mixed_lengths_with_mask(self):
        """A padded batch of mixed real lengths matches per-sequence references."""
        lengths = [30, 50]
        batch = len(lengths)
        q, k, v, g, beta = _rand_inputs(batch, CHUNK, seed=11)

        mask = torch.zeros(batch, CHUNK)
        for i, length in enumerate(lengths):
            mask[i, CHUNK - length :] = 1

        out, _ = exportable_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            chunk_size=CHUNK,
            use_qk_l2norm_in_kernel=True,
            attention_mask=mask,
        )

        for i, length in enumerate(lengths):
            real = slice(CHUNK - length, None)
            ref_i = _reference(
                q[i : i + 1, real],
                k[i : i + 1, real],
                v[i : i + 1, real],
                g[i : i + 1, real],
                beta[i : i + 1, real],
            )
            assert _max_err(out[i : i + 1, real], ref_i) < TOL

    def test_multi_chunk_prefill_parity(self):
        """Sequences spanning several chunks still match the reference."""
        seq = 3 * CHUNK + 17  # not a chunk multiple → exercises internal padding
        q, k, v, g, beta = _rand_inputs(1, seq, seed=5)
        out, _ = exportable_gated_delta_rule(
            q, k, v, g=g, beta=beta, chunk_size=CHUNK, use_qk_l2norm_in_kernel=True
        )
        assert out.shape[1] == seq
        assert _max_err(out, _reference(q, k, v, g, beta)) < TOL

    def test_output_final_state_flag(self):
        q, k, v, g, beta = _rand_inputs(1, CHUNK)
        _, state_none = exportable_gated_delta_rule(
            q, k, v, g=g, beta=beta, chunk_size=CHUNK, output_final_state=False
        )
        _, state = exportable_gated_delta_rule(
            q, k, v, g=g, beta=beta, chunk_size=CHUNK, output_final_state=True
        )
        assert state_none is None
        assert state is not None
        assert state.shape == (1, N_HEADS, K_DIM, V_DIM)

    def test_decode_step_matches_recurrent_state(self):
        """chunk_size=1 decode from a carried state == the same token in-context."""
        # Process a full chunk, capturing the final recurrent state.
        q, k, v, g, beta = _rand_inputs(1, CHUNK + 1, seed=2)
        _, state = exportable_gated_delta_rule(
            q[:, :CHUNK],
            k[:, :CHUNK],
            v[:, :CHUNK],
            g=g[:, :CHUNK],
            beta=beta[:, :CHUNK],
            chunk_size=CHUNK,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        # Decode the next token from that state (chunk_size=1 path).
        out_decode, _ = exportable_gated_delta_rule(
            q[:, CHUNK:],
            k[:, CHUNK:],
            v[:, CHUNK:],
            g=g[:, CHUNK:],
            beta=beta[:, CHUNK:],
            chunk_size=1,
            initial_state=state,
            use_qk_l2norm_in_kernel=True,
        )
        # Reference: the full sequence at once, take the last token.
        ref_full = _reference(q, k, v, g, beta)
        assert _max_err(out_decode[:, 0], ref_full[:, CHUNK]) < TOL

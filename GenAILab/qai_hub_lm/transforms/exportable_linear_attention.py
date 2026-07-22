# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ExportableLinearAttention adaptation for Qwen 3.5 models.

Monkey-patches the gated delta rule functions on each GatedDeltaNet layer
instance to use a single unified code path suitable for all export backends
(torch.export, torch.jit.trace, eager).

The unified function replaces both:
  - chunk_gated_delta_rule: prefill path (any seq_len, chunked internally)
  - recurrent_gated_delta_rule: generation path (seq_len == 1)

Within each chunk the math follows QNN-cores' formulation: cumsum via
lower-triangular matmul, and a product-form Newton refinement for the
triangular solve (~11 MatMul ONNX ops per chunk). This is compact enough
that the inter-chunk loop can be unrolled without graph explosion.

Sequence-length handling
------------------------
Export targets a single, fixed sequence length (no dynamic shapes): inputs
are padded up to a multiple of ``chunk_size`` and the chunk count is a
compile-time constant, so the inter-chunk loop unrolls cleanly.  Variable
*real* lengths within a fixed-length graph are carried by ``attention_mask``
(a data input), which zeros the key/value/decay contributions of padded
positions.  Without this, left-padded inputs would let padded-token garbage
pollute the recurrent state; with it, masked positions are inert and results
match the unmasked variable-length reference to machine precision.
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from GenAILab.bench.yaml_config_parser import YAMLConfigParser


def l2norm(x, dim=-1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True).clamp(min=eps))


# ---------------------------------------------------------------------------
# Triangular solve: (I - A)^{-1} via product-form Newton refinement
# ---------------------------------------------------------------------------


def _solve_triangular(attn, chunk_size, order=4):
    """Approximate (I - A)^{-1} via banded Taylor + product-form refinement.

    Uses an order-``order`` banded initial approximation M0, then refines with
    M0 @ (I+E0)(I+E0^2)(I+E0^4)(I+E0^8) where E0 = I - (I-A)@M0.
    Produces ~11 MatMul ONNX ops and survives INT16 quantization.

    ``attn`` is strictly lower triangular (the diagonal is zero), so ``I - A``
    is unit lower triangular and invertible; the banded Taylor seed plus four
    product-form refinement steps converge for the chunk sizes used here.
    """
    I = torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    mask_acc = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=attn.dtype, device=attn.device),
        diagonal=-order,
    )

    M_mat = I - attn
    acc = I + attn
    power = attn
    for _ in range(2, order + 1):
        power = torch.matmul(power, attn)
        acc = acc + power
    M0 = mask_acc * acc

    E0 = I - torch.matmul(M_mat, M0)
    E1 = torch.matmul(E0, E0)
    E2 = torch.matmul(E1, E1)
    E3 = torch.matmul(E2, E2)
    return torch.matmul(
        torch.matmul(
            torch.matmul(
                torch.matmul(M0, I + E0),
                I + E1,
            ),
            I + E2,
        ),
        I + E3,
    )


# ---------------------------------------------------------------------------
# Unified exportable gated delta rule
# ---------------------------------------------------------------------------


def exportable_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    attention_mask=None,
    **kwargs,
):
    """Unified gated delta rule for both prefill and decode.

    For prefill (seq_len > 1): pads to a multiple of chunk_size, processes
    each chunk with QNN-cores-style matrix ops in a plain Python loop.
    For decode (seq_len == 1): pads to one chunk; the matrix ops degenerate
    to vector ops trivially.

    :param attention_mask: Optional ``[batch, seq_len]`` mask (1 = real token,
        0 = padding).  Padded positions have their key/value/decay zeroed so
        they contribute nothing to the recurrent state or intra-chunk
        attention.  Required for correctness when inputs are left-padded to a
        fixed export length.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, seq_len, k_head_dim = key.shape
    v_head_dim = value.shape[-1]

    # Zero out padded positions before chunking. Masking key/value/g makes a
    # padded token's contribution to the recurrent state and to intra-chunk
    # attention exactly zero, so a fixed-length (padded) graph matches the
    # variable-length reference regardless of left/right padding.
    if attention_mask is not None:
        mask = attention_mask[:, None, :].to(torch.float32)  # [B, 1, S]
        key = key * mask.unsqueeze(-1)
        value = value * mask.unsqueeze(-1)
        g = g * mask

    # Pad to a multiple of chunk_size
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_seq_len = seq_len + pad_size

    query = query * (k_head_dim**-0.5)
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # Reshape into chunks: [B, H, num_chunks, chunk_size, ...]
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    num_chunks = total_seq_len // chunk_size

    # Precompute masks (shared across all chunks)
    tril_mask = torch.tril(
        torch.ones(chunk_size, chunk_size, dtype=torch.float32, device=query.device)
    )
    eye = torch.eye(chunk_size, dtype=torch.float32, device=query.device)
    strict_lower_tri = tril_mask - eye

    # Per-chunk cumsum via lower-triangular matmul
    g_cum = (tril_mask @ g.unsqueeze(-1)).squeeze(-1)
    decay_mask = (
        (g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)) * tril_mask
    ).exp() * tril_mask

    # Intra-chunk triangular solve: (I - A)^{-1}
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask) * strict_lower_tri
    attn = _solve_triangular(attn, chunk_size)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g_cum.exp().unsqueeze(-1))

    # State initialization
    state = (
        torch.zeros(
            batch_size,
            num_heads,
            k_head_dim,
            v_head_dim,
            device=value.device,
            dtype=value.dtype,
        )
        if initial_state is None
        else initial_state.to(value)
    )

    # Inter-chunk loop (unrolled — ~15 ops per chunk is small enough)
    outputs = []
    for i in range(num_chunks):
        q_i = query[:, :, i]
        k_i = key[:, :, i]
        v_i = value[:, :, i]
        kc_i = k_cumdecay[:, :, i]
        dm_i = decay_mask[:, :, i]
        g_i = g_cum[:, :, i]

        attn_i = (q_i @ k_i.transpose(-1, -2) * dm_i) * tril_mask
        v_new = v_i - kc_i @ state
        o_i = (q_i * g_i.unsqueeze(-1).exp()) @ state + attn_i @ v_new

        state = (
            state * g_i[:, :, -1, None, None].exp()
            + (k_i * (g_i[:, :, -1, None] - g_i).exp().unsqueeze(-1)).transpose(-1, -2)
            @ v_new
        )
        outputs.append(o_i)

    core_attn_out = torch.stack(outputs, dim=2)

    if not output_final_state:
        state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :seq_len]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, state


# ---------------------------------------------------------------------------
# Exportable GatedDeltaNet.forward
# ---------------------------------------------------------------------------


def _roll_indices(npad, seq_len, device):
    """Gather indices that roll a [..., seq_len] tensor left by ``npad`` columns.

    ``npad`` is a per-batch int64 tensor of shape ``(B,)``. Implemented as a
    modular index so it exports to a single ONNX ``GatherElements`` (no Loop /
    data-dependent control flow), unlike ``torch.roll`` with a tensor shift.
    """
    import torch

    base = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, S)
    return (base + npad.unsqueeze(1)) % seq_len  # (B, S)


def exportable_gated_delta_net_forward(
    self, hidden_states, cache_params=None, attention_mask=None, **kwargs
):
    """Single-graph GatedDeltaNet forward for export.

    The stock forward gates its conv / recurrent-state handling on
    ``use_precomputed_states = cache_params.has_previous_state(layer_idx)``.
    At trace time the cache is freshly built, so that is False and the graph
    bakes in the from-scratch prefill path: the incoming ``conv_state`` and
    ``recurrent_state`` graph inputs become dead code and are ignored at
    inference, so decode has no memory and collapses after the first token.

    This replacement always threads the cached states (conv + recurrent) as
    live inputs, with no data-dependent branch, so a single static graph serves
    both prefill and decode:
      - delta rule: pass ``initial_state=recurrent_state`` unconditionally, and
        let ``attention_mask`` zero the padded positions (the kernel handles
        left-padding exactly).
      - conv: the generator left-pads inputs to a fixed width with the real
        tokens right-aligned, but the depthwise causal conv mixes *adjacent*
        columns, so the cached ``conv_state`` history must sit immediately
        before the first real token. We roll the real tokens to the front
        (via a gather), prepend ``conv_state`` there, convolve, capture the new
        conv window, then roll the outputs back. Padding is never between the
        history and the real tokens, so the conv sees the correct context.
    """
    import torch
    import torch.nn.functional as F
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        apply_mask_to_padding_states,
    )

    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape

    conv_state = cache_params.layers[self.layer_idx].conv_states
    recurrent_state = cache_params.layers[self.layer_idx].recurrent_states

    # Recover a 2D real-token mask [B, S] (1 = real) from whatever form the
    # model hands down. The generator feeds a 4D additive causal mask
    # (B, 1, S, KV); query row i is a real token iff its own diagonal key
    # (kv index q_offset + i, q_offset = KV - S) is unmasked (== 0). A plain 2D
    # mask is used directly; ``None`` means all-real.
    if attention_mask is None:
        mask2d = torch.ones(
            batch_size, seq_len, dtype=hidden_states.dtype, device=hidden_states.device
        )
    elif attention_mask.dim() == 2:
        mask2d = attention_mask.to(hidden_states.dtype)
    else:
        mask4d = attention_mask
        kv_len = mask4d.shape[-1]
        q_offset = kv_len - seq_len
        diag_idx = torch.arange(seq_len, device=mask4d.device) + q_offset  # (S,)
        diag = (
            mask4d[:, 0]
            .gather(-1, diag_idx.view(1, seq_len, 1).expand(batch_size, seq_len, 1))
            .squeeze(-1)
        )  # (B, S)
        mask2d = (diag >= 0).to(hidden_states.dtype)
    n_real = mask2d.sum(dim=-1).to(torch.int64)  # (B,)
    npad = seq_len - n_real

    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # (B, conv_dim, S)
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    # Roll the real tokens (right-aligned) to the front so the cached conv_state
    # sits immediately before them, convolve over [conv_state | real | pad],
    # then roll the per-position outputs back to their original alignment.
    K = self.conv_kernel_size
    fwd_idx = _roll_indices(npad, seq_len, mixed_qkv.device)  # roll left by npad
    back_idx = _roll_indices(
        seq_len - npad, seq_len, mixed_qkv.device
    )  # roll right by npad
    mixed_rolled = torch.gather(
        mixed_qkv, -1, fwd_idx.unsqueeze(1).expand_as(mixed_qkv)
    )

    buf = torch.cat([conv_state, mixed_rolled], dim=-1)  # (B, conv_dim, K + S)
    # New conv window = the last K real columns. In ``buf`` the real region is
    # cols [K : K + n_real], so the trailing K of it start at ``n_real``. Gather
    # those K columns per batch (tensor index → exportable, no dynamic slice).
    cs_cols = torch.arange(K, device=buf.device).unsqueeze(0) + n_real.unsqueeze(
        1
    )  # (B, K)
    new_conv_state = torch.gather(
        buf, -1, cs_cols.unsqueeze(1).expand(buf.shape[0], buf.shape[1], K)
    )
    cache_params.update_conv_state(new_conv_state, self.layer_idx)

    # conv1d has built-in left padding (K-1); slice to the input width then drop
    # the K leading conv_state columns, leaving S outputs aligned to mixed_rolled.
    conv_out = F.silu(self.conv1d(buf)[:, :, : buf.shape[-1]])[:, :, K:]
    mixed_qkv = torch.gather(conv_out, -1, back_idx.unsqueeze(1).expand_as(conv_out))

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
    )
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=recurrent_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        attention_mask=mask2d,
    )
    cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return self.out_proj(core_attn_out)


# ---------------------------------------------------------------------------
# Adaptation registration
# ---------------------------------------------------------------------------


def _passthrough_linear_attn_mask(self, attention_mask, past_key_values):
    """Always hand the linear-attention layers the real mask.

    The stock ``_update_linear_attn_mask`` nulls the mask when it is all-ones or
    when the cache has prior state. During export the sample input is all-real,
    so the mask is nulled and the exportable forward bakes in npad=0 — breaking
    decode, where padded positions must be located from the mask. Passing the
    mask through unconditionally keeps that information live in the graph.
    """
    return attention_mask


def _patch_gated_delta_net_instances(
    model: PreTrainedModel, chunk_size: int = 64
) -> None:
    """Walk all modules and replace the gated delta rule + forward on GatedDeltaNet instances."""
    import types

    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    for module in model.modules():
        if isinstance(module, Qwen3_5GatedDeltaNet):
            module.chunk_gated_delta_rule = functools.partial(
                exportable_gated_delta_rule, chunk_size=chunk_size
            )
            module.recurrent_gated_delta_rule = functools.partial(
                exportable_gated_delta_rule, chunk_size=1
            )
            module.forward = types.MethodType(
                exportable_gated_delta_net_forward, module
            )

    # Keep the linear-attention mask live (the stock model nulls it).
    for module in model.modules():
        if hasattr(module, "_update_linear_attn_mask"):
            module._update_linear_attn_mask = types.MethodType(
                _passthrough_linear_attn_mask, module
            )


@YAMLConfigParser.register_adaptation(
    "ExportableLinearAttention", model_type="qwen3_5", required_for_export=True
)
class Qwen3_5ExportableLinearAttentionAdaptation:
    """ExportableLinearAttention adaptation for Qwen 3.5 models.

    Replaces both gated delta rule functions with a single unified
    implementation that works under all export backends without graph
    explosion.

    Configurable via YAML adaptation kwargs:
        chunk_size (int): Prefill chunk size for the triangular solve.
            Defaults to 64.
    """

    chunk_size = 64

    @classmethod
    def instantiate_model(cls, *args, **kwargs) -> PreTrainedModel:
        model = super().instantiate_model(*args, **kwargs)
        _patch_gated_delta_net_instances(model, chunk_size=cls.chunk_size)
        return model

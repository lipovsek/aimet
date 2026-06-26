# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import io

from onnxscript import script, opset17 as op, FLOAT, INT64
from onnxscript.values import Opset
import onnx_ir as ir
import onnx
import onnx.inliner
import numpy as np
import torch
import torch.nn as nn


local = Opset("local", 1)


@script(local, default_opset=op)
def normalize(x: FLOAT[1, 1, 2048, 128], weight: FLOAT[128]) -> FLOAT[1, 1, 2048, 128]:
    eps = op.Constant(
        value=onnx.helper.make_tensor("epsilon", onnx.TensorProto.FLOAT, (), [1e-6])
    )
    one = op.Constant(
        value=onnx.helper.make_tensor("const_one", onnx.TensorProto.FLOAT, (), [1.0])
    )
    two = op.Constant(
        value=onnx.helper.make_tensor("const_two", onnx.TensorProto.FLOAT, (), [2.0])
    )

    x_cast = op.Cast(x, to=ir.DataType.FLOAT)
    x_sqr = op.Pow(x_cast, two)
    mean_sqr = op.ReduceMean(x_sqr, axes=[-1], keepdims=1)
    mean_sqr_eps = mean_sqr + eps
    sqrt = op.Sqrt(mean_sqr_eps)
    inv_sqrt = one / sqrt
    x_norm = x_cast * inv_sqrt
    x_norm_cast = op.Cast(x_norm, to=ir.DataType.FLOAT)
    return x_norm_cast * weight


@script(local, default_opset=op)
def rot_embed(
    x: FLOAT[1, 1, 2048, 128],
    position_ids_cos: FLOAT[1, 1, 2048, 64],
    position_ids_sin: FLOAT[1, 1, 2048, 64],
) -> FLOAT[1, 1, 2048, 128]:
    x_1 = op.Slice(x, 0, 64, 3, 1)
    x_2 = op.Slice(x, 64, 128, 3, 1)
    x_1_sin = x_1 * position_ids_sin
    x_1_cos = x_1 * position_ids_cos
    x_2_sin = x_2 * position_ids_sin
    x_2_cos = x_2 * position_ids_cos

    emb_sub = x_1_cos - x_2_sin
    emb_add = x_1_sin + x_2_cos
    return op.Concat(emb_sub, emb_add, axis=3)


@script(local, default_opset=op)
def sha_qkv(
    hidden: FLOAT[1, 2048, 256],
    position_ids_cos: FLOAT[1, 1, 2048, 64],
    position_ids_sin: FLOAT[1, 1, 2048, 64],
    k_proj_sha_weight: FLOAT[128, 256, 1, 1],
    q_proj_sha_weight: FLOAT[128, 256, 1, 1],
    v_proj_sha_weight: FLOAT[128, 256, 1, 1],
    q_norm_sha_weight: FLOAT[128],
):
    q_proj = op.Conv(hidden, q_proj_sha_weight)
    k_proj = op.Conv(hidden, k_proj_sha_weight)
    v_proj = op.Conv(hidden, v_proj_sha_weight)

    q_proj_t = op.Transpose(q_proj, perm=[0, 2, 3, 1])
    k_proj_t = op.Transpose(k_proj, perm=[0, 2, 3, 1])
    v_proj_t = op.Transpose(v_proj, perm=[0, 2, 3, 1])

    q_proj_norm = normalize(q_proj_t, q_norm_sha_weight)
    k_proj_norm = normalize(k_proj_t, q_norm_sha_weight)

    q_proj_emb = rot_embed(q_proj_norm, position_ids_cos, position_ids_sin)
    k_proj_emb = rot_embed(k_proj_norm, position_ids_cos, position_ids_sin)

    q_proj_emb = op.Reshape(q_proj_emb, [1, 1, 2048, 128])
    k_proj_emb = op.Reshape(k_proj_emb, [1, 1, 2048, 128])

    k_proj_emb_t = op.Transpose(k_proj_emb, perm=[0, 1, 3, 2])
    return q_proj_emb, k_proj_emb_t, v_proj_t


@script(local, default_opset=op)
def sha_head(
    hidden: FLOAT[1, 2048, 256],
    attention_mask: FLOAT[1, 1, 2048, 4096],
    position_ids_cos: FLOAT[1, 1, 2048, 64],
    position_ids_sin: FLOAT[1, 1, 2048, 64],
    past_key_in: FLOAT[1, 1, 128, 2048],
    past_value_in: FLOAT[1, 1, 2048, 128],
    k_proj_sha_weight: FLOAT[128, 256, 1, 1],
    q_proj_sha_weight: FLOAT[128, 256, 1, 1],
    v_proj_sha_weight: FLOAT[128, 256, 1, 1],
    q_norm_sha_weight: FLOAT[128],
):
    div_factor = op.Constant(
        value=onnx.helper.make_tensor(
            "div_factor_0", onnx.TensorProto.FLOAT, (), [11.31]
        )
    )
    const_one = op.Constant(
        value=onnx.helper.make_tensor("const_one", onnx.TensorProto.FLOAT, (), [1.0])
    )

    q_proj_emb, k_proj_emb_t, v_proj_t = sha_qkv(
        hidden,
        position_ids_cos,
        position_ids_sin,
        k_proj_sha_weight,
        q_proj_sha_weight,
        v_proj_sha_weight,
        q_norm_sha_weight,
    )

    total_key = op.Concat(past_key_in, k_proj_emb_t, axis=3)
    key_scaled = total_key / div_factor

    qk_matmul_out = op.MatMul(q_proj_emb, key_scaled)

    # Unnecessary mul, kept for structure
    attn_mask_mul = op.Mul(attention_mask, const_one)
    qk_matmul_out_masked = qk_matmul_out + attn_mask_mul

    attn_score = op.Softmax(qk_matmul_out_masked, axis=-1)

    total_value = op.Concat(past_value_in, v_proj_t, axis=2)

    self_attn_out = op.MatMul(attn_score, total_value)
    return self_attn_out, k_proj_emb_t, v_proj_t


@script(local, default_opset=op)
def sha_head_native_kvcache(
    hidden: FLOAT[1, 2048, 256],
    attention_mask: FLOAT[1, 1, 2048, 4096],
    position_ids_cos: FLOAT[1, 1, 2048, 64],
    position_ids_sin: FLOAT[1, 1, 2048, 64],
    past_key_in: FLOAT[1, 1, 128, 4096],
    past_value_in: FLOAT[1, 1, 4096, 128],
    cache_index: INT64[1],
    k_proj_sha_weight: FLOAT[128, 256, 1, 1],
    q_proj_sha_weight: FLOAT[128, 256, 1, 1],
    v_proj_sha_weight: FLOAT[128, 256, 1, 1],
    q_norm_sha_weight: FLOAT[128],
):
    div_factor = op.Constant(
        value=onnx.helper.make_tensor(
            "div_factor_0", onnx.TensorProto.FLOAT, (), [11.31]
        )
    )
    const_one = op.Constant(
        value=onnx.helper.make_tensor("const_one", onnx.TensorProto.FLOAT, (), [1.0])
    )
    arange = op.Constant(
        value=onnx.helper.make_tensor(
            "arange", onnx.TensorProto.INT64, (2048,), np.arange(2048)
        )
    )
    index_ones = op.Constant(
        value=onnx.helper.make_tensor(
            "index_ones",
            onnx.TensorProto.INT64,
            (1, 1, 128, 2048),
            np.ones(1 * 1 * 128 * 2048, dtype=np.int64),
        )
    )

    q_proj_emb, k_proj_emb_t, v_proj_t = sha_qkv(
        hidden,
        position_ids_cos,
        position_ids_sin,
        k_proj_sha_weight,
        q_proj_sha_weight,
        v_proj_sha_weight,
        q_norm_sha_weight,
    )

    # Native kv-cache: scatter the current key/value into the pre-allocated cache
    # at positions cache_index + arange, instead of concatenating with the past.
    scatter_index = cache_index + arange
    key_index = scatter_index * index_ones
    value_index = op.Transpose(key_index, perm=[0, 1, 3, 2])

    total_key = op.ScatterElements(past_key_in, key_index, k_proj_emb_t, axis=3)
    key_scaled = total_key / div_factor

    qk_matmul_out = op.MatMul(q_proj_emb, key_scaled)

    # Unnecessary mul, kept for structure
    attn_mask_mul = op.Mul(attention_mask, const_one)
    qk_matmul_out_masked = qk_matmul_out + attn_mask_mul

    attn_score = op.Softmax(qk_matmul_out_masked, axis=-1)

    total_value = op.ScatterElements(past_value_in, value_index, v_proj_t, axis=2)

    self_attn_out = op.MatMul(attn_score, total_value)
    return self_attn_out, k_proj_emb_t, v_proj_t


@script()
def sha_block(
    x: FLOAT[1, 2048, 256],
    attention_mask: FLOAT[1, 1, 2048, 4096],
    position_ids_cos: FLOAT[1, 1, 2048, 64],
    position_ids_sin: FLOAT[1, 1, 2048, 64],
    past_key_in: FLOAT[2, 1, 128, 2048],
    past_value_in: FLOAT[2, 1, 2048, 128],
    # initializers
    k_proj_sha_0_weight: FLOAT[128, 256, 1, 1],
    k_proj_sha_1_weight: FLOAT[128, 256, 1, 1],
    q_proj_sha_0_weight: FLOAT[128, 256, 1, 1],
    q_proj_sha_1_weight: FLOAT[128, 256, 1, 1],
    v_proj_sha_0_weight: FLOAT[128, 256, 1, 1],
    v_proj_sha_1_weight: FLOAT[128, 256, 1, 1],
    self_attn_o_proj_conv_weight: FLOAT[256, 256, 1, 1],
    q_norm_sha_0_weight: FLOAT[128],
) -> tuple[FLOAT[1, 256, 1, 2048], FLOAT[2, 1, 128, 2048], FLOAT[2, 1, 2048, 128]]:
    past_key_0_in_0 = op.Slice(past_key_in, 0, 1, 0, 1)
    past_key_0_in_1 = op.Slice(past_key_in, 1, 2, 0, 1)

    past_value_0_in_0 = op.Slice(past_value_in, 0, 1, 0, 1)
    past_value_0_in_1 = op.Slice(past_value_in, 1, 2, 0, 1)

    hidden = op.Reshape(x, [1, -1, 1, 256])
    hidden_t = op.Transpose(hidden, perm=[0, 3, 2, 1])

    self_attn_out_0, k_proj_emb_t_0, v_proj_t_0 = sha_head(
        hidden_t,
        attention_mask,
        position_ids_cos,
        position_ids_sin,
        past_key_0_in_0,
        past_value_0_in_0,
        k_proj_sha_0_weight,
        q_proj_sha_0_weight,
        v_proj_sha_0_weight,
        q_norm_sha_0_weight,
    )
    self_attn_out_1, k_proj_emb_t_1, v_proj_t_1 = sha_head(
        hidden_t,
        attention_mask,
        position_ids_cos,
        position_ids_sin,
        past_key_0_in_1,
        past_value_0_in_1,
        k_proj_sha_1_weight,
        q_proj_sha_1_weight,
        v_proj_sha_1_weight,
        q_norm_sha_0_weight,
    )

    self_attn_out = op.Concat(self_attn_out_0, self_attn_out_1, axis=3)
    self_attn_out_t = op.Transpose(self_attn_out, perm=[0, 3, 1, 2])
    out_proj = op.Conv(self_attn_out_t, self_attn_o_proj_conv_weight)

    past_value_out = op.Concat(v_proj_t_0, v_proj_t_1, axis=0)
    past_key_out = op.Concat(k_proj_emb_t_0, k_proj_emb_t_1, axis=0)
    return out_proj, past_key_out, past_value_out


@script()
def sha_block_native_kvcache(
    x: FLOAT[1, 2048, 256],
    attention_mask: FLOAT[1, 1, 2048, 4096],
    position_ids_cos: FLOAT[1, 1, 2048, 64],
    position_ids_sin: FLOAT[1, 1, 2048, 64],
    past_key_in: FLOAT[2, 1, 128, 4096],
    past_value_in: FLOAT[2, 1, 4096, 128],
    cache_index: INT64[1],
    # initializers
    k_proj_sha_0_weight: FLOAT[128, 256, 1, 1],
    k_proj_sha_1_weight: FLOAT[128, 256, 1, 1],
    q_proj_sha_0_weight: FLOAT[128, 256, 1, 1],
    q_proj_sha_1_weight: FLOAT[128, 256, 1, 1],
    v_proj_sha_0_weight: FLOAT[128, 256, 1, 1],
    v_proj_sha_1_weight: FLOAT[128, 256, 1, 1],
    self_attn_o_proj_conv_weight: FLOAT[256, 256, 1, 1],
    q_norm_sha_0_weight: FLOAT[128],
) -> tuple[FLOAT[1, 256, 1, 2048], FLOAT[2, 1, 128, 2048], FLOAT[2, 1, 2048, 128]]:
    past_key_0_in_0 = op.Slice(past_key_in, 0, 1, 0, 1)
    past_key_0_in_1 = op.Slice(past_key_in, 1, 2, 0, 1)

    past_value_0_in_0 = op.Slice(past_value_in, 0, 1, 0, 1)
    past_value_0_in_1 = op.Slice(past_value_in, 1, 2, 0, 1)

    hidden = op.Reshape(x, [1, -1, 1, 256])
    hidden_t = op.Transpose(hidden, perm=[0, 3, 2, 1])

    self_attn_out_0, k_proj_emb_t_0, v_proj_t_0 = sha_head_native_kvcache(
        hidden_t,
        attention_mask,
        position_ids_cos,
        position_ids_sin,
        past_key_0_in_0,
        past_value_0_in_0,
        cache_index,
        k_proj_sha_0_weight,
        q_proj_sha_0_weight,
        v_proj_sha_0_weight,
        q_norm_sha_0_weight,
    )
    self_attn_out_1, k_proj_emb_t_1, v_proj_t_1 = sha_head_native_kvcache(
        hidden_t,
        attention_mask,
        position_ids_cos,
        position_ids_sin,
        past_key_0_in_1,
        past_value_0_in_1,
        cache_index,
        k_proj_sha_1_weight,
        q_proj_sha_1_weight,
        v_proj_sha_1_weight,
        q_norm_sha_0_weight,
    )

    self_attn_out = op.Concat(self_attn_out_0, self_attn_out_1, axis=3)
    self_attn_out_t = op.Transpose(self_attn_out, perm=[0, 3, 1, 2])
    out_proj = op.Conv(self_attn_out_t, self_attn_o_proj_conv_weight)

    past_value_out = op.Concat(v_proj_t_0, v_proj_t_1, axis=0)
    past_key_out = op.Concat(k_proj_emb_t_0, k_proj_emb_t_1, axis=0)
    return out_proj, past_key_out, past_value_out


def _build_block_model(block_script, num_model_inputs):
    """Turn a block onnxscript into a checked, inlined model proto.

    The first ``num_model_inputs`` parameters are kept as graph inputs; the
    remaining parameters are weight matrices and are moved into initializers
    filled with random values.
    """
    proto = block_script.to_model_proto()
    for inp in proto.graph.input[num_model_inputs:]:
        shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        init = onnx.numpy_helper.from_array(
            np.random.randn(*shape).astype(np.float32), name=inp.name
        )
        proto.graph.initializer.append(init)

    inps = proto.graph.input[:num_model_inputs]
    proto.graph.ClearField("input")
    proto.graph.input.extend(inps)
    proto = onnx.inliner.inline_local_functions(proto)
    onnx.checker.check_model(proto)
    return proto


def sha_2_head_block():
    return _build_block_model(sha_block, num_model_inputs=6)


def sha_2_head_block_native_kvcache():
    return _build_block_model(sha_block_native_kvcache, num_model_inputs=7)


# -----------------------------------
# Scaled SHA + GQA decoder (torch)
# -----------------------------------


def _sha_rms(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def _sha_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE on a [1, 1, S, head_dim] tensor with split-half cos/sin."""
    rot = x.shape[-1] // 2
    x_1, x_2 = x[..., :rot], x[..., rot:]
    return torch.cat([x_1 * cos - x_2 * sin, x_1 * sin + x_2 * cos], dim=-1)


class _ShaGqaLayer(nn.Module):
    """SHA + GQA decoder layer using Conv1d projections"""

    def __init__(self, head_dim, num_kv, group, intermediate, rescale_key_tensor):
        super().__init__()
        self.head_dim = head_dim
        self.num_kv = num_kv
        self.group = group
        self.rescale_key_tensor = rescale_key_tensor
        hidden = num_kv * group * head_dim
        num_q = num_kv * group
        self.input_norm = nn.Parameter(torch.randn(hidden))
        self.post_attn_norm = nn.Parameter(torch.randn(hidden))
        self.q_proj = nn.ModuleList(
            nn.Conv1d(hidden, head_dim, 1, bias=False) for _ in range(num_q)
        )
        self.k_proj = nn.ModuleList(
            nn.Conv1d(hidden, head_dim, 1, bias=False) for _ in range(num_kv)
        )
        self.v_proj = nn.ModuleList(
            nn.Conv1d(hidden, head_dim, 1, bias=False) for _ in range(num_kv)
        )
        self.q_norm = nn.ParameterList(
            nn.Parameter(torch.randn(head_dim)) for _ in range(num_q)
        )
        self.k_norm = nn.ParameterList(
            nn.Parameter(torch.randn(head_dim)) for _ in range(num_kv)
        )
        self.o_proj = nn.Conv1d(hidden, hidden, 1, bias=False)
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Conv1d(intermediate, hidden, 1, bias=False)

    def forward(self, x, attention_mask, cos, sin, past_key, past_value):
        # x: [1, S, H]; attention_mask: [1, 1, S, total]
        # past_key: [num_kv, 1, head_dim, past]; past_value: [num_kv, 1, past, head_dim]
        h = _sha_rms(x, self.input_norm).transpose(1, 2)  # [1, H, S] for Conv1d
        attn_heads, key_outs, value_outs = [], [], []
        for kv in range(self.num_kv):
            k = self.k_proj[kv](h).transpose(1, 2)  # [1, S, head_dim]
            k = _sha_rope(_sha_rms(k, self.k_norm[kv]).unsqueeze(1), cos, sin)
            k_t = k.transpose(-1, -2)  # [1, 1, head_dim, S] — head_dim on axis -2
            v = self.v_proj[kv](h).transpose(1, 2).unsqueeze(1)  # [1, 1, S, head_dim]
            total_key = torch.cat([past_key[kv : kv + 1], k_t], dim=-1)
            total_value = torch.cat([past_value[kv : kv + 1], v], dim=-2)
            key_outs.append(k_t)
            value_outs.append(v)
            for g in range(self.group):
                qi = kv * self.group + g
                q = self.q_proj[qi](h).transpose(1, 2)  # [1, S, head_dim]
                q = _sha_rope(_sha_rms(q, self.q_norm[qi]).unsqueeze(1), cos, sin)
                if self.rescale_key_tensor:
                    score = q @ (total_key / self.head_dim**0.5) + attention_mask
                else:
                    score = q @ total_key / (self.head_dim**0.5) + attention_mask
                score = torch.softmax(score, dim=-1)  # [1, 1, S, total]
                attn_heads.append((score @ total_value).squeeze(1))  # [1, S, head_dim]
        attn = torch.cat(attn_heads, dim=-1).transpose(1, 2)  # [1, H, S]
        x = x + self.o_proj(attn).transpose(1, 2)
        h2 = _sha_rms(x, self.post_attn_norm)
        mlp = (self.gate(h2) * torch.sigmoid(self.gate(h2))) * self.up(h2)
        x = x + self.down(mlp.transpose(1, 2)).transpose(1, 2)
        return x, torch.cat(key_outs, dim=0), torch.cat(value_outs, dim=0)


class ShaGqaDecoder(nn.Module):
    """Scaled SHA + GQA decoder LM with a concat-style KV cache.

    Structure:
     - tied embed/lm_head,
     - per-head Conv q/k/v projections
     - GQA
     - RoPE

    :param head_dim: per-head size (production: 128).
    :param num_kv: key/value heads (production: 8).
    :param group: query heads per kv head (production: 2).
    :param intermediate: SwiGLU MLP intermediate size (production: 6144).
    :param vocab: vocabulary size.
    :param num_layers: decoder layers.
    """

    def __init__(
        self,
        head_dim=8,
        num_kv=2,
        group=2,
        intermediate=16,
        vocab=16,
        num_layers=2,
        rescale_key_tensor=False,
    ):
        super().__init__()
        hidden = num_kv * group * head_dim
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(
            _ShaGqaLayer(head_dim, num_kv, group, intermediate, rescale_key_tensor)
            for _ in range(num_layers)
        )
        self.final_norm = nn.Parameter(torch.randn(hidden))
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # tied embed/lm_head

    def forward(self, input_ids, attention_mask, cos, sin, *past_kv):
        x = self.embed_tokens(input_ids)
        present = []
        for i, layer in enumerate(self.layers):
            x, present_key, present_value = layer(
                x, attention_mask, cos, sin, past_kv[2 * i], past_kv[2 * i + 1]
            )
            present += [present_key, present_value]
        logits = self.lm_head(_sha_rms(x, self.final_norm))
        return (logits, *present)


def sha_gqa_decoder(
    head_dim=8,
    num_kv=2,
    group=2,
    intermediate=16,
    vocab=16,
    num_layers=2,
    seq=4,
    past=3,
    dynamo=False,
    rescale_key_tensor=False,
) -> onnx.ModelProto:
    """Build and export a :class:`ShaGqaDecoder` to ONNX with named KV-cache I/O.

    ``seq`` and ``past`` set the example sequence / past-cache lengths used for the export trace.
    """
    module = ShaGqaDecoder(
        head_dim, num_kv, group, intermediate, vocab, num_layers, rescale_key_tensor
    )
    rot = head_dim // 2
    attention_mask = torch.randn(1, 1, seq, seq + past)
    cos = torch.randn(1, 1, seq, rot)
    sin = torch.randn(1, 1, seq, rot)
    past_kv = []
    for _ in range(num_layers):
        past_kv += [
            torch.randn(num_kv, 1, head_dim, past),
            torch.randn(num_kv, 1, past, head_dim),
        ]
    input_names = [
        "input_ids",
        "attention_mask",
        "position_ids_cos",
        "position_ids_sin",
    ]
    output_names = ["logits"]
    for i in range(num_layers):
        input_names += [f"past_key_{i}_in", f"past_value_{i}_in"]
        output_names += [f"past_key_{i}_out", f"past_value_{i}_out"]

    token_ids = torch.randint(0, vocab, (1, seq))
    buf = io.BytesIO()
    torch.onnx.export(
        module.eval(),
        (token_ids, attention_mask, cos, sin, *past_kv),
        buf,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
        dynamo=dynamo,
    )
    buf.seek(0)
    return onnx.load_model(buf)

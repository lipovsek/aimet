# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from onnxscript import script, opset17 as op, FLOAT
from onnxscript.values import Opset
import onnx_ir as ir
import onnx
import onnx.inliner
import numpy as np


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


def sha_2_head_block():
    proto = sha_block.to_model_proto()
    for inp in proto.graph.input[6:]:
        shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        init = onnx.numpy_helper.from_array(
            np.random.randn(*shape).astype(np.float32), name=inp.name
        )
        proto.graph.initializer.append(init)

    inps = proto.graph.input[:6]
    proto.graph.ClearField("input")
    proto.graph.input.extend(inps)
    proto = onnx.inliner.inline_local_functions(proto)
    onnx.checker.check_model(proto)
    return proto

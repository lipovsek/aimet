# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-docstring


class _AttentionInterface:
    def __init__(self, attn):
        self.attn = attn

    @property
    def o_proj(self):
        return self.attn.o_proj

    @o_proj.setter
    def o_proj(self, value):
        self.attn.o_proj = value

    def qkv_layers(self):
        raise NotImplementedError()

    def attention_layer_names(self):
        raise NotImplementedError()


class _SeparatedProjAttentionInterface(_AttentionInterface):
    @property
    def q_proj(self):
        return self.attn.q_proj

    @q_proj.setter
    def q_proj(self, value):
        self.attn.q_proj = value

    @property
    def k_proj(self):
        return self.attn.k_proj

    @k_proj.setter
    def k_proj(self, value):
        self.attn.k_proj = value

    @property
    def v_proj(self):
        return self.attn.v_proj

    @v_proj.setter
    def v_proj(self, value):
        self.attn.v_proj = value

    def qkv_layers(self):
        yield self.q_proj
        yield self.k_proj
        yield self.v_proj

    def attention_layer_names(self):
        return ["q_proj", "k_proj", "v_proj", "o_proj"]


class _JointProjAttentionInterface(_AttentionInterface):
    @property
    def qkv_proj(self):
        return self.attn.qkv_proj

    @qkv_proj.setter
    def qkv_proj(self, value):
        self.attn.qkv_proj = value

    def qkv_layers(self):
        yield self.qkv_proj

    def attention_layer_names(self):
        return ["qkv_proj", "o_proj"]


class _MLPInterface:
    def __init__(self, mlp):
        self.mlp = mlp

    @property
    def down_proj(self):
        return self.mlp.down_proj

    @down_proj.setter
    def down_proj(self, value):
        self.mlp.down_proj = value

    def gate_up_layers(self):
        raise NotImplementedError()

    def mlp_layer_names(self):
        raise NotImplementedError()


class _SeparatedProjMLPInterface(_MLPInterface):
    @property
    def up_proj(self):
        return self.mlp.up_proj

    @up_proj.setter
    def up_proj(self, value):
        self.mlp.up_proj = value

    @property
    def gate_proj(self):
        return self.mlp.gate_proj

    @gate_proj.setter
    def gate_proj(self, value):
        self.mlp.gate_proj = value

    def gate_up_layers(self):
        yield self.gate_proj
        yield self.up_proj

    def mlp_layer_names(self):
        return ["gate_proj", "up_proj", "down_proj"]


class _JointProjMLPInterface(_MLPInterface):
    @property
    def gate_up_proj(self):
        return self.mlp.gate_up_proj

    @gate_up_proj.setter
    def gate_up_proj(self, value):
        self.mlp.gate_up_proj = value

    def gate_up_layers(self):
        yield self.gate_up_proj

    def mlp_layer_names(self):
        return ["gate_up_proj", "down_proj"]


class _NormInterface:
    def __init__(self, block):
        self.block = block

    @property
    def input_norm(self):
        return self.block.input_layernorm

    @input_norm.setter
    def input_norm(self, value):
        self.block.input_layernorm = value

    @property
    def post_attention_norm(self):
        return self.block.post_attention_layernorm

    @post_attention_norm.setter
    def post_attention_norm(self, value):
        self.block.post_attention_layernorm = value


# This class is just used for type checking. When constructing a new BlockInterface class, inherit from the correct
# Attention/MLP/Norm interfaces directly.
# pylint: disable=abstract-method
class BlockInterface(_AttentionInterface, _MLPInterface, _NormInterface):
    pass


def get_block_dtype(block: BlockInterface):
    return block.o_proj.weight.data.dtype


# Same as default, so don't need to do anything
class LlamaBlockInterface(
    _SeparatedProjAttentionInterface, _SeparatedProjMLPInterface, _NormInterface
):
    def __init__(self, block):
        _SeparatedProjAttentionInterface.__init__(self, block.self_attn)
        _SeparatedProjMLPInterface.__init__(self, block.mlp)
        _NormInterface.__init__(self, block)


# Same as default, so don't need to do anything
class Qwen2BlockInterface(
    _SeparatedProjAttentionInterface, _SeparatedProjMLPInterface, _NormInterface
):
    def __init__(self, block):
        _SeparatedProjAttentionInterface.__init__(self, block.self_attn)
        _SeparatedProjMLPInterface.__init__(self, block.mlp)
        _NormInterface.__init__(self, block)


class Qwen3BlockInterface(
    _SeparatedProjAttentionInterface, _SeparatedProjMLPInterface, _NormInterface
):
    def __init__(self, block):
        _SeparatedProjAttentionInterface.__init__(self, block.self_attn)
        _SeparatedProjMLPInterface.__init__(self, block.mlp)
        _NormInterface.__init__(self, block)

    @property
    def q_norm(self):
        return self.attn.q_norm

    @q_norm.setter
    def q_norm(self, value):
        self.attn.q_norm = value

    @property
    def k_norm(self):
        return self.attn.k_norm

    @k_norm.setter
    def k_norm(self, value):
        self.attn.k_norm = value


class Phi3BlockInterface(
    _JointProjAttentionInterface, _JointProjMLPInterface, _NormInterface
):
    def __init__(self, block):
        _JointProjAttentionInterface.__init__(self, block.self_attn)
        _JointProjMLPInterface.__init__(self, block.mlp)
        _NormInterface.__init__(self, block)


class Qwen2dot5VLViTBlockInterface(
    _JointProjAttentionInterface, _SeparatedProjMLPInterface, _NormInterface
):
    def __init__(self, block):
        _JointProjAttentionInterface.__init__(self, block.attn)
        _SeparatedProjMLPInterface.__init__(self, block.mlp)
        _NormInterface.__init__(self, block)

    @property
    def qkv_proj(self):
        return self.attn.qkv

    @qkv_proj.setter
    def qkv_proj(self, value):
        self.attn.qkv = value

    @property
    def o_proj(self):
        return self.attn.proj

    @o_proj.setter
    def o_proj(self, value):
        self.attn.proj = value

    @property
    def input_norm(self):
        return self.block.norm1

    @input_norm.setter
    def input_norm(self, value):
        self.block.norm1 = value

    @property
    def post_attention_norm(self):
        return self.block.norm2

    @post_attention_norm.setter
    def post_attention_norm(self, value):
        self.block.norm2 = value


class Qwen2dot5VLBackboneBlockInterface(
    _SeparatedProjAttentionInterface, _SeparatedProjMLPInterface, _NormInterface
):
    def __init__(self, block):
        _SeparatedProjAttentionInterface.__init__(self, block.self_attn)
        _SeparatedProjMLPInterface.__init__(self, block.mlp)
        _NormInterface.__init__(self, block)

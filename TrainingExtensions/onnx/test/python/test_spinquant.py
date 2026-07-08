# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import io
import logging
import shutil
from types import SimpleNamespace
from typing import Optional

import numpy as np
import scipy.linalg
import pytest
import torch
import torch.nn as nn
import onnx
import onnx_ir
from onnx import load_model, numpy_helper
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

from .models.test_models import RMSNorm
from .models import transformer_blocks
from .models.transformer_blocks import qwen3_causal_lm
from .utils import add_genai_tests_path
from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.common.hadamard import get_hadamard_matrix
from aimet_onnx.graph_passes.fusions import fuse_supergroups
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.utils import ParamUtils, make_dummy_input

from aimet_onnx.experimental.spinquant.model_analysis import (
    ActiveNorm,
    DecoderBlockRoleMap,
    DecoderModelRoleMap,
    find_active_norms,
    find_merger_linear2,
    find_r3_anchors,
    get_decoder_block_boundaries,
    get_decoder_role_map,
    get_bias_product as _get_bias_product,
    get_weight_product as _get_weight_product,
    infer_head_dim as _infer_head_dim,
    infer_hidden_size as _infer_hidden_size,
)
from aimet_onnx.experimental.spinquant.model_analysis.norm_detection import (
    _find_norm_scale_and_consumers,
)
from aimet_onnx.experimental.spinquant.transforms import (
    apply_transform as _apply_transform,
    fuse_norm_layers_into_linears,
    left_multiply as _left_multiply,
    right_multiply as _right_multiply,
)
from aimet_onnx.experimental.spinquant.passes.r1 import (
    _rotate_backbone,
    _rotate_merger_linear2,
    _validate_merger_linear2,
)
from aimet_onnx.experimental.spinquant.transforms.rotation_primitives import (
    hadamard_rotation_matrix,
)
from aimet_onnx.experimental.spinquant import apply_spinquant
from aimet_onnx.experimental.spinquant import is_online_rotation_op
from aimet_onnx.experimental.spinquant.model_analysis import find_attention_topology

from aimet_onnx.prepare_passes.fix_node_names_in_dynamo_exported_onnx import (
    fix_node_names_pass,
)


def apply_r1_rotation(model, role_map, backbone_hidden_size):
    """Test shim for the legacy ``apply_r1_rotation`` API."""
    _rotate_backbone(model, role_map, hadamard_rotation_matrix(backbone_hidden_size))


def apply_r1_rotation_merger(model, merger_linear2, backbone_hidden_size):
    """Test shim for the legacy ``apply_r1_rotation_merger`` API."""
    _rotate_merger_linear2(
        model, merger_linear2, hadamard_rotation_matrix(backbone_hidden_size)
    )


AimetLogger.set_level_for_all_areas(logging.INFO)


def _export_to_onnx(
    module: nn.Module,
    dummy_input: torch.Tensor,
    opset: int = 17,
    do_constant_folding: bool = True,
):
    buf = io.BytesIO()
    torch.onnx.export(
        module.eval(),
        dummy_input,
        buf,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=do_constant_folding,
        dynamo=False,
    )
    buf.seek(0)
    return load_model(buf)


def _build_session(model: onnx.ModelProto):
    sess_options = SessionOptions()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
    return InferenceSession(
        path_or_bytes=model.SerializeToString(),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )


def _pad_dummy_input(model: onnx.ModelProto, **named_inputs) -> dict:
    """Build a feed dict that satisfies every graph input.

    Caller-supplied entries in ``named_inputs`` win; dangling inputs (e.g. the
    ``past_value_0`` tensor we attach for head_dim derivation) get zero-filled
    placeholders. The graph never reads those, so the values are irrelevant.
    """
    feeds = dict(named_inputs)
    for graph_input in model.graph.input:
        if graph_input.name in feeds:
            continue
        shape = [
            d.dim_value if d.HasField("dim_value") else 1
            for d in graph_input.type.tensor_type.shape.dim
        ]
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(
            graph_input.type.tensor_type.elem_type
        )
        feeds[graph_input.name] = np.zeros(shape, dtype=np_dtype)
    return feeds


def _run_model(model: onnx.ModelProto, inp: np.ndarray) -> np.ndarray:
    session = _build_session(model)
    return session.run(None, _pad_dummy_input(model, input=inp))[0]


def _collect_pre_fusion_state(
    model: onnx.ModelProto, connected_graph: ConnectedGraph
) -> dict:
    pre_fusion_state = {}
    for op in connected_graph.ordered_ops:
        result = _find_norm_scale_and_consumers(op, model)
        if result is None:
            continue
        scale_name, linear_ops = result
        if not linear_ops:
            continue

        scale = numpy_helper.to_array(
            ParamUtils.get_param_by_name(model, scale_name)
        ).copy()

        downstream = {}
        for linear_op in linear_ops:
            weight_inp, is_transposed = _get_weight_product(linear_op)
            if weight_inp is None:
                continue
            weight_tensor = ParamUtils.get_param_by_name(model, weight_inp.name)
            if weight_tensor is None:
                continue
            downstream[weight_inp.name] = (
                numpy_helper.to_array(weight_tensor).copy(),
                linear_op,
                is_transposed,
            )

        if downstream:
            pre_fusion_state[scale_name] = (scale, downstream)

    return pre_fusion_state


def _verify_fusion(model: onnx.ModelProto, pre_state: dict):
    """
    When: fuse_norm_layers_into_linears has been called on the model.
    Then: every RMSNorm gamma initializer is reset to ones, and every downstream
          linear weight has been scaled by the corresponding pre-fusion gamma values.
    """
    assert pre_state

    for scale_name, (scale_before, weights_before) in pre_state.items():
        scale_after = numpy_helper.to_array(
            ParamUtils.get_param_by_name(model, scale_name)
        )
        assert np.array_equal(scale_after, np.ones_like(scale_after))

        for wname, (w_before, linear_op, is_transposed) in weights_before.items():
            w_after = numpy_helper.to_array(ParamUtils.get_param_by_name(model, wname))
            scale_f64 = scale_before.astype(np.float64)

            if linear_op.type == "Conv":
                # W[out, in, *kernel]: absorb gamma along axis 1 (in_channels)
                bc = scale_f64.reshape(1, -1, *([1] * (w_before.ndim - 2)))
            elif is_transposed:
                # Gemm transB=1 or W→Transpose→MatMul: stored W[out, in], absorb along axis 1
                bc = scale_f64[None, :]
            else:
                # MatMul / Gemm transB=0: stored W[in, out], absorb along axis 0
                bc = scale_f64[:, None]

            w_expected = (bc * w_before.astype(np.float64)).astype(w_before.dtype)
            assert np.allclose(
                w_after,
                w_expected,
            )


def _collect_all_weights(model: onnx.ModelProto, role_map: DecoderModelRoleMap) -> dict:
    weights = {}

    def _store_linear(op):
        weight_inp, _ = _get_weight_product(op)
        if weight_inp is not None:
            tensor = ParamUtils.get_param_by_name(model, weight_inp.name)
            if tensor is not None:
                weights[weight_inp.name] = numpy_helper.to_array(tensor).copy()
        bias_inp = _get_bias_product(op)
        if bias_inp is not None:
            tensor = ParamUtils.get_param_by_name(model, bias_inp.name)
            if tensor is not None:
                weights[bias_inp.name] = numpy_helper.to_array(tensor).copy()

    def _store_gather(op):
        for inp in op.inputs:
            if inp.is_parm or inp.is_const:
                tensor = ParamUtils.get_param_by_name(model, inp.name)
                if tensor is not None:
                    weights[inp.name] = numpy_helper.to_array(tensor).copy()
                    return

    for op in role_map.embed_tokens:
        _store_gather(op)
    for op in role_map.lm_head:
        _store_linear(op)
    for block in role_map.blocks:
        for op in (
            block.qkv_linears + block.o_proj + block.gate_up_linears + block.down_proj
        ):
            _store_linear(op)
    return weights


class RMSNormMatMul(nn.Module):
    """RMSNorm followed by torch.matmul — exports as MatMul (transB=0)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mul_for_pow: bool,
        mul_rsqrt_pattern: str,
    ):
        super().__init__()
        self.norm = RMSNorm(
            in_features, mul_for_pow=mul_for_pow, mul_rsqrt_pattern=mul_rsqrt_pattern
        )
        self.W = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, x):
        return torch.matmul(self.norm(x), self.W)


class RMSNormLinear(nn.Module):
    """RMSNorm followed by nn.Linear — exports as Gemm with transB=1."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mul_for_pow: bool,
        mul_rsqrt_pattern: str,
        bias: bool = False,
    ):
        super().__init__()
        self.norm = RMSNorm(
            in_features, mul_for_pow=mul_for_pow, mul_rsqrt_pattern=mul_rsqrt_pattern
        )
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(self.norm(x))


class RMSNormProjectionLayers(nn.Module):
    """One RMSNorm feeding three independent Linear layers (Q / K / V pattern)."""

    def __init__(self, H: int, mul_for_pow: bool, mul_rsqrt_pattern: str):
        super().__init__()
        self.norm = RMSNorm(
            H, mul_for_pow=mul_for_pow, mul_rsqrt_pattern=mul_rsqrt_pattern
        )
        self.q = nn.Linear(H, H, bias=False)
        self.k = nn.Linear(H, H, bias=False)
        self.v = nn.Linear(H, H, bias=False)

    def forward(self, x):
        y = self.norm(x)
        return self.q(y) + self.k(y) + self.v(y)


class RMSNormConvViaTranspose(nn.Module):
    """SHA_Conv ConvInplaceLinear pattern"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mul_for_pow: bool,
        mul_rsqrt_pattern: str,
    ):
        super().__init__()
        self.norm = RMSNorm(
            in_channels, mul_for_pow=mul_for_pow, mul_rsqrt_pattern=mul_rsqrt_pattern
        )
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):  # x: [B, seq, in_channels]
        y = self.norm(x)  # [B, seq, in_channels]
        y = y.transpose(-1, -2)  # [B, in_channels, seq]  — Transpose in ONNX graph
        return self.conv(y)  # [B, out_channels, seq]


class RMSNormReshapeLinear(nn.Module):
    """RMSNorm(d_v) -> Reshape(1, s_sq*d_v) -> Gemm(s_sq*d_v)"""

    def __init__(
        self,
        d_v: int,
        s_sq: int,
        mul_for_pow: bool,
        mul_rsqrt_pattern: str,
    ):
        super().__init__()
        self.norm = RMSNorm(
            d_v, mul_for_pow=mul_for_pow, mul_rsqrt_pattern=mul_rsqrt_pattern
        )
        self.linear = nn.Linear(d_v * s_sq, d_v * s_sq, bias=True)

    def forward(self, x):
        y = self.norm(x)
        y = y.reshape(1, -1)
        return self.linear(y)


class TestFuseNormLayers:
    """Unit tests for fuse_norm_layers_into_linears.

    1. RMSNorm → MatMul  (via torch.matmul, transB=0)
    2. RMSNorm → Gemm    (via nn.Linear, transB=1)
    3. RMSNorm → MatMul  (via nn.Linear, transB=1)
    4. RMSNorm → three parallel Gemm ops  (Q / K / V)
    5. RMSNorm → Transpose → Conv  (SHA_Conv reshape-chain pattern)
    6. Non-affine RMSNorm (no gamma)  → no-op, weights unchanged
    7. RMSNorm → Reshape → Linear    (gamma tiling, VLM merger ln_q pattern)
    """

    IN = 8
    OUT = 6
    B, SEQ = 1, 4

    @pytest.mark.parametrize("mul_for_pow", [True, False])
    @pytest.mark.parametrize(
        "mul_rsqrt_pattern", ["mul_rsqrt", "div_sqrt", "mul_reciprocal_sqrt"]
    )
    def test_matmul(self, mul_for_pow, mul_rsqrt_pattern):
        """RMSNorm → MatMul[in_features, out_features]: gamma absorbed along axis 0 (in_features).

        nn.Linear with 3D input and do_constant_folding=True, exports as MatMul[in_features, out_features] , which sets
        transposed_params=False in ConnectedGraph
        """
        torch.manual_seed(0)
        np.random.seed(0)
        module = RMSNormMatMul(self.IN, self.OUT, mul_for_pow, mul_rsqrt_pattern)
        x = np.random.randn(self.B, self.SEQ, self.IN).astype(np.float32)
        model = _export_to_onnx(module, torch.from_numpy(x))
        cg = ConnectedGraph(model)

        y_before = _run_model(model, x)
        pre = _collect_pre_fusion_state(model, cg)
        fuse_norm_layers_into_linears(model, find_active_norms(model, cg))
        _verify_fusion(model, pre)
        assert np.allclose(_run_model(model, x), y_before, atol=1e-6)

    @pytest.mark.parametrize("mul_for_pow", [True, False])
    @pytest.mark.parametrize(
        "mul_rsqrt_pattern", ["mul_rsqrt", "div_sqrt", "mul_reciprocal_sqrt"]
    )
    def test_gemm_transb(self, mul_for_pow, mul_rsqrt_pattern):
        """RMSNorm → Gemm[out, H] transB=1: gamma absorbed along axis 1.

        nn.Linear with bias and 2D input exports as Gemm(transB=1), which sets
        transposed_params=True in ConnectedGraph
        """
        torch.manual_seed(0)
        np.random.seed(0)
        module = RMSNormLinear(
            self.IN, self.OUT, mul_for_pow, mul_rsqrt_pattern, bias=True
        )
        x = np.random.randn(self.B, self.IN).astype(np.float32)
        model = _export_to_onnx(module, torch.from_numpy(x))
        cg = ConnectedGraph(model)

        y_before = _run_model(model, x)
        pre = _collect_pre_fusion_state(model, cg)
        fuse_norm_layers_into_linears(model, find_active_norms(model, cg))
        _verify_fusion(model, pre)
        assert np.allclose(_run_model(model, x), y_before, atol=1e-6)

    @pytest.mark.parametrize("mul_for_pow", [True, False])
    @pytest.mark.parametrize(
        "mul_rsqrt_pattern", ["mul_rsqrt", "div_sqrt", "mul_reciprocal_sqrt"]
    )
    def test_matmul_transb(self, mul_for_pow, mul_rsqrt_pattern):
        """RMSNorm → MatMul[out_features, in_features] transB=1: gamma absorbed along axis 1.

        nn.Linear with 3D input and do_constant_folding=False, exports as MatMul[out_features, in_features] , which sets
        transposed_params=True in ConnectedGraph
        """
        torch.manual_seed(0)
        np.random.seed(0)
        module = RMSNormLinear(
            self.IN, self.OUT, mul_for_pow, mul_rsqrt_pattern, bias=False
        )
        x = np.random.randn(self.B, self.SEQ, self.IN).astype(np.float32)
        model = _export_to_onnx(module, torch.from_numpy(x), do_constant_folding=False)
        cg = ConnectedGraph(model)

        y_before = _run_model(model, x)
        pre = _collect_pre_fusion_state(model, cg)
        fuse_norm_layers_into_linears(model, find_active_norms(model, cg))
        _verify_fusion(model, pre)
        assert np.allclose(_run_model(model, x), y_before, atol=1e-6)

    @pytest.mark.parametrize("mul_for_pow", [True, False])
    @pytest.mark.parametrize(
        "mul_rsqrt_pattern", ["mul_rsqrt", "div_sqrt", "mul_reciprocal_sqrt"]
    )
    def test_multiple_downstream_linears(self, mul_for_pow, mul_rsqrt_pattern):
        """One norm scale Mul feeds three Gemm ops — all must be fused."""
        torch.manual_seed(0)
        np.random.seed(0)
        module = RMSNormProjectionLayers(self.IN, mul_for_pow, mul_rsqrt_pattern)
        x = np.random.randn(self.B, self.SEQ, self.IN).astype(np.float32)
        model = _export_to_onnx(module, torch.from_numpy(x))
        cg = ConnectedGraph(model)

        y_before = _run_model(model, x)
        pre = _collect_pre_fusion_state(model, cg)
        assert len(next(iter(pre.values()))[1]) == 3
        fuse_norm_layers_into_linears(model, find_active_norms(model, cg))
        _verify_fusion(model, pre)
        assert np.allclose(_run_model(model, x), y_before, atol=1e-6)

    @pytest.mark.parametrize("mul_for_pow", [True, False])
    @pytest.mark.parametrize(
        "mul_rsqrt_pattern", ["mul_rsqrt", "div_sqrt", "mul_reciprocal_sqrt"]
    )
    def test_conv_via_reshape_chain(self, mul_for_pow, mul_rsqrt_pattern):
        """scale_Mul → Transpose → Conv1d: gamma absorbed along Conv axis 1 (in_channels)."""
        torch.manual_seed(0)
        np.random.seed(0)
        module = RMSNormConvViaTranspose(
            self.IN, self.OUT, mul_for_pow, mul_rsqrt_pattern
        )
        x = np.random.randn(self.B, self.SEQ, self.IN).astype(np.float32)
        model = _export_to_onnx(module, torch.from_numpy(x))
        cg = ConnectedGraph(model)

        y_before = _run_model(model, x)
        pre = _collect_pre_fusion_state(model, cg)
        fuse_norm_layers_into_linears(model, find_active_norms(model, cg))
        _verify_fusion(model, pre)
        assert np.allclose(_run_model(model, x), y_before, atol=1e-6)

    @pytest.mark.parametrize("mul_for_pow", [True, False])
    @pytest.mark.parametrize(
        "mul_rsqrt_pattern", ["mul_rsqrt", "div_sqrt", "mul_reciprocal_sqrt"]
    )
    def test_non_affine_norm_no_op(self, mul_for_pow, mul_rsqrt_pattern):
        """RMSNorm without a scale weight leaves all downstream weights unchanged."""
        torch.manual_seed(0)
        np.random.seed(0)

        class NoGammaNormLinear(nn.Module):
            def __init__(self, mul_for_pow, mul_rsqrt_pattern):
                super().__init__()
                self.norm = RMSNorm(
                    TestFuseNormLayers.IN,
                    elementwise_affine=False,
                    mul_for_pow=mul_for_pow,
                    mul_rsqrt_pattern=mul_rsqrt_pattern,
                )
                self.linear = nn.Linear(
                    TestFuseNormLayers.IN, TestFuseNormLayers.OUT, bias=False
                )

            def forward(self, x):
                return self.linear(self.norm(x))

        x = np.random.randn(self.B, self.SEQ, self.IN).astype(np.float32)
        module = NoGammaNormLinear(mul_for_pow, mul_rsqrt_pattern)
        model = _export_to_onnx(module, torch.from_numpy(x))
        w_name = next(t.name for t in model.graph.initializer)
        w_before = numpy_helper.to_array(
            ParamUtils.get_param_by_name(model, w_name)
        ).copy()

        y_before = _run_model(model, x)
        cg = ConnectedGraph(model)
        fuse_norm_layers_into_linears(model, find_active_norms(model, cg))

        w_after = numpy_helper.to_array(ParamUtils.get_param_by_name(model, w_name))
        assert np.array_equal(w_after, w_before)
        assert np.allclose(_run_model(model, x), y_before, atol=1e-6)

    @pytest.mark.parametrize("mul_for_pow", [True, False])
    @pytest.mark.parametrize(
        "mul_rsqrt_pattern", ["mul_rsqrt", "div_sqrt", "mul_reciprocal_sqrt"]
    )
    def test_gamma_tiling(self, mul_for_pow, mul_rsqrt_pattern):
        """RMSNorm(d_v) -> Reshape -> Gemm(s_sq*d_v): gamma tiled s_sq times before fusion."""
        d_v = 4
        s_sq = 2
        torch.manual_seed(0)
        np.random.seed(0)
        module = RMSNormReshapeLinear(d_v, s_sq, mul_for_pow, mul_rsqrt_pattern)
        x = np.random.randn(s_sq, d_v).astype(np.float32)
        model = _export_to_onnx(module, torch.from_numpy(x))
        cg = ConnectedGraph(model)

        y_before = _run_model(model, x)

        # Collect pre-fusion state manually: gamma is [d_v] but weight in_features is s_sq*d_v
        active_norms = find_active_norms(model, cg)
        assert len(active_norms) == 1
        scale_name = active_norms[0].scale_name
        gamma_before = numpy_helper.to_array(
            ParamUtils.get_param_by_name(model, scale_name)
        ).copy()
        assert gamma_before.shape == (d_v,)

        linear_ops = active_norms[0].downstream_linears
        assert len(linear_ops) == 1
        weight_inp, is_transposed = _get_weight_product(linear_ops[0])
        W_before = numpy_helper.to_array(
            ParamUtils.get_param_by_name(model, weight_inp.name)
        ).copy()
        in_features = W_before.shape[1] if is_transposed else W_before.shape[0]
        assert in_features == d_v * s_sq

        """
        When: fuse_norm_layers_into_linears is applied
        Then: Gamma must be reset to ones (original d_v shape),
         weight must be scaled by gamma tiled s_sq times
        """

        fuse_norm_layers_into_linears(model, active_norms)

        gamma_after = numpy_helper.to_array(
            ParamUtils.get_param_by_name(model, scale_name)
        )
        assert np.array_equal(gamma_after, np.ones(d_v, dtype=gamma_before.dtype))

        gamma_tiled = np.tile(gamma_before.astype(np.float64), s_sq)
        W_after = numpy_helper.to_array(
            ParamUtils.get_param_by_name(model, weight_inp.name)
        )
        if is_transposed:
            # Gemm transB=1: stored W[out, in], gamma absorbed along axis 1
            W_expected = (gamma_tiled[None, :] * W_before.astype(np.float64)).astype(
                W_before.dtype
            )
        else:
            # MatMul: stored W[in, out], gamma absorbed along axis 0
            W_expected = (gamma_tiled[:, None] * W_before.astype(np.float64)).astype(
                W_before.dtype
            )
        assert np.allclose(W_after, W_expected)
        assert np.allclose(_run_model(model, x), y_before)


_NORM_KW = dict(mul_for_pow=False, mul_rsqrt_pattern="mul_rsqrt")
_H, _I = 8, 16  # hidden dim, intermediate dim
_NUM_HEADS, _HEAD_DIM = 2, 4  # _H == _NUM_HEADS * _HEAD_DIM
_VOCAB = 16
_B, _SEQ = 1, 4  # batch, sequence length


def _mha(q, k, v, num_heads, head_dim):
    """Standard multi-head attention: split into heads, softmax(QK^T/sqrt(d))V, merge.

    :param q: [B, S, hidden]
    :param k: [B, S, hidden]
    :param v: [B, S, hidden]
    :return: [B, S, hidden]
    """
    B, S, _ = q.shape
    q = q.reshape(B, S, num_heads, head_dim).transpose(1, 2)  # [B, H, S, D]
    k = k.reshape(B, S, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(B, S, num_heads, head_dim).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim**0.5)
    attn = torch.softmax(scores, dim=-1)
    y = torch.matmul(attn, v)  # [B, H, S, D]
    return y.transpose(1, 2).reshape(B, S, num_heads * head_dim)


_VIT_D, _VIT_I = 8, 12  # ViT hidden size, intermediate dim
_VIT_S_SQ = 4
_VIT_N = 8
_VIT_D_L = _H  # language hidden size; must equal backbone _H for VLM integration tests


def _export_vit(module: nn.Module) -> onnx.ModelProto:
    x = torch.randn(_VIT_N, _VIT_D)
    return _export_to_onnx(module, x)


def _attach_past_value_input(model: onnx.ModelProto) -> onnx.ModelProto:
    """Attach a dangling ``past_value_0`` graph input whose last dim is ``_HEAD_DIM``.

    The input is unused by any node in the graph; it exists solely so that
    ``infer_head_dim`` can derive ``head_dim`` from a real export-style input,
    matching the HF/optimum convention where ``past_value_*`` tensors carry the
    KV cache. Static shape ``[1, 1, 1, _HEAD_DIM]`` is the smallest valid
    rank-4 KV-cache layout that gives the helper a static last dim to read.
    """
    past_value = onnx.helper.make_tensor_value_info(
        name="past_value_0",
        elem_type=onnx.TensorProto.FLOAT,
        shape=[1, 1, 1, _HEAD_DIM],
    )
    model.graph.input.append(past_value)
    return model


def _export_decoder(module: nn.Module) -> onnx.ModelProto:
    x = torch.randn(_B, _SEQ, _H)
    return _attach_past_value_input(_export_to_onnx(module, x))


def _export_decoder_with_ids(module: nn.Module) -> onnx.ModelProto:
    token_ids = torch.randint(0, _VOCAB, (_B, _SEQ))
    return _attach_past_value_input(_export_to_onnx(module, token_ids))


def _fuse_rms_norms(model: onnx.ModelProto) -> onnx.ModelProto:
    """Coalesce decomposed RMSNorm patterns into ``RMSNormalization`` supergroup ops.

    QuantizationSimModel applies this fusion before constructing its ConnectedGraph,
    so the spinquant analyzers see fused ops in production. Tests that build a bare
    ConnectedGraph need this shim to exercise the same branches.
    """
    ir_model = onnx_ir.from_proto(model)
    fused = fuse_supergroups(ir_model, patterns=["RMSNormalization"])
    return onnx_ir.to_proto(fused)


class _LlamaBlock(nn.Module):
    """Simplified LLaMA/Qwen2 decoder block: 2 active norms per block.

    input_norm feeds q/k/v projections.
    post_attn_norm feeds gate/up projections.

    :param bias: When True, enables bias on the writing layers (o_proj, down_proj).
        Reading layers (q, k, v, gate, up) always have bias=False.
    """

    def __init__(self, bias: bool = False):
        super().__init__()
        self.input_norm = RMSNorm(_H, **_NORM_KW)
        self.q = nn.Linear(_H, _H, bias=False)
        self.k = nn.Linear(_H, _H, bias=False)
        self.v = nn.Linear(_H, _H, bias=False)
        self.o = nn.Linear(_H, _H, bias=bias)
        self.post_attn_norm = RMSNorm(_H, **_NORM_KW)
        self.gate = nn.Linear(_H, _I, bias=False)
        self.up = nn.Linear(_H, _I, bias=False)
        self.down = nn.Linear(_I, _H, bias=bias)

    def forward(self, x):
        h = self.input_norm(x)
        y = _mha(self.q(h), self.k(h), self.v(h), _NUM_HEADS, _HEAD_DIM)
        x = x + self.o(y)
        h2 = self.post_attn_norm(x)
        return x + self.down(self.gate(h2) * self.up(h2))


class LlamaStyleDecoder(nn.Module):
    """2-block LLaMA decoder + embed_tokens + final norm + lm_head: 5 active norms total.

    :param bias: When True, enables bias on the writing layers (o_proj, down_proj) in each block.
    """

    def __init__(self, bias: bool = False):
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB, _H)
        self.block0 = _LlamaBlock(bias=bias)
        self.block1 = _LlamaBlock(bias=bias)
        self.norm = RMSNorm(_H, **_NORM_KW)
        self.lm_head = nn.Linear(_H, _VOCAB, bias=False)

    def forward(self, token_ids):
        x = self.embed_tokens(token_ids)
        x = self.block0(x)
        x = self.block1(x)
        return self.lm_head(self.norm(x))


class _Qwen3Block(nn.Module):
    """Qwen3-style decoder block: 4 norms per block, only 2 active (q_norm / k_norm are internal and their output not fed directly into a weight MatMul/Gemm/Conv )."""

    def __init__(self):
        super().__init__()
        self.input_norm = RMSNorm(_H, **_NORM_KW)
        self.q_proj = nn.Linear(_H, _H, bias=False)
        self.k_proj = nn.Linear(_H, _H, bias=False)
        self.v = nn.Linear(_H, _H, bias=False)
        self.q_norm = RMSNorm(_H, **_NORM_KW)
        self.k_norm = RMSNorm(_H, **_NORM_KW)
        self.o = nn.Linear(_H, _H, bias=False)
        self.post_attn_norm = RMSNorm(_H, **_NORM_KW)
        self.gate = nn.Linear(_H, _I, bias=False)
        self.up = nn.Linear(_H, _I, bias=False)
        self.down = nn.Linear(_I, _H, bias=False)

    def forward(self, x):
        h = self.input_norm(x)
        q = self.q_norm(self.q_proj(h))
        k = self.k_norm(self.k_proj(h))
        v = self.v(h)
        y = _mha(q, k, v, _NUM_HEADS, _HEAD_DIM)
        x = x + self.o(y)
        h2 = self.post_attn_norm(x)
        return x + self.down(self.gate(h2) * self.up(h2))


class Qwen3StyleDecoder(nn.Module):
    """2-block Qwen3 decoder + embed_tokens + final norm + lm_head: 5 active norms total."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB, _H)
        self.block0 = _Qwen3Block()
        self.block1 = _Qwen3Block()
        self.norm = RMSNorm(_H, **_NORM_KW)
        self.lm_head = nn.Linear(_H, _VOCAB, bias=False)

    def forward(self, token_ids):
        x = self.embed_tokens(token_ids)
        x = self.block0(x)
        x = self.block1(x)
        return self.lm_head(self.norm(x))


class _Phi3Block(nn.Module):
    """Phi3-style decoder block: fused qkv_proj (single linear) and fused gate_up_proj (single linear)."""

    def __init__(self):
        super().__init__()
        self.input_norm = RMSNorm(_H, **_NORM_KW)
        self.qkv_proj = nn.Linear(_H, 3 * _H, bias=False)
        self.o = nn.Linear(_H, _H, bias=False)
        self.post_attn_norm = RMSNorm(_H, **_NORM_KW)
        self.gate_up_proj = nn.Linear(_H, 2 * _I, bias=False)
        self.down = nn.Linear(_I, _H, bias=False)

    def forward(self, x):
        h = self.input_norm(x)
        q, k, v = self.qkv_proj(h).chunk(3, dim=-1)
        attn = self.o(q + k + v)
        x = x + attn
        h2 = self.post_attn_norm(x)
        gate, up = self.gate_up_proj(h2).chunk(2, dim=-1)
        return x + self.down(gate * up)


class Phi3StyleDecoder(nn.Module):
    """2-block Phi3 decoder + embed_tokens + final norm + lm_head: 5 active norms total."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB, _H)
        self.block0 = _Phi3Block()
        self.block1 = _Phi3Block()
        self.norm = RMSNorm(_H, **_NORM_KW)
        self.lm_head = nn.Linear(_H, _VOCAB, bias=False)

    def forward(self, token_ids):
        x = self.embed_tokens(token_ids)
        x = self.block0(x)
        x = self.block1(x)
        return self.lm_head(self.norm(x))


class _ViTBlock(nn.Module):
    """Minimal ViT transformer block."""

    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(_VIT_D, **_NORM_KW)
        self.qkv = nn.Linear(_VIT_D, _VIT_D, bias=False)  # reading — no bias rotation
        self.proj = nn.Linear(
            _VIT_D, _VIT_D, bias=True
        )  # writing — bias rotated with R_V
        self.norm2 = RMSNorm(_VIT_D, **_NORM_KW)
        self.gate_proj = nn.Linear(
            _VIT_D, _VIT_I, bias=False
        )  # reading — no bias rotation
        self.up_proj = nn.Linear(
            _VIT_D, _VIT_I, bias=False
        )  # reading — no bias rotation
        self.down_proj = nn.Linear(
            _VIT_I, _VIT_D, bias=True
        )  # writing — bias rotated with R_V

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.proj(self.qkv(h))
        h2 = self.norm2(x)
        return x + self.down_proj(self.gate_proj(h2) * self.up_proj(h2))


class _ViTMerger(nn.Module):
    """PatchMerger: ln_q -> view(-1, s²·d_V) -> linear1 -> GELU -> linear2."""

    def __init__(self):
        super().__init__()
        self.ln_q = RMSNorm(_VIT_D, **_NORM_KW)
        self.linear1 = nn.Linear(
            _VIT_D * _VIT_S_SQ, _VIT_D * _VIT_S_SQ, bias=False
        )  # reading — no bias rotation
        self.linear2 = nn.Linear(
            _VIT_D * _VIT_S_SQ, _VIT_D_L, bias=True
        )  # writing — bias rotated with R_L

    def forward(self, x):
        x = self.ln_q(x)
        x = x.reshape(-1, _VIT_D * _VIT_S_SQ)
        x = torch.nn.functional.gelu(self.linear1(x))
        return self.linear2(x)


class ViTEncoder(nn.Module):
    """ViT encoder: Conv patch_embed -> 2 ViT blocks -> PatchMerger."""

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv1d(_VIT_D, _VIT_D, kernel_size=1, bias=False)
        self.block0 = _ViTBlock()
        self.block1 = _ViTBlock()
        self.merger = _ViTMerger()

    def forward(self, x):
        h = self.patch_embed(x.T.unsqueeze(0)).squeeze(0).T
        h = self.block0(h)
        h = self.block1(h)
        return self.merger(h)


class _LayerNormViTBlock(nn.Module):
    """Qwen3-VL style ViT block using LayerNorm."""

    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(_VIT_D)
        self.qkv = nn.Linear(_VIT_D, _VIT_D, bias=False)
        self.proj = nn.Linear(_VIT_D, _VIT_D, bias=False)
        self.norm2 = nn.LayerNorm(_VIT_D)
        self.gate_proj = nn.Linear(_VIT_D, _VIT_I, bias=False)
        self.up_proj = nn.Linear(_VIT_D, _VIT_I, bias=False)
        self.down_proj = nn.Linear(_VIT_I, _VIT_D, bias=False)

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.proj(self.qkv(h))
        h2 = self.norm2(x)
        return x + self.down_proj(self.gate_proj(h2) * self.up_proj(h2))


class LayerNormViTEncoder(nn.Module):
    """Qwen3-VL style ViT encoder."""

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv1d(_VIT_D, _VIT_D, kernel_size=1, bias=False)
        self.block0 = _LayerNormViTBlock()
        self.block1 = _LayerNormViTBlock()
        self.linear2 = nn.Linear(_VIT_D, _VIT_D_L, bias=False)

    def forward(self, x):
        h = self.patch_embed(x.T.unsqueeze(0)).squeeze(0).T
        h = self.block0(h)
        h = self.block1(h)
        return self.linear2(h)


class _Gemma3Block(nn.Module):
    """Gemma3-style decoder block: post-writing norms after o_proj and down_proj."""

    def __init__(self):
        super().__init__()
        self.input_norm = RMSNorm(_H, **_NORM_KW)
        self.q = nn.Linear(_H, _H, bias=False)
        self.k = nn.Linear(_H, _H, bias=False)
        self.v = nn.Linear(_H, _H, bias=False)
        self.o = nn.Linear(_H, _H, bias=False)
        self.post_attn_norm = RMSNorm(_H, **_NORM_KW)
        self.pre_ffn_norm = RMSNorm(_H, **_NORM_KW)
        self.gate = nn.Linear(_H, _I, bias=False)
        self.up = nn.Linear(_H, _I, bias=False)
        self.down = nn.Linear(_I, _H, bias=False)
        self.post_ffn_norm = RMSNorm(_H, **_NORM_KW)

    def forward(self, x):
        h = self.input_norm(x)
        attn = self.o(self.q(h) + self.k(h) + self.v(h))
        x = x + self.post_attn_norm(attn)
        h2 = self.pre_ffn_norm(x)
        ffn = self.down(self.gate(h2) * self.up(h2))
        return x + self.post_ffn_norm(ffn)


class Gemma3StyleDecoder(nn.Module):
    """2-block Gemma3 decoder"""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB, _H)
        self.block0 = _Gemma3Block()
        self.block1 = _Gemma3Block()
        self.norm = RMSNorm(_H, **_NORM_KW)
        self.lm_head = nn.Linear(_H, _VOCAB, bias=False)

    def forward(self, token_ids):
        x = self.embed_tokens(token_ids)
        x = self.block0(x)
        x = self.block1(x)
        return self.lm_head(self.norm(x))


class VLMBackbone(nn.Module):
    """Backbone for VLM: takes inputs_embeds, no embed_tokens Gather."""

    def __init__(self):
        super().__init__()
        self.block0 = _LlamaBlock()
        self.block1 = _LlamaBlock()
        self.norm = RMSNorm(_H, **_NORM_KW)
        self.lm_head = nn.Linear(_H, _VOCAB, bias=False)

    def forward(self, inputs_embeds):
        x = self.block0(inputs_embeds)
        x = self.block1(x)
        return self.lm_head(self.norm(x))


def _export_vlm_backbone(module: nn.Module) -> onnx.ModelProto:
    inputs_embeds = torch.randn(_B, _SEQ, _H)
    return _export_to_onnx(module, inputs_embeds)


class TestBlockIdentifier:
    """Tests for find_active_norms and decoder block detection boundaries."""

    def test_llama_active_norm_count(self):
        """2 blocks × 2 active norms/block + 1 final norm → 5 active norms."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(LlamaStyleDecoder())
        cg = ConnectedGraph(model)
        active_norms = find_active_norms(model, cg)
        assert len(active_norms) == 5

    def test_llama_all_have_downstream_linears(self):
        """Every returned ActiveNorm must expose at least one downstream weight linear."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(LlamaStyleDecoder())
        cg = ConnectedGraph(model)
        for active_norm in find_active_norms(model, cg):
            assert active_norm.downstream_linears

    def test_qwen3_active_norm_count(self):
        """9 total norms but only 5 active; q_norm/k_norm (internal) are excluded."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(Qwen3StyleDecoder())
        cg = ConnectedGraph(model)
        active_norms = find_active_norms(model, cg)
        assert len(active_norms) == 5

    def test_qwen3_internal_norms_excluded(self):
        """9 total norms but only 5 active; q_norm/k_norm (internal) are excluded."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(Qwen3StyleDecoder())
        cg = ConnectedGraph(model)
        for active_norm in find_active_norms(model, cg):
            assert active_norm.downstream_linears

    def test_phi3_active_norm_count(self):
        """2 blocks × 2 active norms/block + 1 final norm → 5 active norms; fused qkv/gate_up do not affect norm detection."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(Phi3StyleDecoder())
        cg = ConnectedGraph(model)
        active_norms = find_active_norms(model, cg)
        assert len(active_norms) == 5

    def test_phi3_block_count(self):
        """Phi3StyleDecoder with 2 blocks → 2 boundaries."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(Phi3StyleDecoder())
        cg = ConnectedGraph(model)
        blocks, _ = get_decoder_block_boundaries(model, cg)
        assert len(blocks) == 2

    def test_llama_block_count(self):
        """LlamaStyleDecoder with 2 blocks → 2 boundaries."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(LlamaStyleDecoder())
        cg = ConnectedGraph(model)
        blocks, _ = get_decoder_block_boundaries(model, cg)
        assert len(blocks) == 2

    def test_qwen3_block_count(self):
        """Qwen3StyleDecoder with 2 blocks → 2 boundaries (internal norms ignored)."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(Qwen3StyleDecoder())
        cg = ConnectedGraph(model)
        blocks, _ = get_decoder_block_boundaries(model, cg)
        assert len(blocks) == 2

    def test_boundaries_are_active_norm_ops(self):
        """Both start_op and end_op of every boundary must be active norm start ops."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(LlamaStyleDecoder())
        cg = ConnectedGraph(model)
        blocks, active_norms = get_decoder_block_boundaries(model, cg)
        active_op_ids = {id(an.norm_op) for an in active_norms}
        for start_op, end_op in blocks:
            assert id(start_op) in active_op_ids
            assert id(end_op) in active_op_ids

    def test_boundaries_non_overlapping(self):
        """end_op of block i must be the same object as start_op of block i+1."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(LlamaStyleDecoder())
        cg = ConnectedGraph(model)
        blocks, _ = get_decoder_block_boundaries(model, cg)
        for i in range(len(blocks) - 1):
            assert id(blocks[i][1]) == id(blocks[i + 1][0])

    def test_even_active_norms(self):
        """Even active norm count must raise ValueError."""
        torch.manual_seed(0)

        class _NoFinalNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.block0 = _LlamaBlock()
                self.block1 = _LlamaBlock()

            def forward(self, x):
                return self.block1(self.block0(x))

        model = _export_decoder(_NoFinalNorm())
        cg = ConnectedGraph(model)
        with pytest.raises(ValueError):
            get_decoder_block_boundaries(model, cg)

    @pytest.mark.skip_on_windows_amd64("Fails with OSError, no space left on device")
    @pytest.mark.skip_on_windows_arm64("transformers is not available on Windows ARM64")
    @pytest.mark.parametrize(
        "model_id, model_type, adaptations",
        [
            ["Qwen/Qwen2-0.5B", "qwen2", []],
            ["Qwen/Qwen3-0.6B", "qwen3", ["SHA_Conv"]],
        ],
    )
    def test_get_decoder_block_boundaries(
        self, add_genai_tests_path, model_id, model_type, adaptations
    ):
        from GenAILab.qai_hub_lm.backends.onnx.llm import LLM_ONNX
        from GenAILab.qai_hub_lm.backends.onnx.export_utils import (
            get_model_checkpoint_path,
        )
        from aimet_onnx.experimental.adascale.find_blocks import (
            get_decoder_blocks_end_points,
        )

        cache_dir = get_model_checkpoint_path(model_id)
        try:
            if adaptations:
                import GenAILab.qai_hub_lm.transforms.sha_conv
                from GenAILab.bench.yaml_config_parser import (
                    YAMLConfigParser,
                )

                model_cls = YAMLConfigParser.get_model_class(model_type, adaptations)
            else:
                model_cls = LLM_ONNX

            entry = model_cls.instantiate_float_model(
                model_id, 32, 16, small_model=True
            )
            collection = model_cls.instantiate_quantsim(entry)
            blocks, _ = get_decoder_block_boundaries(
                collection.backbone.model.model,
                collection.backbone.connected_graph,
            )
            assert len(blocks) == 2
            blocks_old = get_decoder_blocks_end_points(collection.backbone, model_type)
            assert len(blocks_old) == 2

            # Both methods must identify the same block boundary ops.
            assert blocks == blocks_old
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)


class TestDecoderRoleMap:
    """Tests for get_decoder_role_map.

    Each test parametrizes ``fuse_rmsnorm``: False keeps the decomposed RMSNorm pattern
    (ReduceMean / Sqrt / Mul chain), True coalesces it into a single ``RMSNormalization``
    supergroup op, mirroring what QuantizationSimModel does before constructing its
    ConnectedGraph. Both paths must produce identical role maps.
    """

    @pytest.mark.parametrize("fuse_rmsnorm", [False, True])
    def test_llama_role_map_structure(self, fuse_rmsnorm):
        """LlamaStyleDecoder: verify per-block and model-level role counts.

        2 blocks × (3 qkv, 1 o_proj, 2 gate_up, 1 down_proj) + 1 lm_head + 1 embed_tokens.
        """
        torch.manual_seed(0)
        model = _export_decoder_with_ids(LlamaStyleDecoder())
        if fuse_rmsnorm:
            model = _fuse_rms_norms(model)
        cg = ConnectedGraph(model)
        blocks, active_norms = get_decoder_block_boundaries(model, cg)
        role_map = get_decoder_role_map(cg, blocks, active_norms)

        assert len(role_map.blocks) == 2
        assert len(role_map.lm_head) == 1
        assert len(role_map.embed_tokens) == 1
        for block in role_map.blocks:
            assert len(block.qkv_linears) == 3
            assert len(block.o_proj) == 1
            assert len(block.gate_up_linears) == 2
            assert len(block.down_proj) == 1

    @pytest.mark.parametrize("fuse_rmsnorm", [False, True])
    def test_qwen3_qkv_count(self, fuse_rmsnorm):
        """Qwen3 q_norm/k_norm are internal; qkv_linears still has 3 (q_proj, k_proj, v)."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(Qwen3StyleDecoder())
        if fuse_rmsnorm:
            model = _fuse_rms_norms(model)
        cg = ConnectedGraph(model)
        blocks, active_norms = get_decoder_block_boundaries(model, cg)
        role_map = get_decoder_role_map(cg, blocks, active_norms)
        for block in role_map.blocks:
            assert len(block.qkv_linears) == 3

    @pytest.mark.parametrize("fuse_rmsnorm", [False, True])
    def test_missing_embed_tokens_warns(self, fuse_rmsnorm):
        """A model without a Gather embedding (e.g. VLM backbone) warns, not raises."""
        torch.manual_seed(0)

        class _NoEmbedDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.block0 = _LlamaBlock()
                self.block1 = _LlamaBlock()
                self.norm = RMSNorm(_H, **_NORM_KW)
                self.lm_head = nn.Linear(_H, _VOCAB, bias=False)

            def forward(self, x):
                x = self.block0(x)
                x = self.block1(x)
                return self.lm_head(self.norm(x))

        model = _export_decoder(_NoEmbedDecoder())
        if fuse_rmsnorm:
            model = _fuse_rms_norms(model)
        cg = ConnectedGraph(model)
        blocks, active_norms = get_decoder_block_boundaries(model, cg)
        # Must not raise — VLM backbones exported with use_inputs_embeds=True have no Gather.
        role_map = get_decoder_role_map(cg, blocks, active_norms)
        assert role_map.embed_tokens == []

    @pytest.mark.parametrize("fuse_rmsnorm", [False, True])
    def test_extra_prologue_gather_with_scalar_constant_excluded(self, fuse_rmsnorm):
        """Non-embedding Gather ops (e.g. position-id / shape-derived lookups) with
        scalar or 1-D static constants must not be admitted into ``role_map.embed_tokens``.

        Real ONNX exports (e.g. Qwen3-0.6B with rotary preprocessing) produce extra
        ``Gather(constant_table, dynamic_index)`` ops in the prologue whose static
        input is a small 1-D or scalar tensor — not a [vocab, hidden] embedding
        table. Those must be filtered out so ``infer_hidden_size`` doesn't read
        ``shape[-1]`` of a 0-/1-D tensor.
        """
        torch.manual_seed(0)

        class _DecoderWithPrologueGather(nn.Module):
            """LLaMA decoder + a non-embedding Gather over a 1-D constant in the prologue.

            The auxiliary Gather output participates in the model output, so ONNX
            constant-folding can't strip it. This mimics what real exports produce
            in the rotary / position-id preprocessing — Gathers over scalar / 1-D
            static constants that must not be confused with the embedding table.
            """

            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(_VOCAB, _H)
                # 1-D table that gets indexed dynamically; exports as a Gather
                # whose static data input is a 1-D constant — NOT an embedding.
                self.register_buffer(
                    "aux_table", torch.arange(_SEQ, dtype=torch.float32)
                )
                self.block0 = _LlamaBlock()
                self.block1 = _LlamaBlock()
                self.norm = RMSNorm(_H, **_NORM_KW)
                self.lm_head = nn.Linear(_H, _VOCAB, bias=False)

            def forward(self, token_ids):
                x = self.embed_tokens(token_ids)
                # Gather(aux_table, token_ids[:, 0]): static data is 1-D, dynamic index.
                # ids[:, 0] also produces a Gather(input, scalar_const_index).
                aux = self.aux_table[token_ids[:, 0]]  # [B]
                x = self.block0(x)
                x = self.block1(x)
                # Use aux in the output so constant folding can't remove the Gathers.
                return self.lm_head(self.norm(x)) + aux.unsqueeze(-1).unsqueeze(-1)

        model = _export_decoder_with_ids(_DecoderWithPrologueGather())
        if fuse_rmsnorm:
            model = _fuse_rms_norms(model)
        cg = ConnectedGraph(model)
        blocks, active_norms = get_decoder_block_boundaries(model, cg)
        role_map = get_decoder_role_map(cg, blocks, active_norms)

        # Exactly one embed_tokens — the [vocab, hidden] embedding, not the 1-D Gather.
        assert len(role_map.embed_tokens) == 1
        # And infer_hidden_size doesn't IndexError on a scalar/1-D shape.
        assert _infer_hidden_size(model, role_map) == _H

    @pytest.mark.parametrize("fuse_rmsnorm", [False, True])
    def test_wrong_active_norms_per_block_raises(self, fuse_rmsnorm):
        """Passing active_norms_per_block inconsistent with detected norms raises ValueError."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(LlamaStyleDecoder())
        if fuse_rmsnorm:
            model = _fuse_rms_norms(model)
        cg = ConnectedGraph(model)
        blocks, active_norms = get_decoder_block_boundaries(model, cg)
        with pytest.raises(ValueError):
            get_decoder_role_map(cg, blocks, active_norms, active_norms_per_block=3)

    @pytest.mark.skip_on_windows_amd64("Fails with OSError, no space left on device")
    @pytest.mark.skip_on_windows_arm64("transformers is not available on Windows ARM64")
    def test_qwen3_role_map(self, add_genai_tests_path):
        """Qwen/Qwen3-0.6B with no adaptations."""
        from GenAILab.qai_hub_lm.backends.onnx.export_utils import (
            get_model_checkpoint_path,
        )
        from GenAILab.qai_hub_lm.backends.onnx.llm import LLM_ONNX
        import GenAILab.qai_hub_lm.transforms.sha_conv
        from GenAILab.bench.yaml_config_parser import YAMLConfigParser

        model_id = "Qwen/Qwen3-0.6B"
        cache_dir = get_model_checkpoint_path(model_id)
        try:
            model_cls = YAMLConfigParser.get_model_class("qwen3")
            entry = model_cls.instantiate_float_model(
                model_id, 32, 16, small_model=True
            )
            collection = model_cls.instantiate_quantsim(entry)
            onnx_model = collection.backbone.model.model
            cg = collection.backbone.connected_graph

            blocks, active_norms = get_decoder_block_boundaries(onnx_model, cg)
            role_map = get_decoder_role_map(cg, blocks, active_norms)

            assert len(role_map.blocks) == 2
            assert len(role_map.lm_head) == 1
            assert len(role_map.embed_tokens) == 1
            for block in role_map.blocks:
                assert len(block.qkv_linears) == 3
                assert len(block.o_proj) == 1
                assert len(block.gate_up_linears) == 2
                assert len(block.down_proj) == 1
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)

    @pytest.mark.parametrize("fuse_rmsnorm", [False, True])
    def test_find_merger_linear2_rmsorm_vit(self, fuse_rmsnorm):
        """ViTEncoder (RMSNorm blocks + PatchMerger): find_merger_linear2 returns linear_fc2."""
        torch.manual_seed(0)
        model = _export_vit(ViTEncoder())
        if fuse_rmsnorm:
            model = _fuse_rms_norms(model)
        cg = ConnectedGraph(model)

        """
        When: find_merger_linear2 is called on a ViT with a PatchMerger.
        Then: exactly 1 merger_linear2 op (linear_fc2) detected as the leaf weighted linear.
        """
        merger_linear2 = find_merger_linear2(cg)

        assert len(merger_linear2) == 1
        assert merger_linear2[0].type in ("MatMul", "Gemm")

    def test_find_merger_linear2_layernorm_vit(self):
        """Qwen3-VL style (LayerNorm): find_merger_linear2 detects the single output projection."""
        torch.manual_seed(0)
        model = _export_vit(LayerNormViTEncoder())
        cg = ConnectedGraph(model)

        """
        When: find_merger_linear2 is called on a ViT with LayerNorm (no active RMSNorms).
        Then: exactly 1 merger_linear2 op detected (leaf linear / graph output).
        """
        merger_linear2 = find_merger_linear2(cg)

        assert len(merger_linear2) == 1
        assert merger_linear2[0].type in ("MatMul", "Gemm")


class TestApplyR1Rotation:
    """Tests for apply_r1_rotation."""

    @pytest.mark.parametrize(
        "decoder_cls",
        [
            LlamaStyleDecoder,
            lambda: LlamaStyleDecoder(bias=True),
            Qwen3StyleDecoder,
            Phi3StyleDecoder,
        ],
        ids=["llama", "llama_bias", "qwen3", "phi3"],
    )
    def test_output_preserved_after_rotation(self, decoder_cls):
        """Model output must be numerically same before and after R1 rotation."""
        torch.manual_seed(0)
        np.random.seed(0)
        model = _export_decoder_with_ids(decoder_cls())
        cg = ConnectedGraph(model)

        blocks, active_norms = get_decoder_block_boundaries(model, cg)
        role_map = get_decoder_role_map(cg, blocks, active_norms)

        """
        When: fuse_norm_layers_into_linears is applied
        Then: RMSNorm's scale weight (gamma) is fused into downstream linear layers.
        """
        fuse_norm_layers_into_linears(model, active_norms)

        token_ids = np.random.randint(0, _VOCAB, (_B, _SEQ)).astype(np.int64)
        y_before = _run_model(model, token_ids)

        """
        When: apply_r1_rotation is applied
        Then: Model output is preserved numerically after rotation (R1 @ R1^T = I).
        """
        apply_r1_rotation(model, role_map, backbone_hidden_size=_H)

        y_after = _run_model(model, token_ids)
        assert np.allclose(y_after, y_before, atol=1e-5)

    @pytest.mark.parametrize(
        "decoder_cls",
        [
            LlamaStyleDecoder,
            lambda: LlamaStyleDecoder(bias=True),
            Qwen3StyleDecoder,
            Phi3StyleDecoder,
        ],
        ids=["llama", "llama_bias", "qwen3", "phi3"],
    )
    def test_double_rotation_recovers_original_weights(self, decoder_cls):
        """Applying R1 rotation twice must recover the original weights (R1 @ R1^T = I)."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(decoder_cls())
        cg = ConnectedGraph(model)

        blocks, active_norms = get_decoder_block_boundaries(model, cg)
        role_map = get_decoder_role_map(cg, blocks, active_norms)
        fuse_norm_layers_into_linears(model, active_norms)
        weights_original = _collect_all_weights(model, role_map)

        """
        When: apply_r1_rotation is applied twice
        Then: Linear layer weights are recovered.
        """
        apply_r1_rotation(model, role_map, backbone_hidden_size=_H)
        apply_r1_rotation(model, role_map, backbone_hidden_size=_H)

        weights_recovered = _collect_all_weights(model, role_map)
        for name, W_orig in weights_original.items():
            W_rec = weights_recovered[name]
            assert np.allclose(W_rec, W_orig, atol=1e-5)

    def test_right_multiply_formula(self):
        """_right_multiply: W_new = W @ R."""
        np.random.seed(0)
        H = 8
        R = np.array(scipy.linalg.hadamard(H) / np.sqrt(H), dtype=np.float64)

        # 2D [out, in], axis=1 - reading
        W = np.random.randn(5, H)
        assert np.allclose(_right_multiply(W, R, axis=1), W @ R)

        # 2D [in, out], axis=-1 - writing, (out=H, rotated on output side)
        W = np.random.randn(5, H)
        assert np.allclose(_right_multiply(W, R, axis=-1), W @ R)

        # Conv [out, in, k], axis=1 - reading
        W = np.random.randn(4, H, 3)
        expected = (W.transpose(0, 2, 1) @ R).transpose(0, 2, 1)
        assert np.allclose(_right_multiply(W, R, axis=1), expected)

    def test_left_multiply_formula(self):
        """_left_multiply: W_new = R^T @ W."""
        np.random.seed(0)
        H = 8
        R = np.array(scipy.linalg.hadamard(H) / np.sqrt(H), dtype=np.float64)

        # 2D [out, in], axis=0 - writing
        W = np.random.randn(H, 5)
        assert np.allclose(_left_multiply(W, R, axis=0), R.T @ W)

        # 2D [in, out], axis=0 - reading
        W = np.random.randn(H, 5)
        assert np.allclose(_left_multiply(W, R, axis=0), R.T @ W)

        # Conv [out, in, k], axis=0 - writing
        W = np.random.randn(H, 4, 3)
        expected = (R.T @ W.reshape(W.shape[0], -1)).reshape(W.shape)
        assert np.allclose(_left_multiply(W, R, axis=0), expected)

    def test_right_and_left_multiply_twice(self):
        """Applying the same multiply twice with R then R^T recovers W (R @ R^T = I)."""
        np.random.seed(0)
        H = 8
        R = np.array(scipy.linalg.hadamard(H) / np.sqrt(H), dtype=np.float64)
        W = np.random.randn(H, H)

        """
        When: _right_multiply is applied with R then R^T on the same axis, and
              _left_multiply is applied with R then R^T on the same axis.
        Then: The rotated weights is same as original weights.
        """
        assert np.allclose(
            _right_multiply(_right_multiply(W, R, axis=1), R.T, axis=1), W
        )
        assert np.allclose(_left_multiply(_left_multiply(W, R, axis=0), R.T, axis=0), W)

    @pytest.mark.parametrize(
        "vit_cls,vit_input_shape",
        [
            (ViTEncoder, (_VIT_N, _VIT_D)),
            (LayerNormViTEncoder, (_VIT_N, _VIT_D)),
        ],
        ids=["rmsnorm_vit", "layernorm_vit"],
    )
    def test_merger_output_rotated_by_r_l(self, vit_cls, vit_input_shape):
        """apply_r1_rotation_merger: ViT output must equal y_before @ R_L."""
        torch.manual_seed(0)
        np.random.seed(0)
        model = _export_vit(vit_cls())
        cg = ConnectedGraph(model)
        merger_linear2 = find_merger_linear2(cg)

        x = np.random.randn(*vit_input_shape).astype(np.float32)
        y_before = _run_model(model, x)

        """
        When: apply_r1_rotation_merger is applied to merger_linear2 with R_L.
        Then: output equals y_before @ R_L (merger_linear2 writes into R_L-rotated language space).
        """
        apply_r1_rotation_merger(model, merger_linear2, backbone_hidden_size=_VIT_D_L)

        R_L = (get_hadamard_matrix(_VIT_D_L) / np.sqrt(_VIT_D_L)).astype(np.float32)
        y_after = _run_model(model, x)
        assert np.allclose(y_after, y_before @ R_L)

    @pytest.mark.parametrize(
        "vit_cls",
        [ViTEncoder, LayerNormViTEncoder],
        ids=["rmsnorm_vit", "layernorm_vit"],
    )
    def test_merger_double_rotation_recovers_weights(self, vit_cls):
        """Applying apply_r1_rotation_merger twice must recover the original merger_linear2 weight."""
        torch.manual_seed(0)
        model = _export_vit(vit_cls())
        cg = ConnectedGraph(model)
        merger_linear2 = find_merger_linear2(cg)

        weights_original = {
            t.name: numpy_helper.to_array(t).copy() for t in model.graph.initializer
        }

        """
        When: apply_r1_rotation_merger is applied twice.
        Then: all initializers are recovered (R @ R^T = I).
        """
        apply_r1_rotation_merger(model, merger_linear2, backbone_hidden_size=_VIT_D_L)
        apply_r1_rotation_merger(model, merger_linear2, backbone_hidden_size=_VIT_D_L)

        for name, W_orig in weights_original.items():
            W_rec = numpy_helper.to_array(ParamUtils.get_param_by_name(model, name))
            assert np.allclose(W_rec, W_orig, atol=1e-5)

    def test_validate_merger_linear2_wrong_shape_raises(self):
        """_validate_merger_linear2 must raise RuntimeError when backbone_hidden_size doesn't match."""
        torch.manual_seed(0)
        model = _export_vit(ViTEncoder())
        cg = ConnectedGraph(model)
        merger_linear2 = find_merger_linear2(cg)

        """
        When: _validate_merger_linear2 is called with wrong backbone_hidden_size.
        Then: RuntimeError is raised.
        """
        with pytest.raises(RuntimeError):
            _validate_merger_linear2(
                model, merger_linear2, backbone_hidden_size=_VIT_D_L + 1
            )


_R2_DECODER_PARAMS = [
    pytest.param(LlamaStyleDecoder, id="llama"),
    pytest.param(lambda: LlamaStyleDecoder(bias=True), id="llama_bias"),
    pytest.param(Qwen3StyleDecoder, id="qwen3"),
]


class TestApplyR2Rotation:
    """Tests for R2RotationPass."""

    def test_find_attention_topology_returns_v(self):
        """Topology helper must identify the V projection (not Q or K)."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(LlamaStyleDecoder())
        cg = ConnectedGraph(model)
        blocks, active_norms = get_decoder_block_boundaries(model, cg)
        role_map = get_decoder_role_map(cg, blocks, active_norms)

        topology = find_attention_topology(role_map)

        assert len(topology) == len(role_map.blocks)
        for t in topology:
            assert len(t.v_ops) == 1
            assert "/v/" in t.v_ops[0].name

    def test_infer_head_dim_from_past_value_input(self):
        """``infer_head_dim`` must read head_dim from the attached past_value_0 input."""
        model = _export_decoder_with_ids(LlamaStyleDecoder())
        assert _infer_head_dim(model) == _HEAD_DIM

    def test_find_attention_topology_rejects_fused_qkv(self):
        """Phi3-style fused QKV must be rejected with a clear error."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(Phi3StyleDecoder())
        cg = ConnectedGraph(model)
        blocks, active_norms = get_decoder_block_boundaries(model, cg)
        role_map = get_decoder_role_map(cg, blocks, active_norms)

        with pytest.raises(ValueError, match="R2 rotation"):
            find_attention_topology(role_map)

    @pytest.mark.parametrize("decoder_cls", _R2_DECODER_PARAMS)
    def test_r2_alone_preserves_output(self, decoder_cls):
        """R2 alone must preserve model output (it cancels through softmax(QK^T)V)."""
        torch.manual_seed(0)
        np.random.seed(0)
        model = _export_decoder_with_ids(decoder_cls())
        token_ids = np.random.randint(0, _VOCAB, (_B, _SEQ)).astype(np.int64)
        y_before = _run_model(model, token_ids)

        apply_spinquant(model, enable_r1=False, enable_r2=True)

        y_after = _run_model(model, token_ids)
        assert np.allclose(y_after, y_before, atol=1e-5)

    @pytest.mark.parametrize("decoder_cls", _R2_DECODER_PARAMS)
    def test_r2_twice_recovers_original_output(self, decoder_cls):
        """Applying R2 twice must recover the original output (R2 @ R2^T = I per head)."""
        torch.manual_seed(0)
        np.random.seed(0)
        model = _export_decoder_with_ids(decoder_cls())
        token_ids = np.random.randint(0, _VOCAB, (_B, _SEQ)).astype(np.int64)
        y_before = _run_model(model, token_ids)

        apply_spinquant(model, enable_r1=False, enable_r2=True)
        apply_spinquant(model, enable_r1=False, enable_r2=True)

        y_after = _run_model(model, token_ids)
        assert np.allclose(y_after, y_before, atol=1e-5)

    @pytest.mark.parametrize("decoder_cls", _R2_DECODER_PARAMS)
    def test_r1_then_r2_preserves_output(self, decoder_cls):
        """R1 + R2 stacked must preserve model output: R1 acts on hidden, R2 on head_dim."""
        torch.manual_seed(0)
        np.random.seed(0)
        model = _export_decoder_with_ids(decoder_cls())
        token_ids = np.random.randint(0, _VOCAB, (_B, _SEQ)).astype(np.int64)
        y_before = _run_model(model, token_ids)

        apply_spinquant(model, enable_r1=True, enable_r2=True)

        y_after = _run_model(model, token_ids)
        assert np.allclose(y_after, y_before, atol=1e-5)

    def test_r2_rejects_fused_qkv_at_validate(self):
        """apply_spinquant with R2 on a Phi3-style model must error before mutating."""
        torch.manual_seed(0)
        np.random.seed(0)
        model = _export_decoder_with_ids(Phi3StyleDecoder())

        with pytest.raises(ValueError, match="R2 rotation"):
            apply_spinquant(model, enable_r1=False, enable_r2=True)


_R3_DECODER_PARAMS = [
    pytest.param(LlamaStyleDecoder, id="llama"),
    pytest.param(lambda: LlamaStyleDecoder(bias=True), id="llama_bias"),
]


def _attach_past_key_concats(model: onnx.ModelProto) -> onnx.ModelProto:
    """Inject ``past_key_<i>`` graph inputs and matching Concats on each block's K path.

    Real HF exports place the past-key Concat where ``head_dim`` is the last
    axis, so R3's MatMul-on-last-axis rotates the right thing. Torch's ONNX
    exporter fuses our test model's two K transposes (``transpose(1,2) ->
    transpose(-1,-2)``) into one Transpose with ``perm=[0,2,3,1]``, leaving no
    place on the K path where head_dim is last and head_idx isn't.

    Workaround: splice the Concat BEFORE the fused Transpose (right after the
    Reshape). At that point the layout is ``[B, S, H, D]`` so head_dim is the
    last axis. We concat on the seq axis (``axis=1``) with past_seq=0.
    """

    def _find_consumer(tensor_name: str, op_type: str) -> Optional[onnx.NodeProto]:
        for n in model.graph.node:
            if n.op_type == op_type and tensor_name in n.input:
                return n
        return None

    block_idx = 0
    while True:
        k_matmul = next(
            (
                n
                for n in model.graph.node
                if n.op_type == "MatMul"
                and n.name.endswith(f"/block{block_idx}/k/MatMul")
            ),
            None,
        )
        if k_matmul is None:
            break

        reshape = _find_consumer(k_matmul.output[0], "Reshape")
        assert reshape is not None, f"block{block_idx} K Reshape not found"

        reshape_out = reshape.output[0]
        new_reshape_out = f"{reshape_out}_pre_pkconcat"
        reshape.output[0] = new_reshape_out

        past_key_name = f"past_key_{block_idx}"
        # Layout at insertion point is [B, S, H, D] (post-Reshape, pre-Transpose).
        past_key = onnx.helper.make_tensor_value_info(
            name=past_key_name,
            elem_type=onnx.TensorProto.FLOAT,
            shape=[_B, 0, _NUM_HEADS, _HEAD_DIM],
        )
        model.graph.input.append(past_key)

        concat_node = onnx.helper.make_node(
            op_type="Concat",
            inputs=[past_key_name, new_reshape_out],
            outputs=[reshape_out],
            name=f"block{block_idx}_past_key_concat",
            axis=1,
        )
        nodes = model.graph.node
        for idx, node in enumerate(nodes):
            if node is reshape:
                nodes.insert(idx + 1, concat_node)
                break
        block_idx += 1
    assert block_idx > 0, "no decoder blocks found in model"
    return model


def _export_decoder_with_pkv(module: nn.Module) -> onnx.ModelProto:
    """Export a decoder and attach ``past_key_<i>`` Concats so R3 has anchors."""
    return _attach_past_key_concats(_export_decoder_with_ids(module))


class TestApplyR3Rotation:
    """Tests for R3RotationPass (online Hadamard on Q/K paths into QK^T)."""

    @pytest.mark.parametrize("decoder_cls", _R3_DECODER_PARAMS)
    def test_r3_alone_preserves_output(self, decoder_cls):
        """R3 alone must preserve model output ((Q@H)(K@H)^T = QK^T cancels in float).

        Purpose: R3 inserts an online Hadamard H on both the Q and K paths feeding
            QK^T. Because (Q@H)(K@H)^T == Q@(H@H^T)@K^T == QK^T, the rotation is a
            mathematical no-op in float — this guards that the graph splice is
            value-preserving and didn't rotate only one side.
        Pass criteria: float output of the R3-rotated model (quantizers removed)
            matches the pre-rotation output within atol=1e-5.
        """
        torch.manual_seed(0)
        np.random.seed(0)
        model = _export_decoder_with_pkv(decoder_cls())
        token_ids = np.random.randint(0, _VOCAB, (_B, _SEQ)).astype(np.int64)
        y_before = _run_model(model, token_ids)

        apply_spinquant(model, enable_r1=False, enable_r3=True)

        y_after = _run_model(model, token_ids)
        # Pass: R3 cancels through QK^T, so output is numerically unchanged.
        assert np.allclose(y_after, y_before, atol=1e-5)

    @pytest.mark.parametrize("decoder_cls", _R3_DECODER_PARAMS)
    def test_r3_inserts_two_matmuls_per_block(self, decoder_cls):
        """R3 must add 4 new MatMul nodes (2 blocks × {Q-side, K-side}).

        Purpose: structural check that R3 actually splices the online Hadamard
            into the graph on both Q and K paths, in every decoder block — not a
            numerical check. Each rotation is a MatMul node tagged with ``_R3``.
        Pass criteria: exactly 4 MatMul nodes whose name contains ``_R3`` exist
            (2 blocks × {Q-side, K-side}).
        """
        torch.manual_seed(0)
        np.random.seed(0)
        model = _export_decoder_with_pkv(decoder_cls())

        apply_spinquant(model, enable_r1=False, enable_r3=True)

        n_r3_matmuls = sum(
            1 for n in model.graph.node if "_R3" in n.name and n.op_type == "MatMul"
        )
        # Pass: one Q-side + one K-side Hadamard MatMul per block, 2 blocks → 4.
        assert n_r3_matmuls == 4

    @pytest.mark.skip_on_windows_arm64("transformers is not available on Windows ARM64")
    @pytest.mark.parametrize("layout", ["mha", "gqa"])
    def test_r3_inserts_two_matmuls_per_block_qwen3(self, layout):
        """R3 via apply_spinquant must add 2*num_layers MatMuls on a real Qwen3 graph.

        Purpose: structural check that R3 actually splices the online Hadamard
            into the graph on both Q and K paths, in every decoder block — not a
            numerical check. Each rotation is a MatMul node tagged with ``_R3``.
            Unlike ``test_r3_inserts_two_matmuls_per_block`` (synthetic spliced
            Concats on a toy decoder), this drives the genuine ``Qwen3DecoderLayer``
            attention graph through the top-level ``apply_spinquant`` API, so the
            full analyze -> validate -> find_r3_anchors -> insert path is exercised
            end-to-end against a real ``past_key_* -> Concat -> QK^T`` topology.
            Parametrized over MHA (kv_heads == attn_heads, clean K-path) and GQA
            (kv_heads < attn_heads, whose ``repeat_kv`` inserts Unsqueeze/Expand
            ops the forward walk must traverse).
        Pass criteria: exactly 2*_QWEN3_NUM_LAYERS MatMul nodes whose name
            contains ``_R3`` exist (per layer × {Q-side, K-side}).
        """
        torch.manual_seed(0)
        np.random.seed(0)
        num_kv_heads = _QWEN3_HEADS if layout == "mha" else _QWEN3_HEADS // 2
        model = _export_qwen3_causal_lm_with_kv_cache(num_kv_heads=num_kv_heads)

        apply_spinquant(model, enable_r1=False, enable_r3=True)

        n_r3_matmuls = sum(
            1 for n in model.graph.node if "_R3" in n.name and n.op_type == "MatMul"
        )
        # Pass: one Q-side + one K-side Hadamard MatMul per layer.
        assert n_r3_matmuls == 2 * _QWEN3_NUM_LAYERS

    @pytest.mark.skip_on_windows_arm64("transformers is not available on Windows ARM64")
    @pytest.mark.parametrize("layout", ["mha", "gqa"])
    def test_r3_preserves_output_qwen3(self, layout):
        """R3 must preserve the float logits of a real Qwen3 LM, MHA and GQA alike.

        Purpose: numerical counterpart to ``test_r3_inserts_two_matmuls_per_block_qwen3``.
            R3 inserts an online Hadamard H on both the Q and K paths into QK^T;
            because ``(Q@H)(K@H)^T == Q@(H@H^T)@K^T == QK^T`` the rotation cancels
            in float, so the model's logits must be unchanged. This guards that
            the splice is value-preserving on a genuine ``Qwen3DecoderLayer``
            graph — and, for GQA, that inserting R3 *upstream* of ``repeat_kv``
            (so each broadcast K head carries the same rotation as its Q group)
            still cancels.

            The comparison mirrors R3's cache convention. R3 rotates the
            *current* K before the past-key Concat, so in production the cache
            stores rotated K and ``present_key`` (rotated) is fed back as
            ``past_key`` next step. The two models therefore use *different* cache
            conventions: the original consumes un-rotated past K, the R3 model
            consumes ``past_key @ H``. Fed each in its own convention they compute
            identical attention scores. The fed values are random (not zero) so
            QK^T is non-trivial and a one-sided rotation would be caught;
            ``past_value`` carries no R3 and is identical for both.
        Pass criteria: float logits of the R3-rotated model (quantizers removed),
            fed rotated past K, match the original model's logits, fed un-rotated
            past K, within atol=1e-4.
        """
        torch.manual_seed(0)
        np.random.seed(0)
        num_kv_heads = _QWEN3_HEADS if layout == "mha" else _QWEN3_HEADS // 2
        model = _export_qwen3_causal_lm_with_kv_cache(num_kv_heads=num_kv_heads)
        token_ids = np.random.randint(0, _VOCAB, (_B, _SEQ)).astype(np.int64)

        # H rotates the per-head head_dim axis. The original model gets un-rotated
        # past K; the R3 model gets past K @ H (the rotated-cache convention R3
        # produces). past_value is shared and unrotated.
        H = hadamard_rotation_matrix(_QWEN3_HEAD_DIM)
        feeds_orig = {"input_ids": token_ids}
        feeds_r3 = {"input_ids": token_ids}
        for i in range(_QWEN3_NUM_LAYERS):
            past_key = np.random.randn(
                _B, num_kv_heads, _QWEN3_PAST_SEQ, _QWEN3_HEAD_DIM
            ).astype(np.float32)
            past_value = np.random.randn(
                _B, num_kv_heads, _QWEN3_PAST_SEQ, _QWEN3_HEAD_DIM
            ).astype(np.float32)
            feeds_orig[f"past_key_{i}"] = past_key
            feeds_r3[f"past_key_{i}"] = (past_key @ H).astype(np.float32)
            feeds_orig[f"past_value_{i}"] = past_value
            feeds_r3[f"past_value_{i}"] = past_value
        feeds_orig = _pad_dummy_input(model, **feeds_orig)
        feeds_r3 = _pad_dummy_input(model, **feeds_r3)

        y_before = _build_session(model).run(None, feeds_orig)[0]

        apply_spinquant(model, enable_r1=False, enable_r3=True)

        y_after = _build_session(model).run(None, feeds_r3)[0]
        # Pass: R3 cancels through QK^T, so logits are numerically unchanged.
        assert np.allclose(y_after, y_before, atol=1e-4)

    def test_r3_rejects_missing_past_key_inputs(self):
        """A model without past_key_* inputs must error in R3 validate.

        Purpose: R3 pins its K-side anchor to the ``past_key_* -> Concat`` site in
            the attention graph. A model exported without a KV cache has no such
            inputs, so there is nothing to anchor on — R3 must reject it up front
            (during validation) rather than silently producing a wrong rotation.
        Pass criteria: ``apply_spinquant`` raises ``ValueError`` whose message
            matches ``"R3 rotation"``.
        """
        torch.manual_seed(0)
        np.random.seed(0)
        model = _export_decoder_with_ids(LlamaStyleDecoder())  # no past_key inputs

        # Pass: validation rejects the anchor-less model with an R3-specific error.
        with pytest.raises(ValueError, match="R3 rotation"):
            apply_spinquant(model, enable_r1=False, enable_r3=True)

    @pytest.mark.parametrize("decoder_cls", _R3_DECODER_PARAMS)
    def test_r1_r2_r3_stacked_preserves_output(self, decoder_cls):
        """R1 + R2 + R3 stacked must preserve model output.

        Purpose: the three rotations act on independent axes (R1 on the residual
            hidden dim, R2 on the per-head V/o_proj dim, R3 on the Q/K head_dim
            into QK^T). Applying all three at once must still be a float no-op —
            this guards that the passes compose without interfering.
        Pass criteria: float output of the fully-rotated model (quantizers
            removed) matches the pre-rotation output within atol=1e-5.
        """
        torch.manual_seed(0)
        np.random.seed(0)
        model = _export_decoder_with_pkv(decoder_cls())
        token_ids = np.random.randint(0, _VOCAB, (_B, _SEQ)).astype(np.int64)
        y_before = _run_model(model, token_ids)

        apply_spinquant(model, enable_r1=True, enable_r2=True, enable_r3=True)

        y_after = _run_model(model, token_ids)
        # Pass: stacked rotations on independent axes leave the output unchanged.
        assert np.allclose(y_after, y_before, atol=1e-5)

    def test_r3_anchors_pinned_to_past_key_concats(self):
        """K-side anchor must be the past-key Concat for every block.

        Purpose: directly exercise ``find_r3_anchors`` (rather than the full
            ``apply_spinquant`` path) to confirm it locates, per block, the
            ``past_key_<i>`` graph input, the Concat that splices the KV cache
            onto the K path, and the downstream QK^T MatMul. Uses the synthetic
            spliced Concats from ``_export_decoder_with_pkv``.
        Pass criteria: one anchor per block, and for block ``i`` the anchor's
            ``past_key_input_name == "past_key_{i}"``, its ``k_concat_node`` is a
            Concat, and its ``qk_matmul_node`` is a MatMul.
        """
        from aimet_onnx.experimental.spinquant.model_analysis import (
            find_r3_anchors,
            get_decoder_block_boundaries,
            get_decoder_role_map,
        )

        torch.manual_seed(0)
        np.random.seed(0)
        model = _export_decoder_with_pkv(LlamaStyleDecoder())

        cg = ConnectedGraph(model)
        blocks, active_norms = get_decoder_block_boundaries(model, cg)
        role_map = get_decoder_role_map(cg, blocks, active_norms)
        anchors = find_r3_anchors(role_map, model)

        # Pass: exactly one anchor per block, each pinned to that block's
        # past_key input / Concat / QK^T MatMul.
        assert len(anchors) == len(role_map.blocks)
        for i, anchor in enumerate(anchors):
            assert anchor.past_key_input_name == f"past_key_{i}"
            assert len(anchor.k_consumers) == 1
            assert anchor.k_consumers[0].op_type == "Concat"
            assert len(anchor.qk_matmul_nodes) == 1
            assert anchor.qk_matmul_nodes[0].op_type == "MatMul"


# Tiny Qwen3 shape used by the find_r3_anchors tests below. These are NOT the
# real Qwen3-0.6B dims — they're the smallest config that still exports the
# genuine ``past_key_* -> Concat -> QK^T`` attention topology cheaply.
_QWEN3_HIDDEN = 32
_QWEN3_HEADS = 4
_QWEN3_HEAD_DIM = 8
_QWEN3_INTERMEDIATE = 64
_QWEN3_NUM_LAYERS = 2
_QWEN3_PAST_SEQ = 3


def _export_qwen3_layers_with_kv_cache(num_kv_heads: int) -> onnx.ModelProto:
    """Export real ``Qwen3DecoderLayer`` stack with a KV cache to ONNX.

    Instantiates ``_QWEN3_NUM_LAYERS`` genuine ``Qwen3DecoderLayer`` modules
    (random weights — no checkpoint download) and traces them with a
    ``DynamicCache`` primed by ``past_key_*`` / ``past_value_*`` inputs. Tracing
    through ``Cache.update`` is what makes torch emit the real
    ``past_key_<i> -> Concat -> QK^T MatMul`` topology that ``find_r3_anchors``
    pins on, instead of the synthetic Concats other R3 tests splice in.

    ``num_kv_heads`` toggles attention layout: equal to ``_QWEN3_HEADS`` gives
    MHA (``repeat_kv`` is a no-op, clean K-path); smaller gives GQA, whose
    ``repeat_kv`` expansion inserts Unsqueeze/Expand/Reshape ops on the K-path.

    transformers is imported lazily so this module still collects where it is
    unavailable (e.g. Windows ARM64).
    """
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3Config,
        Qwen3DecoderLayer,
        Qwen3RotaryEmbedding,
    )
    from transformers.cache_utils import DynamicCache

    cfg = Qwen3Config(
        hidden_size=_QWEN3_HIDDEN,
        num_attention_heads=_QWEN3_HEADS,
        num_key_value_heads=num_kv_heads,
        head_dim=_QWEN3_HEAD_DIM,
        intermediate_size=_QWEN3_INTERMEDIATE,
        num_hidden_layers=_QWEN3_NUM_LAYERS,
        vocab_size=64,
        max_position_embeddings=64,
        _attn_implementation="eager",
    )
    layers = nn.ModuleList(
        [Qwen3DecoderLayer(cfg, layer_idx=i).eval() for i in range(_QWEN3_NUM_LAYERS)]
    )
    rotary = Qwen3RotaryEmbedding(cfg)

    seq, past = _SEQ, _QWEN3_PAST_SEQ
    hidden = torch.randn(_B, seq, _QWEN3_HIDDEN)
    position_ids = torch.arange(past, past + seq).unsqueeze(0)
    cos, sin = rotary(hidden, position_ids)
    cache_position = torch.arange(past, past + seq)
    past_kv = [
        torch.randn(_B, num_kv_heads, past, _QWEN3_HEAD_DIM)
        for _ in range(2 * _QWEN3_NUM_LAYERS)
    ]

    class _Wrapper(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = layers

        def forward(self, hidden, cos, sin, *past_kv):
            cache = DynamicCache()
            for i in range(_QWEN3_NUM_LAYERS):
                cache.update(past_kv[2 * i], past_kv[2 * i + 1], i)
            x = hidden
            for layer in self.layers:
                x = layer(
                    x,
                    position_embeddings=(cos, sin),
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=cache_position,
                )
            return x

    input_names = ["hidden", "cos", "sin"]
    for i in range(_QWEN3_NUM_LAYERS):
        input_names += [f"past_key_{i}", f"past_value_{i}"]

    buf = io.BytesIO()
    torch.onnx.export(
        _Wrapper(layers).eval(),
        (hidden, cos, sin, *past_kv),
        buf,
        input_names=input_names,
        output_names=["out"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    buf.seek(0)
    return load_model(buf)


def _cache_keys(cache, layer_idx):
    """Read layer ``layer_idx``'s updated keys from a DynamicCache, version-robustly.

    transformers >= 5 stores per-layer tensors under ``cache.layers[i].keys``;
    older versions use the ``cache.key_cache[i]`` list.
    """
    if hasattr(cache, "layers"):
        return cache.layers[layer_idx].keys
    return cache.key_cache[layer_idx]


def _cache_values(cache, layer_idx):
    """Read layer ``layer_idx``'s updated values from a DynamicCache (see ``_cache_keys``)."""
    if hasattr(cache, "layers"):
        return cache.layers[layer_idx].values
    return cache.value_cache[layer_idx]


def _export_qwen3_causal_lm_with_kv_cache(
    num_kv_heads: int = _QWEN3_HEADS,
) -> onnx.ModelProto:
    """Export a full real-Qwen3 causal LM (embed -> layers -> norm -> lm_head) with a KV cache.

    ``num_kv_heads`` toggles attention layout: equal to ``_QWEN3_HEADS`` (the
    default) gives MHA, where ``repeat_kv`` is a no-op and the K-path is clean;
    a smaller value gives GQA, whose ``repeat_kv`` expansion inserts
    Unsqueeze/Expand/Reshape ops on the K-path between the past-key Concat and
    QK^T.

    Wraps a genuine ``Qwen3DecoderLayer`` stack with the prologue/epilogue that
    ``apply_spinquant`` needs to run end-to-end: a ``Gather`` embed_tokens, a
    final ``RMSNorm``, and an ``lm_head`` MatMul. With those present,
    ``get_decoder_block_boundaries`` / ``get_decoder_role_map`` detect 2 blocks,
    and the export still emits the genuine ``past_key_<i> -> Concat -> QK^T``
    topology ``find_r3_anchors`` pins on — so the top-level ``apply_spinquant``
    R3 path can be exercised against a real attention graph rather than the
    synthetic Concats spliced in by ``_export_decoder_with_pkv``.

    Two details make block detection work on this tiny export:

    * cos/sin are computed *inside* the graph from the embedded tokens (as a
      real causal-LM forward does). Feeding ``position_ids`` and letting the
      first layer build the K cache from a graph-input rotary instead degenerates
      layer 0's past-key Concat into a single-input Concat that R3 can't anchor.
    * Every RMSNorm gamma is randomized. Qwen3 initializes all gammas to ones,
      so the export/folding dedupes the identical initializers and only the first
      norm keeps a distinct (detectable) scale — randomizing keeps all of them
      distinct so ``find_active_norms`` sees every block's norms.

    transformers is imported lazily so this module still collects where it is
    unavailable (e.g. Windows ARM64).
    """
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3Config,
        Qwen3DecoderLayer,
        Qwen3RMSNorm,
        Qwen3RotaryEmbedding,
    )
    from transformers.cache_utils import DynamicCache

    cfg = Qwen3Config(
        hidden_size=_QWEN3_HIDDEN,
        num_attention_heads=_QWEN3_HEADS,
        num_key_value_heads=num_kv_heads,
        head_dim=_QWEN3_HEAD_DIM,
        intermediate_size=_QWEN3_INTERMEDIATE,
        num_hidden_layers=_QWEN3_NUM_LAYERS,
        vocab_size=_VOCAB,
        max_position_embeddings=64,
        _attn_implementation="eager",
    )

    class _CausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(_VOCAB, _QWEN3_HIDDEN)
            self.layers = nn.ModuleList(
                [
                    Qwen3DecoderLayer(cfg, layer_idx=i).eval()
                    for i in range(_QWEN3_NUM_LAYERS)
                ]
            )
            self.norm = Qwen3RMSNorm(_QWEN3_HIDDEN)
            self.lm_head = nn.Linear(_QWEN3_HIDDEN, _VOCAB, bias=False)
            self.rotary = Qwen3RotaryEmbedding(cfg)
            # Randomize every RMSNorm gamma so the (otherwise all-ones, hence
            # deduped) scale initializers stay distinct and detectable.
            for module in (*self.layers, self.norm):
                for name, param in module.named_parameters():
                    if "norm" in name.lower():
                        with torch.no_grad():
                            param.copy_(torch.randn_like(param))

        def forward(self, token_ids, *past_kv):
            seq, past = _SEQ, _QWEN3_PAST_SEQ
            position_ids = torch.arange(past, past + seq).unsqueeze(0)
            cache_position = torch.arange(past, past + seq)
            x = self.embed_tokens(token_ids)
            cos, sin = self.rotary(x, position_ids)
            cache = DynamicCache()
            for i in range(_QWEN3_NUM_LAYERS):
                cache.update(past_kv[2 * i], past_kv[2 * i + 1], i)
            for layer in self.layers:
                x = layer(
                    x,
                    position_embeddings=(cos, sin),
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=cache_position,
                )
            logits = self.lm_head(self.norm(x))
            # Surface the updated ("present") KV so the exporter emits past-KV
            # *outputs* to match the past-KV inputs. The cache is mutated in
            # place inside each attention block; unless its tensors are part of
            # the returned values, torch.onnx.export traces only logits and the
            # present_key_*/present_value_* Concats stay buried as intermediates.
            present = []
            for i in range(_QWEN3_NUM_LAYERS):
                present += [_cache_keys(cache, i), _cache_values(cache, i)]
            return (logits, *present)

    token_ids = torch.randint(0, _VOCAB, (_B, _SEQ))
    past_kv = [
        torch.randn(_B, num_kv_heads, _QWEN3_PAST_SEQ, _QWEN3_HEAD_DIM)
        for _ in range(2 * _QWEN3_NUM_LAYERS)
    ]

    input_names = ["input_ids"]
    output_names = ["logits"]
    for i in range(_QWEN3_NUM_LAYERS):
        input_names += [f"past_key_{i}", f"past_value_{i}"]
        output_names += [f"present_key_{i}", f"present_value_{i}"]

    buf = io.BytesIO()
    torch.onnx.export(
        _CausalLM().eval(),
        (token_ids, *past_kv),
        buf,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    buf.seek(0)
    return load_model(buf)


class TestFindR3Anchors:
    """Unit tests for find_r3_anchors against a real Qwen3 attention graph.

    find_r3_anchors only reads ``role_map.blocks`` (for the count) and
    ``role_map.past_key_input_names`` (collected at role-map build time), so
    these tests pass a lightweight stub with the right block count and past_key
    names and exercise the anchor-finding logic purely against the exported
    ONNX graph.
    """

    @pytest.mark.skip_on_windows_arm64("transformers is not available on Windows ARM64")
    def test_find_r3_anchors_mha(self):
        """MHA Qwen3 (kv_heads == attn_heads): one anchor per block on past_key_*.

        Purpose: validate ``find_r3_anchors`` against a genuine ``Qwen3DecoderLayer``
            attention graph (traced through a DynamicCache) in MHA layout, where
            ``repeat_kv`` is a no-op and the K-path from past_key to QK^T is clean.
            This is the realistic topology the synthetic-Concat tests approximate.
        Pass criteria: one anchor per layer; for layer ``i`` the anchor pins
            ``past_key_{i}`` with a Concat ``k_concat_node`` and a MatMul
            ``qk_matmul_node``, and the Q-side and K-side feed *different* inputs
            of that MatMul (``q_consumer_input_idx != k_consumer_input_idx``).
        """
        import types

        torch.manual_seed(0)
        model = _export_qwen3_layers_with_kv_cache(num_kv_heads=_QWEN3_HEADS)

        role_map = types.SimpleNamespace(
            blocks=[None] * _QWEN3_NUM_LAYERS,
            past_key_input_names=[
                inp.name for inp in model.graph.input if "past_key" in inp.name
            ],
        )
        anchors = find_r3_anchors(role_map, model)

        # Pass: anchors found for every layer, correctly identifying past_key
        # input, Concat, QK^T MatMul, and distinct Q/K operand positions.
        assert len(anchors) == _QWEN3_NUM_LAYERS
        for i, anchor in enumerate(anchors):
            assert anchor.past_key_input_name == f"past_key_{i}"
            assert len(anchor.k_consumers) == 1
            assert anchor.k_consumers[0].op_type == "Concat"
            assert len(anchor.qk_matmul_nodes) == 1
            assert anchor.qk_matmul_nodes[0].op_type == "MatMul"
            # Q-side and K-side must be distinct inputs of the QK^T MatMul.
            assert anchor.q_input_tensors[0] != anchor.k_consumers[0].output[0]

    @pytest.mark.skip_on_windows_arm64("transformers is not available on Windows ARM64")
    def test_find_r3_anchors_gqa(self):
        """GQA Qwen3 (kv_heads < attn_heads): one anchor per block through repeat_kv.

        Purpose: validate ``find_r3_anchors`` against a GQA ``Qwen3DecoderLayer``
            attention graph. When kv_heads < attn_heads, ``repeat_kv`` inserts a
            clean ``Unsqueeze -> Expand -> Reshape`` chain between the past_key
            Concat and QK^T to broadcast each KV head across its query-head group.
            The forward walk now traverses those ops, so anchor finding resolves
            the same per-block ``past_key_* -> Concat -> QK^T`` topology as MHA.
        Pass criteria: one anchor per layer; for layer ``i`` the anchor pins
            ``past_key_{i}`` with a Concat ``k_concat_node`` and a MatMul
            ``qk_matmul_node``, and the Q-side and K-side feed *different* inputs
            of that MatMul (``q_consumer_input_idx != k_consumer_input_idx``).
        """
        import types

        torch.manual_seed(0)
        model = _export_qwen3_layers_with_kv_cache(num_kv_heads=_QWEN3_HEADS // 2)

        role_map = types.SimpleNamespace(
            blocks=[None] * _QWEN3_NUM_LAYERS,
            past_key_input_names=[
                inp.name for inp in model.graph.input if "past_key" in inp.name
            ],
        )
        anchors = find_r3_anchors(role_map, model)

        assert len(anchors) == _QWEN3_NUM_LAYERS
        for i, anchor in enumerate(anchors):
            assert anchor.past_key_input_name == f"past_key_{i}"
            assert len(anchor.k_consumers) == 1
            assert anchor.k_consumers[0].op_type == "Concat"
            assert len(anchor.qk_matmul_nodes) == 1
            assert anchor.qk_matmul_nodes[0].op_type == "MatMul"
            # Q-side and K-side must be distinct inputs of the QK^T MatMul.
            assert anchor.q_input_tensors[0] != anchor.k_consumers[0].output[0]


class TestApplySpinquant:
    """End-to-end tests for apply_spinquant"""

    @pytest.mark.parametrize(
        "decoder_cls",
        [LlamaStyleDecoder, Qwen3StyleDecoder, Phi3StyleDecoder],
    )
    def test_float_output_preserved(self, decoder_cls):
        """apply_spinquant must preserve model output numrically."""
        torch.manual_seed(0)
        np.random.seed(0)
        model = _export_decoder_with_ids(decoder_cls())
        token_ids = np.random.randint(0, _VOCAB, (_B, _SEQ)).astype(np.int64)

        y_before = _run_model(model, token_ids)

        """
        When: apply_spinquant correctly applied R1 rotation
        Then: The rotated model is mathematically equivalent to original FP32 model.
        """

        apply_spinquant(model)

        y_after = _run_model(model, token_ids)
        assert np.allclose(y_before, y_after, atol=1e-5)

    @pytest.mark.parametrize(
        "decoder_cls",
        [LlamaStyleDecoder, Qwen3StyleDecoder, Phi3StyleDecoder],
    )
    def test_weights_changed(self, decoder_cls):
        """apply_spinquant must modify weight initializers in the model."""
        torch.manual_seed(0)
        np.random.seed(0)
        model = _export_decoder_with_ids(decoder_cls())

        # Find the embed_tokens initializer (Gather weight: shape [_VOCAB, _H]).
        embed_name = next(
            init.name
            for init in model.graph.initializer
            if numpy_helper.to_array(init).shape == (_VOCAB, _H)
        )
        w_before = numpy_helper.to_array(
            ParamUtils.get_param_by_name(model, embed_name)
        ).copy()

        apply_spinquant(model)

        w_after = numpy_helper.to_array(ParamUtils.get_param_by_name(model, embed_name))
        assert not np.array_equal(w_before, w_after)

    def test_validation_failure_leaves_model_untouched(self):
        """Model shouldn't be corrupted if the validation fails"""
        torch.manual_seed(0)
        np.random.seed(0)

        class _OddNormDecoder(nn.Module):
            """1 block + embed_tokens + lm_head, no final norm → 2 active norms.
            (2-1) % 2 = 1, so block detection raises ValueError."""

            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(_VOCAB, _H)
                self.block0 = _LlamaBlock()
                self.lm_head = nn.Linear(_H, _VOCAB, bias=False)

            def forward(self, token_ids):
                return self.lm_head(self.block0(self.embed_tokens(token_ids)))

        model = _export_decoder_with_ids(_OddNormDecoder())

        # Snapshot all initializers before the failed call.
        init_before = {
            t.name: numpy_helper.to_array(t).copy() for t in model.graph.initializer
        }

        """
        When: Block detection/classification raises an error.
        Then: The model is unchanged.
        """

        with pytest.raises(ValueError):
            apply_spinquant(model)

        # Every initializer must be bit-exact same.
        for name, arr_before in init_before.items():
            arr_after = numpy_helper.to_array(ParamUtils.get_param_by_name(model, name))
            assert np.array_equal(arr_before, arr_after)

    def test_embedding_weight_rotated(self):
        """External embedding.pth weight (raw tensor) must be rotated by R_L after apply_spinquant."""
        torch.manual_seed(0)

        backbone_model = _export_vlm_backbone(VLMBackbone())

        # Simulate torch.load("embedding.pth") → raw tensor [vocab, hidden]
        embedding = torch.randn(_VOCAB, _H)
        W_before = embedding.clone()

        """
        When: apply_spinquant is called with an external embedding tensor.
        Then: tensor is modified in-place (rotated by R_L).
        """
        apply_spinquant(backbone_model, embedding=embedding)

        assert not torch.equal(embedding, W_before)

    def test_embedding_rotation_consistent_with_backbone(self):
        """Backbone(embedding_rotated[i]) == backbone_rotated(embedding_orig[i])."""
        torch.manual_seed(0)
        np.random.seed(0)

        backbone_model = _export_vlm_backbone(VLMBackbone())

        embedding = torch.randn(_VOCAB, _H)
        W_orig = embedding.clone().numpy()  # [vocab, H]

        # Baseline: original backbone on original embedding rows
        y_before = _run_model(backbone_model, W_orig[:_SEQ].reshape(1, _SEQ, _H))

        apply_spinquant(backbone_model, embedding=embedding)

        W_rot = embedding.numpy()  # rotated in-place by R_L

        """
        When: backbone_rotated receives embedding_rotated rows as inputs_embeds.
        Then: output matches baseline (original backbone on original embedding).
        """
        y_after = _run_model(backbone_model, W_rot[:_SEQ].reshape(1, _SEQ, _H))
        assert np.allclose(y_before, y_after, atol=1e-5)

    def test_visual_output_rotated_by_r_l(self):
        """Visual encoder output must equal y_before @ R_L after apply_spinquant."""
        torch.manual_seed(0)
        np.random.seed(0)

        backbone_model = _export_vlm_backbone(VLMBackbone())
        visual_model = _export_vit(ViTEncoder())

        x_vit = np.random.randn(_VIT_N, _VIT_D).astype(np.float32)

        y_vit_before = _run_model(visual_model, x_vit)

        """
        When: apply_spinquant is called with backbone and visual models.
        Then: visual encoder output equals y_before @ R_L.
        """
        embedding = torch.randn(_VOCAB, _H)
        apply_spinquant(backbone_model, visual_model=visual_model, embedding=embedding)

        R_L = (get_hadamard_matrix(_VIT_D_L) / np.sqrt(_VIT_D_L)).astype(np.float32)

        y_vit_after = _run_model(visual_model, x_vit)
        assert np.allclose(y_vit_after, y_vit_before @ R_L)

    @pytest.mark.skip_on_windows_arm64("transformers is not available on Windows ARM64")
    @pytest.mark.parametrize(
        "with_lm_head",
        [True, False],
    )
    @pytest.mark.parametrize("hidden_size", [1024, 1536])
    def test_r1_preserves_output_for_headless_model(self, with_lm_head, hidden_size):
        """
        R1 must preserve the output of a real 4-layer Qwen3 export with or without lm_head.
        """
        torch.manual_seed(0)
        np.random.seed(0)
        model = qwen3_causal_lm(
            num_hidden_layers=4,
            with_lm_head=with_lm_head,
            vocab_size=_VOCAB,
            hidden_size=hidden_size,
        )
        output_name = model.graph.output[0].name

        dummy_input = _pad_dummy_input(model)

        y_before = _build_session(model).run([output_name], dummy_input)[0]

        apply_spinquant(model, enable_r1=True)

        y_after = _build_session(model).run([output_name], dummy_input)[0]
        assert np.allclose(y_before, y_after, atol=1e-4)

    @pytest.mark.parametrize("dynamo", [True, False])
    def test_spinquant_r2_sha_gqa(self, dynamo):
        """R2 on a SHA + GQA decoder must preserve output (rotated-cache)"""
        torch.manual_seed(0)
        np.random.seed(0)
        head_dim, num_kv, group, num_layers = 8, 2, 2, 2
        seq, past = 4, 3
        model = transformer_blocks.sha_gqa_decoder(
            head_dim=head_dim,
            num_kv=num_kv,
            group=group,
            vocab=_VOCAB,
            num_layers=num_layers,
            seq=seq,
            past=past,
            dynamo=dynamo,
        )
        if dynamo:
            fix_node_names_pass(model)

        # Make dummy input
        input_dict = make_dummy_input(model)
        input_dict["input_ids"] = np.random.randint(0, _VOCAB, (1, seq)).astype(
            np.int64
        )
        input_dict["attention_mask"] = np.zeros_like(input_dict["attention_mask"])

        y_before = _build_session(model).run(None, input_dict)

        apply_spinquant(model, enable_r2=True)
        # After R2, expects V-Cache inputs to be rotated
        # Rotate value cache from dummy input
        H = hadamard_rotation_matrix(head_dim).astype(np.float32)

        def _rotate_cache(tensor):
            return (tensor @ H).astype(np.float32)

        input_dict_r2 = dict(input_dict)
        for name, tensor in input_dict.items():
            if name.startswith("past_value_"):
                input_dict_r2[name] = _rotate_cache(tensor)

        y_after = _build_session(model).run(None, input_dict_r2)
        # Logits match
        assert np.allclose(y_before[0], y_after[0], atol=1e-4, rtol=1e-4)
        # Key cache matches
        for key_before, key_after in zip(y_before[1::2], y_after[1::2]):
            assert np.allclose(key_before, key_after, atol=1e-4, rtol=1e-4)
        # Value cache is rotated
        for value_before, value_after in zip(y_before[2::2], y_after[2::2]):
            assert np.allclose(
                _rotate_cache(value_before), value_after, atol=1e-4, rtol=1e-4
            )

    @pytest.mark.parametrize("rescale_key_tensor", [True, False])
    def test_spinquant_r3_sha_gqa(self, rescale_key_tensor):
        """R3 on a SHA + GQA decoder must preserve output (rotated-cache)"""
        torch.manual_seed(0)
        np.random.seed(0)
        head_dim, num_kv, group, num_layers = 8, 2, 2, 2
        seq, past = 4, 3
        model = transformer_blocks.sha_gqa_decoder(
            head_dim=head_dim,
            num_kv=num_kv,
            group=group,
            vocab=_VOCAB,
            num_layers=num_layers,
            seq=seq,
            past=past,
            dynamo=True,
            rescale_key_tensor=rescale_key_tensor,
        )

        # Make dummy input
        input_dict = make_dummy_input(model)
        input_dict["input_ids"] = np.random.randint(0, _VOCAB, (1, seq)).astype(
            np.int64
        )
        input_dict["attention_mask"] = np.zeros_like(input_dict["attention_mask"])

        y_before = _build_session(model).run(None, input_dict)

        apply_spinquant(model, enable_r3=True)

        r3_matmuls = [
            n for n in model.graph.node if "_R3" in n.name and n.op_type == "MatMul"
        ]
        for n in r3_matmuls:
            assert is_online_rotation_op(SimpleNamespace(type=n.op_type, name=n.name))
        assert len(r3_matmuls) == num_layers * (num_kv + group * num_kv)
        # no name collisions
        assert len({n.name for n in r3_matmuls}) == len(r3_matmuls)

        # After R3, expects K-Cache inputs to be rotated
        # Rotate key cache from dummy input
        H = hadamard_rotation_matrix(head_dim).astype(np.float32)

        def _rotate_cache(tensor):
            return np.swapaxes(np.swapaxes(tensor, -1, -2) @ H, -1, -2).astype(
                np.float32
            )

        input_dict_r3 = dict(input_dict)
        for name, tensor in input_dict.items():
            if name.startswith("past_key_"):
                input_dict_r3[name] = _rotate_cache(tensor)

        y_after = _build_session(model).run(None, input_dict_r3)
        # Logits match
        assert np.allclose(y_before[0], y_after[0], atol=1e-4, rtol=1e-4)
        # Key cache is rotated
        for key_before, key_after in zip(y_before[1::2], y_after[1::2]):
            assert np.allclose(
                _rotate_cache(key_before), key_after, atol=1e-4, rtol=1e-4
            )
        # Value cache matches
        for value_before, value_after in zip(y_before[2::2], y_after[2::2]):
            assert np.allclose(value_before, value_after, atol=1e-4, rtol=1e-4)

    def test_post_writing_norm_raises_and_leaves_model_untouched(self):
        """apply_spinquant must raise ValueError for Gemma-style architectures (post-writing norms)
        and must not modify any initializers before raising."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(Gemma3StyleDecoder())

        init_before = {
            t.name: numpy_helper.to_array(t).copy() for t in model.graph.initializer
        }

        """
        When: apply_spinquant is applied for Gemma3/Gemma4 style architectures.
        Then: Must raise ValueError and initializers are not modified.
        """

        with pytest.raises(ValueError):
            apply_spinquant(model)

        for name, arr_before in init_before.items():
            arr_after = numpy_helper.to_array(ParamUtils.get_param_by_name(model, name))
            assert np.array_equal(arr_before, arr_after)

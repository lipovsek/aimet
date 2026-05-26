# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import io
import logging
import shutil
import numpy as np
import scipy.linalg
import pytest
import torch
import torch.nn as nn
import onnx
from onnx import load_model, numpy_helper
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

from .models.test_models import RMSNorm
from .utils import add_genai_tests_path
from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.common.hadamard import get_hadamard_matrix
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.utils import ParamUtils

from aimet_onnx.experimental.spinquant.model_analysis import (
    ActiveNorm,
    DecoderBlockRoleMap,
    DecoderModelRoleMap,
    find_active_norms,
    find_merger_linear2,
    get_decoder_block_boundaries,
    get_decoder_role_map,
    get_bias_product as _get_bias_product,
    get_weight_product as _get_weight_product,
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


def _run_model(model: onnx.ModelProto, inp: np.ndarray) -> np.ndarray:
    session = _build_session(model)
    return session.run(None, {"input": inp})[0]


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
_VOCAB = 16
_B, _SEQ = 1, 4  # batch, sequence length

_VIT_D, _VIT_I = 8, 12  # ViT hidden size, intermediate dim
_VIT_S_SQ = 4
_VIT_N = 8
_VIT_D_L = _H  # language hidden size; must equal backbone _H for VLM integration tests


def _export_vit(module: nn.Module) -> onnx.ModelProto:
    x = torch.randn(_VIT_N, _VIT_D)
    return _export_to_onnx(module, x)


def _export_decoder(module: nn.Module) -> onnx.ModelProto:
    x = torch.randn(_B, _SEQ, _H)
    return _export_to_onnx(module, x)


def _export_decoder_with_ids(module: nn.Module) -> onnx.ModelProto:
    token_ids = torch.randint(0, _VOCAB, (_B, _SEQ))
    return _export_to_onnx(module, token_ids)


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
        attn = self.o(self.q(h) + self.k(h) + self.v(h))
        x = x + attn
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
        attn = self.o(q + k + v)
        x = x + attn
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

            collection = model_cls.instantiate_quantsim(
                model_id, 32, 16, small_model=True
            )
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
    """Tests for get_decoder_role_map."""

    def test_llama_role_map_structure(self):
        """LlamaStyleDecoder: verify per-block and model-level role counts.

        2 blocks × (3 qkv, 1 o_proj, 2 gate_up, 1 down_proj) + 1 lm_head + 1 embed_tokens.
        """
        torch.manual_seed(0)
        model = _export_decoder_with_ids(LlamaStyleDecoder())
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

    def test_qwen3_qkv_count(self):
        """Qwen3 q_norm/k_norm are internal; qkv_linears still has 3 (q_proj, k_proj, v)."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(Qwen3StyleDecoder())
        cg = ConnectedGraph(model)
        blocks, active_norms = get_decoder_block_boundaries(model, cg)
        role_map = get_decoder_role_map(cg, blocks, active_norms)
        for block in role_map.blocks:
            assert len(block.qkv_linears) == 3

    def test_missing_embed_tokens_warns(self):
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
        cg = ConnectedGraph(model)
        blocks, active_norms = get_decoder_block_boundaries(model, cg)
        # Must not raise — VLM backbones exported with use_inputs_embeds=True have no Gather.
        role_map = get_decoder_role_map(cg, blocks, active_norms)
        assert role_map.embed_tokens == []

    def test_wrong_active_norms_per_block_raises(self):
        """Passing active_norms_per_block inconsistent with detected norms raises ValueError."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(LlamaStyleDecoder())
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
            collection = model_cls.instantiate_quantsim(
                model_id, 32, 16, small_model=True
            )
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

    def test_find_merger_linear2_rmsorm_vit(self):
        """ViTEncoder (RMSNorm blocks + PatchMerger): find_merger_linear2 returns linear_fc2."""
        torch.manual_seed(0)
        model = _export_vit(ViTEncoder())
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
        assert np.allclose(y_after, y_before)

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

        # Capture float output BEFORE sim creation
        y_before = _run_model(model, token_ids)

        """
        When: apply_spinquant correctly applied R1 rotation
        Then: The rotated model is mathematically equivalent to original FP32 model.
        """

        sim = QuantizationSimModel(model, dummy_input={"input": token_ids})
        apply_spinquant(sim)

        # Strip QcQuantizeOp to get the rotated float model for comparison.
        rotated_float = QuantizationSimModel.remove_quantizers(
            copy.deepcopy(sim.model.model)
        )
        y_after = _run_model(rotated_float, token_ids)
        assert np.allclose(y_before, y_after)

    @pytest.mark.parametrize(
        "decoder_cls",
        [LlamaStyleDecoder, Qwen3StyleDecoder, Phi3StyleDecoder],
    )
    def test_weights_changed(self, decoder_cls):
        """apply_spinquant must modify weight initializers in sim.model.model."""
        torch.manual_seed(0)
        np.random.seed(0)
        model = _export_decoder_with_ids(decoder_cls())
        token_ids = np.random.randint(0, _VOCAB, (_B, _SEQ)).astype(np.int64)

        # Find the embed_tokens initializer (Gather weight: shape [_VOCAB, _H]).
        embed_name = next(
            init.name
            for init in model.graph.initializer
            if numpy_helper.to_array(init).shape == (_VOCAB, _H)
        )
        w_before = numpy_helper.to_array(
            ParamUtils.get_param_by_name(model, embed_name)
        ).copy()

        sim = QuantizationSimModel(model, dummy_input={"input": token_ids})
        apply_spinquant(sim)

        w_after = numpy_helper.to_array(
            ParamUtils.get_param_by_name(sim.model.model, embed_name)
        )
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
        token_ids = np.random.randint(0, _VOCAB, (_B, _SEQ)).astype(np.int64)

        sim = QuantizationSimModel(model, dummy_input={"input": token_ids})

        # Snapshot all initializers before the failed call.
        init_before = {
            t.name: numpy_helper.to_array(t).copy()
            for t in sim.model.model.graph.initializer
        }

        """
        When: Block detection/classification raises an error.
        Then: The model is unchanged.
        """

        with pytest.raises(ValueError):
            apply_spinquant(sim)

        # Every initializer must be bit-exact same.
        for name, arr_before in init_before.items():
            arr_after = numpy_helper.to_array(
                ParamUtils.get_param_by_name(sim.model.model, name)
            )
            assert np.array_equal(arr_before, arr_after)

    def test_embedding_weight_rotated(self):
        """External embedding.pth weight (raw tensor) must be rotated by R_L after apply_spinquant."""
        torch.manual_seed(0)

        backbone_model = _export_vlm_backbone(VLMBackbone())
        inputs_embeds = np.random.randn(_B, _SEQ, _H).astype(np.float32)

        # Simulate torch.load("embedding.pth") → raw tensor [vocab, hidden]
        embedding = torch.randn(_VOCAB, _H)
        W_before = embedding.clone()

        backbone_sim = QuantizationSimModel(
            backbone_model, dummy_input={"input": inputs_embeds}
        )

        """
        When: apply_spinquant is called with an external embedding tensor.
        Then: tensor is modified in-place (rotated by R_L).
        """
        apply_spinquant(backbone_sim, embedding=embedding)

        assert not torch.equal(embedding, W_before)

    def test_embedding_rotation_consistent_with_backbone(self):
        """Backbone(embedding_rotated[i]) == backbone_rotated(embedding_orig[i])."""
        torch.manual_seed(0)
        np.random.seed(0)

        backbone_model = _export_vlm_backbone(VLMBackbone())
        inputs_embeds = np.random.randn(_B, _SEQ, _H).astype(np.float32)

        embedding = torch.randn(_VOCAB, _H)
        W_orig = embedding.clone().numpy()  # [vocab, H]

        # Baseline: original backbone on original embedding rows
        y_before = _run_model(backbone_model, W_orig[:_SEQ].reshape(1, _SEQ, _H))

        backbone_sim = QuantizationSimModel(
            backbone_model, dummy_input={"input": inputs_embeds}
        )
        apply_spinquant(backbone_sim, embedding=embedding)

        rotated_float = QuantizationSimModel.remove_quantizers(
            copy.deepcopy(backbone_sim.model.model)
        )
        W_rot = embedding.numpy()  # rotated in-place by R_L

        """
        When: backbone_rotated receives embedding_rotated rows as inputs_embeds.
        Then: output matches baseline (original backbone on original embedding).
        """
        y_after = _run_model(rotated_float, W_rot[:_SEQ].reshape(1, _SEQ, _H))
        assert np.allclose(y_before, y_after, atol=1e-5)

    def test_visual_output_rotated_by_r_l(self):
        """Visual encoder output must equal y_before @ R_L after apply_spinquant."""
        torch.manual_seed(0)
        np.random.seed(0)

        backbone_model = _export_vlm_backbone(VLMBackbone())
        visual_model = _export_vit(ViTEncoder())

        inputs_embeds = np.random.randn(_B, _SEQ, _H).astype(np.float32)
        x_vit = np.random.randn(_VIT_N, _VIT_D).astype(np.float32)

        y_vit_before = _run_model(visual_model, x_vit)

        backbone_sim = QuantizationSimModel(
            backbone_model, dummy_input={"input": inputs_embeds}
        )
        visual_sim = QuantizationSimModel(visual_model, dummy_input={"input": x_vit})

        """
        When: apply_spinquant is called with backbone_sim and visual_sim.
        Then: visual encoder output equals y_before @ R_L.
        """
        embedding = torch.randn(_VOCAB, _H)
        apply_spinquant(backbone_sim, visual_sim=visual_sim, embedding=embedding)

        R_L = (get_hadamard_matrix(_VIT_D_L) / np.sqrt(_VIT_D_L)).astype(np.float32)

        rotated_visual_float = QuantizationSimModel.remove_quantizers(
            copy.deepcopy(visual_sim.model.model)
        )
        y_vit_after = _run_model(rotated_visual_float, x_vit)
        assert np.allclose(y_vit_after, y_vit_before @ R_L)

    def test_post_writing_norm_raises_and_leaves_model_untouched(self):
        """apply_spinquant must raise ValueError for Gemma-style architectures (post-writing norms)
        and must not modify any initializers before raising."""
        torch.manual_seed(0)
        model = _export_decoder_with_ids(Gemma3StyleDecoder())
        token_ids = np.random.randint(0, _VOCAB, (_B, _SEQ)).astype(np.int64)
        sim = QuantizationSimModel(model, dummy_input={"input": token_ids})

        init_before = {
            t.name: numpy_helper.to_array(t).copy()
            for t in sim.model.model.graph.initializer
        }

        """
        When: apply_spinquant is applied for Gemma3/Gemma4 style architectures.
        Then: Must raise ValueError and initializers are not modified.
        """

        with pytest.raises(ValueError):
            apply_spinquant(sim)

        for name, arr_before in init_before.items():
            arr_after = numpy_helper.to_array(
                ParamUtils.get_param_by_name(sim.model.model, name)
            )
            assert np.array_equal(arr_before, arr_after)

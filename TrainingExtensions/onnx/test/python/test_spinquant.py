# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import io
import numpy as np
import pytest
import torch
import torch.nn as nn
import onnx
from onnx import load_model, numpy_helper
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

from .models.test_models import RMSNorm
from aimet_onnx.experimental.spinquant.fuse_norm import (
    _OP_OUTPUTS_TO_IGNORE,
    _find_norm_scale_and_consumers,
    _get_weight_product,
    fuse_norm_layers_into_linears,
)
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.utils import ParamUtils


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


class TestFuseNormLayers:
    """Unit tests for fuse_norm_layers_into_linears.

    1. RMSNorm → MatMul  (via torch.matmul, transB=0)
    2. RMSNorm → Gemm    (via nn.Linear, transB=1)
    3. RMSNorm → MatMul  (via nn.Linear, transB=1)
    4. RMSNorm → three parallel Gemm ops  (Q / K / V)
    5. RMSNorm → Transpose → Conv  (SHA_Conv reshape-chain pattern)
    6. Non-affine RMSNorm (no gamma)  → no-op, weights unchanged
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
        fuse_norm_layers_into_linears(model, cg)
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
        fuse_norm_layers_into_linears(model, cg)
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
        fuse_norm_layers_into_linears(model, cg)
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
        fuse_norm_layers_into_linears(model, cg)
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
        fuse_norm_layers_into_linears(model, cg)
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
        fuse_norm_layers_into_linears(model, cg)

        w_after = numpy_helper.to_array(ParamUtils.get_param_by_name(model, w_name))
        assert np.array_equal(w_after, w_before)
        assert np.allclose(_run_model(model, x), y_before, atol=1e-6)

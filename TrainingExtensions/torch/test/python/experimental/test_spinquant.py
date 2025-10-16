# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Test SpinQuant functions"""

import contextlib
import copy
import math
import pytest
import torch
from torch.nn import functional as F
from transformers import PreTrainedModel, DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaForCausalLM,
    LlamaConfig,
)
from aimet_torch.experimental.spinquant import spinquant_optimizer
from aimet_torch.experimental.spinquant.hadamard_utils import get_hadamard_matrix
from aimet_torch.experimental.spinquant import apply_spinquant
from aimet_torch.experimental.spinquant.spinquant_optimizer import (
    _fuse_rmsnorm_into_linear,
    _identify_and_fuse_rmsnorms_into_linears,
    RMSNORM_LINEAR_PAIRS,
    R1_FUSION_PAIRS,
)
from aimet_torch.quantsim import QuantizationSimModel


class TorchExportableModuleWithCache(torch.nn.Module):
    """
    Helper class to enable Torch JIT trace and ONNX export of HuggingFace models that produce and consume Cache objects
    """

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model

    @property
    def device(self):
        """Return model device"""
        return self.model.device

    @property
    def dtype(self):
        """Return model dtype"""
        return self.model.dtype

    @property
    def config(self):
        """Return model config"""
        return self.model.config

    # pylint: disable=keyword-arg-before-vararg
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        past_key_values: DynamicCache = None,
        *args,
        **kwargs,
    ):
        """Redefine model forward to convert to/from Huggingface DynamicCache objects"""
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        lm_logits, new_past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            num_logits_to_return=0,
            return_dict=False,
            *args,
            **kwargs,
        )
        return lm_logits, new_past_key_values.to_legacy_cache()


@contextlib.contextmanager
def _register_module_for_spinquant(
    module, rmsnorm_linear_identifier, r1_fusion_identifier
):
    try:
        spinquant_optimizer.SUPPORTED_MODULE_DICT[module] = {}
        spinquant_optimizer.SUPPORTED_MODULE_DICT[module][RMSNORM_LINEAR_PAIRS] = (
            rmsnorm_linear_identifier
        )
        spinquant_optimizer.SUPPORTED_MODULE_DICT[module][R1_FUSION_PAIRS] = (
            r1_fusion_identifier
        )
        yield

    finally:
        del spinquant_optimizer.SUPPORTED_MODULE_DICT[module]


class RMSNormWithLinear(torch.nn.Module):
    def __init__(self, bias):
        super(RMSNormWithLinear, self).__init__()
        self.rmsnorm = LlamaRMSNorm(10)
        self.q = torch.nn.Linear(10, 10, bias=bias)
        self.k = torch.nn.Linear(10, 10, bias=bias)
        self.v = torch.nn.Linear(10, 10, bias=bias)

    def forward(self, x):
        x = self.rmsnorm(x)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return q, k, v


@pytest.mark.parametrize("bias", [True, False])
def test_fuse_rmsnorm_into_linear(bias):
    """Test fuse_rmsnorm_into_linear"""
    torch.manual_seed(0)
    dummy_input = torch.randn(10, 10)
    model = RMSNormWithLinear(bias)
    with torch.no_grad():
        model.rmsnorm.weight.copy_(torch.randn(model.rmsnorm.weight.shape))

    orig_q_out, orig_k_out, orig_v_out = model(dummy_input)
    orig_rmsnorm_weight = model.rmsnorm.weight.clone()
    orig_q_weight = model.q.weight.clone()
    orig_k_weight = model.k.weight.clone()
    orig_v_weight = model.v.weight.clone()
    _fuse_rmsnorm_into_linear(model.rmsnorm, [model.q, model.k, model.v])

    new_q_out, new_k_out, new_v_out = model(dummy_input)
    assert torch.allclose(orig_q_out, new_q_out, atol=1e-6)
    assert torch.allclose(orig_k_out, new_k_out, atol=1e-6)
    assert torch.allclose(orig_v_out, new_v_out, atol=1e-6)

    assert torch.equal(model.rmsnorm.weight, torch.ones(orig_rmsnorm_weight.shape))
    assert not torch.equal(orig_rmsnorm_weight, model.rmsnorm.weight)
    assert not torch.equal(orig_q_weight, model.q.weight)
    assert not torch.equal(orig_k_weight, model.k.weight)
    assert not torch.equal(orig_v_weight, model.v.weight)


def test_identify_and_fuse_llama_model_rmsnorms():
    torch.manual_seed(0)
    with _register_module_for_spinquant(
        RMSNormWithLinear, lambda m: [(m.rmsnorm, [m.q, m.k, m.v])], lambda _: []
    ):
        model = RMSNormWithLinear(bias=True)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, LlamaRMSNorm):
                    module.weight.copy_(torch.randn(module.weight.shape))

        dummy_input = torch.randn(10, 10)
        orig_q_out, orig_k_out, orig_v_out = model(dummy_input)

        orig_rmsnorm_weight = model.rmsnorm.weight.clone()

        _identify_and_fuse_rmsnorms_into_linears(model)

        new_q_out, new_k_out, new_v_out = model(dummy_input)
        new_rmsnorm_weight = model.rmsnorm.weight.clone()

        assert torch.allclose(orig_q_out, new_q_out, atol=1e-6)
        assert torch.allclose(orig_k_out, new_k_out, atol=1e-6)
        assert torch.allclose(orig_v_out, new_v_out, atol=1e-6)
        assert not torch.equal(orig_rmsnorm_weight, new_rmsnorm_weight)


def test_fuse_rmsnorm_fp_quantsim_equivalence():
    torch.manual_seed(0)
    with _register_module_for_spinquant(
        RMSNormWithLinear, lambda m: [(m.rmsnorm, [m.q, m.k, m.v])], lambda _: []
    ):
        model = RMSNormWithLinear(bias=True)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, LlamaRMSNorm):
                    module.weight.copy_(torch.randn(module.weight.shape))
        model_2 = copy.deepcopy(model)
        dummy_input = torch.randn(10, 10)

        _identify_and_fuse_rmsnorms_into_linears(model)
        qsim = QuantizationSimModel(model, dummy_input=dummy_input)
        qsim.compute_encodings(lambda m: m(dummy_input))

        qsim_2 = QuantizationSimModel(model_2, dummy_input=dummy_input)
        _identify_and_fuse_rmsnorms_into_linears(qsim_2.model)
        qsim_2.compute_encodings(lambda m: m(dummy_input))

        qsim_out_q, qsim_out_k, qsim_out_v = qsim.model(dummy_input)
        qsim_2_out_q, qsim_2_out_k, qsim_2_out_v = qsim_2.model(dummy_input)
        assert torch.allclose(qsim_out_q, qsim_2_out_q, atol=1e-6)
        assert torch.allclose(qsim_out_k, qsim_2_out_k, atol=1e-6)
        assert torch.allclose(qsim_out_v, qsim_2_out_v, atol=1e-6)


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_raise_error_on_tied_word_embeddings(tie_word_embeddings):
    config = LlamaConfig(
        vocab_size=10,
        hidden_size=64,
        num_hidden_layers=1,
        tie_word_embeddings=tie_word_embeddings,
        intermediate_size=100,
    )
    model = LlamaForCausalLM(config=config)
    if tie_word_embeddings:
        with pytest.raises(RuntimeError):
            apply_spinquant(model)
    else:
        apply_spinquant(model)


@pytest.mark.parametrize("hidden_size", [64, 192])
def test_fuse_r1_rotation(hidden_size):
    torch.manual_seed(0)

    # Given w, w2 as randomized matrices, test:
    # w * w2 == (w * R) * (R^-1 * w2)

    # R * R^-1 is expected to be I due to Hadamard matrix property
    # Produce (w * R) via fusing after, and R^-1 * w2 via fusing before with R^-1 * w2 = (w2.T * R).T
    w = torch.randn(hidden_size, hidden_size)
    w2 = torch.randn(hidden_size, hidden_size)

    had_matrix = get_hadamard_matrix(hidden_size) / torch.sqrt(
        torch.tensor(hidden_size)
    )

    fused_w = w @ had_matrix.T
    fused_w2 = (w2.T @ had_matrix.T).T

    w_times_w2 = torch.matmul(w, w2)
    fused_w_times_w2 = torch.matmul(fused_w, fused_w2)

    assert torch.allclose(w_times_w2, fused_w_times_w2, atol=1e-4)


@pytest.mark.parametrize(
    "hidden_size", [64, 192]
)  # hidden_size 192 will utilize hadamard matrix size 12 with factor 16
@pytest.mark.parametrize("use_bias", [True, False])
def test_apply_spinquant(hidden_size, use_bias):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = LlamaConfig(
        vocab_size=10,
        hidden_size=hidden_size,
        num_hidden_layers=1,
        tie_word_embeddings=False,
        intermediate_size=100,
        attention_bias=use_bias,
        mlp_bias=use_bias,
    )
    dummy_input = torch.randint(0, 10, (1, 200)).to(device)
    model = LlamaForCausalLM(config=config).to(device)
    if use_bias:
        for module in model.modules():
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                with torch.no_grad():
                    module.bias.copy_(torch.randn(module.bias.shape).to(device))
    orig_out = model(input_ids=dummy_input)
    orig_q = model.model.layers[0].self_attn.q_proj.weight.clone()
    orig_embed_tokens = model.model.embed_tokens.weight.clone()
    apply_spinquant(model)
    new_out = model(input_ids=dummy_input)
    new_embed_tokens = model.model.embed_tokens.weight.clone()
    new_q = model.model.layers[0].self_attn.q_proj.weight.clone()
    assert not torch.equal(orig_embed_tokens, new_embed_tokens)
    assert not torch.equal(orig_q, new_q)
    assert torch.allclose(orig_out.logits, new_out.logits, atol=1e-6)


@pytest.mark.parametrize("hidden_size", [192, 1536, 2560, 3584])
def test_get_hadamard_matrix(hidden_size):
    had_matrix = get_hadamard_matrix(hidden_size)

    ones = torch.ones(hidden_size)
    eye = torch.eye(hidden_size) * hidden_size

    # Check that had_matrix consists of all 1s and -1s
    assert torch.allclose(had_matrix * had_matrix, ones, atol=1e-4)

    # Check that had_matrix conforms to expected hadamard
    assert torch.allclose(had_matrix @ had_matrix.T, eye, atol=1e-4)


def test_apply_spinquant_quantsim_equivalence():
    torch.manual_seed(0)

    config = LlamaConfig(
        vocab_size=10,
        hidden_size=192,
        num_hidden_layers=1,
        tie_word_embeddings=False,
        intermediate_size=100,
        attention_bias=True,
        mlp_bias=True,
    )
    dummy_input = torch.randint(0, 10, (1, 200))
    model = LlamaForCausalLM(config=config)
    model = TorchExportableModuleWithCache(model)
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            with torch.no_grad():
                module.bias.copy_(torch.randn(module.bias.shape))

    model_2 = copy.deepcopy(model)

    apply_spinquant(model)
    qsim = QuantizationSimModel(model, dummy_input=dummy_input)
    qsim.compute_encodings(lambda m: m(dummy_input))

    qsim_2 = QuantizationSimModel(model_2, dummy_input=dummy_input)
    apply_spinquant(qsim_2.model)
    qsim_2.compute_encodings(lambda m: m(dummy_input))

    qsim_out = qsim.model(input_ids=dummy_input)
    qsim_out_2 = qsim_2.model(input_ids=dummy_input)
    assert torch.equal(qsim_out[0], qsim_out_2[0])

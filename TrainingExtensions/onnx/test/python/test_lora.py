# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the aimet_onnx.experimental.lora module."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn

from .conftest import skip_module_on_windows_amd64, skip_module_on_windows_arm64

skip_module_on_windows_amd64(
    "torch dynamo ONNX export packaging issue on Windows AMD64 CI"
)
skip_module_on_windows_arm64("peft and safetensors not available on Windows ARM64")

from peft import LoraConfig, get_peft_model
from safetensors.numpy import load_file, save_file

import aimet_onnx
from aimet_onnx import int4, int16
from aimet_onnx.experimental.lora import (
    export_peft_to_onnx,
    set_lora_bitwidth,
    freeze_base_param_quantizers,
    freeze_base_activation_quantizers,
    freeze_base_model,
    unfreeze_lora_quantizers,
    get_lora_encodings,
    set_lora_encodings,
    get_zero_weights,
    write_lora_weight_list,
    write_lora_config,
    write_adaptor_list,
)
from aimet_onnx.experimental.lora.peft_to_onnx import (
    _extract_output,
    _infer_dynamic_shapes,
    _infer_input_names,
    _load_adapter_safetensors,
    _onnx_name_to_safetensors_key,
)
from aimet_onnx.quantsim import QuantizationSimModel

# =========================================================================
# Model generation utilities
# =========================================================================


class _Attention(nn.Module):
    """Minimal multi-head attention with explicit q/k/v/o projections (LLM-style)."""

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)


class _FeedForward(nn.Module):
    """Simple two-layer FFN (up_proj / down_proj)."""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.up_proj(x)))


class _TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.self_attn = _Attention(hidden_size, num_heads)
        self.mlp = _FeedForward(hidden_size, intermediate_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def _build_tiny_transformer(
    vocab_size: int = 512,
    hidden_size: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    intermediate_size: int = 128,
) -> nn.Module:
    """Build a minimal LLM-style transformer for testing. No HuggingFace dependency."""

    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.layers = nn.ModuleList(
                [
                    _TransformerBlock(hidden_size, num_heads, intermediate_size)
                    for _ in range(num_layers)
                ]
            )
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        def forward(self, input_ids):
            x = self.embedding(input_ids)
            for layer in self.layers:
                x = layer(x)
            return self.lm_head(x)

    return TinyTransformer()


def _apply_peft_lora(model: nn.Module, rank: int = 8) -> "PeftModel":
    """Apply PEFT LoRA to all linear layers in the model."""
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            attr_name = name.split(".")[-1]
            if attr_name not in target_modules:
                target_modules.append(attr_name)

    config = LoraConfig(
        r=rank,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    return get_peft_model(model, config)


def _randomize_lora_weights(peft_model, seed: int) -> None:
    """Re-initialize LoRA weights with a different random seed to simulate a different adapter."""
    with torch.no_grad():
        torch.manual_seed(seed)
        for name, param in peft_model.named_parameters():
            if "lora_A" in name:
                nn.init.kaiming_uniform_(param.data)
            elif "lora_B" in name:
                nn.init.zeros_(param.data)
                param.data += torch.randn_like(param.data) * 0.01


def generate_lora_test_artifacts(
    output_dir: str,
    rank: int = 8,
    seq_len: int = 32,
) -> dict:
    """Generate a test ONNX model with LoRA adapters via export_peft_to_onnx.

    Returns a dict with model_path, lora_names, and output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model = _build_tiny_transformer()
    peft_model = _apply_peft_lora(base_model, rank=rank)
    peft_model.eval()

    # Snapshot default LoRA weights before randomizing for B/C
    original_lora_state = {
        name: param.detach().clone()
        for name, param in peft_model.named_parameters()
        if "lora_" in name
    }

    # Save adapters B and C to disk in PEFT format
    adapter_dirs = {}
    for i, name in enumerate(["B", "C"]):
        _randomize_lora_weights(peft_model, seed=42 + i + 1)
        adapter_dir = str(output_dir / f"adapter_{name}")
        peft_model.save_pretrained(adapter_dir)
        adapter_dirs[name] = adapter_dir

    # Restore default weights so export captures the original adapter
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if name in original_lora_state:
                param.copy_(original_lora_state[name])

    sample_inputs = (torch.randint(0, 512, (1, seq_len)),)
    model_proto, lora_names = export_peft_to_onnx(
        peft_model,
        sample_inputs,
        adapter_dirs,
        str(output_dir),
    )

    return {
        "model_path": str(output_dir / "model.onnx"),
        "lora_names": lora_names,
        "output_dir": str(output_dir),
    }


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture(scope="session")
def lora_artifacts(tmp_path_factory):
    """Generate LoRA test artifacts once per test session via export_peft_to_onnx."""
    artifacts_dir = tmp_path_factory.mktemp("lora_artifacts")
    return generate_lora_test_artifacts(output_dir=str(artifacts_dir), rank=8)


@pytest.fixture
def prepared(lora_artifacts) -> tuple:
    """Fresh model proto and lora_names for each test.

    Loads a new model proto from disk so QuantSim mutations don't leak between tests.
    """
    model_proto = onnx.load(lora_artifacts["model_path"])
    onnx.load_external_data_for_model(model_proto, lora_artifacts["output_dir"])
    lora_names = lora_artifacts["lora_names"]
    output_dir = lora_artifacts["output_dir"]
    return model_proto, lora_names, output_dir


def _make_sim(model):
    """Create a QuantizationSimModel from a model proto."""
    dummy_input = {
        "input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64),
    }
    return QuantizationSimModel(
        model, dummy_input=dummy_input, param_type=int4, activation_type=int16
    )


def _calibrate(sim, weights, num_batches=5):
    """Run calibration with given weights."""

    def _fn(session, w=weights):
        for _ in range(num_batches):
            batch = {
                "input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64),
            }
            batch.update(w)
            session.run(None, batch)

    sim.compute_encodings(_fn)


def _load_adapters(output_dir, expected=("B", "C")):
    """Load adapter safetensors from output_dir and verify expected adapters exist."""
    adapters = {}
    for name in expected:
        path = os.path.join(output_dir, f"{name}.safetensors")
        assert os.path.exists(path), (
            f"Expected adapter file {name}.safetensors not found"
        )
        adapters[name] = load_file(path)
    return adapters


# =========================================================================
# Test: V1 workflow (single encodings across adapters)
# =========================================================================


def test_full_lora_v1_workflow(prepared):
    """V1: LoRA as inputs, single weight + activation encoding across adapters."""
    model, lora_names, output_dir = prepared
    assert len(lora_names["params"]) > 0
    assert len(lora_names["activations"]) > 0

    adapters = _load_adapters(output_dir)

    sim = QuantizationSimModel(
        model,
        dummy_input={"input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64)},
        param_type=int4,
        activation_type=int16,
    )

    set_lora_bitwidth(sim, lora_names, param_type=int16, activation_type=int16)

    # Calibrate with all adapters in a loop (V1 strategy)
    def calibrate_adapters(session):
        for adapter_name, weights in adapters.items():
            for _ in range(5):
                batch = {
                    "input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64),
                }
                batch.update(weights)
                session.run(None, batch)

    sim.compute_encodings(calibrate_adapters)

    # Freeze everything (V1: single encoding for all adapters)
    frozen = freeze_base_model(sim, lora_names)
    assert frozen > 0

    with tempfile.TemporaryDirectory() as export_dir:
        sim.export(export_dir, "model_lora", export_model=False)
        assert os.path.exists(os.path.join(export_dir, "model_lora.encodings"))


# =========================================================================
# Test: V2 workflow (full recalibration per adapter)
# =========================================================================


def test_full_lora_v2_workflow(prepared):
    """V2: Full recalibration per adapter, no freezing."""
    model, lora_names, output_dir = prepared
    assert len(lora_names["params"]) > 0

    adapters = _load_adapters(output_dir)

    sim = _make_sim(model)
    set_lora_bitwidth(sim, lora_names, param_type=int16, activation_type=int16)

    for name in lora_names["params"]:
        if name in sim.qc_quantize_op_dict:
            assert sim.qc_quantize_op_dict[name].bitwidth == 16

    with tempfile.TemporaryDirectory() as export_dir:
        for adapter_name, weights in adapters.items():
            _calibrate(sim, weights)

            sim.export(export_dir, f"model_{adapter_name}", export_model=False)
            assert os.path.exists(
                os.path.join(export_dir, f"model_{adapter_name}.encodings")
            )


# =========================================================================
# Test: V3 workflow (base frozen + per-adapter LoRA recalibration)
# =========================================================================


def test_full_lora_v3_workflow(prepared):
    """V3: Base frozen, per-adapter LoRA recalibration."""
    model, lora_names, output_dir = prepared
    assert len(lora_names["params"]) > 0
    assert len(lora_names["activations"]) > 0

    adapters = _load_adapters(output_dir)

    sim = _make_sim(model)
    set_lora_bitwidth(sim, lora_names, param_type=int16, activation_type=int16)

    # Base calibration with LoRA disabled
    zero_weights = get_zero_weights(model, lora_names)
    _calibrate(sim, zero_weights)

    # Freeze base (V3: freeze all base, then per-adapter LoRA recalibration)
    frozen_params = freeze_base_param_quantizers(sim, lora_names)
    assert frozen_params > 0
    frozen_activations = freeze_base_activation_quantizers(sim, lora_names)
    assert frozen_activations > 0

    # Snapshot a base param encoding to verify it stays frozen
    base_param_name = next(
        name
        for name in sim.param_names
        if name not in set(lora_names["params"])
        and name in sim.qc_quantize_op_dict
        and sim.qc_quantize_op_dict[name].enabled
    )
    base_encoding_before = sim.qc_quantize_op_dict[base_param_name].export_encodings(
        "1.0.0"
    )

    # Per-adapter calibration
    adapter_encodings = {}
    with tempfile.TemporaryDirectory() as export_dir:
        sim.export(export_dir, "model")
        assert os.path.exists(os.path.join(export_dir, "model.encodings"))

        for adapter_name, weights in adapters.items():
            unfrozen = unfreeze_lora_quantizers(sim, lora_names)
            assert unfrozen > 0

            _calibrate(sim, weights)
            adapter_encodings[adapter_name] = get_lora_encodings(sim, lora_names)

            sim.export(export_dir, f"model_{adapter_name}", export_model=False)
            assert os.path.exists(
                os.path.join(export_dir, f"model_{adapter_name}.encodings")
            )

        # Verify base encodings did NOT change after per-adapter calibration
        base_encoding_after = sim.qc_quantize_op_dict[base_param_name].export_encodings(
            "1.0.0"
        )
        assert base_encoding_before == base_encoding_after, (
            "Base param encoding changed during per-adapter calibration — freeze is broken"
        )

        # Verify adapter encodings differ between B and C
        enc_b = adapter_encodings["B"]
        enc_c = adapter_encodings["C"]
        assert len(enc_b) > 0 and len(enc_c) > 0
        any_differ = any(enc_b.get(name) != enc_c.get(name) for name in enc_b)
        assert any_differ, (
            "Adapter B and C encodings are identical — per-adapter calibration may be broken"
        )

        # Verify output structure
        output_files = sorted(os.listdir(export_dir))
        assert "model.onnx" in output_files
        assert "model.encodings" in output_files
        for adapter_name in adapters:
            assert f"model_{adapter_name}.encodings" in output_files


# =========================================================================
# Test: Encoding roundtrip (get/set_lora_encodings)
# =========================================================================


def test_lora_encoding_roundtrip(prepared):
    """get_lora_encodings / set_lora_encodings round-trip correctly."""
    model, lora_names, output_dir = prepared
    sim = _make_sim(model)

    set_lora_bitwidth(sim, lora_names, param_type=int16, activation_type=int16)
    zero_weights = get_zero_weights(model, lora_names)
    _calibrate(sim, zero_weights)
    freeze_base_model(sim, lora_names)

    # Calibrate with adapter B
    adapter_b = load_file(os.path.join(output_dir, "B.safetensors"))
    unfreeze_lora_quantizers(sim, lora_names)
    _calibrate(sim, adapter_b)

    encodings_b = get_lora_encodings(sim, lora_names)
    assert len(encodings_b) > 0

    # Calibrate with adapter C (overwrites B's encodings)
    adapter_c = load_file(os.path.join(output_dir, "C.safetensors"))
    unfreeze_lora_quantizers(sim, lora_names)
    _calibrate(sim, adapter_c)

    # Restore adapter B encodings
    restored = set_lora_encodings(sim, encodings_b)
    assert restored == len(encodings_b)

    encodings_b_restored = get_lora_encodings(sim, lora_names)
    for name in encodings_b:
        assert name in encodings_b_restored
        assert encodings_b[name] == encodings_b_restored[name]


# =========================================================================
# Test: Per-channel quantization for LoRA weights
# =========================================================================


def test_lora_per_channel(prepared):
    """LoRA quantizers get per-channel mode (dual-listed → classified as params)."""
    model, lora_names, output_dir = prepared

    # Verify dual-listing: LoRA names in both initializer and graph.input
    init_names = {init.name for init in model.graph.initializer}
    input_names = {inp.name for inp in model.graph.input}
    for name in lora_names["params"]:
        assert name in init_names, f"{name} should be an initializer (dual-listed)"
        assert name in input_names, f"{name} should be a graph input (dual-listed)"

    # Create QuantSim — LoRA classified as params → per-channel
    sim = _make_sim(model)

    for name in lora_names["params"]:
        if name in sim.qc_quantize_op_dict:
            qtzr = sim.qc_quantize_op_dict[name]
            assert qtzr.quant_info.usePerChannelMode, (
                f"{name} should be per-channel (dual-listed as param)"
            )

    # set_lora_bitwidth should not break per-channel
    set_lora_bitwidth(sim, lora_names, param_type=int16, activation_type=int16)

    for name in lora_names["params"]:
        if name in sim.qc_quantize_op_dict:
            qtzr = sim.qc_quantize_op_dict[name]
            assert qtzr.bitwidth == 16
            assert qtzr.quant_info.usePerChannelMode, (
                f"{name} should still be per-channel after set_lora_bitwidth"
            )


# =========================================================================
# Test: lora_names structure
# =========================================================================


def test_lora_names_structure(prepared):
    """lora_names has correct keys and non-empty lists."""
    _model, lora_names, _output_dir = prepared

    assert "params" in lora_names
    assert "activations" in lora_names
    assert len(lora_names["params"]) > 0
    assert len(lora_names["activations"]) > 0

    # All param names should contain lora_A or lora_B
    for name in lora_names["params"]:
        assert "lora_A" in name or "lora_B" in name, (
            f"Param name '{name}' should contain lora_A or lora_B"
        )

    # Activations should be distinct from params
    param_set = set(lora_names["params"])
    for name in lora_names["activations"]:
        assert name not in param_set, f"Activation '{name}' should not also be a param"


# =========================================================================
# Test: get_zero_weights
# =========================================================================


def test_get_zero_weights(prepared):
    """get_zero_weights returns zero arrays for all LoRA params."""
    model, lora_names, _output_dir = prepared

    zero_weights = get_zero_weights(model, lora_names)

    assert len(zero_weights) == len(lora_names["params"])
    for name, arr in zero_weights.items():
        assert name in lora_names["params"]
        assert isinstance(arr, np.ndarray)
        assert np.all(arr == 0)


# =========================================================================
# Test: QAIRT artifact helpers
# =========================================================================


def test_write_lora_weight_list(prepared):
    """write_lora_weight_list writes all LoRA param names."""
    _model, lora_names, _output_dir = prepared

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "lora_weight_list.txt")
        write_lora_weight_list(lora_names, path)

        with open(path) as f:
            weight_names = [line.strip() for line in f if line.strip()]

        assert set(weight_names) == set(lora_names["params"])


def test_write_lora_config():
    """write_lora_config produces valid YAML structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        write_lora_config(["adapter_A", "adapter_B"], tmpdir, "model")

        config_path = os.path.join(tmpdir, "lora_config.yaml")
        assert os.path.exists(config_path)
        with open(config_path) as f:
            content = f.read()
        assert "use_case:" in content
        assert "adapter_A" in content
        assert "adapter_B" in content
        assert "model.onnx" in content


def test_write_adaptor_list():
    """write_adaptor_list produces valid YAML structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        write_adaptor_list(["adapter_A", "adapter_B"], tmpdir, "model")

        adaptor_path = os.path.join(tmpdir, "lora_adaptor_list.yaml")
        assert os.path.exists(adaptor_path)
        with open(adaptor_path) as f:
            content = f.read()
        assert "adapter_A" in content
        assert "adapter_B" in content


# =========================================================================
# Test: _onnx_name_to_safetensors_key name transform
# =========================================================================


def test_onnx_name_to_safetensors_key():
    """Deterministic name transform strips model. prefix and adapter name."""
    # Standard case: model. prefix + .default. adapter name
    assert (
        _onnx_name_to_safetensors_key(
            "model.base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
        )
        == "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    )

    # lora_B variant
    assert (
        _onnx_name_to_safetensors_key(
            "model.base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"
        )
        == "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
    )

    # Non-default adapter name
    assert (
        _onnx_name_to_safetensors_key(
            "model.base_model.model.model.layers.0.self_attn.q_proj.lora_A.style.weight"
        )
        == "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    )

    # No model. prefix (edge case)
    assert (
        _onnx_name_to_safetensors_key("base_model.model.layers.0.lora_A.default.weight")
        == "base_model.model.layers.0.lora_A.weight"
    )

    # Higher layer index
    assert (
        _onnx_name_to_safetensors_key(
            "model.base_model.model.model.layers.31.self_attn.v_proj.lora_B.default.weight"
        )
        == "base_model.model.model.layers.31.self_attn.v_proj.lora_B.weight"
    )


# =========================================================================
# Test: _load_adapter_safetensors from local directory
# =========================================================================


def test_load_adapter_safetensors_local():
    """Load adapter weights from a local safetensors file and map to ONNX names."""
    sf_weights = {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": np.random.randn(
            8, 64
        ).astype(np.float32),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": np.random.randn(
            64, 8
        ).astype(np.float32),
        "base_model.model.layers.0.self_attn.v_proj.lora_A.weight": np.random.randn(
            8, 64
        ).astype(np.float32),
        "base_model.model.layers.0.self_attn.v_proj.lora_B.weight": np.random.randn(
            64, 8
        ).astype(np.float32),
    }

    onnx_input_names = [
        "model.base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
        "model.base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight",
        "model.base_model.model.layers.0.self_attn.v_proj.lora_A.default.weight",
        "model.base_model.model.layers.0.self_attn.v_proj.lora_B.default.weight",
    ]

    with tempfile.TemporaryDirectory() as adapter_dir:
        save_file(sf_weights, os.path.join(adapter_dir, "adapter_model.safetensors"))
        mapped = _load_adapter_safetensors(adapter_dir, onnx_input_names)

    assert len(mapped) == 4
    for onnx_name in onnx_input_names:
        assert onnx_name in mapped
        assert isinstance(mapped[onnx_name], np.ndarray)

    np.testing.assert_array_equal(
        mapped[
            "model.base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
        ],
        sf_weights["base_model.model.layers.0.self_attn.q_proj.lora_A.weight"],
    )


def test_load_adapter_safetensors_missing_file():
    """Raises FileNotFoundError when adapter_model.safetensors is missing."""
    with tempfile.TemporaryDirectory() as empty_dir:
        with pytest.raises(FileNotFoundError, match="No adapter_model.safetensors"):
            _load_adapter_safetensors(empty_dir, ["some_name"])


def test_load_adapter_safetensors_partial_match():
    """Raises ValueError when not all ONNX names can be mapped."""
    sf_weights = {
        "base_model.model.layers.0.lora_A.weight": np.zeros((4, 8), dtype=np.float32),
    }

    onnx_names = [
        "model.base_model.model.layers.0.lora_A.default.weight",
        "model.base_model.model.layers.0.lora_B.default.weight",
    ]

    with tempfile.TemporaryDirectory() as adapter_dir:
        save_file(sf_weights, os.path.join(adapter_dir, "adapter_model.safetensors"))
        with pytest.raises(ValueError, match="Only mapped 1/2"):
            _load_adapter_safetensors(adapter_dir, onnx_names)


def test_load_adapter_safetensors_direct_file():
    """Load adapter weights from a direct .safetensors file path."""
    sf_weights = {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": np.random.randn(
            8, 64
        ).astype(np.float32),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": np.random.randn(
            64, 8
        ).astype(np.float32),
    }

    onnx_input_names = [
        "model.base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
        "model.base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "my_adapter.safetensors")
        save_file(sf_weights, file_path)
        mapped = _load_adapter_safetensors(file_path, onnx_input_names)

    assert len(mapped) == 2
    for name in onnx_input_names:
        assert name in mapped


# =========================================================================
# Test: _extract_output handles various model output types
# =========================================================================


def test_extract_output():
    """_extract_output handles tensors, structured outputs, and tuples."""
    t = torch.randn(2, 3)
    assert _extract_output(t) is t

    class CausalLMOutput:
        def __init__(self):
            self.logits = torch.randn(2, 3)

    out = CausalLMOutput()
    assert _extract_output(out) is out.logits

    class UNetOutput:
        def __init__(self):
            self.sample = torch.randn(1, 4, 64, 64)

    out = UNetOutput()
    assert _extract_output(out) is out.sample

    class EncoderOutput:
        def __init__(self):
            self.last_hidden_state = torch.randn(1, 10, 768)

    out = EncoderOutput()
    assert _extract_output(out) is out.last_hidden_state

    t0 = torch.randn(2, 3)
    assert _extract_output((t0, torch.randn(2, 3))) is t0


# =========================================================================
# Test: _infer_input_names from forward signature
# =========================================================================


def test_infer_input_names():
    """_infer_input_names extracts required param names from forward signature."""

    class FakeBaseModel(nn.Module):
        def forward(self, input_ids, attention_mask=None, labels=None):
            pass

    class FakePeftModel:
        def get_base_model(self):
            return FakeBaseModel()

    names = _infer_input_names(FakePeftModel(), (torch.zeros(1, 10),))
    assert names == ["input_ids"]


def test_infer_input_names_multiple_required():
    """_infer_input_names with multiple required args (e.g. UNet)."""

    class FakeUNet(nn.Module):
        def forward(self, sample, timestep, encoder_hidden_states, return_dict=True):
            pass

    class FakePeftModel:
        def get_base_model(self):
            return FakeUNet()

    inputs = (torch.zeros(1, 4, 64, 64), torch.tensor(1.0), torch.zeros(1, 77, 768))
    names = _infer_input_names(FakePeftModel(), inputs)
    assert names == ["sample", "timestep", "encoder_hidden_states"]


def test_infer_input_names_fallback():
    """Falls back to input_0, input_1, ... when signature has no required params."""

    class FakeModel(nn.Module):
        def forward(self, *args, **kwargs):
            pass

    class FakePeftModel:
        def get_base_model(self):
            return FakeModel()

    names = _infer_input_names(FakePeftModel(), (torch.zeros(1), torch.zeros(2)))
    assert names == ["input_0", "input_1"]


# =========================================================================
# Test: _infer_dynamic_shapes
# =========================================================================


def test_infer_dynamic_shapes():
    """_infer_dynamic_shapes returns a tuple with dim 0 dynamic for tensors with dim > 0."""
    inputs = (torch.zeros(1, 10), torch.tensor(1.0))
    names = ["input_ids", "timestep"]
    shapes = _infer_dynamic_shapes(inputs, names)

    assert isinstance(shapes, tuple)
    assert len(shapes) == 2
    assert 0 in shapes[0]
    assert shapes[1] == {}


# =========================================================================
# Test: Different adapters produce different inference outputs
# =========================================================================


def test_different_adapters_different_outputs(prepared):
    """Feed dict with different adapter weights must produce different outputs."""
    model, lora_names, output_dir = prepared

    sim = _make_sim(model)
    set_lora_bitwidth(sim, lora_names, param_type=int16, activation_type=int16)

    # Calibrate with zero weights so encodings are valid
    zero_weights = get_zero_weights(model, lora_names)
    _calibrate(sim, zero_weights)

    # Run inference with adapter B
    adapter_b = load_file(os.path.join(output_dir, "B.safetensors"))
    input_data = {"input_ids": np.array([[1, 2, 3, 4, 5] + [0] * 27], dtype=np.int64)}
    output_b = sim.session.run(None, {**input_data, **adapter_b})[0]

    # Run inference with adapter C
    adapter_c = load_file(os.path.join(output_dir, "C.safetensors"))
    output_c = sim.session.run(None, {**input_data, **adapter_c})[0]

    # Run inference with zero weights (LoRA disabled)
    output_zero = sim.session.run(None, {**input_data, **zero_weights})[0]

    # All three should be different
    assert not np.allclose(output_b, output_c, atol=1e-6), (
        "Adapter B and C should produce different outputs"
    )
    assert not np.allclose(output_b, output_zero, atol=1e-6), (
        "Adapter B and zero weights should produce different outputs"
    )


# =========================================================================
# Test: set_lora_bitwidth affects both param and activation quantizers
# =========================================================================


def test_set_lora_bitwidth_params_and_activations(prepared):
    """set_lora_bitwidth sets bitwidth on BOTH param and activation quantizers."""
    model, lora_names, _output_dir = prepared

    sim = _make_sim(model)
    count = set_lora_bitwidth(sim, lora_names, param_type=int16, activation_type=int16)
    assert count > 0

    # Verify param quantizers
    param_checked = 0
    for name in lora_names["params"]:
        if name in sim.qc_quantize_op_dict:
            assert sim.qc_quantize_op_dict[name].bitwidth == 16, (
                f"Param {name} should be 16-bit"
            )
            param_checked += 1
    assert param_checked > 0, "No param quantizers found to check"

    # Verify activation quantizers
    activation_checked = 0
    for name in lora_names["activations"]:
        if name in sim.qc_quantize_op_dict:
            assert sim.qc_quantize_op_dict[name].bitwidth == 16, (
                f"Activation {name} should be 16-bit"
            )
            activation_checked += 1
    assert activation_checked > 0, "No activation quantizers found to check"


def test_set_lora_bitwidth_string_type(prepared):
    """set_lora_bitwidth accepts string types (Union[str, qtype] pattern)."""
    model, lora_names, _output_dir = prepared

    sim = _make_sim(model)
    count = set_lora_bitwidth(
        sim, lora_names, param_type="int16", activation_type="int8"
    )
    assert count > 0

    for name in lora_names["params"]:
        if name in sim.qc_quantize_op_dict:
            assert sim.qc_quantize_op_dict[name].bitwidth == 16

    for name in lora_names["activations"]:
        if name in sim.qc_quantize_op_dict:
            assert sim.qc_quantize_op_dict[name].bitwidth == 8


# =========================================================================
# Test: Expected activation count per LoRA layer
# =========================================================================


def test_lora_activation_count(prepared):
    """Each LoRA target module produces expected number of activation tensors.

    Per layer, the LoRA branch has: lora_A MatMul output, lora_B MatMul output,
    Mul (scaling) output, and Add (base + LoRA merge) output = 4 activations.
    With 2 params (lora_A.weight, lora_B.weight) per target, we expect
    activations >= 2 * num_params (i.e., at least 2 activations per param).
    """
    _model, lora_names, _output_dir = prepared

    num_params = len(lora_names["params"])
    num_activations = len(lora_names["activations"])

    # Each pair of lora_A/lora_B params should produce at least 2 activations
    # (the MatMul outputs). Typically 4 (MatMul_A, MatMul_B, Mul, Add).
    assert num_activations >= num_params, (
        f"Expected at least {num_params} activations for {num_params} params, "
        f"got {num_activations}"
    )

    # With our model: each target has 2 params and 4 activations
    # So activations should be ~2x params
    num_lora_pairs = num_params // 2
    assert num_activations >= num_lora_pairs * 2, (
        f"Expected at least {num_lora_pairs * 2} activations for {num_lora_pairs} "
        f"LoRA pairs, got {num_activations}"
    )


# =========================================================================
# Test: get_zero_weights error on missing initializer
# =========================================================================


def test_get_zero_weights_missing_initializer(prepared):
    """get_zero_weights raises ValueError when param names aren't in the model."""
    model, _lora_names, _output_dir = prepared

    bad_lora_names = {
        "params": ["nonexistent_param_name"],
        "activations": [],
    }

    with pytest.raises(ValueError, match="not found in model initializers"):
        get_zero_weights(model, bad_lora_names)


# =========================================================================
# Test: unfreeze actually resets encodings
# =========================================================================


def test_unfreeze_resets_lora_encodings(prepared):
    """unfreeze_lora_quantizers resets encoding stats so recalibration works."""
    model, lora_names, output_dir = prepared

    sim = _make_sim(model)
    set_lora_bitwidth(sim, lora_names, param_type=int16, activation_type=int16)

    # Calibrate base + freeze
    zero_weights = get_zero_weights(model, lora_names)
    _calibrate(sim, zero_weights)
    freeze_base_model(sim, lora_names)

    # Calibrate with adapter B
    adapter_b = load_file(os.path.join(output_dir, "B.safetensors"))
    _calibrate(sim, adapter_b)
    encodings_b = get_lora_encodings(sim, lora_names)

    # Unfreeze and calibrate with adapter C
    unfreeze_lora_quantizers(sim, lora_names)
    adapter_c = load_file(os.path.join(output_dir, "C.safetensors"))
    _calibrate(sim, adapter_c)
    encodings_c = get_lora_encodings(sim, lora_names)

    # After unfreeze + recalibrate with different adapter, encodings should differ
    any_differ = any(
        encodings_b.get(name) != encodings_c.get(name) for name in encodings_b
    )
    assert any_differ, (
        "Encodings should differ after unfreeze + recalibrate with different adapter"
    )


# =========================================================================
# Test: write_lora_weight_list natural sort order
# =========================================================================


def test_write_lora_weight_list_natural_sort():
    """write_lora_weight_list sorts layers.2 before layers.10 (natural sort)."""
    lora_names = {
        "params": [
            "model.layers.10.lora_A.default.weight",
            "model.layers.2.lora_A.default.weight",
            "model.layers.1.lora_A.default.weight",
            "model.layers.20.lora_A.default.weight",
        ],
        "activations": [],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "lora_weight_list.txt")
        write_lora_weight_list(lora_names, path)

        with open(path) as f:
            lines = [line.strip() for line in f if line.strip()]

    assert lines == [
        "model.layers.1.lora_A.default.weight",
        "model.layers.2.lora_A.default.weight",
        "model.layers.10.lora_A.default.weight",
        "model.layers.20.lora_A.default.weight",
    ], f"Expected natural sort order, got: {lines}"


# =========================================================================
# Test: _infer_input_names HuggingFace all-defaults path
# =========================================================================


def test_infer_input_names_all_defaults():
    """HuggingFace models where all params have defaults use positional order."""

    class FakeHFModel(nn.Module):
        def forward(self, input_ids=None, attention_mask=None, position_ids=None):
            pass

    class FakePeftModel:
        def get_base_model(self):
            return FakeHFModel()

    # 2 sample inputs → first 2 positional params
    names = _infer_input_names(
        FakePeftModel(), (torch.zeros(1, 10), torch.zeros(1, 10))
    )
    assert names == ["input_ids", "attention_mask"]

    # 1 sample input → first positional param only
    names = _infer_input_names(FakePeftModel(), (torch.zeros(1, 10),))
    assert names == ["input_ids"]


# =========================================================================
# Error-path tests: _validate_lora_names
# =========================================================================


def test_validate_lora_names_missing_keys(prepared):
    """Functions raise ValueError when lora_names is missing required keys."""
    model, _lora_names, _output_dir = prepared
    sim = _make_sim(model)

    bad_names = {"params": ["some_name"]}  # missing "activations"
    with pytest.raises(ValueError, match="must have 'params' and 'activations' keys"):
        freeze_base_model(sim, bad_names)


def test_validate_lora_names_empty_lists(prepared):
    """Functions raise ValueError when both params and activations are empty."""
    model, _lora_names, _output_dir = prepared
    sim = _make_sim(model)

    empty_names = {"params": [], "activations": []}
    with pytest.raises(ValueError, match="empty 'params' and 'activations' lists"):
        set_lora_bitwidth(sim, empty_names, param_type=int16, activation_type=int16)


# =========================================================================
# Error-path tests: set_lora_bitwidth on frozen quantizers
# =========================================================================


def test_set_lora_bitwidth_raises_on_frozen(prepared):
    """set_lora_bitwidth raises RuntimeError if LoRA quantizers are already frozen."""
    model, lora_names, output_dir = prepared
    sim = _make_sim(model)

    # Calibrate and freeze everything first
    zero_weights = get_zero_weights(model, lora_names)
    _calibrate(sim, zero_weights)
    freeze_base_model(sim, lora_names)

    # Now try to set bitwidth on frozen LoRA quantizers — should fail
    # First freeze LoRA quantizers too (they aren't frozen by freeze_base_model)
    for name in lora_names["params"] + lora_names["activations"]:
        if name in sim.qc_quantize_op_dict:
            sim.qc_quantize_op_dict[name].freeze_encodings()

    with pytest.raises(RuntimeError, match="is frozen"):
        set_lora_bitwidth(sim, lora_names, param_type=int16, activation_type=int16)


# =========================================================================
# Error-path tests: get_lora_encodings on uncalibrated quantizers
# =========================================================================


def test_get_lora_encodings_raises_uncalibrated(prepared):
    """get_lora_encodings raises RuntimeError if quantizers have no encoding."""
    model, lora_names, _output_dir = prepared
    sim = _make_sim(model)

    # Don't calibrate — quantizers have no encodings
    with pytest.raises(RuntimeError, match="has no encoding"):
        get_lora_encodings(sim, lora_names)


# =========================================================================
# Error-path tests: set_lora_encodings with unknown names
# =========================================================================


def test_set_lora_encodings_raises_missing_names(prepared):
    """set_lora_encodings raises ValueError if encoding names aren't in sim."""
    model, _lora_names, _output_dir = prepared
    sim = _make_sim(model)

    fake_encodings = {"nonexistent_quantizer_name": {"some": "data"}}
    with pytest.raises(ValueError, match="encoding names not found in sim"):
        set_lora_encodings(sim, fake_encodings)


# =========================================================================
# Output correctness: FP32 vs quantized outputs
# =========================================================================


def test_quantized_output_close_to_fp32(prepared):
    """Quantized LoRA output should be close to FP32 (not garbage)."""
    model, lora_names, output_dir = prepared

    # FP32 baseline: run original model without quantization
    fp32_session = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    adapter_b = load_file(os.path.join(output_dir, "B.safetensors"))
    input_data = {"input_ids": np.array([[1, 2, 3, 4, 5] + [0] * 27], dtype=np.int64)}
    fp32_feed = {**input_data, **adapter_b}
    fp32_output = fp32_session.run(None, fp32_feed)[0]

    # Quantized: create QuantSim, calibrate, run
    sim = _make_sim(model)
    set_lora_bitwidth(sim, lora_names, param_type=int16, activation_type=int16)
    _calibrate(sim, adapter_b)
    quant_output = sim.session.run(None, fp32_feed)[0]

    # Quantized output should differ from FP32 (quantization has an effect)
    assert not np.array_equal(fp32_output, quant_output), (
        "Quantized output is identical to FP32 — quantization may not be applied"
    )

    # But should be reasonably close (not garbage)
    assert np.allclose(fp32_output, quant_output, atol=1.0), (
        f"Quantized output diverges too much from FP32. "
        f"Max diff: {np.max(np.abs(fp32_output - quant_output)):.4f}"
    )


# =========================================================================
# Output correctness: export roundtrip preserves inference
# =========================================================================


def test_export_preserves_inference(prepared):
    """sim.session.run() produces same output before and after sim.export()."""
    model, lora_names, output_dir = prepared

    sim = _make_sim(model)
    set_lora_bitwidth(sim, lora_names, param_type=int16, activation_type=int16)
    adapter_b = load_file(os.path.join(output_dir, "B.safetensors"))
    _calibrate(sim, adapter_b)

    input_data = {"input_ids": np.array([[1, 2, 3, 4, 5] + [0] * 27], dtype=np.int64)}
    feed = {**input_data, **adapter_b}

    output_before = sim.session.run(None, feed)[0]

    with tempfile.TemporaryDirectory() as export_dir:
        sim.export(export_dir, "model_test")

        # After export, session should still produce same output
        output_after = sim.session.run(None, feed)[0]
        assert np.allclose(output_before, output_after), (
            f"Export changed inference output. "
            f"Max diff: {np.max(np.abs(output_before - output_after)):.6f}"
        )

        # Exported encodings file should exist
        assert os.path.exists(os.path.join(export_dir, "model_test.encodings"))


# =========================================================================
# lora_names JSON serialization roundtrip
# =========================================================================


def test_lora_names_json_serializable(prepared):
    """lora_names must be JSON-serializable (71st adapter scenario)."""
    _model, lora_names, _output_dir = prepared

    # Serialize to JSON and back
    json_str = json.dumps(lora_names)
    restored = json.loads(json_str)

    assert restored == lora_names
    assert isinstance(restored["params"], list)
    assert isinstance(restored["activations"], list)
    assert all(isinstance(name, str) for name in restored["params"])
    assert all(isinstance(name, str) for name in restored["activations"])

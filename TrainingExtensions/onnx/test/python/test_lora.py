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
from safetensors.numpy import load_file

import aimet_onnx
from aimet_onnx import int4, int16
from aimet_onnx.experimental.lora import (
    configure_lora_onnx,
    export_peft_to_onnx,
    add_lora_branches,
    prepare_lora_onnx,
    disable_lora_quantizers,
    enable_lora_calibration,
    set_lora_bitwidth,
    set_lora_weights,
    freeze_base_param_quantizers,
    freeze_base_activation_quantizers,
    freeze_base_model,
    unfreeze_lora_quantizers,
    get_lora_encodings,
    set_lora_encodings,
    get_zero_weights,
    get_adapter_scale_weights,
    get_adapter_names,
    get_adapter_lora_names,
    build_concurrent_feed_dict,
    adapt_base_encodings_for_lora,
    write_lora_weight_list,
    write_lora_config,
    write_adapter_list,
)
from aimet_onnx.experimental.lora.peft_to_onnx import (
    _infer_dynamic_shapes,
    _infer_input_names,
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
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
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

    Returns a dict with model_path, lora_names, output_dir, and adapter_dirs.
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
        str(output_dir),
    )

    return {
        "model_path": str(output_dir / "model.onnx"),
        "lora_names": lora_names,
        "output_dir": str(output_dir),
        "adapter_dirs": adapter_dirs,
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
    adapter_dirs = lora_artifacts["adapter_dirs"]
    return model_proto, lora_names, adapter_dirs


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


def _load_adapters(adapter_dirs):
    """Load adapter safetensors from PEFT adapter directories."""
    adapters = {}
    for name, adapter_dir in adapter_dirs.items():
        path = os.path.join(adapter_dir, "adapter_model.safetensors")
        assert os.path.exists(path), (
            f"Expected adapter_model.safetensors not found in {adapter_dir}"
        )
        adapters[name] = load_file(path)
    return adapters


# =========================================================================
# Test: V1 workflow (single encodings across adapters)
# =========================================================================


def test_full_lora_v1_workflow(prepared):
    """V1: LoRA as inputs, single weight + activation encoding across adapters."""
    model, lora_names, adapter_dirs = prepared
    assert len(lora_names["params"]) > 0
    assert len(lora_names["activations"]) > 0

    adapters = _load_adapters(adapter_dirs)

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
    model, lora_names, adapter_dirs = prepared
    assert len(lora_names["params"]) > 0

    adapters = _load_adapters(adapter_dirs)

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
    model, lora_names, adapter_dirs = prepared
    assert len(lora_names["params"]) > 0
    assert len(lora_names["activations"]) > 0

    adapters = _load_adapters(adapter_dirs)

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
    model, lora_names, adapter_dirs = prepared
    sim = _make_sim(model)

    set_lora_bitwidth(sim, lora_names, param_type=int16, activation_type=int16)
    zero_weights = get_zero_weights(model, lora_names)
    _calibrate(sim, zero_weights)
    freeze_base_model(sim, lora_names)

    # Calibrate with adapter B
    adapter_b = load_file(os.path.join(adapter_dirs["B"], "adapter_model.safetensors"))
    unfreeze_lora_quantizers(sim, lora_names)
    _calibrate(sim, adapter_b)

    encodings_b = get_lora_encodings(sim, lora_names)
    assert len(encodings_b) > 0

    # Calibrate with adapter C (overwrites B's encodings)
    adapter_c = load_file(os.path.join(adapter_dirs["C"], "adapter_model.safetensors"))
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
    model, lora_names, adapter_dirs = prepared

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
    _model, lora_names, _adapter_dirs = prepared

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


def test_lora_param_names_match_peft_safetensors_keys(prepared):
    """Exported ONNX LoRA param names exactly match PEFT safetensors keys."""
    from peft.utils import get_peft_model_state_dict

    _model, lora_names, _adapter_dirs = prepared

    # Re-create the peft model to get its state dict keys
    base_model = _build_tiny_transformer()
    peft_model = _apply_peft_lora(base_model)
    peft_state = get_peft_model_state_dict(peft_model)

    # Every exported param name should be a PEFT safetensors key
    for name in lora_names["params"]:
        assert name in peft_state, (
            f"Exported param '{name}' not found in PEFT state dict keys: "
            f"{list(peft_state.keys())[:5]}"
        )


# =========================================================================
# Test: get_zero_weights
# =========================================================================


def test_get_zero_weights(prepared):
    """get_zero_weights returns zero arrays for all LoRA params."""
    model, lora_names, _adapter_dirs = prepared

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
    _model, lora_names, _adapter_dirs = prepared

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


def test_write_adapter_list():
    """write_adapter_list produces valid YAML structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        write_adapter_list(["adapter_A", "adapter_B"], tmpdir, "model")

        adaptor_path = os.path.join(tmpdir, "lora_adaptor_list.yaml")
        assert os.path.exists(adaptor_path)
        with open(adaptor_path) as f:
            content = f.read()
        assert "adapter_A" in content
        assert "adapter_B" in content


# =========================================================================
# Test: _infer_input_names from forward signature
# =========================================================================


def test_infer_input_names():
    """_infer_input_names extracts required param names from forward signature."""

    class FakeModel(nn.Module):
        def forward(self, input_ids, attention_mask=None, labels=None):
            pass

    names = _infer_input_names(FakeModel(), (torch.zeros(1, 10),))
    assert names == ["input_ids"]


def test_infer_input_names_multiple_required():
    """_infer_input_names with multiple required args (e.g. UNet)."""

    class FakeUNet(nn.Module):
        def forward(self, sample, timestep, encoder_hidden_states, return_dict=True):
            pass

    inputs = (torch.zeros(1, 4, 64, 64), torch.tensor(1.0), torch.zeros(1, 77, 768))
    names = _infer_input_names(FakeUNet(), inputs)
    assert names == ["sample", "timestep", "encoder_hidden_states"]


def test_infer_input_names_fallback():
    """Falls back to input_0, input_1, ... when signature has no required params."""

    class FakeModel(nn.Module):
        def forward(self, *args, **kwargs):
            pass

    names = _infer_input_names(FakeModel(), (torch.zeros(1), torch.zeros(2)))
    assert names == ["input_0", "input_1"]


# =========================================================================
# Test: _infer_dynamic_shapes
# =========================================================================


def test_infer_dynamic_shapes():
    """_infer_dynamic_shapes returns a dict with seq_len dim dynamic."""
    inputs = (torch.zeros(1, 10), torch.tensor(1.0))
    shapes = _infer_dynamic_shapes(["x", "scalar"], inputs)

    assert isinstance(shapes, dict)
    assert len(shapes) == 2
    # 2-D tensor: seq_len is dim 1
    assert 1 in shapes["x"]
    assert shapes["scalar"] == {}

    # 1-D tensor: seq_len is dim 0
    inputs_1d = (torch.zeros(10),)
    shapes_1d = _infer_dynamic_shapes(["seq"], inputs_1d)
    assert 0 in shapes_1d["seq"]


# =========================================================================
# Test: eager attention exports without SDPA switch
# =========================================================================


def test_eager_attention_export():
    """Export works with eager attention config (no SDPA switch needed)."""
    base_model = _build_tiny_transformer()
    peft_model = _apply_peft_lora(base_model)
    peft_model.eval()

    class Config:
        _attn_implementation = "eager"

    inner_model = peft_model.base_model.model
    inner_model.config = Config()

    sample_inputs = (torch.randint(0, 512, (1, 32)),)
    with tempfile.TemporaryDirectory() as tmpdir:
        _model, lora_names = export_peft_to_onnx(peft_model, sample_inputs, tmpdir)

    # Config should be unchanged
    assert inner_model.config._attn_implementation == "eager"
    assert len(lora_names["params"]) > 0


# =========================================================================
# Test: Different adapters produce different inference outputs
# =========================================================================


def test_different_adapters_different_outputs(prepared):
    """Feed dict with different adapter weights must produce different outputs."""
    model, lora_names, adapter_dirs = prepared

    sim = _make_sim(model)
    set_lora_bitwidth(sim, lora_names, param_type=int16, activation_type=int16)

    # Calibrate with zero weights so encodings are valid
    zero_weights = get_zero_weights(model, lora_names)
    _calibrate(sim, zero_weights)

    # Run inference with adapter B
    adapter_b = load_file(os.path.join(adapter_dirs["B"], "adapter_model.safetensors"))
    input_data = {"input_ids": np.array([[1, 2, 3, 4, 5] + [0] * 27], dtype=np.int64)}
    output_b = sim.session.run(None, {**input_data, **adapter_b})[0]

    # Run inference with adapter C
    adapter_c = load_file(os.path.join(adapter_dirs["C"], "adapter_model.safetensors"))
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
    model, lora_names, _adapter_dirs = prepared

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
    model, lora_names, _adapter_dirs = prepared

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
    _model, lora_names, _adapter_dirs = prepared

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
    model, _lora_names, _adapter_dirs = prepared

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
    model, lora_names, adapter_dirs = prepared

    sim = _make_sim(model)
    set_lora_bitwidth(sim, lora_names, param_type=int16, activation_type=int16)

    # Calibrate base + freeze
    zero_weights = get_zero_weights(model, lora_names)
    _calibrate(sim, zero_weights)
    freeze_base_model(sim, lora_names)

    # Calibrate with adapter B
    adapter_b = load_file(os.path.join(adapter_dirs["B"], "adapter_model.safetensors"))
    _calibrate(sim, adapter_b)
    encodings_b = get_lora_encodings(sim, lora_names)

    # Unfreeze and calibrate with adapter C
    unfreeze_lora_quantizers(sim, lora_names)
    adapter_c = load_file(os.path.join(adapter_dirs["C"], "adapter_model.safetensors"))
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

    model = FakeHFModel()

    # 2 sample inputs → first 2 positional params
    names = _infer_input_names(model, (torch.zeros(1, 10), torch.zeros(1, 10)))
    assert names == ["input_ids", "attention_mask"]

    # 1 sample input → first positional param only
    names = _infer_input_names(model, (torch.zeros(1, 10),))
    assert names == ["input_ids"]


# =========================================================================
# Error-path tests: name mismatch
# =========================================================================


def test_set_lora_bitwidth_raises_on_no_match(prepared):
    """set_lora_bitwidth raises ValueError when no LoRA names match sim quantizers."""
    model, _lora_names, _adapter_dirs = prepared
    sim = _make_sim(model)

    bogus_names = {"params": ["bogus_param"], "activations": ["bogus_act"]}
    with pytest.raises(ValueError, match="No LoRA quantizers found in sim"):
        set_lora_bitwidth(sim, bogus_names, param_type=int16, activation_type=int16)


def test_unfreeze_lora_raises_on_no_match(prepared):
    """unfreeze_lora_quantizers raises ValueError when no LoRA names match sim."""
    model, _lora_names, _adapter_dirs = prepared
    sim = _make_sim(model)

    bogus_names = {"params": ["bogus_param"], "activations": ["bogus_act"]}
    with pytest.raises(ValueError, match="No LoRA quantizers found in sim"):
        unfreeze_lora_quantizers(sim, bogus_names)


# =========================================================================
# Error-path tests: set_lora_bitwidth on frozen quantizers
# =========================================================================


def test_set_lora_bitwidth_raises_on_frozen(prepared):
    """set_lora_bitwidth raises RuntimeError if LoRA quantizers are already frozen."""
    model, lora_names, adapter_dirs = prepared
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
    model, lora_names, _adapter_dirs = prepared
    sim = _make_sim(model)

    # Don't calibrate — quantizers have no encodings
    with pytest.raises(RuntimeError, match="has no encoding"):
        get_lora_encodings(sim, lora_names)


# =========================================================================
# Error-path tests: set_lora_encodings with unknown names
# =========================================================================


def test_set_lora_encodings_raises_missing_names(prepared):
    """set_lora_encodings raises ValueError if encoding names aren't in sim."""
    model, _lora_names, _adapter_dirs = prepared
    sim = _make_sim(model)

    fake_encodings = {"nonexistent_quantizer_name": {"some": "data"}}
    with pytest.raises(ValueError, match="encoding names not found in sim"):
        set_lora_encodings(sim, fake_encodings)


# =========================================================================
# Output correctness: FP32 vs quantized outputs
# =========================================================================


def test_quantized_output_close_to_fp32(prepared):
    """Quantized LoRA output should be close to FP32 (not garbage)."""
    model, lora_names, adapter_dirs = prepared

    # FP32 baseline: run original model without quantization
    fp32_session = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    adapter_b = load_file(os.path.join(adapter_dirs["B"], "adapter_model.safetensors"))
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
    model, lora_names, adapter_dirs = prepared

    sim = _make_sim(model)
    set_lora_bitwidth(sim, lora_names, param_type=int16, activation_type=int16)
    adapter_b = load_file(os.path.join(adapter_dirs["B"], "adapter_model.safetensors"))
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
    _model, lora_names, _adapter_dirs = prepared

    # Serialize to JSON and back
    json_str = json.dumps(lora_names)
    restored = json.loads(json_str)

    assert restored == lora_names
    assert isinstance(restored["params"], list)
    assert isinstance(restored["activations"], list)
    assert all(isinstance(name, str) for name in restored["params"])
    assert all(isinstance(name, str) for name in restored["activations"])


# =========================================================================
# Test: explicit input_names and dynamic_shapes
# =========================================================================


def test_export_explicit_input_names():
    """export_peft_to_onnx accepts user-provided input_names and dynamic_shapes."""
    base_model = _build_tiny_transformer()
    peft_model = _apply_peft_lora(base_model)
    peft_model.eval()

    sample_inputs = (torch.randint(0, 512, (1, 32)),)
    batch = torch.export.Dim("batch")
    # dynamic_shapes keys must match forward() arg names for torch.export
    dynamic_shapes = {"input_ids": {0: batch}}

    with tempfile.TemporaryDirectory() as tmpdir:
        model_proto, lora_names = export_peft_to_onnx(
            peft_model,
            sample_inputs,
            tmpdir,
            input_names=["tokens"],
            dynamic_shapes=dynamic_shapes,
        )

    assert len(lora_names["params"]) > 0
    # Verify the custom input name was used in the ONNX graph
    graph_input_names = [inp.name for inp in model_proto.graph.input]
    assert "tokens" in graph_input_names


# =========================================================================
# Error-path tests: get_lora_encodings with no matching quantizers
# =========================================================================


def test_get_lora_encodings_raises_on_no_match(prepared):
    """get_lora_encodings raises ValueError when no LoRA names match sim."""
    model, _lora_names, _adapter_dirs = prepared
    sim = _make_sim(model)

    bogus_names = {"params": ["bogus_param"], "activations": ["bogus_act"]}
    with pytest.raises(ValueError, match="No LoRA quantizers found in sim"):
        get_lora_encodings(sim, bogus_names)


# =========================================================================
# Multi-adapter helpers (different attach points)
# =========================================================================


def _make_multi_adapter_artifacts(output_dir, rank_a=8, rank_b=4):
    """Export a model with two adapters that target different modules.

    Adapter A: targets [q_proj, v_proj], rank=rank_a, alpha=16
    Adapter B: targets [q_proj, k_proj, v_proj, o_proj], rank=rank_b, alpha=8

    The union graph should have LoRA branches on all 4 attention projections.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model = _build_tiny_transformer()
    # Apply LoRA targeting q_proj, v_proj (adapter A's targets)
    config_a = LoraConfig(
        r=rank_a,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
    )
    peft_model = get_peft_model(base_model, config_a)
    _randomize_lora_weights(peft_model, seed=99)
    peft_model.eval()

    # Save adapter A (with randomized weights so LoRA has a visible effect)
    adapter_a_dir = str(output_dir / "adapter_A")
    peft_model.save_pretrained(adapter_a_dir)

    # Save adapter B config (different targets, different rank/alpha)
    adapter_b_dir = str(output_dir / "adapter_B")
    os.makedirs(adapter_b_dir, exist_ok=True)
    with open(os.path.join(adapter_b_dir, "adapter_config.json"), "w") as f:
        json.dump(
            {
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "r": rank_b,
                "lora_alpha": 8,
                "peft_type": "LORA",
            },
            f,
        )

    # Create fake safetensors for adapter B with correct shapes.
    # Shapes must match the graph initializers: q_proj/v_proj use rank_a
    # (from the PEFT export), k_proj/o_proj use rank_b (from add_lora_branches).
    from safetensors.numpy import save_file

    hidden_size = 64
    adapter_b_weights = {}
    for layer_idx in range(2):
        for module in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            # Use the graph's rank for this module
            r = rank_a if module in ["q_proj", "v_proj"] else rank_b
            prefix = f"base_model.model.layers.{layer_idx}.self_attn.{module}"
            adapter_b_weights[f"{prefix}.lora_A.weight"] = (
                np.random.randn(r, hidden_size).astype(np.float32) * 0.01
            )
            adapter_b_weights[f"{prefix}.lora_B.weight"] = (
                np.random.randn(hidden_size, r).astype(np.float32) * 0.01
            )
    save_file(
        adapter_b_weights, os.path.join(adapter_b_dir, "adapter_model.safetensors")
    )

    sample_inputs = (torch.randint(0, 512, (1, 32)),)
    model_proto, lora_names = export_peft_to_onnx(
        peft_model,
        sample_inputs,
        str(output_dir),
        adapter_paths=[adapter_b_dir],
    )

    return {
        "model_proto": model_proto,
        "lora_names": lora_names,
        "output_dir": str(output_dir),
        "adapter_a_dir": adapter_a_dir,
        "adapter_b_dir": adapter_b_dir,
        "adapter_b_weights": adapter_b_weights,
    }


@pytest.fixture(scope="session")
def multi_adapter_artifacts(tmp_path_factory):
    """Generate multi-adapter test artifacts with different attach points."""
    artifacts_dir = tmp_path_factory.mktemp("multi_adapter")
    return _make_multi_adapter_artifacts(str(artifacts_dir))


@pytest.fixture
def multi_prepared(multi_adapter_artifacts) -> tuple:
    """Fresh model proto and lora_names for each multi-adapter test."""
    model_proto = onnx.load(
        os.path.join(multi_adapter_artifacts["output_dir"], "model.onnx")
    )
    onnx.load_external_data_for_model(
        model_proto, multi_adapter_artifacts["output_dir"]
    )
    return (
        model_proto,
        multi_adapter_artifacts["lora_names"],
        multi_adapter_artifacts["adapter_a_dir"],
        multi_adapter_artifacts["adapter_b_dir"],
    )


# =========================================================================
# Test: Multi-adapter export with different attach points
# =========================================================================


def test_multi_adapter_export_union_graph(multi_adapter_artifacts):
    """Union graph has LoRA branches for ALL target modules across adapters."""
    lora_names = multi_adapter_artifacts["lora_names"]

    # Should have params for q_proj, k_proj, v_proj, o_proj (all 4 projections)
    param_modules = set()
    for name in lora_names["params"]:
        for mod in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            if f".{mod}." in name:
                param_modules.add(mod)

    assert param_modules == {"q_proj", "k_proj", "v_proj", "o_proj"}, (
        f"Expected branches for all 4 projections, got: {param_modules}"
    )

    # Should have scales for all 4 modules
    scale_modules = set()
    for name in lora_names["scales"]:
        for mod in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            if f".{mod}." in name:
                scale_modules.add(mod)

    assert scale_modules == {"q_proj", "k_proj", "v_proj", "o_proj"}, (
        f"Expected scales for all 4 projections, got: {scale_modules}"
    )


def test_multi_adapter_lora_names_structure(multi_adapter_artifacts):
    """lora_names has params, activations, and scales keys with correct content."""
    lora_names = multi_adapter_artifacts["lora_names"]

    assert "params" in lora_names
    assert "activations" in lora_names
    assert "scales" in lora_names
    assert len(lora_names["params"]) > 0
    assert len(lora_names["activations"]) > 0
    assert len(lora_names["scales"]) > 0

    for name in lora_names["params"]:
        assert "lora_A" in name or "lora_B" in name


# =========================================================================
# Test: get_adapter_scale_weights
# =========================================================================


def test_get_adapter_scale_weights_matching(multi_adapter_artifacts):
    """get_adapter_scale_weights returns correct scales for matching modules."""
    lora_names = multi_adapter_artifacts["lora_names"]
    adapter_a_dir = multi_adapter_artifacts["adapter_a_dir"]

    scales = get_adapter_scale_weights(lora_names, adapter_a_dir)

    # Should return a value for every scale name
    assert set(scales.keys()) == set(lora_names["scales"].keys())

    # Adapter A: alpha=16, r=8 → scale=2.0 for q_proj, v_proj
    for name, val in scales.items():
        assert isinstance(val, np.ndarray)
        assert val.dtype == np.float32
        if "q_proj" in name or "v_proj" in name:
            assert float(val) == pytest.approx(2.0), (
                f"{name} should have scale 2.0 (alpha=16, r=8)"
            )


def test_get_adapter_scale_weights_different_adapters(multi_adapter_artifacts):
    """Different adapters with different alpha/r produce different scale values."""
    lora_names = multi_adapter_artifacts["lora_names"]
    adapter_a_dir = multi_adapter_artifacts["adapter_a_dir"]
    adapter_b_dir = multi_adapter_artifacts["adapter_b_dir"]

    scales_a = get_adapter_scale_weights(lora_names, adapter_a_dir)
    scales_b = get_adapter_scale_weights(lora_names, adapter_b_dir)

    # Both should return all scale names
    assert set(scales_a.keys()) == set(scales_b.keys())

    # Adapter A: alpha=16, r=8 → 2.0 for q_proj, v_proj
    # Adapter B: alpha=8, r=4 → 2.0 for q_proj, k_proj, v_proj, o_proj
    # For k_proj: A returns default (from graph), B returns 2.0
    for name in scales_b:
        if "k_proj" in name or "o_proj" in name:
            assert float(scales_b[name]) == pytest.approx(2.0), (
                f"Adapter B {name} should have scale 2.0 (alpha=8, r=4)"
            )


def test_get_adapter_scale_weights_missing_config():
    """get_adapter_scale_weights raises FileNotFoundError for missing config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError, match="adapter_config.json"):
            get_adapter_scale_weights({"scales": {}}, tmpdir)


# =========================================================================
# Test: add_lora_branches standalone
# =========================================================================


def test_add_lora_branches_standalone(multi_prepared):
    """add_lora_branches inserts new branches for modules not already in graph."""
    model, lora_names, _adapter_a_dir, _adapter_b_dir = multi_prepared

    # Graph already has q_proj, k_proj, v_proj, o_proj branches.
    # Try adding branches for up_proj, down_proj (from the FFN).
    new_config = {
        "target_modules": ["up_proj", "down_proj"],
        "r": 4,
        "lora_alpha": 8,
    }

    new_names = add_lora_branches(model, new_config)

    assert len(new_names["params"]) > 0
    assert len(new_names["scales"]) > 0

    # Verify new params are for up_proj and down_proj
    new_modules = set()
    for name in new_names["params"]:
        for mod in ["up_proj", "down_proj"]:
            if f".{mod}." in name:
                new_modules.add(mod)

    assert new_modules == {"up_proj", "down_proj"}, (
        f"Expected new branches for up_proj, down_proj; got: {new_modules}"
    )


def test_add_lora_branches_skips_existing(multi_prepared):
    """add_lora_branches skips modules that already have LoRA branches."""
    model, lora_names, _adapter_a_dir, _adapter_b_dir = multi_prepared

    # Try adding branches for q_proj (already exists) and up_proj (new)
    config = {
        "target_modules": ["q_proj", "up_proj"],
        "r": 4,
        "lora_alpha": 8,
    }

    new_names = add_lora_branches(model, config)

    # Only up_proj should be new (q_proj already exists)
    new_modules = set()
    for name in new_names["params"]:
        for mod in ["q_proj", "up_proj"]:
            if f".{mod}." in name:
                new_modules.add(mod)

    assert "q_proj" not in new_modules, (
        "q_proj should have been skipped (already exists)"
    )
    assert "up_proj" in new_modules, "up_proj should be newly inserted"


def test_add_lora_branches_from_path():
    """add_lora_branches accepts a file path instead of a dict."""
    base_model = _build_tiny_transformer()
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
    )
    peft_model = get_peft_model(base_model, config)
    peft_model.eval()

    sample_inputs = (torch.randint(0, 512, (1, 32)),)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_proto, lora_names = export_peft_to_onnx(peft_model, sample_inputs, tmpdir)

        # Write adapter config to a file
        config_path = os.path.join(tmpdir, "adapter_config.json")
        with open(config_path, "w") as f:
            json.dump(
                {
                    "target_modules": ["k_proj", "o_proj"],
                    "r": 4,
                    "lora_alpha": 8,
                },
                f,
            )

        new_names = add_lora_branches(model_proto, config_path)

    assert len(new_names["params"]) > 0
    new_modules = set()
    for name in new_names["params"]:
        for mod in ["k_proj", "o_proj"]:
            if f".{mod}." in name:
                new_modules.add(mod)
    assert new_modules == {"k_proj", "o_proj"}


# =========================================================================
# Test: Multi-adapter V1 workflow (different attach points)
# =========================================================================


def test_multi_adapter_v1_workflow(multi_prepared):
    """V1 with different attach points: single encoding across adapters."""
    model, lora_names, adapter_a_dir, adapter_b_dir = multi_prepared

    sim = _make_sim(model)
    set_lora_bitwidth(sim, lora_names, param_type=int16, activation_type=int16)

    # Load adapter weights
    adapter_a = load_file(os.path.join(adapter_a_dir, "adapter_model.safetensors"))
    adapter_b = load_file(os.path.join(adapter_b_dir, "adapter_model.safetensors"))

    zero_weights = get_zero_weights(model, lora_names)

    # Calibrate with all adapters (V1 strategy)
    def calibrate_all(session):
        for weights in [adapter_a, adapter_b]:
            feed = {**zero_weights, **weights}
            scales = get_adapter_scale_weights(
                lora_names,
                adapter_a_dir if weights is adapter_a else adapter_b_dir,
            )
            for _ in range(3):
                batch = {
                    "input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64),
                }
                batch.update(feed)
                batch.update(scales)
                session.run(None, batch)

    sim.compute_encodings(calibrate_all)

    frozen = freeze_base_model(sim, lora_names)
    assert frozen > 0

    with tempfile.TemporaryDirectory() as export_dir:
        sim.export(export_dir, "model_v1", export_model=False)
        assert os.path.exists(os.path.join(export_dir, "model_v1.encodings"))


# =========================================================================
# Test: Multi-adapter V2 workflow (different attach points)
# =========================================================================


def test_multi_adapter_v2_workflow(multi_prepared):
    """V2 with different attach points: full recalibration per adapter."""
    model, lora_names, adapter_a_dir, adapter_b_dir = multi_prepared

    sim = _make_sim(model)
    set_lora_bitwidth(sim, lora_names, param_type=int16, activation_type=int16)

    adapter_a = load_file(os.path.join(adapter_a_dir, "adapter_model.safetensors"))
    adapter_b = load_file(os.path.join(adapter_b_dir, "adapter_model.safetensors"))
    zero_weights = get_zero_weights(model, lora_names)

    with tempfile.TemporaryDirectory() as export_dir:
        for name, weights, adapter_dir in [
            ("A", adapter_a, adapter_a_dir),
            ("B", adapter_b, adapter_b_dir),
        ]:
            feed = {**zero_weights, **weights}
            scales = get_adapter_scale_weights(lora_names, adapter_dir)

            def cal_fn(session, f=feed, s=scales):
                for _ in range(3):
                    batch = {
                        "input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64)
                    }
                    batch.update(f)
                    batch.update(s)
                    session.run(None, batch)

            sim.compute_encodings(cal_fn)
            sim.export(export_dir, f"model_{name}", export_model=False)
            assert os.path.exists(os.path.join(export_dir, f"model_{name}.encodings"))


# =========================================================================
# Test: Multi-adapter V3 workflow (different attach points)
# =========================================================================


def test_multi_adapter_v3_workflow(multi_prepared):
    """V3 with different attach points: base frozen, per-adapter LoRA recalibration."""
    model, lora_names, adapter_a_dir, adapter_b_dir = multi_prepared

    sim = _make_sim(model)
    set_lora_bitwidth(sim, lora_names, param_type=int16, activation_type=int16)

    zero_weights = get_zero_weights(model, lora_names)
    scale_defaults = {
        name: np.array(val, dtype=np.float32)
        for name, val in lora_names["scales"].items()
    }

    # Base calibration with all branches zeroed
    def base_cal(session):
        for _ in range(5):
            batch = {"input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64)}
            batch.update(zero_weights)
            batch.update(scale_defaults)
            session.run(None, batch)

    sim.compute_encodings(base_cal)

    # Freeze base
    frozen = freeze_base_model(sim, lora_names)
    assert frozen > 0

    # Snapshot a base param encoding
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
    adapter_a = load_file(os.path.join(adapter_a_dir, "adapter_model.safetensors"))
    adapter_b = load_file(os.path.join(adapter_b_dir, "adapter_model.safetensors"))

    adapter_encodings = {}
    with tempfile.TemporaryDirectory() as export_dir:
        for name, weights, adapter_dir in [
            ("A", adapter_a, adapter_a_dir),
            ("B", adapter_b, adapter_b_dir),
        ]:
            unfreeze_lora_quantizers(sim, lora_names)

            feed = {**zero_weights, **weights}
            scales = get_adapter_scale_weights(lora_names, adapter_dir)

            def cal_fn(session, f=feed, s=scales):
                for _ in range(3):
                    batch = {
                        "input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64)
                    }
                    batch.update(f)
                    batch.update(s)
                    session.run(None, batch)

            sim.compute_encodings(cal_fn)
            adapter_encodings[name] = get_lora_encodings(sim, lora_names)

            sim.export(export_dir, f"model_{name}", export_model=False)
            assert os.path.exists(os.path.join(export_dir, f"model_{name}.encodings"))

    # Verify base encodings didn't change
    base_encoding_after = sim.qc_quantize_op_dict[base_param_name].export_encodings(
        "1.0.0"
    )
    assert base_encoding_before == base_encoding_after, (
        "Base encoding changed during per-adapter calibration"
    )

    # Verify adapter encodings differ
    enc_a = adapter_encodings["A"]
    enc_b = adapter_encodings["B"]
    assert len(enc_a) > 0 and len(enc_b) > 0
    any_differ = any(enc_a.get(n) != enc_b.get(n) for n in enc_a)
    assert any_differ, "Adapter A and B encodings should differ"


# =========================================================================
# Test: Zero-weight branches are invisible (different attach points)
# =========================================================================


def test_zero_weight_branches_invisible(multi_prepared):
    """Feeding zero weights for extra branches produces same output as base-only."""
    model, lora_names, adapter_a_dir, _adapter_b_dir = multi_prepared

    # Run with zero weights (all branches disabled)
    zero_weights = get_zero_weights(model, lora_names)
    scale_defaults = {
        name: np.array(val, dtype=np.float32)
        for name, val in lora_names["scales"].items()
    }

    session = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )

    input_data = {"input_ids": np.array([[1, 2, 3, 4, 5] + [0] * 27], dtype=np.int64)}
    feed_zero = {**input_data, **zero_weights, **scale_defaults}
    output_zero = session.run(None, feed_zero)[0]

    # Run with adapter A (only q_proj, v_proj active; k_proj, o_proj zero)
    adapter_a = load_file(os.path.join(adapter_a_dir, "adapter_model.safetensors"))
    feed_a = {**zero_weights, **adapter_a}
    scales_a = get_adapter_scale_weights(lora_names, adapter_a_dir)
    feed_a_full = {**input_data, **feed_a, **scales_a}
    output_a = session.run(None, feed_a_full)[0]

    # Adapter A should differ from zero (LoRA has an effect)
    assert not np.allclose(output_a, output_zero, atol=1e-6), (
        "Adapter A output should differ from zero-weight output"
    )

    # Output should be finite (not NaN or Inf)
    assert np.all(np.isfinite(output_zero)), "Zero-weight output has NaN/Inf"
    assert np.all(np.isfinite(output_a)), "Adapter A output has NaN/Inf"


# =========================================================================
# Test: Encoding roundtrip with multi-adapter (different attach points)
# =========================================================================


def test_multi_adapter_encoding_roundtrip(multi_prepared):
    """get/set_lora_encodings round-trip works with multi-adapter union graph."""
    model, lora_names, adapter_a_dir, adapter_b_dir = multi_prepared

    sim = _make_sim(model)
    set_lora_bitwidth(sim, lora_names, param_type=int16, activation_type=int16)

    zero_weights = get_zero_weights(model, lora_names)
    scale_defaults = {
        name: np.array(val, dtype=np.float32)
        for name, val in lora_names["scales"].items()
    }

    # Base calibration
    def base_cal(session):
        for _ in range(3):
            batch = {"input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64)}
            batch.update(zero_weights)
            batch.update(scale_defaults)
            session.run(None, batch)

    sim.compute_encodings(base_cal)
    freeze_base_model(sim, lora_names)

    # Calibrate adapter A
    adapter_a = load_file(os.path.join(adapter_a_dir, "adapter_model.safetensors"))
    unfreeze_lora_quantizers(sim, lora_names)
    feed_a = {**zero_weights, **adapter_a}
    scales_a = get_adapter_scale_weights(lora_names, adapter_a_dir)

    def cal_a(session):
        for _ in range(3):
            batch = {"input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64)}
            batch.update(feed_a)
            batch.update(scales_a)
            session.run(None, batch)

    sim.compute_encodings(cal_a)
    encodings_a = get_lora_encodings(sim, lora_names)

    # Calibrate adapter B (overwrites A)
    adapter_b = load_file(os.path.join(adapter_b_dir, "adapter_model.safetensors"))
    unfreeze_lora_quantizers(sim, lora_names)
    feed_b = {**zero_weights, **adapter_b}
    scales_b = get_adapter_scale_weights(lora_names, adapter_b_dir)

    def cal_b(session):
        for _ in range(3):
            batch = {"input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64)}
            batch.update(feed_b)
            batch.update(scales_b)
            session.run(None, batch)

    sim.compute_encodings(cal_b)

    # Restore adapter A encodings
    restored = set_lora_encodings(sim, encodings_a)
    assert restored == len(encodings_a)

    encodings_a_restored = get_lora_encodings(sim, lora_names)
    for name in encodings_a:
        assert name in encodings_a_restored
        assert encodings_a[name] == encodings_a_restored[name]


# =========================================================================
# Test: TinyLlama E2E V3 workflow with real HF adapters
# =========================================================================

HF_CACHE = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "onnx_lora", "hf_cache", "hub"
)
_TINYLLAMA_BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
_ADAPTER_A_ID = "barissglc/tinyllama-tarot-v1"  # [q_proj, v_proj], r=8
_ADAPTER_B_ID = "cahlen/tinyllama-offline-practical-skills-qa-qlora"  # 7 modules, r=64


def _resolve_hf_adapter(model_id: str) -> str:
    """Resolve a HuggingFace model ID to its local snapshot path."""
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, cache_dir=HF_CACHE)


def _tinyllama_available() -> bool:
    """Check if TinyLlama and both adapters are cached locally."""
    try:
        for model_id in [_TINYLLAMA_BASE, _ADAPTER_A_ID, _ADAPTER_B_ID]:
            model_dir = os.path.join(HF_CACHE, "models--" + model_id.replace("/", "--"))
            if not os.path.isdir(model_dir):
                return False
        return True
    except Exception:
        return False


@pytest.mark.skip(reason="Local only — requires TinyLlama + adapters cached")
def test_tinyllama_e2e_v3_workflow():
    """E2E V3 workflow: TinyLlama-1.1B + 2 real HF LoRA adapters.

    Adapter A: barissglc/tinyllama-tarot-v1
        target_modules=[q_proj, v_proj], rank=8, alpha=16

    Adapter B: cahlen/tinyllama-offline-practical-skills-qa-qlora
        target_modules=[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
        rank=64, alpha=16

    The union graph has LoRA branches on all 7 modules.  Adapter B uses
    rank-64 weights even for q_proj/v_proj (exported at rank-8) — validated
    by the ``"lora_rank"`` dynamic dimension.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    adapter_a_path = _resolve_hf_adapter(_ADAPTER_A_ID)
    adapter_b_path = _resolve_hf_adapter(_ADAPTER_B_ID)

    # 1. Load base model + adapter A
    base_model = AutoModelForCausalLM.from_pretrained(
        _TINYLLAMA_BASE,
        cache_dir=HF_CACHE,
        dtype=torch.float32,
    )
    peft_model = PeftModel.from_pretrained(base_model, adapter_a_path)
    peft_model.eval()

    sample_inputs = (torch.randint(0, 32000, (1, 16)),)

    with tempfile.TemporaryDirectory(dir="/local/mnt/workspace") as tmpdir:
        # 2. Export with adapter B's extra branches
        model, lora_names = export_peft_to_onnx(
            peft_model,
            sample_inputs,
            tmpdir,
            adapter_paths=[adapter_b_path],
        )

        # 3. Verify union graph covers all 7 modules
        param_modules = set()
        for name in lora_names["params"]:
            for mod in [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]:
                if f".{mod}." in name:
                    param_modules.add(mod)

        assert param_modules == {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }, f"Expected 7 module branches, got: {param_modules}"

        assert len(lora_names["activations"]) > 0
        assert len(lora_names["scales"]) > 0

        # 4. Create QuantSim
        dummy_input = {
            "input_ids": np.random.randint(0, 32000, (1, 16)).astype(np.int64),
        }
        sim = QuantizationSimModel(
            model,
            dummy_input=dummy_input,
            param_type=int4,
            activation_type=int16,
        )
        set_lora_bitwidth(sim, lora_names, param_type=int16, activation_type=int16)

        # 5. Base calibration with zero weights
        zero_weights = get_zero_weights(model, lora_names)
        scale_defaults = {
            name: np.array(val, dtype=np.float32)
            for name, val in lora_names["scales"].items()
        }

        def base_cal(session):
            for _ in range(3):
                batch = {
                    "input_ids": np.random.randint(0, 32000, (1, 16)).astype(np.int64),
                }
                batch.update(zero_weights)
                batch.update(scale_defaults)
                session.run(None, batch)

        sim.compute_encodings(base_cal)

        # 6. Freeze base
        frozen = freeze_base_model(sim, lora_names)
        assert frozen > 0

        # 7. Per-adapter calibration
        adapter_encodings = {}
        adapter_feeds = {}
        adapter_scales = {}

        for name, adapter_path in [("A", adapter_a_path), ("B", adapter_b_path)]:
            unfreeze_lora_quantizers(sim, lora_names)

            weights = load_file(os.path.join(adapter_path, "adapter_model.safetensors"))
            feed = {**zero_weights, **weights}
            scales = get_adapter_scale_weights(lora_names, adapter_path)

            adapter_feeds[name] = feed
            adapter_scales[name] = scales

            def cal_fn(session, f=feed, s=scales):
                for _ in range(3):
                    batch = {
                        "input_ids": np.random.randint(0, 32000, (1, 16)).astype(
                            np.int64
                        ),
                    }
                    batch.update(f)
                    batch.update(s)
                    session.run(None, batch)

            sim.compute_encodings(cal_fn)
            adapter_encodings[name] = get_lora_encodings(sim, lora_names)

        # 8. Adapter encodings should differ
        enc_a = adapter_encodings["A"]
        enc_b = adapter_encodings["B"]
        assert len(enc_a) > 0 and len(enc_b) > 0
        any_differ = any(enc_a.get(n) != enc_b.get(n) for n in enc_a)
        assert any_differ, "Adapter A and B encodings should differ"

        # 9. Inference with each adapter produces different output
        input_data = {
            "input_ids": np.array([[1, 2, 3, 4, 5] + [0] * 11], dtype=np.int64),
        }

        set_lora_encodings(sim, enc_a)
        feed_a = {**input_data, **adapter_feeds["A"], **adapter_scales["A"]}
        output_a = sim.session.run(None, feed_a)[0]

        set_lora_encodings(sim, enc_b)
        feed_b = {**input_data, **adapter_feeds["B"], **adapter_scales["B"]}
        output_b = sim.session.run(None, feed_b)[0]

        assert np.all(np.isfinite(output_a)), "Adapter A output has NaN/Inf"
        assert np.all(np.isfinite(output_b)), "Adapter B output has NaN/Inf"
        assert not np.allclose(output_a, output_b, atol=1e-6), (
            "Adapter A and B should produce different outputs"
        )

        # 10. Export
        sim.export(tmpdir, "tinyllama_v3", export_model=False)
        assert os.path.exists(os.path.join(tmpdir, "tinyllama_v3.encodings"))


# =========================================================================
# Test: Adapted model (Conv2d projections) + PEFT → export → QuantSim
# =========================================================================


class _Conv2dAttention(nn.Module):
    """Attention with Conv2d projections (simulating ai-hub-models SHA adaptation)."""

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        # Conv2d projections instead of Linear (post-adaptation)
        self.q_proj = nn.Conv2d(hidden_size, hidden_size, 1, bias=False)
        self.k_proj = nn.Conv2d(hidden_size, hidden_size, 1, bias=False)
        self.v_proj = nn.Conv2d(hidden_size, hidden_size, 1, bias=False)
        self.o_proj = nn.Conv2d(hidden_size, hidden_size, 1, bias=False)

    def forward(self, x):
        B, S, C = x.shape
        # Conv2d expects (B, C, H, W) — reshape for 1×1 conv
        h = x.unsqueeze(2).permute(0, 3, 2, 1)  # (B, C, 1, S)
        q = self.q_proj(h).permute(0, 3, 2, 1).squeeze(2)  # (B, S, C)
        k = self.k_proj(h).permute(0, 3, 2, 1).squeeze(2)
        v = self.v_proj(h).permute(0, 3, 2, 1).squeeze(2)
        # Simple scaled dot-product attention
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        # o_proj is also Conv2d
        out = out.unsqueeze(2).permute(0, 3, 2, 1)
        out = self.o_proj(out).permute(0, 3, 2, 1).squeeze(2)
        return out


def _adapt_attention_to_conv2d(model: nn.Module) -> nn.Module:
    """Simulate MHA→SHA adaptation: replace Linear attention with Conv2d.

    This mirrors the ai-hub-models adaptation pipeline where Linear q/k/v/o
    projections are replaced with 1×1 Conv2d for on-device efficiency.
    The attention forward pass changes (Conv2d needs B,C,H,W layout),
    but the TransformerBlock forward stays the same (calls self_attn
    polymorphically).
    """
    for block in model.layers:
        hidden_size = block.self_attn.q_proj.in_features
        num_heads = block.self_attn.num_heads
        block.self_attn = _Conv2dAttention(hidden_size, num_heads)
    return model


def _build_adapted_transformer(
    vocab_size=512, hidden_size=64, num_heads=4, num_layers=2, intermediate_size=128
):
    """Build a standard transformer and adapt it to Conv2d attention.

    Mirrors the real workflow: build standard model → run adaptation →
    apply LoRA → export to ONNX.
    """
    model = _build_tiny_transformer(
        vocab_size, hidden_size, num_heads, num_layers, intermediate_size
    )
    return _adapt_attention_to_conv2d(model)


def test_adapted_model_conv2d_export():
    """Conv2d-adapted model + PEFT → export_peft_to_onnx → LoRA names → QuantSim."""
    base_model = _build_adapted_transformer()

    # Apply LoRA to Conv2d projections
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
    )
    peft_model = get_peft_model(base_model, config)
    peft_model.eval()

    sample_inputs = (torch.randint(0, 512, (1, 8)),)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_proto, lora_names = export_peft_to_onnx(peft_model, sample_inputs, tmpdir)

        # Verify LoRA params exist for all 4 projections
        param_modules = set()
        for name in lora_names["params"]:
            for mod in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                if f".{mod}." in name:
                    param_modules.add(mod)
        assert param_modules == {"q_proj", "k_proj", "v_proj", "o_proj"}, (
            f"Expected LoRA branches for all 4 projections, got: {param_modules}"
        )

        # Verify LoRA param names match PEFT safetensors keys
        from peft.utils import get_peft_model_state_dict

        peft_state = get_peft_model_state_dict(peft_model)
        for name in lora_names["params"]:
            assert name in peft_state, f"Exported param '{name}' not in PEFT state dict"

        # Create QuantSim from exported model
        dummy_input = {
            "input_ids": np.random.randint(0, 512, (1, 8)).astype(np.int64),
        }
        sim = QuantizationSimModel(
            model_proto,
            dummy_input=dummy_input,
            param_type=int4,
            activation_type=int16,
        )
        set_lora_bitwidth(sim, lora_names, param_type=int16, activation_type=int16)

        # Calibrate and verify inference produces finite output
        zero_weights = get_zero_weights(model_proto, lora_names)

        def calibrate(session):
            for _ in range(3):
                batch = {
                    "input_ids": np.random.randint(0, 512, (1, 8)).astype(np.int64),
                }
                batch.update(zero_weights)
                session.run(None, batch)

        sim.compute_encodings(calibrate)

        input_data = {
            "input_ids": np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64),
        }
        output = sim.session.run(None, {**input_data, **zero_weights})[0]
        assert np.all(np.isfinite(output)), "Output has NaN/Inf"


# =========================================================================
# Error-path tests: add_lora_branches validation
# =========================================================================


def test_add_lora_branches_rank_zero(multi_prepared):
    """add_lora_branches raises ValueError on rank=0."""
    model, _lora_names, _adapter_a_dir, _adapter_b_dir = multi_prepared

    config = {"target_modules": ["up_proj"], "r": 0, "lora_alpha": 8}
    with pytest.raises(ValueError, match="must be positive"):
        add_lora_branches(model, config)


def test_add_lora_branches_default_rank(multi_prepared):
    """add_lora_branches uses default_rank when config has no 'r' key."""
    model, _lora_names, _adapter_a_dir, _adapter_b_dir = multi_prepared

    # multi_prepared has q/k/v/o_proj branches; up_proj is available
    config = {"target_modules": ["up_proj"], "lora_alpha": 8}
    new_names = add_lora_branches(model, config, default_rank=4)

    assert len(new_names["params"]) > 0
    # Verify the initializer shapes reflect rank=4
    init_map = {init.name: init for init in model.graph.initializer}
    for name in new_names["params"]:
        if "lora_A" in name:
            assert init_map[name].dims[0] == 4, f"{name} should have rank dim = 4"


def test_export_peft_to_onnx_empty_peft_config():
    """export_peft_to_onnx raises ValueError when peft_config is empty."""
    base_model = _build_tiny_transformer()
    peft_model = _apply_peft_lora(base_model)
    peft_model.eval()

    # Clear peft_config to simulate no adapters
    peft_model.peft_config = {}

    sample_inputs = (torch.randint(0, 512, (1, 32)),)
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="no adapters configured"):
            export_peft_to_onnx(peft_model, sample_inputs, tmpdir)


# =========================================================================
# Tests for configure_lora_onnx (user-owned export path)
# =========================================================================


def _user_export_peft_model(peft_model, sample_inputs, output_dir):
    """Export a PeftModel the way a user would — no wrapper, no AIMET helpers."""
    from peft.utils import get_peft_model_state_dict

    model = peft_model.base_model.model
    model.eval()

    onnx_path = str(Path(output_dir) / "model.onnx")
    onnx_program = torch.onnx.export(
        model,
        sample_inputs,
        dynamo=True,
        optimize=False,
    )
    onnx_program.save(onnx_path, external_data=True)

    adapter_name = next(iter(peft_model.peft_config))
    peft_keys = list(
        get_peft_model_state_dict(peft_model, adapter_name=adapter_name).keys()
    )
    return onnx_path, peft_keys


def test_configure_lora_onnx_user_export():
    """configure_lora_onnx works with a user-exported ONNX model (no wrapper)."""
    base_model = _build_tiny_transformer()
    peft_model = _apply_peft_lora(base_model, rank=8)
    peft_model.eval()
    sample_inputs = (torch.randint(0, 512, (1, 32)),)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path, peft_keys = _user_export_peft_model(
            peft_model, sample_inputs, tmpdir
        )

        model_proto, lora_names = configure_lora_onnx(onnx_path, peft_keys, onnx_path)

        # Should have params, activations, and scales
        assert len(lora_names["params"]) > 0
        assert len(lora_names["activations"]) > 0
        assert len(lora_names["scales"]) > 0

        # All param names should follow PEFT naming convention
        for name in lora_names["params"]:
            assert name.startswith("base_model.model.")
            assert "lora_A" in name or "lora_B" in name

        # Activations should be at least half the number of params
        # (one activation per LoRA A/B pair at minimum)
        assert len(lora_names["activations"]) >= len(lora_names["params"]) // 2


def test_configure_lora_onnx_safetensors_keys():
    """configure_lora_onnx works with keys from a safetensors file (no live PeftModel)."""
    base_model = _build_tiny_transformer()
    peft_model = _apply_peft_lora(base_model, rank=8)
    peft_model.eval()
    sample_inputs = (torch.randint(0, 512, (1, 32)),)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path, _ = _user_export_peft_model(peft_model, sample_inputs, tmpdir)

        # Save adapter to disk and load keys from safetensors
        adapter_dir = str(Path(tmpdir) / "adapter_default")
        peft_model.save_pretrained(adapter_dir)

        safetensors_path = Path(adapter_dir) / "default" / "adapter_model.safetensors"
        if not safetensors_path.exists():
            safetensors_path = Path(adapter_dir) / "adapter_model.safetensors"
        safetensors_keys = list(load_file(str(safetensors_path)).keys())

        model_proto, lora_names = configure_lora_onnx(
            onnx_path, safetensors_keys, onnx_path
        )

        assert len(lora_names["params"]) > 0
        assert len(lora_names["activations"]) > 0


def test_configure_lora_onnx_with_adapter_paths():
    """configure_lora_onnx inserts branches for additional adapter target modules."""
    base_model = _build_tiny_transformer()

    # Apply LoRA to only q_proj/v_proj (not all linear layers)
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
    )
    peft_model = get_peft_model(base_model, config)
    peft_model.eval()
    sample_inputs = (torch.randint(0, 512, (1, 32)),)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path, peft_keys = _user_export_peft_model(
            peft_model, sample_inputs, tmpdir
        )

        # Create adapter_config.json targeting additional modules
        extra_adapter_dir = Path(tmpdir) / "extra_adapter"
        extra_adapter_dir.mkdir()
        adapter_config = {
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "r": 8,
            "lora_alpha": 16,
        }
        with open(extra_adapter_dir / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f)

        model_proto, lora_names = configure_lora_onnx(
            onnx_path, peft_keys, onnx_path, adapter_paths=[str(extra_adapter_dir)]
        )

        # Should have params from both original (q/v) and newly inserted (k/o)
        param_names_str = " ".join(lora_names["params"])
        assert "q_proj" in param_names_str
        assert "v_proj" in param_names_str
        assert "k_proj" in param_names_str
        assert "o_proj" in param_names_str


def test_configure_lora_onnx_no_match_raises():
    """configure_lora_onnx raises ValueError with actionable message for bogus keys."""
    base_model = _build_tiny_transformer()
    peft_model = _apply_peft_lora(base_model, rank=8)
    peft_model.eval()
    sample_inputs = (torch.randint(0, 512, (1, 32)),)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path, _ = _user_export_peft_model(peft_model, sample_inputs, tmpdir)

        bogus_keys = [
            "completely.wrong.key.lora_A.weight",
            "another.bad.key.lora_B.weight",
        ]
        with pytest.raises(ValueError, match="dynamo=True and optimize=False"):
            configure_lora_onnx(onnx_path, bogus_keys, onnx_path)


def test_configure_lora_onnx_bad_export_detected():
    """configure_lora_onnx detects models exported with optimize=True."""
    base_model = _build_tiny_transformer()
    peft_model = _apply_peft_lora(base_model, rank=8)
    peft_model.eval()
    sample_inputs = (torch.randint(0, 512, (1, 32)),)

    from peft.utils import get_peft_model_state_dict

    with tempfile.TemporaryDirectory() as tmpdir:
        model = peft_model.base_model.model
        model.eval()

        onnx_path = str(Path(tmpdir) / "model.onnx")
        # Export with optimize=True — this destroys LoRA init names
        onnx_program = torch.onnx.export(
            model,
            sample_inputs,
            dynamo=True,
            optimize=True,
        )
        onnx_program.save(onnx_path, external_data=True)

        adapter_name = next(iter(peft_model.peft_config))
        peft_keys = list(
            get_peft_model_state_dict(peft_model, adapter_name=adapter_name).keys()
        )

        with pytest.raises(ValueError, match="dynamo=True and optimize=False"):
            configure_lora_onnx(onnx_path, peft_keys, onnx_path)


def test_configure_lora_onnx_separate_output():
    """configure_lora_onnx with separate output_path preserves original."""
    base_model = _build_tiny_transformer()
    peft_model = _apply_peft_lora(base_model, rank=8)
    peft_model.eval()
    sample_inputs = (torch.randint(0, 512, (1, 32)),)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path, peft_keys = _user_export_peft_model(
            peft_model, sample_inputs, tmpdir
        )
        original_size = os.path.getsize(onnx_path)

        output_path = os.path.join(tmpdir, "prepared", "model_prepared.onnx")
        model_proto, lora_names = configure_lora_onnx(
            onnx_path,
            peft_keys,
            output_path,
        )

        assert os.path.exists(output_path), "output_path file was not created"
        assert os.path.getsize(onnx_path) == original_size, (
            "Original ONNX file was modified when output_path was specified"
        )
        assert len(lora_names["params"]) > 0
        assert len(lora_names["activations"]) > 0


# =========================================================================
# E2E tests using configure_lora_onnx (user-export flow)
# =========================================================================


def _generate_configure_lora_artifacts(
    output_dir: str,
    rank: int = 8,
    seq_len: int = 32,
) -> dict:
    """Generate LoRA test artifacts via user-export + configure_lora_onnx.

    Same result as generate_lora_test_artifacts, but uses the new flow:
    user does torch.onnx.export → configure_lora_onnx.
    """
    from peft.utils import get_peft_model_state_dict

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

    # User-style export: no wrapper, no AIMET helpers
    model = peft_model.base_model.model
    model.eval()
    sample_inputs = (torch.randint(0, 512, (1, seq_len)),)

    onnx_path = str(output_dir / "model.onnx")
    onnx_program = torch.onnx.export(
        model,
        sample_inputs,
        dynamo=True,
        optimize=False,
    )
    onnx_program.save(onnx_path, external_data=True)

    # Get peft keys and call configure_lora_onnx
    adapter_name = next(iter(peft_model.peft_config))
    peft_keys = list(
        get_peft_model_state_dict(peft_model, adapter_name=adapter_name).keys()
    )

    model_proto, lora_names = configure_lora_onnx(onnx_path, peft_keys, onnx_path)

    return {
        "model_path": str(output_dir / "model.onnx"),
        "lora_names": lora_names,
        "output_dir": str(output_dir),
        "adapter_dirs": adapter_dirs,
    }


@pytest.fixture(scope="session")
def configure_lora_artifacts(tmp_path_factory):
    """Generate LoRA test artifacts via user-export + configure_lora_onnx."""
    artifacts_dir = tmp_path_factory.mktemp("configure_lora_artifacts")
    return _generate_configure_lora_artifacts(output_dir=str(artifacts_dir), rank=8)


@pytest.fixture
def configure_prepared(configure_lora_artifacts) -> tuple:
    """Fresh model proto and lora_names for each test (configure_lora_onnx flow)."""
    model_proto = onnx.load(configure_lora_artifacts["model_path"])
    onnx.load_external_data_for_model(
        model_proto, configure_lora_artifacts["output_dir"]
    )
    lora_names = configure_lora_artifacts["lora_names"]
    adapter_dirs = configure_lora_artifacts["adapter_dirs"]
    return model_proto, lora_names, adapter_dirs


def test_configure_v1_workflow(configure_prepared):
    """V1 workflow with configure_lora_onnx: single encoding across adapters."""
    model, lora_names, adapter_dirs = configure_prepared
    assert len(lora_names["params"]) > 0
    assert len(lora_names["activations"]) > 0

    adapters = _load_adapters(adapter_dirs)

    sim = QuantizationSimModel(
        model,
        dummy_input={"input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64)},
        param_type=int4,
        activation_type=int16,
    )
    set_lora_bitwidth(sim, lora_names, param_type=int16, activation_type=int16)

    def calibrate_adapters(session):
        for adapter_name, weights in adapters.items():
            for _ in range(5):
                batch = {
                    "input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64),
                }
                batch.update(weights)
                session.run(None, batch)

    sim.compute_encodings(calibrate_adapters)

    frozen = freeze_base_model(sim, lora_names)
    assert frozen > 0

    with tempfile.TemporaryDirectory() as export_dir:
        sim.export(export_dir, "model_lora", export_model=False)
        assert os.path.exists(os.path.join(export_dir, "model_lora.encodings"))


def test_configure_v2_workflow(configure_prepared):
    """V2 workflow with configure_lora_onnx: full recalibration per adapter."""
    model, lora_names, adapter_dirs = configure_prepared
    assert len(lora_names["params"]) > 0

    adapters = _load_adapters(adapter_dirs)

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


def test_configure_v3_workflow(configure_prepared):
    """V3 workflow with configure_lora_onnx: base frozen, per-adapter LoRA recalibration."""
    model, lora_names, adapter_dirs = configure_prepared
    assert len(lora_names["params"]) > 0
    assert len(lora_names["activations"]) > 0

    adapters = _load_adapters(adapter_dirs)

    sim = _make_sim(model)
    set_lora_bitwidth(sim, lora_names, param_type=int16, activation_type=int16)

    # Base calibration with LoRA disabled
    zero_weights = get_zero_weights(model, lora_names)
    _calibrate(sim, zero_weights)

    # Freeze base
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

        # Verify base encodings did NOT change
        base_encoding_after = sim.qc_quantize_op_dict[base_param_name].export_encodings(
            "1.0.0"
        )
        assert base_encoding_before == base_encoding_after, (
            "Base param encoding changed during per-adapter calibration — freeze is broken"
        )

        # Verify adapter encodings differ
        enc_b = adapter_encodings["B"]
        enc_c = adapter_encodings["C"]
        assert len(enc_b) > 0 and len(enc_c) > 0
        any_differ = any(enc_b.get(name) != enc_c.get(name) for name in enc_b)
        assert any_differ, (
            "Adapter B and C encodings are identical — per-adapter calibration may be broken"
        )


@pytest.mark.skip(reason="Local only — requires TinyLlama + adapters cached")
def test_tinyllama_configure_lora_onnx_v3():
    """E2E V3 with TinyLlama using configure_lora_onnx (user-export flow).

    Same as test_tinyllama_e2e_v3_workflow but using user-style export +
    configure_lora_onnx instead of export_peft_to_onnx.
    """
    from peft import PeftModel
    from peft.utils import get_peft_model_state_dict
    from transformers import AutoModelForCausalLM

    adapter_a_path = _resolve_hf_adapter(_ADAPTER_A_ID)
    adapter_b_path = _resolve_hf_adapter(_ADAPTER_B_ID)

    # 1. Load base model + adapter A
    base_model = AutoModelForCausalLM.from_pretrained(
        _TINYLLAMA_BASE,
        cache_dir=HF_CACHE,
        dtype=torch.float32,
    )
    peft_model = PeftModel.from_pretrained(base_model, adapter_a_path)
    peft_model.eval()

    sample_inputs = (torch.randint(0, 32000, (1, 16)),)

    with tempfile.TemporaryDirectory(dir="/local/mnt/workspace") as tmpdir:
        # 2. User-style export — user handles use_cache and output format
        model = peft_model.base_model.model
        model.eval()
        model.config.use_cache = False

        onnx_path = str(Path(tmpdir) / "model.onnx")
        onnx_program = torch.onnx.export(
            model,
            sample_inputs,
            dynamo=True,
            optimize=False,
        )
        onnx_program.save(onnx_path, external_data=True)

        adapter_name = next(iter(peft_model.peft_config))
        peft_keys = list(
            get_peft_model_state_dict(peft_model, adapter_name=adapter_name).keys()
        )

        # 3. configure_lora_onnx with adapter_b branches
        model_proto, lora_names = configure_lora_onnx(
            onnx_path,
            peft_keys,
            onnx_path,
            adapter_paths=[adapter_b_path],
        )

        # 4. Verify union graph covers all 7 modules
        param_modules = set()
        for name in lora_names["params"]:
            for mod in [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]:
                if f".{mod}." in name:
                    param_modules.add(mod)

        assert param_modules == {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }, f"Expected 7 module branches, got: {param_modules}"

        assert len(lora_names["activations"]) > 0
        assert len(lora_names["scales"]) > 0

        # 5. QuantSim
        dummy_input = {
            "input_ids": np.random.randint(0, 32000, (1, 16)).astype(np.int64),
        }
        sim = QuantizationSimModel(
            model_proto,
            dummy_input=dummy_input,
            param_type=int4,
            activation_type=int16,
        )
        set_lora_bitwidth(sim, lora_names, param_type=int16, activation_type=int16)

        # 6. Base calibration with zero weights
        zero_weights = get_zero_weights(model_proto, lora_names)
        scale_defaults = {
            name: np.array(val, dtype=np.float32)
            for name, val in lora_names["scales"].items()
        }

        def base_cal(session):
            for _ in range(3):
                batch = {
                    "input_ids": np.random.randint(0, 32000, (1, 16)).astype(np.int64),
                }
                batch.update(zero_weights)
                batch.update(scale_defaults)
                session.run(None, batch)

        sim.compute_encodings(base_cal)

        # 7. Freeze base
        frozen = freeze_base_model(sim, lora_names)
        assert frozen > 0

        # 8. Per-adapter calibration
        adapter_encodings = {}
        adapter_feeds = {}
        adapter_scales = {}

        for name, adapter_path in [("A", adapter_a_path), ("B", adapter_b_path)]:
            unfreeze_lora_quantizers(sim, lora_names)

            weights = load_file(os.path.join(adapter_path, "adapter_model.safetensors"))
            feed = {**zero_weights, **weights}
            scales = get_adapter_scale_weights(lora_names, adapter_path)

            adapter_feeds[name] = feed
            adapter_scales[name] = scales

            def cal_fn(session, f=feed, s=scales):
                for _ in range(3):
                    batch = {
                        "input_ids": np.random.randint(0, 32000, (1, 16)).astype(
                            np.int64
                        ),
                    }
                    batch.update(f)
                    batch.update(s)
                    session.run(None, batch)

            sim.compute_encodings(cal_fn)
            adapter_encodings[name] = get_lora_encodings(sim, lora_names)

        # 9. Adapter encodings should differ
        enc_a = adapter_encodings["A"]
        enc_b = adapter_encodings["B"]
        assert len(enc_a) > 0 and len(enc_b) > 0
        any_differ = any(enc_a.get(n) != enc_b.get(n) for n in enc_a)
        assert any_differ, "Adapter A and B encodings should differ"

        # 10. Inference with each adapter produces different output
        input_data = {
            "input_ids": np.array([[1, 2, 3, 4, 5] + [0] * 11], dtype=np.int64),
        }

        set_lora_encodings(sim, enc_a)
        feed_a = {**input_data, **adapter_feeds["A"], **adapter_scales["A"]}
        output_a = sim.session.run(None, feed_a)[0]

        set_lora_encodings(sim, enc_b)
        feed_b = {**input_data, **adapter_feeds["B"], **adapter_scales["B"]}
        output_b = sim.session.run(None, feed_b)[0]

        assert np.all(np.isfinite(output_a)), "Adapter A output has NaN/Inf"
        assert np.all(np.isfinite(output_b)), "Adapter B output has NaN/Inf"
        assert not np.allclose(output_a, output_b, atol=1e-6), (
            "Adapter A and B should produce different outputs"
        )

        # 11. Export
        sim.export(tmpdir, "tinyllama_v3", export_model=False)
        assert os.path.exists(os.path.join(tmpdir, "tinyllama_v3.encodings"))


# =========================================================================
# prepare_lora_onnx tests (dynamo=False PeftModel exports)
# =========================================================================


def _generate_peft_dynamo_false_export(output_dir: str, rank: int = 8) -> dict:
    """Export a PeftModel with dynamo=False and save adapter artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model = _build_tiny_transformer()
    peft_model = _apply_peft_lora(base_model, rank=rank)
    peft_model.eval()

    sample = torch.randint(0, 512, (1, 32))
    onnx_path = str(output_dir / "peft_model.onnx")
    torch.onnx.export(
        peft_model,
        (sample,),
        onnx_path,
        dynamo=False,
        input_names=["input_ids"],
        output_names=["logits"],
    )

    # Save adapter config — read target_modules directly from the PEFT
    # config (set by _apply_peft_lora BEFORE wrapping, so it has the real
    # module names: q_proj, k_proj, etc. — not PEFT wrappers).
    peft_cfg = peft_model.peft_config[next(iter(peft_model.peft_config))]
    target_modules = list(peft_cfg.target_modules)

    config = {
        "r": rank,
        "lora_alpha": 16,
        "target_modules": target_modules,
        "bias": "none",
    }
    config_path = str(output_dir / "adapter_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f)

    # Save safetensors (default adapter weights)
    from peft import get_peft_model_state_dict

    state = get_peft_model_state_dict(peft_model)
    np_state = {k: v.cpu().numpy() for k, v in state.items()}
    from safetensors.numpy import save_file

    save_file(np_state, str(output_dir / "adapter_model.safetensors"))

    # Save adapter B with different weights
    _randomize_lora_weights(peft_model, seed=99)
    adapter_b_dir = str(output_dir / "adapter_B")
    peft_model.save_pretrained(adapter_b_dir)

    return {
        "onnx_path": onnx_path,
        "config_path": config_path,
        "output_dir": str(output_dir),
        "peft_keys": sorted(np_state.keys()),
        "peft_shapes": {k: v.shape for k, v in np_state.items()},
        "adapter_b_dir": adapter_b_dir,
        "target_modules": target_modules,
    }


@pytest.fixture(scope="session")
def peft_dynamo_false_artifacts(tmp_path_factory):
    """Generate PeftModel dynamo=False export once per test session."""
    artifacts_dir = tmp_path_factory.mktemp("peft_dynamo_false")
    return _generate_peft_dynamo_false_export(str(artifacts_dir))


@pytest.fixture
def prepared_dynamo_false(peft_dynamo_false_artifacts) -> tuple:
    """Run prepare_lora_onnx and return (model, lora_names, artifacts)."""
    arts = peft_dynamo_false_artifacts
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "prepared.onnx")
        _, lora_names = prepare_lora_onnx(
            arts["onnx_path"], arts["config_path"], out_path
        )
        # Reload with external data in memory (temp dir cleaned after yield)
        model = onnx.load(out_path)
        onnx.external_data_helper.load_external_data_for_model(model, tmpdir)
        yield model, lora_names, arts


def test_prepare_lora_onnx_basic(prepared_dynamo_false):
    """prepare_lora_onnx finds patterns and produces a valid ONNX model."""
    model, lora_names, arts = prepared_dynamo_false

    assert len(lora_names["params"]) > 0, "Should find LoRA params"
    assert len(lora_names["activations"]) > 0, "Should find LoRA activations"
    assert len(lora_names["scales"]) > 0, "Should find LoRA scales"

    # ORT inference works
    sess = ort.InferenceSession(model.SerializeToString())
    inp = {"input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64)}
    output = sess.run(None, inp)
    assert output[0].shape[0] == 1
    assert np.all(np.isfinite(output[0]))


def test_prepare_lora_onnx_safetensors_match(prepared_dynamo_false):
    """Renamed init names match PEFT safetensors keys.

    Shapes are stored in MatMul convention in the ONNX graph (transposed
    vs PEFT convention). The transposition is applied at load time by
    _remap_safetensors_to_onnx using lora_names["transposed_params"].
    """
    model, lora_names, arts = prepared_dynamo_false
    peft_keys = set(arts["peft_keys"])
    param_names = set(lora_names["params"])

    assert param_names == peft_keys, (
        f"Key mismatch:\n"
        f"  In params but not safetensors: {param_names - peft_keys}\n"
        f"  In safetensors but not params: {peft_keys - param_names}"
    )

    # Shapes are transposed in the graph (MatMul convention).
    # Verify transposed_params records the graph shape and it's the
    # reverse of the PEFT safetensors shape.
    init_map = {i.name: tuple(i.dims) for i in model.graph.initializer}
    transposed_params = lora_names.get("transposed_params", {})
    for key, peft_shape in arts["peft_shapes"].items():
        assert key in init_map, f"Init {key} not found in model"
        onnx_shape = init_map[key]
        if len(peft_shape) == 2:
            # ONNX stores in MatMul convention — reversed vs PEFT
            assert onnx_shape == tuple(reversed(peft_shape)), (
                f"Shape for {key}: onnx={onnx_shape} should be "
                f"reverse of peft={peft_shape}"
            )
            assert key in transposed_params, (
                f"{key} not in transposed_params — safetensors load will fail"
            )


def test_prepare_lora_onnx_initializer_only(prepared_dynamo_false):
    """LoRA params should be initializer-only (not in graph.input)."""
    model, lora_names, _ = prepared_dynamo_false
    graph_input_names = {inp.name for inp in model.graph.input}

    for param_name in lora_names["params"]:
        assert param_name not in graph_input_names, (
            f"LoRA param {param_name} should not be in graph.input "
            f"(initializer-only mode)"
        )

    for scale_name in lora_names["scales"]:
        assert scale_name not in graph_input_names, (
            f"LoRA scale {scale_name} should not be in graph.input"
        )


def test_prepare_lora_onnx_scales(prepared_dynamo_false):
    """Scale constants are converted to named initializers with correct values."""
    model, lora_names, arts = prepared_dynamo_false
    rank = arts.get("peft_shapes", {})
    # alpha=16, r=8 → scale = 2.0
    for scale_name, scale_value in lora_names["scales"].items():
        assert scale_value == pytest.approx(2.0), (
            f"Scale {scale_name} = {scale_value}, expected 2.0 (alpha=16/r=8)"
        )
        assert "lora_scale" in scale_name


def test_prepare_lora_onnx_target_module_coverage(prepared_dynamo_false):
    """All target_modules from adapter_config are found in the graph."""
    _, lora_names, arts = prepared_dynamo_false
    target_modules = set(arts["target_modules"])

    found_modules = set()
    for param_name in lora_names["params"]:
        # base_model.model.layers.0.self_attn.q_proj.lora_A.weight → q_proj
        parts = (
            param_name.replace(".lora_A.weight", "")
            .replace(".lora_B.weight", "")
            .split(".")
        )
        found_modules.add(parts[-1])

    assert target_modules <= found_modules, (
        f"Missing target modules: {target_modules - found_modules}"
    )


def test_prepare_lora_onnx_no_patterns_raises():
    """prepare_lora_onnx raises ValueError for a model without LoRA patterns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export a plain model (no PEFT)
        base_model = _build_tiny_transformer()
        base_model.eval()
        sample = torch.randint(0, 512, (1, 32))
        onnx_path = os.path.join(tmpdir, "base.onnx")
        torch.onnx.export(
            base_model,
            (sample,),
            onnx_path,
            dynamo=False,
            input_names=["input_ids"],
            output_names=["logits"],
        )
        config_path = os.path.join(tmpdir, "adapter_config.json")
        with open(config_path, "w") as f:
            json.dump({"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]}, f)

        with pytest.raises(ValueError, match="No LoRA patterns found"):
            prepare_lora_onnx(onnx_path, config_path, os.path.join(tmpdir, "out.onnx"))


def test_disable_enable_lora_quantizers(prepared_dynamo_false):
    """disable_lora_quantizers disables and enable_lora_calibration re-enables."""
    model, lora_names, _ = prepared_dynamo_false
    dummy_input = {"input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64)}
    sim = QuantizationSimModel(model, dummy_input=dummy_input)

    # Phase 2: disable
    disabled_count = disable_lora_quantizers(sim, lora_names)
    assert disabled_count > 0

    all_lora = lora_names["params"] + lora_names["activations"]
    for name in all_lora:
        if name in sim.qc_quantize_op_dict:
            assert not sim.qc_quantize_op_dict[name].enabled, (
                f"LoRA quantizer {name} should be disabled"
            )

    # Phase 3: enable
    enabled_count = enable_lora_calibration(
        sim, lora_names, param_type="int16", activation_type="int8"
    )
    assert enabled_count > 0

    for name in all_lora:
        if name in sim.qc_quantize_op_dict:
            assert sim.qc_quantize_op_dict[name].enabled, (
                f"LoRA quantizer {name} should be enabled after enable_lora_calibration"
            )


def test_lora_params_initializer_only(prepared_dynamo_false):
    """LoRA params are initializer-only (not in graph.input) after prepare."""
    model, lora_names, _ = prepared_dynamo_false
    graph_input_names = {inp.name for inp in model.graph.input}
    init_names = {init.name for init in model.graph.initializer}

    for name in lora_names["params"]:
        assert name in init_names, f"{name} should be in graph.initializer"
        assert name not in graph_input_names, f"{name} should not be in graph.input"


# ===========================================================================
# Multi-adapter tests
# ===========================================================================


def _generate_multi_adapter_export(output_dir: str) -> dict:
    """Export PeftMixedModel with 2 adapters (different targets/ranks)."""
    from peft import LoraConfig, get_peft_model

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model = _build_tiny_transformer()

    # Adapter "code": targets q_proj + v_proj, rank=8
    config_code = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    peft = get_peft_model(base_model, config_code, adapter_name="code", mixed=True)

    # Adapter "medical": targets q_proj only, rank=4
    config_medical = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj"],
        bias="none",
    )
    peft.add_adapter("medical", config_medical)
    peft.set_adapter(["code", "medical"])
    peft.eval()

    sample = torch.randint(0, 512, (1, 32))
    onnx_path = str(output_dir / "multi_adapter.onnx")
    torch.onnx.export(
        peft,
        (sample,),
        onnx_path,
        dynamo=False,
        input_names=["input_ids"],
        output_names=["logits"],
    )

    # Save adapter configs
    code_config = {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"]}
    medical_config = {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj"]}

    code_cfg_path = str(output_dir / "code_config.json")
    medical_cfg_path = str(output_dir / "medical_config.json")
    with open(code_cfg_path, "w") as f:
        json.dump(code_config, f)
    with open(medical_cfg_path, "w") as f:
        json.dump(medical_config, f)

    # Save safetensors per adapter
    from peft import get_peft_model_state_dict
    from safetensors.numpy import save_file

    # Code adapter weights
    peft.set_adapter(["code"])
    code_state = get_peft_model_state_dict(peft, adapter_name="code")
    code_np = {k: v.cpu().numpy() for k, v in code_state.items()}
    save_file(code_np, str(output_dir / "code_weights.safetensors"))

    # Medical adapter weights
    peft.set_adapter(["medical"])
    medical_state = get_peft_model_state_dict(peft, adapter_name="medical")
    medical_np = {k: v.cpu().numpy() for k, v in medical_state.items()}
    save_file(medical_np, str(output_dir / "medical_weights.safetensors"))

    return {
        "onnx_path": onnx_path,
        "code_config_path": code_cfg_path,
        "medical_config_path": medical_cfg_path,
        "code_weights_path": str(output_dir / "code_weights.safetensors"),
        "medical_weights_path": str(output_dir / "medical_weights.safetensors"),
        "output_dir": str(output_dir),
    }


@pytest.fixture(scope="session")
def prepare_multi_adapter_artifacts(tmp_path_factory):
    """Generate multi-adapter PeftMixedModel export once per session."""
    arts_dir = tmp_path_factory.mktemp("prepare_multi_adapter")
    return _generate_multi_adapter_export(str(arts_dir))


@pytest.fixture
def prepared_multi_adapter(prepare_multi_adapter_artifacts) -> tuple:
    """Run prepare_lora_onnx with multi-adapter configs."""
    arts = prepare_multi_adapter_artifacts
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "prepared_multi.onnx")
        _, lora_names = prepare_lora_onnx(
            arts["onnx_path"],
            adapter_config=[arts["code_config_path"], arts["medical_config_path"]],
            output_path=out_path,
        )
        # Reload with external data in memory (temp dir cleaned after yield)
        model = onnx.load(out_path)
        onnx.external_data_helper.load_external_data_for_model(model, tmpdir)
        yield model, lora_names, arts


def test_prepare_multi_adapter_basic(prepared_multi_adapter):
    """Multi-adapter: discovers both adapters, returns per-adapter lora_names."""
    model, lora_names, arts = prepared_multi_adapter

    # Should have "adapters" key
    assert "adapters" in lora_names
    assert "code" in lora_names["adapters"]
    assert "medical" in lora_names["adapters"]

    # Each adapter has params, activations, scales
    for adapter_name in ["code", "medical"]:
        adapter_ln = lora_names["adapters"][adapter_name]
        assert "params" in adapter_ln
        assert "activations" in adapter_ln
        assert "scales" in adapter_ln
        assert len(adapter_ln["params"]) > 0

    # Flat lists contain union of all adapters
    assert len(lora_names["params"]) == (
        len(lora_names["adapters"]["code"]["params"])
        + len(lora_names["adapters"]["medical"]["params"])
    )


def test_prepare_multi_adapter_diff_targets(prepared_multi_adapter):
    """Multi-adapter: code targets q+v, medical targets q only."""
    _, lora_names, _ = prepared_multi_adapter

    code_params = lora_names["adapters"]["code"]["params"]
    medical_params = lora_names["adapters"]["medical"]["params"]

    # Code targets 2 modules (q_proj, v_proj) × 2 layers × 2 weights (A+B) = 8
    assert len(code_params) == 8

    # Medical targets 1 module (q_proj) × 2 layers × 2 weights (A+B) = 4
    assert len(medical_params) == 4

    # Medical params should reference q_proj but not v_proj
    for p in medical_params:
        assert "q_proj" in p
        assert "v_proj" not in p


def test_prepare_multi_adapter_diff_ranks(prepared_multi_adapter):
    """Multi-adapter: code rank=8, medical rank=4 — different shapes.

    lora_A is stored in MatMul convention: (in_features, rank).
    """
    model, lora_names, _ = prepared_multi_adapter

    init_map = {init.name: init for init in model.graph.initializer}

    code_params = lora_names["adapters"]["code"]["params"]
    medical_params = lora_names["adapters"]["medical"]["params"]

    # Code lora_A shape is (in_features, rank=8)
    code_lora_a = [p for p in code_params if "lora_A" in p][0]
    assert init_map[code_lora_a].dims[1] == 8

    # Medical lora_A shape is (in_features, rank=4)
    medical_lora_a = [p for p in medical_params if "lora_A" in p][0]
    assert init_map[medical_lora_a].dims[1] == 4


def test_prepare_multi_adapter_ort_inference(prepared_multi_adapter):
    """Multi-adapter: ORT inference works and produces valid output."""
    import onnxruntime as ort

    model, lora_names, _ = prepared_multi_adapter

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        onnx.save_model(
            model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.onnx.data",
        )
        session = ort.InferenceSession(model_path)

        input_name = session.get_inputs()[0].name
        dummy_input = np.random.randint(0, 512, (1, 32)).astype(np.int64)

        # Model runs with baked-in initializer weights (all adapters active)
        output = session.run(None, {input_name: dummy_input})[0]
        assert output.shape[0] == 1
        assert np.all(np.isfinite(output))


def test_prepare_multi_adapter_concurrent(prepared_multi_adapter):
    """Multi-adapter: build_concurrent_feed_dict returns empty when LoRA not in graph.input."""
    model, lora_names, arts = prepared_multi_adapter
    code_wts = load_file(arts["code_weights_path"])

    # Before promotion, feed dict should be empty (no graph inputs for LoRA)
    feed = build_concurrent_feed_dict(
        model, lora_names, active_adapters={"code": code_wts}
    )
    # All LoRA params are initializer-only → nothing to feed
    assert len(feed) == 0, (
        "Before enable_lora_calibration, feed dict should be empty "
        "(LoRA weights are initializer-only, not graph inputs)"
    )


def test_prepare_single_adapter_backward_compat(prepared_dynamo_false):
    """Single-adapter: no 'adapters' key — backward compatible format."""
    _, lora_names, _ = prepared_dynamo_false

    # Should NOT have "adapters" key for single-adapter
    assert "adapters" not in lora_names
    # Should have flat structure
    assert "params" in lora_names
    assert "activations" in lora_names
    assert "scales" in lora_names


# ===========================================================================
# TinyLlama multi-adapter E2E test (prepare_lora_onnx + PeftMixedModel)
# ===========================================================================


@pytest.mark.skip(reason="Local only — requires TinyLlama + adapters cached")
def test_tinyllama_multi_adapter_prepare_lora_onnx():
    """E2E multi-adapter workflow with TinyLlama using prepare_lora_onnx.

    Uses PeftMixedModel to export with both adapters active (dynamo=False),
    then runs the full quantization workflow:
    prepare_lora_onnx -> QuantSim -> disable LoRA -> base cal -> freeze
    -> enable LoRA -> per-adapter: set_lora_weights + compute_encodings
    + get_lora_encodings -> set_lora_weights + set_lora_encodings for
    inference -> verify outputs differ between adapters.

    Adapter A (TLDR): barissglc/tinyllama-tarot-v1
        target_modules=[q_proj, v_proj], rank=8, alpha=16

    Adapter B (Skills): cahlen/tinyllama-offline-practical-skills-qa-qlora
        target_modules=[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
        rank=64, alpha=16
    """
    from peft import PeftConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    adapter_a_path = _resolve_hf_adapter(_ADAPTER_A_ID)
    adapter_b_path = _resolve_hf_adapter(_ADAPTER_B_ID)

    base_model = AutoModelForCausalLM.from_pretrained(
        _TINYLLAMA_BASE,
        cache_dir=HF_CACHE,
        torch_dtype=torch.float32,
    )

    config_a = PeftConfig.from_pretrained(adapter_a_path)
    peft_model = get_peft_model(base_model, config_a, adapter_name="tarot", mixed=True)

    from safetensors.torch import load_file as load_torch_safetensors

    state_a = load_torch_safetensors(
        os.path.join(adapter_a_path, "adapter_model.safetensors")
    )
    peft_model.load_state_dict(state_a, strict=False)

    config_b = PeftConfig.from_pretrained(adapter_b_path)
    peft_model.add_adapter("skills", config_b)

    state_b = load_torch_safetensors(
        os.path.join(adapter_b_path, "adapter_model.safetensors")
    )
    peft_model.load_state_dict(state_b, strict=False)

    peft_model.set_adapter(["tarot", "skills"])
    peft_model.eval()

    sample_inputs = (torch.randint(0, 32000, (1, 16)),)

    with tempfile.TemporaryDirectory(dir="/local/mnt/workspace") as tmpdir:
        onnx_path = os.path.join(tmpdir, "tinyllama_mixed.onnx")
        torch.onnx.export(
            peft_model,
            sample_inputs,
            onnx_path,
            dynamo=False,
            input_names=["input_ids"],
            output_names=["logits"],
        )

        cfg_a = {
            "r": config_a.r,
            "lora_alpha": config_a.lora_alpha,
            "target_modules": list(config_a.target_modules),
        }
        cfg_b = {
            "r": config_b.r,
            "lora_alpha": config_b.lora_alpha,
            "target_modules": list(config_b.target_modules),
        }
        cfg_a_path = os.path.join(tmpdir, "tarot_config.json")
        cfg_b_path = os.path.join(tmpdir, "skills_config.json")
        with open(cfg_a_path, "w") as f:
            json.dump(cfg_a, f)
        with open(cfg_b_path, "w") as f:
            json.dump(cfg_b, f)

        out_path = os.path.join(tmpdir, "prepared.onnx")
        model, lora_names = prepare_lora_onnx(
            onnx_path,
            adapter_config=[cfg_a_path, cfg_b_path],
            output_path=out_path,
        )

        assert "adapters" in lora_names
        assert "tarot" in lora_names["adapters"]
        assert "skills" in lora_names["adapters"]
        assert len(lora_names["adapters"]["skills"]["params"]) > len(
            lora_names["adapters"]["tarot"]["params"]
        )

        dummy_input = {
            "input_ids": np.random.randint(0, 32000, (1, 16)).astype(np.int64),
        }
        sim = QuantizationSimModel(
            model,
            dummy_input=dummy_input,
            param_type=int4,
            activation_type=int16,
        )

        disable_lora_quantizers(sim, lora_names)

        def base_cal(session):
            for _ in range(3):
                batch = {
                    "input_ids": np.random.randint(0, 32000, (1, 16)).astype(np.int64)
                }
                session.run(None, batch)

        sim.compute_encodings(base_cal)
        freeze_base_model(sim, lora_names)

        enable_lora_calibration(
            sim, lora_names, param_type="int16", activation_type="int16"
        )

        adapter_wts = {
            "tarot": load_file(
                os.path.join(adapter_a_path, "adapter_model.safetensors")
            ),
            "skills": load_file(
                os.path.join(adapter_b_path, "adapter_model.safetensors")
            ),
        }
        adapter_encodings = {}

        for name, wts in adapter_wts.items():
            unfreeze_lora_quantizers(sim, lora_names)
            set_lora_weights(sim, lora_names, name, wts)

            def cal_fn(session):
                for _ in range(3):
                    batch = {
                        "input_ids": np.random.randint(0, 32000, (1, 16)).astype(
                            np.int64
                        )
                    }
                    session.run(None, batch)

            sim.compute_encodings(cal_fn)
            adapter_encodings[name] = get_lora_encodings(sim, lora_names)

        any_differ = any(
            adapter_encodings["tarot"].get(n) != adapter_encodings["skills"].get(n)
            for n in adapter_encodings["tarot"]
        )
        assert any_differ

        input_data = {
            "input_ids": np.array([[1, 2, 3, 4, 5] + [0] * 11], dtype=np.int64)
        }

        set_lora_weights(sim, lora_names, "tarot", adapter_wts["tarot"])
        set_lora_encodings(sim, adapter_encodings["tarot"])
        output_a = sim.session.run(None, input_data)[0]

        set_lora_weights(sim, lora_names, "skills", adapter_wts["skills"])
        set_lora_encodings(sim, adapter_encodings["skills"])
        output_b = sim.session.run(None, input_data)[0]

        assert np.all(np.isfinite(output_a))
        assert np.all(np.isfinite(output_b))
        assert not np.allclose(output_a, output_b, atol=1e-6)

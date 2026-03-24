# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the aimet_onnx.experimental.lora module."""

import dataclasses
import os
import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
import torch.nn as nn

pytest.importorskip("peft", reason="peft is required for LoRA test model generation")
pytest.importorskip(
    "safetensors", reason="safetensors is required for LoRA adapter I/O"
)

from aimet_onnx.experimental.lora.lora_adapter_quantization import LoRAResult
from aimet_onnx import int4, int8, int16

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
    from peft import LoraConfig, get_peft_model

    # Target all linear layers
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # PEFT wants the attribute name, not the full path
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
                # Add small random values so adapters are actually different
                param.data += torch.randn_like(param.data) * 0.01


def generate_lora_test_model(
    output_dir: str,
    rank: int = 8,
    seq_len: int = 32,
) -> "LoRAResult":
    """Generate a test ONNX model with LoRA adapters via export_peft_to_onnx.

    Exercises the real public API so that any breakpoint in export_peft_to_onnx
    or its helpers is hit during test runs.

    Returns the LoRAResult with default, B, and C adapters loaded.
    """
    from aimet_onnx.experimental.lora.peft_to_onnx import export_peft_to_onnx

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build base model + apply LoRA
    base_model = _build_tiny_transformer()
    peft_model = _apply_peft_lora(base_model, rank=rank)
    peft_model.eval()

    # Snapshot default LoRA weights before randomizing for B/C
    original_lora_state = {
        name: param.detach().clone()
        for name, param in peft_model.named_parameters()
        if "lora_" in name
    }

    # Save adapters B and C to disk in PEFT format (save_pretrained writes
    # adapter_model.safetensors with PEFT-format keys, which is exactly what
    # _load_adapter_safetensors expects)
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

    # Export via the real public API — this exercises _Wrapper, _infer_input_names,
    # _infer_dynamic_shapes, _onnx_name_to_safetensors_key, _load_adapter_safetensors
    sample_inputs = (torch.randint(0, 512, (1, seq_len)),)
    _model_proto, result = export_peft_to_onnx(
        peft_model,
        sample_inputs,
        adapter_dirs,
        str(output_dir),
    )

    return result


# =========================================================================
# Fixtures
# =========================================================================

import aimet_onnx
from aimet_onnx.experimental.lora import (
    configure_lora_quantizers,
    freeze_base_param_quantizers,
    freeze_base_activation_quantizers,
    freeze_base_model,
    unfreeze_lora_quantizers,
    export_lora_weights,
    get_lora_encodings,
    set_lora_encodings,
    calibrate_lora,
    export_lora,
)
from aimet_onnx.experimental.lora.peft_to_onnx import _load_adapter_safetensors
from aimet_onnx.quantsim import QuantizationSimModel


@pytest.fixture(scope="session")
def lora_artifacts(tmp_path_factory):
    """Generate LoRA test artifacts once per test session via export_peft_to_onnx.

    Returns a LoRAResult with model_path set and adapters (default, B, C) loaded.
    Pytest cleans up tmp_path_factory directories after the session ends.
    """
    artifacts_dir = tmp_path_factory.mktemp("lora_artifacts")
    return generate_lora_test_model(output_dir=str(artifacts_dir), rank=8)


@pytest.fixture
def prepared_result(lora_artifacts) -> tuple[onnx.ModelProto, LoRAResult]:
    """Fresh model proto and LoRAResult copy for each test.

    Loads a new model proto from disk (so QuantSim mutations don't leak between
    tests) and shallow-copies the LoRAResult (so tests can safely mutate
    model_path or adapter_encodings without affecting the session fixture).
    """
    result = lora_artifacts
    model_proto = onnx.load(result.model_path)
    result_copy = dataclasses.replace(
        result,
        adapters=dict(result.adapters),
        adapter_encodings=dict(result.adapter_encodings),
    )
    return model_proto, result_copy


def _make_sim(model):
    """Create a QuantizationSimModel from a model proto."""
    dummy_input = {
        "input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64),
    }
    return QuantizationSimModel(model, dummy_input=dummy_input)


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


# =========================================================================
# Test: Top-level API (the "quick path" from __init__.py)
# =========================================================================


def test_top_level_api(tmp_path):
    """Test the 4-step user-facing workflow with no fixtures or helpers.

    Mirrors the 'Quick path' from the module docstring::

        model, result = export_peft_to_onnx(...)
        sim = QuantizationSimModel(model, ...)
        calibrate_lora(sim, result, dataloader)
        export_lora(sim, result, export_dir, target="ort")
    """
    from aimet_onnx.experimental.lora import (
        export_peft_to_onnx,
        calibrate_lora,
        export_lora,
    )

    # --- Build a PeftModel (user's responsibility) ---
    base_model = _build_tiny_transformer()
    peft_model = _apply_peft_lora(base_model, rank=8)
    peft_model.eval()

    # Save an adapter to disk (simulates a user's fine-tuned adapter)
    adapter_dir = str(tmp_path / "my_adapter")
    _randomize_lora_weights(peft_model, seed=99)
    peft_model.save_pretrained(adapter_dir)

    # Restore original weights for export
    _randomize_lora_weights(peft_model, seed=0)

    sample_inputs = (torch.randint(0, 512, (1, 32)),)

    # --- Step 1: Export ---
    model, result = export_peft_to_onnx(
        peft_model,
        sample_inputs,
        {"my_adapter": adapter_dir},
        str(tmp_path / "export"),
    )

    assert result.model_path is not None
    assert len(result.lora_input_names) > 0
    assert "my_adapter" in result.adapters

    # --- Step 2: QuantSim ---
    dummy_input = {"input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64)}
    sim = QuantizationSimModel(model, dummy_input=dummy_input)

    # --- Step 3: Calibrate ---
    dataloader = [
        {"input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64)}
        for _ in range(3)
    ]
    calibrate_lora(sim, result, dataloader, lora_param_type=aimet_onnx.int16)

    assert "my_adapter" in result.adapter_encodings

    # --- Step 4: Export ---
    with tempfile.TemporaryDirectory() as export_dir:
        export_lora(sim, result, export_dir, target="ort")

        assert os.path.exists(os.path.join(export_dir, "model.onnx"))
        assert os.path.exists(os.path.join(export_dir, "model.encodings"))
        assert os.path.exists(os.path.join(export_dir, "my_adapter.safetensors"))
        assert os.path.exists(os.path.join(export_dir, "model_my_adapter.encodings"))


# =========================================================================
# Test: Composable workflow (individual API functions)
# =========================================================================


def test_full_lora_workflow(prepared_result):
    """Full end-to-end LoRA quantization workflow."""

    # Phase 1: Setup — model and result come from export_peft_to_onnx
    model, result = prepared_result
    assert len(result.lora_input_names) > 0

    adapters = {
        "default": result.get_adapter("default"),
        "B": result.get_adapter("B"),
        "C": result.get_adapter("C"),
    }

    # Phase 2: Create QuantSim (LoRA as initializers → per-channel)
    dummy_input = {
        "input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64),
    }
    sim = QuantizationSimModel(model, dummy_input=dummy_input)

    # Phase 3: Configure LoRA quantizers (sets lora_param_type + converts init→input)
    lora_count = configure_lora_quantizers(
        sim, result, lora_param_type=aimet_onnx.int16
    )
    assert lora_count > 0

    for name in result.lora_input_names:
        if name in sim.qc_quantize_op_dict:
            assert sim.qc_quantize_op_dict[name].bitwidth == 16, (
                f"{name} should be 16-bit (aimet_onnx.int16)"
            )

    # Phase 4: Calibrate base model (LoRA disabled)
    zero_weights = result.get_zero_weights()

    def calibrate_base(session):
        for _ in range(5):
            batch = {
                "input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64),
            }
            batch.update(zero_weights)
            session.run(None, batch)

    sim.compute_encodings(calibrate_base)
    frozen_count = freeze_base_param_quantizers(sim, result)
    assert frozen_count > 0

    # Phase 5: Per-adapter calibration & export
    with tempfile.TemporaryDirectory() as output_dir:
        sim.export(output_dir, "model")
        assert os.path.exists(os.path.join(output_dir, "model.encodings"))

        for adapter_name, weights in adapters.items():
            unfrozen = unfreeze_lora_quantizers(sim, result)
            assert unfrozen > 0

            def calibrate_adapter(session, w=weights):
                for _ in range(5):
                    batch = {
                        "input_ids": np.random.randint(0, 512, (1, 32)).astype(
                            np.int64
                        ),
                    }
                    batch.update(w)
                    session.run(None, batch)

            sim.compute_encodings(calibrate_adapter)

            adapter_path = os.path.join(output_dir, f"{adapter_name}.safetensors")
            export_lora_weights(result, weights, adapter_path)
            assert os.path.exists(adapter_path)

            sim.export(output_dir, f"model_{adapter_name}", export_model=False)
            assert os.path.exists(
                os.path.join(output_dir, f"model_{adapter_name}.encodings")
            )

        # Verify output structure
        output_files = sorted(os.listdir(output_dir))
        assert "model.onnx" in output_files
        assert "model.encodings" in output_files
        for adapter_name in adapters:
            assert f"{adapter_name}.safetensors" in output_files
            assert f"model_{adapter_name}.encodings" in output_files

        from safetensors.numpy import load_file

        for adapter_name, original_weights in adapters.items():
            reloaded = load_file(
                os.path.join(output_dir, f"{adapter_name}.safetensors")
            )
            assert len(reloaded) == len(original_weights), (
                f"Adapter {adapter_name}: expected {len(original_weights)}, got {len(reloaded)}"
            )


def test_full_lora_v1_workflow(prepared_result):
    """Full end-to-end LoRA quantization workflow."""
    """ LoRA v1
        * Lora weights as input
        * Single weight encoding across adapters
        * Single activation encoding across adapters
    """
    # Phase 1: Setup — model and result come from export_peft_to_onnx
    model, result = prepared_result
    assert len(result.lora_input_names) > 0

    adapters = {
        "default": result.get_adapter("default"),
        "B": result.get_adapter("B"),
        "C": result.get_adapter("C"),
    }

    # Phase 2: Create QuantSim (LoRA as initializers → per-channel)
    dummy_input = {
        "input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64),
    }
    sim = QuantizationSimModel(
        model, dummy_input=dummy_input, param_type=int4, activation_type=int16
    )

    # Phase 3: Configure LoRA quantizers (sets lora_param_type + converts init→input)
    # Todo: Check that param tensors (as model input) are per-channel, symmetric
    lora_count = configure_lora_quantizers(
        sim, result, lora_param_type=aimet_onnx.int16
    )
    assert lora_count > 0

    for name in result.lora_input_names:
        if name in sim.qc_quantize_op_dict:
            assert sim.qc_quantize_op_dict[name].bitwidth == 16, (
                f"{name} should be 16-bit (aimet_onnx.int16)"
            )

    # Phase 5: Per-adapter calibration & export
    with tempfile.TemporaryDirectory() as output_dir:

        def calibrate_adapters(session):
            for adapter_name, weights in adapters.items():
                for _ in range(5):
                    batch = {
                        "input_ids": np.random.randint(0, 512, (1, 32)).astype(
                            np.int64
                        ),
                    }
                    batch.update(weights)
                    session.run(None, batch)

        sim.compute_encodings(calibrate_adapters)

        sim.export(output_dir, f"model_lora", export_model=False)
        assert os.path.exists(os.path.join(output_dir, f"model_lora.encodings"))


def test_full_lora_v2_workflow(prepared_result):
    """Full end-to-end LoRA quantization workflow."""
    """ LoRA v2
        * Lora weights as initializers
        * Per-adapter full model encodings (weights + activations)
    """
    # Phase 1: Setup — model and result come from export_peft_to_onnx
    model, result = prepared_result
    assert len(result.lora_input_names) > 0

    adapters = {
        "default": result.get_adapter("default"),
        "B": result.get_adapter("B"),
        "C": result.get_adapter("C"),
    }

    # Phase 2: Create QuantSim (LoRA as initializers → per-channel)
    dummy_input = {
        "input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64),
    }
    sim = QuantizationSimModel(model, dummy_input=dummy_input)

    # Phase 3: Configure LoRA quantizers (sets lora_param_type + converts init→input)
    lora_count = configure_lora_quantizers(
        sim, result, lora_param_type=aimet_onnx.int16
    )
    assert lora_count > 0

    for name in result.lora_input_names:
        if name in sim.qc_quantize_op_dict:
            assert sim.qc_quantize_op_dict[name].bitwidth == 16, (
                f"{name} should be 16-bit (aimet_onnx.int16)"
            )

    # Phase 5: Per-adapter calibration & export
    with tempfile.TemporaryDirectory() as output_dir:
        for adapter_name, weights in adapters.items():

            def calibrate_adapter(session, w=weights):
                for _ in range(5):
                    batch = {
                        "input_ids": np.random.randint(0, 512, (1, 32)).astype(
                            np.int64
                        ),
                    }
                    batch.update(w)
                    session.run(None, batch)

            sim.compute_encodings(calibrate_adapter)

            adapter_path = os.path.join(output_dir, f"{adapter_name}.safetensors")
            export_lora_weights(result, weights, adapter_path)
            assert os.path.exists(adapter_path)

            sim.export(output_dir, f"model_{adapter_name}", export_model=False)
            assert os.path.exists(
                os.path.join(output_dir, f"model_{adapter_name}.encodings")
            )

        # Verify output structure
        output_files = sorted(os.listdir(output_dir))
        for adapter_name in adapters:
            assert f"{adapter_name}.safetensors" in output_files
            assert f"model_{adapter_name}.encodings" in output_files

        from safetensors.numpy import load_file

        for adapter_name, original_weights in adapters.items():
            reloaded = load_file(
                os.path.join(output_dir, f"{adapter_name}.safetensors")
            )
            assert len(reloaded) == len(original_weights), (
                f"Adapter {adapter_name}: expected {len(original_weights)}, got {len(reloaded)}"
            )


def test_full_lora_v3_workflow(prepared_result):
    """Full end-to-end LoRA quantization workflow."""
    """ LoRA v3
        * Lora weights as initializers
        * Single base model encodings (weights + activations)
        * Per-adapter adapter branch encodings (weights + activations)
        * Concurrent adapter branches
    """

    # Phase 1: Setup — model and result come from export_peft_to_onnx
    model, result = prepared_result
    assert len(result.lora_input_names) > 0

    adapters = {
        "default": result.get_adapter("default"),
        "B": result.get_adapter("B"),
        "C": result.get_adapter("C"),
    }

    # Phase 2: Create QuantSim (LoRA as initializers → per-channel)
    dummy_input = {
        "input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64),
    }
    sim = QuantizationSimModel(model, dummy_input=dummy_input)

    # Phase 3: Configure LoRA quantizers (sets lora_param_type + converts init→input)
    lora_count = configure_lora_quantizers(
        sim, result, lora_param_type=aimet_onnx.int16
    )
    assert lora_count > 0

    for name in result.lora_input_names:
        if name in sim.qc_quantize_op_dict:
            assert sim.qc_quantize_op_dict[name].bitwidth == 16, (
                f"{name} should be 16-bit (aimet_onnx.int16)"
            )

    # Phase 4: Calibrate base model (LoRA disabled)
    zero_weights = result.get_zero_weights()

    def calibrate_base(session):
        for _ in range(5):
            batch = {
                "input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64),
            }
            batch.update(zero_weights)
            session.run(None, batch)

    sim.compute_encodings(calibrate_base)
    frozen_count = freeze_base_param_quantizers(sim, result)
    assert frozen_count > 0
    frozen_count = freeze_base_activation_quantizers(sim, result)
    assert frozen_count > 0

    # Phase 5: Per-adapter calibration & export
    with tempfile.TemporaryDirectory() as output_dir:
        for adapter_name, weights in adapters.items():

            def calibrate_adapter(session, w=weights):
                for _ in range(5):
                    batch = {
                        "input_ids": np.random.randint(0, 512, (1, 32)).astype(
                            np.int64
                        ),
                    }
                    batch.update(w)
                    session.run(None, batch)

            sim.compute_encodings(calibrate_adapter)

            adapter_path = os.path.join(output_dir, f"{adapter_name}.safetensors")
            export_lora_weights(result, weights, adapter_path)
            assert os.path.exists(adapter_path)

            sim.export(output_dir, f"model_{adapter_name}", export_model=False)
            assert os.path.exists(
                os.path.join(output_dir, f"model_{adapter_name}.encodings")
            )

        # Verify output structure
        output_files = sorted(os.listdir(output_dir))
        for adapter_name in adapters:
            assert f"{adapter_name}.safetensors" in output_files
            assert f"model_{adapter_name}.encodings" in output_files

        from safetensors.numpy import load_file

        for adapter_name, original_weights in adapters.items():
            reloaded = load_file(
                os.path.join(output_dir, f"{adapter_name}.safetensors")
            )
            assert len(reloaded) == len(original_weights), (
                f"Adapter {adapter_name}: expected {len(original_weights)}, got {len(reloaded)}"
            )


# =========================================================================
# Test: Encoding Roundtrip (get/set_lora_encodings)
# =========================================================================


def test_lora_encoding_roundtrip(prepared_result):
    """get_lora_encodings / set_lora_encodings round-trip correctly."""
    model, result = prepared_result
    sim = _make_sim(model)

    configure_lora_quantizers(sim, result, lora_param_type=aimet_onnx.int16)
    zero_weights = result.get_zero_weights()
    _calibrate(sim, zero_weights)
    freeze_base_model(sim, result)

    # Calibrate with adapter B
    adapter_b = result.get_adapter("B")
    unfreeze_lora_quantizers(sim, result)
    _calibrate(sim, adapter_b)

    encodings_b = get_lora_encodings(sim, result)
    assert len(encodings_b) > 0, "Should have captured at least one encoding"

    # Calibrate with adapter C (overwrites B's encodings)
    adapter_c = result.get_adapter("C")
    unfreeze_lora_quantizers(sim, result)
    _calibrate(sim, adapter_c)

    # Restore adapter B encodings
    restored = set_lora_encodings(sim, result, encodings_b)
    assert restored == len(encodings_b), (
        f"Expected {len(encodings_b)} restored, got {restored}"
    )

    encodings_b_restored = get_lora_encodings(sim, result)
    for name in encodings_b:
        assert name in encodings_b_restored, (
            f"Encoding for '{name}' should be present after restore"
        )
        assert encodings_b[name] == encodings_b_restored[name], (
            f"Encoding values for '{name}' should match after restore"
        )


# =========================================================================
# Test: export_lora_weights validation
# =========================================================================


def test_export_lora_weights_rejects_unknown_names(prepared_result):
    """export_lora_weights raises ValueError for unknown weight names."""
    _model, result = prepared_result
    bad_weights = {"completely_unknown_name": np.zeros((8, 64), dtype=np.float32)}

    with tempfile.TemporaryDirectory() as output_dir:
        path = os.path.join(output_dir, "test.safetensors")
        with pytest.raises(ValueError, match="not in lora_input_names"):
            export_lora_weights(result, bad_weights, path)


# =========================================================================
# Test: Per-channel quantization for LoRA weights
# =========================================================================


def test_lora_per_channel(prepared_result):
    """LoRA quantizers get per-channel mode when kept as initializers during QuantSim creation."""
    model, result = prepared_result

    # LoRA should be initializers at this point (not yet graph inputs)
    init_names = {init.name for init in model.graph.initializer}
    for name in result.lora_input_names:
        assert name in init_names, f"{name} should be an initializer before configure"

    # Create QuantSim — LoRA classified as params → per-channel
    dummy_input = {
        "input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64),
    }
    sim = QuantizationSimModel(model, dummy_input=dummy_input)

    # Verify per-channel mode on LoRA quantizers
    for name in result.lora_input_names:
        if name in sim.qc_quantize_op_dict:
            qtzr = sim.qc_quantize_op_dict[name]
            assert qtzr.quant_info.usePerChannelMode, (
                f"{name} should be per-channel before configure"
            )

    # configure_lora_quantizers converts to graph inputs
    configure_lora_quantizers(sim, result, lora_param_type=aimet_onnx.int16)

    # After conversion, LoRA should be graph inputs (not initializers)
    # Note: sim.model.model is the model proto held by QuantSim internally
    init_names_after = {init.name for init in sim.model.model.graph.initializer}
    input_names_after = {inp.name for inp in sim.model.model.graph.input}
    for name in result.lora_input_names:
        assert name not in init_names_after, (
            f"{name} should not be an initializer after configure"
        )
        assert name in input_names_after, (
            f"{name} should be a graph input after configure"
        )

    # Per-channel mode should persist after conversion
    for name in result.lora_input_names:
        if name in sim.qc_quantize_op_dict:
            qtzr = sim.qc_quantize_op_dict[name]
            assert qtzr.quant_info.usePerChannelMode, (
                f"{name} should still be per-channel after configure"
            )


# =========================================================================
# Test: calibrate_lora + export_lora convenience functions
# =========================================================================


def test_calibrate_and_export_lora_ort(prepared_result):
    """calibrate_lora() + export_lora(target='ort') produce ORT artifacts."""
    model, result = prepared_result

    sim = _make_sim(model)

    dataloader = [
        {"input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64)}
        for _ in range(3)
    ]

    calibrate_lora(sim, result, dataloader, lora_param_type=aimet_onnx.int16)

    assert "B" in result.adapter_encodings
    assert "C" in result.adapter_encodings
    assert len(result.adapter_encodings["B"]) > 0

    with tempfile.TemporaryDirectory() as export_dir:
        export_lora(sim, result, export_dir, target="ort")

        # 1-2. Base model + base encodings
        assert os.path.exists(os.path.join(export_dir, "model.onnx"))
        assert os.path.exists(os.path.join(export_dir, "model.encodings"))

        # 3. Per-adapter encodings
        assert os.path.exists(os.path.join(export_dir, "model_B.encodings"))
        assert os.path.exists(os.path.join(export_dir, "model_C.encodings"))

        # 4. Per-adapter safetensors
        assert os.path.exists(os.path.join(export_dir, "B.safetensors"))
        assert os.path.exists(os.path.join(export_dir, "C.safetensors"))

        from safetensors.numpy import load_file

        for adapter_name in ("B", "C"):
            reloaded = load_file(
                os.path.join(export_dir, f"{adapter_name}.safetensors")
            )
            original = result.get_adapter(adapter_name)
            assert set(reloaded.keys()) == set(original.keys()), (
                f"Adapter {adapter_name}: safetensors keys should match adapter keys"
            )

        # ORT target should NOT produce QAIRT-specific files
        assert not os.path.exists(os.path.join(export_dir, "lora_weight_list.txt"))
        assert not os.path.exists(os.path.join(export_dir, "model_lora_init.onnx"))
        assert not os.path.exists(os.path.join(export_dir, "lora_config.yaml"))
        assert not os.path.exists(os.path.join(export_dir, "lora_adaptor_list.yaml"))


def test_calibrate_and_export_lora_qairt(prepared_result):
    """calibrate_lora() + export_lora(target='qairt') produce all QAIRT artifacts."""
    model, result = prepared_result

    sim = _make_sim(model)

    dataloader = [
        {"input_ids": np.random.randint(0, 512, (1, 32)).astype(np.int64)}
        for _ in range(3)
    ]

    calibrate_lora(sim, result, dataloader, lora_param_type=aimet_onnx.int16)

    with tempfile.TemporaryDirectory() as export_dir:
        export_lora(sim, result, export_dir, target="qairt")

        # 1-2. Base model + base encodings
        assert os.path.exists(os.path.join(export_dir, "model.onnx"))
        assert os.path.exists(os.path.join(export_dir, "model.encodings"))

        # 3. Per-adapter encodings
        assert os.path.exists(os.path.join(export_dir, "model_B.encodings"))
        assert os.path.exists(os.path.join(export_dir, "model_C.encodings"))

        # 4. Per-adapter safetensors
        assert os.path.exists(os.path.join(export_dir, "B.safetensors"))
        assert os.path.exists(os.path.join(export_dir, "C.safetensors"))

        # 5. LoRA weight list
        weight_list_path = os.path.join(export_dir, "lora_weight_list.txt")
        assert os.path.exists(weight_list_path)
        with open(weight_list_path) as f:
            weight_names = [line.strip() for line in f if line.strip()]
        assert set(weight_names) == set(result.lora_input_names)

        # 6. Initializer model
        assert os.path.exists(os.path.join(export_dir, "model_lora_init.onnx"))
        init_model = onnx.load(
            os.path.join(export_dir, "model_lora_init.onnx"),
            load_external_data=False,
        )
        init_names = {init.name for init in init_model.graph.initializer}
        for name in result.lora_input_names:
            assert name in init_names, (
                f"LoRA input '{name}' should be an initializer in the init model"
            )

        # 7. lora_config.yaml
        config_path = os.path.join(export_dir, "lora_config.yaml")
        assert os.path.exists(config_path)
        with open(config_path) as f:
            content = f.read()
        assert "use_case:" in content
        assert "model_lora_init.onnx" in content
        assert "B.safetensors" in content
        assert "C.safetensors" in content
        assert "model_B.encodings" in content
        assert "model_C.encodings" in content

        # 8. lora_adaptor_list.yaml
        adaptor_path = os.path.join(export_dir, "lora_adaptor_list.yaml")
        assert os.path.exists(adaptor_path)
        with open(adaptor_path) as f:
            content = f.read()
        assert "B" in content
        assert "C" in content


# =========================================================================
# Test: _onnx_name_to_safetensors_key name transform
# =========================================================================


def test_onnx_name_to_safetensors_key():
    """Deterministic name transform strips model. prefix and adapter name."""
    from aimet_onnx.experimental.lora.peft_to_onnx import _onnx_name_to_safetensors_key

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
    from safetensors.numpy import save_file

    # Create a fake PEFT adapter directory with safetensors in PEFT-saved format
    # (keys without adapter name, as PEFT saves them)
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

    # ONNX names have model. prefix + .default. adapter name
    onnx_input_names = [
        "model.base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
        "model.base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight",
        "model.base_model.model.layers.0.self_attn.v_proj.lora_A.default.weight",
        "model.base_model.model.layers.0.self_attn.v_proj.lora_B.default.weight",
    ]

    with tempfile.TemporaryDirectory() as adapter_dir:
        save_file(sf_weights, os.path.join(adapter_dir, "adapter_model.safetensors"))

        mapped = _load_adapter_safetensors(adapter_dir, onnx_input_names)

    assert len(mapped) == 4, f"Expected 4 mapped weights, got {len(mapped)}"
    for onnx_name in onnx_input_names:
        assert onnx_name in mapped, f"Missing {onnx_name}"
        assert isinstance(mapped[onnx_name], np.ndarray)

    # Verify the actual weight values match
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
    from safetensors.numpy import save_file

    sf_weights = {
        "base_model.model.layers.0.lora_A.weight": np.zeros((4, 8), dtype=np.float32),
    }

    onnx_names = [
        "model.base_model.model.layers.0.lora_A.default.weight",
        "model.base_model.model.layers.0.lora_B.default.weight",  # no match in safetensors
    ]

    with tempfile.TemporaryDirectory() as adapter_dir:
        save_file(sf_weights, os.path.join(adapter_dir, "adapter_model.safetensors"))

        with pytest.raises(ValueError, match="Only mapped 1/2"):
            _load_adapter_safetensors(adapter_dir, onnx_names)


# =========================================================================
# Test: _extract_output handles various model output types
# =========================================================================


def test_extract_output():
    """_extract_output handles tensors, structured outputs, and tuples."""
    from aimet_onnx.experimental.lora.peft_to_onnx import _extract_output

    # Raw tensor — returned as-is
    t = torch.randn(2, 3)
    assert _extract_output(t) is t

    # Object with .logits
    class CausalLMOutput:
        def __init__(self):
            self.logits = torch.randn(2, 3)

    out = CausalLMOutput()
    assert _extract_output(out) is out.logits

    # Object with .sample (UNet)
    class UNetOutput:
        def __init__(self):
            self.sample = torch.randn(1, 4, 64, 64)

    out = UNetOutput()
    assert _extract_output(out) is out.sample

    # Object with .last_hidden_state
    class EncoderOutput:
        def __init__(self):
            self.last_hidden_state = torch.randn(1, 10, 768)

    out = EncoderOutput()
    assert _extract_output(out) is out.last_hidden_state

    # Tuple fallback — returns first element
    t0 = torch.randn(2, 3)
    assert _extract_output((t0, torch.randn(2, 3))) is t0


# =========================================================================
# Test: _infer_input_names from forward signature
# =========================================================================


def test_infer_input_names():
    """_infer_input_names extracts required param names from forward signature."""
    from aimet_onnx.experimental.lora.peft_to_onnx import _infer_input_names

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
    from aimet_onnx.experimental.lora.peft_to_onnx import _infer_input_names

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
    from aimet_onnx.experimental.lora.peft_to_onnx import _infer_input_names

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
    from aimet_onnx.experimental.lora.peft_to_onnx import _infer_dynamic_shapes

    inputs = (torch.zeros(1, 10), torch.tensor(1.0))
    names = ["input_ids", "timestep"]
    shapes = _infer_dynamic_shapes(inputs, names)

    # Returns tuple (positional) because _Wrapper.forward uses *args
    assert isinstance(shapes, tuple)
    assert len(shapes) == 2
    assert 0 in shapes[0]  # input_ids: dim 0 is dynamic
    assert shapes[1] == {}  # timestep: scalar, no dynamic dims


# =========================================================================
# Test: export_lora target validation
# =========================================================================


def test_load_adapter_safetensors_direct_file():
    """Load adapter weights from a direct .safetensors file path."""
    from safetensors.numpy import save_file

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
        # Save to a direct file path (not adapter_model.safetensors in a dir)
        file_path = os.path.join(tmpdir, "my_adapter.safetensors")
        save_file(sf_weights, file_path)

        mapped = _load_adapter_safetensors(file_path, onnx_input_names)

    assert len(mapped) == 2
    for name in onnx_input_names:
        assert name in mapped


def test_export_lora_invalid_target(prepared_result):
    """export_lora raises ValueError for invalid target."""
    model, result = prepared_result
    sim = _make_sim(model)

    with pytest.raises(ValueError, match="target must be"):
        export_lora(sim, result, "/tmp/test", target="invalid")


def test_export_lora_qairt_requires_model_path(prepared_result):
    """export_lora(target='qairt') raises ValueError when model_path is None."""
    model, result = prepared_result
    result.model_path = None
    sim = _make_sim(model)

    with pytest.raises(ValueError, match="target='qairt' requires result.model_path"):
        export_lora(sim, result, "/tmp/test", target="qairt")

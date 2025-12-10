# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# pylint: disable=missing-module-docstring

"""
Common Utilities for AIMET Torch Feature Runners

This module provides shared functionality used across all AIMET Torch feature runners
(QuantSim, AdaRound, etc.) to ensure consistency and reduce code duplication.

Key Components:
1. Quantization Scheme Mapping - Convert config strings to AIMET enums
2. AIMET QuantSim Construction - Build QuantSim from PyTorch model
3. AIMET Bundle Export - Dual export for validation and AI Hub deployment
4. Cleanup Utilities - Temporary file management

Design Philosophy:
- Device management handled transparently
- QNN compatibility: Export structure matches QNN requirements

Technical Notes:
- Model and inputs must be on CPU for ONNX export
- Dual export: QDQ model for validation, clean bundle for AI Hub
"""

from __future__ import annotations

import os
import shutil
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import onnx

from ONNXRegression.features.torch.utils import ensure_device_patch

from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.common.defs import QuantScheme
from aimet_torch.model_preparer import prepare_model
import aimet_torch
from torch.utils.data import DataLoader
from qai_hub_models.datasets import DatasetSplit, get_dataset_from_name
from qai_hub_models.utils.evaluate import get_deterministic_sample

try:
    from aimet_torch.quantsim_config.quantsim_config import get_path_for_target_config
except ImportError:
    get_path_for_target_config = None

__all__ = [
    "get_torch_model",
    "build_quantsim_torch",
    "export_torch_bundle",
    "map_quant_scheme",
    "bitwidth_from_token",
    "create_dummy_input",
    "create_calibration_dataloader",
]


# ==================== Quantization Scheme Mapping ====================


_QUANT_SCHEME_MAP = {
    "tf": QuantScheme.post_training_tf,
    "tf_enhanced": QuantScheme.post_training_tf_enhanced,
    "percentile": QuantScheme.post_training_percentile,
    "min_max": QuantScheme.post_training_tf,
}


def map_quant_scheme(scheme_str: str) -> QuantScheme:
    """
    Map configuration string to AIMET QuantScheme enum.

    Args:
        scheme_str: Scheme name from config (e.g., "tf_enhanced", "min_max")

    Returns:
        AIMET QuantScheme enum value

    Raises:
        ValueError: If scheme is not recognized
    """
    scheme_lower = scheme_str.lower().strip()

    if scheme_lower not in _QUANT_SCHEME_MAP:
        available = ", ".join(_QUANT_SCHEME_MAP.keys())
        raise ValueError(
            f"Unknown quantization scheme: '{scheme_str}'. Available: {available}"
        )

    return _QUANT_SCHEME_MAP[scheme_lower]


def bitwidth_from_token(token: Optional[Union[str, int]], default: int = 8) -> int:
    """
    Convert various bitwidth representations to integer.

    Args:
        token: Bitwidth specification (None, int, or string like "int8")
        default: Default bitwidth if token is None or unparseable

    Returns:
        Integer bitwidth (typically 4, 8, or 16)
    """
    if token is None:
        return default

    try:
        return int(token)
    except (TypeError, ValueError):
        pass

    token_str = str(token).lower()

    if "16" in token_str:
        return 16
    if "4" in token_str:
        return 4
    if "8" in token_str:
        return 8

    return default


# ==================== Model Extraction ====================


def get_torch_model(model: Any, device: torch.device = None) -> torch.nn.Module:
    """
    Get the PyTorch model from a QAI Hub model wrapper.

    Args:
        model: QAI Hub model instance or nn.Module
        device: Target device (default: CUDA if available, else CPU)

    Returns:
        PyTorch nn.Module ready for AIMET
    """
    ensure_device_patch()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hasattr(model, "to_torch_model"):
        torch_model = model.to_torch_model()
    elif isinstance(model, torch.nn.Module):
        torch_model = model
    else:
        raise TypeError(
            f"Cannot extract torch model from {type(model).__name__}. "
            f"Expected QAI Hub model or nn.Module."
        )

    torch_model = torch_model.to(device).eval()

    return torch_model


# ==================== AIMET QuantSim Construction ====================


def build_quantsim_torch(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    *,
    quant_scheme: str = "tf_enhanced",
    default_param_bw: int = 8,
    default_output_bw: int = 8,
    config_file: Optional[str] = None,
    apply_prepare_model: bool = True,
    use_cuda: bool = True,
) -> QuantizationSimModel:
    """
    Build an AIMET Torch QuantizationSimModel.

    Args:
        model: PyTorch model (nn.Module) in eval mode
        dummy_input: Sample input tensor on same device as model
        quant_scheme: Quantization scheme name
        default_param_bw: Parameter/weight bitwidth
        default_output_bw: Output/activation bitwidth (note: Torch uses output_bw)
        config_file: Optional AIMET config file path or built-in name (e.g., "htp_v79")
        apply_prepare_model: Whether to apply prepare_model() for AIMET compatibility

    Returns:
        Tuple of QuantizationSimModel
    """
    device = next(model.parameters()).device

    if dummy_input.device != device:
        dummy_input = dummy_input.to(device)

    if apply_prepare_model:
        print(f"[QuantSim Torch] Applying prepare_model() for AIMET compatibility...")
        model = prepare_model(model)
        model = model.to(device).eval()

    model.eval()

    scheme_enum = map_quant_scheme(quant_scheme)

    print(
        f"[QuantSim Torch] Building with scheme={quant_scheme}, W{default_param_bw}A{default_output_bw}"
    )
    print(f"[QuantSim Torch] Device: {device}")
    print(f"[QuantSim Torch] Config: {str(config_file)}")

    sim = QuantizationSimModel(
        model=model,
        dummy_input=dummy_input,
        quant_scheme=scheme_enum,
        default_param_bw=default_param_bw,
        default_output_bw=default_output_bw,
        config_file=str(config_file),
    )

    return sim


# ==================== AIMET Bundle Export ====================


def export_torch_bundle(
    sim: QuantizationSimModel,
    dummy_input: torch.Tensor,
    export_dir: Path,
    model_name: str,
    input_spec: Dict[str, Any],
) -> Tuple[Path, Path]:
    """
    Export AIMET Torch QuantSim with dual output format.

    Creates two artifacts:
    1. QDQ ONNX model (for local validation)
    2. AIMET bundle (ONNX + encodings for QNN)

    After export, renames ONNX inputs to match input_spec for QNN compatibility.

    Export structure:
        <export_dir>/
        ├── <model_name>_qdq.onnx          (QDQ for validation)
        └── <model_name>.aimet/            (Bundle for QNN)
            ├── <model_name>.onnx
            └── <model_name>.encodings

    Args:
        sim: QuantizationSimModel after compute_encodings()
        dummy_input: Dummy input (must be on CPU per AIMET requirements)
        export_dir: Parent directory for exports
        model_name: Model name for file naming
        input_spec: Input specification dict to get expected input name

    Returns:
        Tuple of (qdq_path, bundle_dir)

    Important:
        dummy_input MUST be on CPU for both exports, regardless of where
        the model is located. AIMET documentation explicitly requires this.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    dummy_input_cpu = dummy_input.cpu() if dummy_input.is_cuda else dummy_input

    print(f"[AIMET Torch] Exporting QDQ model and bundle")

    qdq_path = export_dir / f"{model_name}_qdq.onnx"

    aimet_torch.onnx.export(
        sim.model,
        (dummy_input_cpu,),
        str(qdq_path),
        dynamo=False,
        opset_version=21,  # For INT4/INT16 support
    )

    print(f"[AIMET Torch] QDQ model: {qdq_path}")

    bundle_dir = export_dir / f"{model_name}.aimet"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_onnx_path = bundle_dir / f"{model_name}.onnx"

    sim.onnx.export(
        (dummy_input_cpu,),
        str(bundle_onnx_path),
        dynamo=False,
        opset_version=20,
        encoding_version="1.0.0",
    )

    print(f"[AIMET Torch] AIMET bundle: {bundle_dir}")

    clean_extra_bundle_files(bundle_dir, model_name)

    print(f"[AIMET Torch] Validating exports...")

    if not qdq_path.exists():
        raise RuntimeError(f"QDQ export failed: {qdq_path}")

    bundle_onnx = bundle_dir / f"{model_name}.onnx"
    if not bundle_onnx.exists():
        actual_files = list(bundle_dir.glob("*"))
        raise RuntimeError(
            f"Bundle ONNX not found: {bundle_onnx}\nBundle contents: {actual_files}"
        )

    enc_candidates = [
        bundle_dir / f"{model_name}.encodings",
        bundle_dir / f"{model_name}.encodings.json",
    ]
    enc_path = next((p for p in enc_candidates if p.exists()), None)

    if not enc_path:
        actual_files = list(bundle_dir.glob("*"))
        raise RuntimeError(f"Encodings file not found\nBundle contents: {actual_files}")

    # Rename ONNX inputs to match input_spec for QNN compatibility
    expected_input_name = next(iter(input_spec.keys())) if input_spec else None
    if expected_input_name:
        _rename_onnx_input(qdq_path, expected_input_name)
        _rename_onnx_input(bundle_onnx, expected_input_name)

    print(f"[AIMET Torch] ✅ Export validation passed")
    print(f"  QDQ: {qdq_path.name}")
    print(f"  Bundle ONNX: {bundle_onnx.name}")
    print(f"  Encodings: {enc_path.name}")

    return qdq_path, bundle_dir


def _rename_onnx_input(onnx_path: Path, new_input_name: str) -> None:
    """
    Rename the first input of an ONNX model to match expected name.

    AIMET export may rename inputs (e.g., "image_tensor" -> "t.1").
    This renames them back for QNN inference compatibility.

    Args:
        onnx_path: Path to ONNX model file
        new_input_name: Expected input name from input_spec
    """
    try:
        onnx_model = onnx.load(str(onnx_path))

        if not onnx_model.graph.input:
            return

        old_name = onnx_model.graph.input[0].name

        if old_name == new_input_name:
            return

        print(f"[AIMET Torch] Renaming ONNX input: '{old_name}' -> '{new_input_name}'")

        # Update input name
        onnx_model.graph.input[0].name = new_input_name

        # Update all nodes that reference this input
        for node in onnx_model.graph.node:
            for i, input_name in enumerate(node.input):
                if input_name == old_name:
                    node.input[i] = new_input_name

        onnx.save(onnx_model, str(onnx_path))

    except Exception as e:
        print(f"[AIMET Torch] Warning: Could not rename input: {e}")


def clean_extra_bundle_files(bundle_dir: Path, model_name: str) -> None:
    """
    Remove extraneous files from AIMET bundle, keeping only .onnx and .encodings.

    AIMET export may create additional files that QNN doesn't need. This function
    ensures the bundle contains only the required files.

    Args:
        bundle_dir: Bundle directory path
        model_name: Model name (for identifying files to keep)

    Keeps:
        - <model_name>.onnx
        - <model_name>.encodings (or .encodings.json)

    Removes:
        - Everything else
    """
    bundle_dir = Path(bundle_dir)

    keep_patterns = [
        f"{model_name}.onnx",
        f"{model_name}.encodings",
        f"{model_name}.encodings.json",
    ]

    all_files = list(bundle_dir.glob("*"))

    for file_path in all_files:
        if file_path.name not in keep_patterns:
            try:
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path, ignore_errors=True)
            except Exception:
                pass


# ==================== Dummy Input Creation ====================


def create_dummy_input(
    input_spec: Dict[str, Any], device: torch.device
) -> torch.Tensor:
    """
    Create a dummy input tensor from input specification.

    Args:
        input_spec: Input specification from model.get_input_spec()
        device: Target device for the tensor

    Returns:
        Dummy input tensor on specified device
    """
    from qai_hub_models.utils.input_spec import make_torch_inputs

    sample_inputs = make_torch_inputs(input_spec)

    if isinstance(sample_inputs, dict):
        dummy_input = next(iter(sample_inputs.values()))
    elif isinstance(sample_inputs, (list, tuple)):
        dummy_input = sample_inputs[0] if sample_inputs else sample_inputs
    else:
        dummy_input = sample_inputs

    if dummy_input.dim() == 3:
        dummy_input = dummy_input.unsqueeze(0)

    return dummy_input.to(device)


def create_calibration_dataloader(
    qai_hub_model: Any,
    dataset_name: str,
    num_samples: int,
    batch_size: int = 1,
) -> DataLoader:
    """
    Create a DataLoader for AIMET calibration/optimization.

    Args:
            qai_hub_model: QAI Hub model object
            dataset_name: Dataset to sample from
            num_samples: Number of samples to use
            batch_size: Batch size for the dataloader

    Returns:
            DataLoader yielding (input_tensor, label) tuples
    """
    dataset = get_dataset_from_name(dataset_name, DatasetSplit.VAL)

    sampler = get_deterministic_sample(
        dataset, num_samples=num_samples, samples_per_job=1
    )

    inputs_list = []
    labels_list = []

    for sample in sampler:
        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
            inputs, label = sample[0], sample[1]
        elif isinstance(sample, (list, tuple)) and len(sample) == 1:
            inputs, label = sample[0], 0
        else:
            inputs, label = sample, 0

        if isinstance(inputs, dict):
            inputs = next(iter(inputs.values()))
        elif isinstance(inputs, (list, tuple)):
            inputs = inputs[0]

        if isinstance(inputs, torch.Tensor):
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(0)
            inputs_list.append(inputs)

        if isinstance(label, torch.Tensor):
            labels_list.append(label.item() if label.numel() == 1 else 0)
        else:
            labels_list.append(int(label) if label is not None else 0)

    if not inputs_list:
        raise ValueError(f"No samples collected from dataset {dataset_name}")

    all_inputs = torch.cat(inputs_list, dim=0)
    all_labels = torch.tensor(labels_list, dtype=torch.long)

    tensor_dataset = torch.utils.data.TensorDataset(all_inputs, all_labels)
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)

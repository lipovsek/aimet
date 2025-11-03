# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# pylint: disable=missing-module-docstring

"""
Common Utilities for AIMET Feature Runners

This module provides shared functionality used across all AIMET feature runners
(QuantSim, Lite-MP, AdaRound, etc.) to ensure consistency and reduce code
duplication.

Key Components:
1. ORT Provider Management - Explicit provider selection without silent fallback
2. AIMET QuantSim Construction - Reliable model building using in-memory ModelProto
3. AIMET Bundle Export - Direct export to Qualcomm AI Hub .aimet bundle format
4. Cleanup Utilities - Temporary file management

Design Philosophy:
- Defensive programming: Validate inputs and handle edge cases
- Explicit behavior: No silent fallbacks or implicit conversions
- Consistent interfaces: All feature runners use the same patterns
- QNN compatibility: Export structure matches QNN requirements

Technical Notes:
- QuantSim uses in-memory ModelProto to avoid temporary file issues seen in
  some AIMET/ORT version combinations
- Bundle structure follows Qualcomm AI Hub specifications for AIMET models
- Provider selection is explicit to prevent performance surprises
- QDQ validation ensures proper quantization export

"""

from __future__ import annotations

import os
import shutil
from glob import glob
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Union

import onnx
import onnxruntime as ort
from aimet_onnx.quantsim import QuantizationSimModel

# Public API - these functions are used by feature runners
__all__ = [
    "pick_providers",
    "make_session",
    "build_quantsim",
    "export_aimet_bundle",
    "clean_dir",
]


# ==================== ORT Provider Management ====================


def pick_providers(preferred: Iterable[str]) -> List[str]:
    """
    Select available ORT execution providers from a preference list.

    This function ensures explicit provider selection without silent CPU fallback.
    Silent fallback can mask configuration issues and lead to unexpected
    performance degradation (e.g., thinking you're using GPU but actually on CPU).

    Args:
        preferred: Ordered list of preferred provider names.
                  Common providers:
                  - "CUDAExecutionProvider" (NVIDIA GPU)
                  - "DmlExecutionProvider" (DirectML for Windows)
                  - "CoreMLExecutionProvider" (macOS/iOS)
                  - "CPUExecutionProvider" (fallback)

    Returns:
        List of available providers from the preference list, maintaining
        the original preference order.

    Raises:
        RuntimeError: If none of the requested providers are available.
                     This is intentional - we want to fail fast rather than
                     silently degrade performance.

    Example:
        >>> providers = pick_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])
        >>> # Returns ["CUDAExecutionProvider"] if CUDA is available
        >>> # Returns ["CPUExecutionProvider"] if only CPU is available
        >>> # Raises RuntimeError if neither is available
    """
    available = set(ort.get_available_providers())
    chosen = [p for p in preferred if p in available]

    if not chosen:
        raise RuntimeError(
            f"No requested ORT providers available. "
            f"Requested={list(preferred)}, Available={sorted(available)}. "
            f"Please install required runtime or adjust provider preferences."
        )

    return chosen


def make_session(
    onnx_path: Union[str, Path], providers: Tuple[str, ...]
) -> ort.InferenceSession:
    """
    Create an ORT InferenceSession with explicit provider configuration.

    This ensures the session uses only the specified providers without
    implicit fallbacks, which is crucial for consistent performance testing
    and debugging.

    Args:
        onnx_path: Path to ONNX model file
        providers: Tuple of provider names to use (order matters for priority)

    Returns:
        Configured ORT InferenceSession ready for inference

    Note:
        Log severity is set to WARNING to reduce noise during evaluation
        while still capturing important issues.
    """
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 2  # 0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR

    return ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=list(providers),  # Convert tuple to list for ORT
    )


# ==================== AIMET QuantSim Construction ====================


def _bitwidth_from_token(token: Optional[Union[str, int]], default: int = 8) -> int:
    """
    Convert various bitwidth representations to integer.

    This function provides flexibility in configuration files, accepting
    multiple formats for specifying precision while normalizing to integers
    for AIMET.

    Args:
        token: Bitwidth specification, can be:
               - None: Returns default
               - Integer: Returned as-is (4, 8, 16)
               - String: "int8", "8", "int16", "16", "int4", "4", etc.
        default: Default bitwidth if token is None or unparseable

    Returns:
        Integer bitwidth (typically 4, 8, or 16)

    Examples:
        >>> _bitwidth_from_token("int8")  # Returns 8
        >>> _bitwidth_from_token("16")    # Returns 16
        >>> _bitwidth_from_token(None)    # Returns 8 (default)
        >>> _bitwidth_from_token("fp16")  # Returns 16 (extracts number)
    """
    if token is None:
        return default

    # Try direct integer conversion first
    try:
        return int(token)
    except (TypeError, ValueError):
        pass

    # Parse string representations
    token_str = str(token).lower()

    # Look for common bitwidth numbers in the string
    if "16" in token_str:
        return 16
    if "4" in token_str:
        return 4
    if "8" in token_str:
        return 8

    return default


def build_quantsim(
    fp32_or_fpN_onnx_path: Union[str, Path],
    *,
    scheme: str,
    param_type: Union[str, int],
    activation_type: Union[str, int],
    config_file: Optional[str],
    use_cuda: bool,
) -> QuantizationSimModel:
    """
    Construct an AIMET QuantizationSimModel from an ONNX file.

    CRITICAL DESIGN DECISION:
    This function loads the ONNX file and passes an in-memory ModelProto to
    AIMET rather than a file path. This avoids a known issue where some
    AIMET/ORT version combinations create temporary files that immediately
    disappear, causing 'NoSuchFile: /tmp/.../model.onnx' errors.

    The in-memory approach is more reliable across different environments and
    AIMET versions, though it uses slightly more memory during construction.

    Args:
        fp32_or_fpN_onnx_path: Path to the source ONNX model (typically FP32)
        scheme: Quantization scheme - one of:
                - "tf_enhanced": TensorFlow-style with enhanced range
                - "tf": Standard TensorFlow quantization
                - "percentile": Percentile-based range selection
                - "entropy": Entropy-based calibration
        param_type: Parameter/weight precision (e.g., "int8", 8, "int4")
        activation_type: Activation precision (e.g., "int8", 8, "int16")
        config_file: Optional path to AIMET JSON configuration file
                    (for advanced per-layer settings)
        use_cuda: Whether to use CUDA acceleration if available

    Returns:
        Configured QuantizationSimModel ready for calibration

    Raises:
        FileNotFoundError: If the ONNX file doesn't exist
        Exception: If AIMET fails to construct the QuantSim

    Note:
        The ModelProto is loaded once and kept in memory during QuantSim
        construction. This is intentional to avoid file system race conditions.
    """
    # Load ONNX model into memory as ModelProto
    if not Path(fp32_or_fpN_onnx_path).exists():
        raise FileNotFoundError(f"ONNX model not found: {fp32_or_fpN_onnx_path}")

    model_proto = onnx.load(str(fp32_or_fpN_onnx_path))

    # Parse bitwidth specifications
    param_bw = _bitwidth_from_token(param_type, 8)
    activation_bw = _bitwidth_from_token(activation_type, 8)

    # Build QuantSim with in-memory model
    # IMPORTANT: Pass ModelProto object, not file path
    return QuantizationSimModel(
        model=model_proto,  # In-memory ModelProto, not file path
        quant_scheme=scheme,
        default_param_bw=param_bw,
        default_activation_bw=activation_bw,
        config_file=config_file,
        use_cuda=bool(use_cuda),
    )


# ==================== AIMET Bundle Export ====================


def export_aimet_bundle(
    sim: QuantizationSimModel, export_dir: Union[str, Path], model_name: str
) -> Path:
    """
    Export AIMET QuantSim to QDQ ONNX format for Qualcomm AI Hub.

    This function exports the quantized model using AIMET's to_onnx_qdq() API, creating:
    1. QDQ ONNX model with QuantizeLinear/DequantizeLinear operators
    2. Encodings JSON file with quantization parameters (scale/zero-point)

    The export uses AIMET's native QDQ conversion which replaces QcQuantizeOp nodes
    with standard QuantizeLinear/DequantizeLinear operators, making it compatible with
    QNN compilation on Qualcomm hardware.

    Bundle structure created:
        <export_dir>/<model_name>.aimet/
        |-- <model_name>.onnx          (QDQ-quantized ONNX model)
        +-- <model_name>.encodings     (quantization parameters JSON)

    Args:
        sim: QuantizationSimModel after compute_encodings() has been called
        export_dir: Parent directory for the .aimet bundle
        model_name: Model name used for bundle and file naming

    Returns:
        Path to created .aimet bundle directory

    Raises:
        RuntimeError: If export fails, files are missing, or QDQ validation fails

    Example:
        >>> sim = build_quantsim(...)
        >>> sim.compute_encodings(...)
        >>> bundle = export_aimet_bundle(sim, "artifacts/resnet50", "resnet50")
        >>> # Creates: artifacts/resnet50/resnet50.aimet/resnet50.onnx
        >>> #          artifacts/resnet50/resnet50.aimet/resnet50.encodings

    Technical Notes:
        - Uses sim.to_onnx_qdq() to convert QcQuantizeOp to QuantizeLinear/DequantizeLinear
        - Uses sim.export() to save encodings file
        - QDQ nodes are inserted at quantization boundaries
        - Encodings file is JSON format with per-layer quantization params
        - Export will fail immediately if QuantSim is not properly calibrated

    Reference:
        AIMET ONNX API: https://quic.github.io/aimet-pages/releases/latest/apiref/onnx/quantsim.html
    """
    export_dir = Path(export_dir)

    # Create .aimet bundle directory
    bundle_dir = export_dir / f"{model_name}.aimet"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    print(f"[AIMET] Exporting QDQ ONNX model to: {bundle_dir}")
    print(f"[AIMET] Converting QcQuantizeOp to QuantizeLinear/DequantizeLinear...")

    # ============================================================
    # Step 1: Convert to QDQ format using to_onnx_qdq()
    # ============================================================
    # This replaces all QcQuantizeOp nodes with standard ONNX
    # QuantizeLinear/DequantizeLinear operators
    qdq_model = sim.to_onnx_qdq(prequantize_constants=False)

    # Save the QDQ model
    onnx_path = bundle_dir / f"{model_name}.onnx"
    onnx.save(qdq_model, str(onnx_path))
    print(f"[AIMET] Saved QDQ ONNX model: {onnx_path}")

    # ============================================================
    # Step 2: Export encodings using sim.export()
    # ============================================================
    # The export() method signature: export(path, filename_prefix, export_model=True)
    # We set export_model=False since we already saved the QDQ model above
    sim.export(
        path=str(bundle_dir),
        filename_prefix=model_name,
        export_model=False,  # Don't export model again, only encodings
    )
    print(f"[AIMET] Exported encodings file")

    print(f"[AIMET] Export completed, validating outputs...")

    # ============================================================
    # Validate ONNX file exists
    # ============================================================
    if not onnx_path.exists():
        # List actual contents for debugging
        actual_files = list(bundle_dir.glob("*"))
        raise RuntimeError(
            f"AIMET export failed: ONNX file not created\n"
            f"Expected: {onnx_path}\n"
            f"Bundle directory contents: {actual_files}\n"
            f"This indicates QDQ conversion failed."
        )

    # ============================================================
    # Validate encodings file exists
    # ============================================================
    # AIMET creates .encodings (JSON format)
    # Handle both .encodings and .encodings.json for compatibility
    enc_candidates = [
        bundle_dir / f"{model_name}.encodings",
        bundle_dir / f"{model_name}.encodings.json",
    ]
    enc_path = next((p for p in enc_candidates if p.exists()), None)

    if not enc_path:
        actual_files = list(bundle_dir.glob("*"))
        raise RuntimeError(
            f"AIMET export failed: Encodings file not created\n"
            f"Expected: {model_name}.encodings or {model_name}.encodings.json\n"
            f"Bundle directory contents: {actual_files}\n"
            f"This indicates QuantSim was not properly calibrated."
        )

    # ============================================================
    # CRITICAL: Validate this is actually a QDQ graph
    # ============================================================
    # Verify that the exported ONNX contains QuantizeLinear and
    # DequantizeLinear operators. Without these, the model is not
    # properly quantized and will fail QNN compilation.
    print(f"[AIMET] Validating QDQ graph structure...")

    model_proto = onnx.load(str(onnx_path))

    # Count QDQ operators in the graph
    quantize_ops = [
        node for node in model_proto.graph.node if node.op_type == "QuantizeLinear"
    ]
    dequantize_ops = [
        node for node in model_proto.graph.node if node.op_type == "DequantizeLinear"
    ]

    total_qdq_ops = len(quantize_ops) + len(dequantize_ops)

    if total_qdq_ops == 0:
        raise RuntimeError(
            f"QDQ validation failed: No QuantizeLinear/DequantizeLinear operators found!\n"
            f"ONNX path: {onnx_path}\n"
            f"Total nodes: {len(model_proto.graph.node)}\n"
            f"This indicates to_onnx_qdq() failed or returned an FP32 model.\n"
            f"Possible causes:\n"
            f"  1. compute_encodings() was not called before export\n"
            f"  2. QuantSim was created with incorrect parameters\n"
            f"  3. AIMET version mismatch\n"
        )

    # Success! Print detailed validation info
    print(f"[AIMET] ✅ QDQ validation passed:")
    print(f"  QuantizeLinear ops:   {len(quantize_ops)}")
    print(f"  DequantizeLinear ops: {len(dequantize_ops)}")
    print(f"  Total QDQ ops:        {total_qdq_ops}")
    print(f"  ONNX file size:       {onnx_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  Encodings file:       {enc_path.name}")

    return bundle_dir


# ==================== Cleanup Utilities ====================


def clean_dir(dir_path: Union[str, Path], pattern: str) -> None:
    """
    Remove files and directories matching a glob pattern.

    Used for cleaning up temporary files after processing (e.g., AIMET
    creates various .json and .pickle files during operation).

    Errors are silently ignored to ensure cleanup doesn't break the main
    pipeline - cleanup is best-effort.

    Args:
        dir_path: Directory to clean
        pattern: Glob pattern for files/dirs to remove
                Examples: "*.tmp", "tmp_*", "*.pickle", "*.json"

    Example:
        >>> clean_dir("./artifacts", "*.tmp")  # Remove all .tmp files
        >>> clean_dir("./artifacts", "tmp_*")  # Remove all tmp_ prefixed items

    Safety:
        This function will not follow symlinks and ignores permission errors.
        It's designed to be safe for automated cleanup tasks.
    """
    base_path = Path(dir_path)

    # Find all matching paths
    for path in glob(str(base_path / pattern)):
        try:
            if os.path.isdir(path):
                # Remove directory tree
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.isfile(path):
                # Remove single file
                os.remove(path)
            # Ignore symlinks and other special files
        except (OSError, PermissionError):
            # Silently ignore errors - cleanup is best-effort
            pass

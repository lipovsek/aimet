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
3. Export and Bundle Creation - Standardized structure for QNN compatibility
4. Cleanup Utilities - Temporary file management

Design Philosophy:
- Defensive programming: Validate inputs and handle edge cases
- Explicit behavior: No silent fallbacks or implicit conversions
- Consistent interfaces: All feature runners use the same patterns
- QNN compatibility: Export structure matches QNN requirements

Technical Notes:
- QuantSim uses in-memory ModelProto to avoid temporary file issues seen in
  some AIMET/ORT version combinations
- Bundle structure is strictly enforced for QNN toolchain compatibility
- Provider selection is explicit to prevent performance surprises

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
    "export_aimet",
    "aimet_bundle_dir",
    "build_bundle",
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


# ==================== Export and Bundle Helpers ====================


def export_aimet(
    sim: QuantizationSimModel, export_dir: Union[str, Path], model_name: str
) -> Tuple[Path, Path]:
    """
    Export AIMET QuantSim model to QDQ ONNX and encodings file.

    AIMET exports two critical files:
    1. QDQ ONNX: Model with Quantize-Dequantize operators inserted
    2. Encodings: JSON file containing quantization parameters (scale/zero-point)
                 for each quantized tensor

    Both files are required for QNN compilation and deployment.

    Args:
        sim: Trained QuantizationSimModel after compute_encodings()
        export_dir: Directory for exported files (created if doesn't exist)
        model_name: Base name for exported files (without extension)

    Returns:
        Tuple of (onnx_path, encodings_path) as Path objects

    Raises:
        RuntimeError: If AIMET fails to create expected export files

    Note:
        AIMET's export naming convention can vary by version:
        - Older: <name>.onnx and <name>.encodings
        - Newer: <name>.onnx and <name>.encodings.json
        This function handles both conventions.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # AIMET exports with the provided prefix
    sim.export(path=str(export_dir), filename_prefix=model_name)

    # Find exported files (handle version differences in naming)
    onnx_candidates = sorted(export_dir.glob(f"{model_name}*.onnx"))
    enc_candidates = sorted(
        [
            *export_dir.glob(f"{model_name}*.encodings"),
            *export_dir.glob(f"{model_name}*.encodings.json"),
        ]
    )

    # Validate exports
    if not onnx_candidates:
        raise RuntimeError(
            f"AIMET export failed: No ONNX file found in {export_dir} "
            f"with prefix '{model_name}'"
        )

    if not enc_candidates:
        raise RuntimeError(
            f"AIMET export failed: No encodings file found in {export_dir} "
            f"with prefix '{model_name}'"
        )

    # Return the most recent files (last in sorted list)
    # This handles cases where multiple exports might exist
    return onnx_candidates[-1], enc_candidates[-1]


def aimet_bundle_dir(export_dir: Union[str, Path], model_name: str) -> Path:
    """
    Get the standard AIMET bundle directory path.

    The bundle directory follows a consistent naming convention for
    organization and QNN toolchain compatibility.

    Convention: <export_dir>/<model_name>.aimet/

    This directory will contain:
    - <model_name>.onnx (QDQ model)
    - <model_name>.encodings (quantization parameters)

    Args:
        export_dir: Base export directory
        model_name: Model name (used in directory name)

    Returns:
        Path to bundle directory (may not exist yet)
    """
    return Path(export_dir) / f"{model_name}.aimet"


def build_bundle(
    exported_onnx_path: Union[str, Path],
    enc_file: Union[str, Path],
    export_dir: Union[str, Path],
    model_name: str,
) -> Path:
    """
    Create a standardized AIMET bundle for QNN compilation.

    QNN toolchain expects a specific directory structure with consistent naming.
    This function creates a clean bundle directory with properly named files,
    regardless of how AIMET named the exports.

    Args:
        exported_onnx_path: Path to ONNX file exported by AIMET
        enc_file: Path to encodings file exported by AIMET
        export_dir: Base directory where bundle should be created
        model_name: Model name (used for consistent file naming)

    Returns:
        Path to created bundle directory

    Bundle structure created:
        <model_name>.aimet/
        ├── <model_name>.onnx
        └── <model_name>.encodings

    Note:
        Files are copied (not moved) to preserve originals.
        Existing bundle directories are overwritten.
    """
    # Create bundle directory
    bundle = aimet_bundle_dir(export_dir, model_name)
    bundle.mkdir(parents=True, exist_ok=True)

    # Standardize file names within bundle
    dst_onnx = bundle / f"{model_name}.onnx"
    dst_enc = bundle / f"{model_name}.encodings"

    # Copy files with consistent naming
    # Use copy2 to preserve metadata (timestamps)
    shutil.copy2(str(exported_onnx_path), dst_onnx)
    shutil.copy2(str(enc_file), dst_enc)

    return bundle


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

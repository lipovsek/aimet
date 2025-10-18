# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# pylint: disable=missing-module-docstring

"""
ONNXRegression Pipeline Runner

This module orchestrates the complete AIMET quantization evaluation pipeline:
1. Load model from QAI Hub Models
2. Export to FP32 ONNX locally (torch.jit.trace + ONNX conversion)
3. Apply AIMET quantization technique
4. Evaluate accuracy at multiple stages
5. Optionally run on-device evaluation via QNN (AI Hub)
6. Generate comprehensive reports

Key Changes from Original:
- FP32 ONNX export is done locally to reduce AI Hub API usage
- AI Hub is only used for on-device QNN evaluation when specified
- Models are still loaded from QAI Hub Models repository

"""

import os
import sys
import yaml
import onnx
import torch
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple, Any

# ==================== External Imports ====================
from qai_hub import Device
from qai_hub_models.utils.input_spec import make_torch_inputs

# ==================== Internal Imports ====================
from ONNXRegression.models.ai_hub_loader import load_model_data
from ONNXRegression.evaluation.eval_onnx import resolve_dataset_name, eval_onnx_model
from ONNXRegression.evaluation.eval_qnn import (
    compile_and_profile_aimet_bundle,
    eval_qnn_accuracy,
)
from ONNXRegression.report.report_writer import write_csv, write_html

# ==================== Feature Runners ====================
from ONNXRegression.features.quantsim_runner import run_quantsim
from ONNXRegression.features.lite_mp_runner import run_lite_mp
from ONNXRegression.features.adaround_runner import run_adaround


# Registry of available feature runners
FEATURE_RUNNERS: Dict[str, Callable] = {
    "quantsim": run_quantsim,
    "lite_mp": run_lite_mp,
    "adaround": run_adaround,
}

# ==================== Configuration ====================
os.environ.setdefault("TORCH_HOME", "./torch_cache")
os.environ.setdefault("QAIHM_CACHE_DIR", "./qaihm_cache")

ARTIFACTS_DIR = Path("./ONNXRegression/artifacts")
REPORTS_DIR = Path("./ONNXRegression/reports")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ==================== Helper Functions ====================


def _resolve_device(device_name: str) -> Device:
    """
    Convert device name string to AI Hub Device enum.

    Args:
        device_name: Device name as string (e.g., "Samsung Galaxy S24 (Family)")

    Returns:
        Device enum instance

    Raises:
        RuntimeError: If device name is not recognized
    """
    try:
        return Device(device_name)
    except Exception as e:
        raise RuntimeError(
            f"Unknown/unsupported device: '{device_name}'. Error: {e}\n"
            f"Check available devices at: https://app.aihub.qualcomm.com/devices"
        )


def _export_torch_to_onnx_local(
    model: Any, input_spec: Dict, out_dir: Path, model_name: str
) -> Path:
    """
    Export PyTorch model to ONNX locally using torch.jit.trace.

    This function creates a TorchScript traced model and converts it to ONNX,
    bypassing AI Hub compilation entirely. This reduces API usage while
    maintaining compatibility with AIMET.

    The approach uses torch.jit.trace (same as AI Hub does internally) followed
    by ONNX serialization, which handles QAI Hub models' preprocessing correctly.

    Args:
        model: QAI Hub model instance
        input_spec: Input specification dictionary
        out_dir: Directory to save exported ONNX file
        model_name: Name for output file

    Returns:
        Path to exported FP32 ONNX file

    Raises:
        Exception: If tracing or ONNX conversion fails
    """
    print(f"\n[INFO] Exporting {model_name} to FP32 ONNX locally...")
    print(f"[INFO] Using torch.jit.trace (no AI Hub compilation)")

    # Extract PyTorch model
    if hasattr(model, "to_torch_model"):
        torch_model = model.to_torch_model().to("cpu").eval()
    else:
        torch_model = model.to("cpu").eval()

    # Create sample inputs for tracing
    sample_inputs = make_torch_inputs(input_spec)

    # Trace the model (handles preprocessing correctly)
    print(f"[INFO] Tracing model with sample inputs...")
    with torch.no_grad():
        traced = torch.jit.trace(torch_model, sample_inputs, check_trace=False)

    # Convert traced model to ONNX
    fp32_path = out_dir / f"{model_name}_fp32.onnx"

    print(f"[INFO] Converting traced model to ONNX...")

    # Handle input format for ONNX export
    # Critical: Use actual input names from input_spec (not generic names)
    # This ensures QNN inference can match input names correctly
    if isinstance(sample_inputs, dict):
        # Use the actual input names from the model's input_spec
        input_names = list(input_spec.keys())
        input_values = list(sample_inputs.values())

        # For single-input models, extract the tensor from the list
        # For multi-input models, keep as tuple for positional args
        if len(input_values) == 1:
            sample_inputs_for_export = input_values[0]
        else:
            sample_inputs_for_export = tuple(input_values)
    else:
        # Non-dict inputs (already in tuple/tensor format)
        if isinstance(sample_inputs, tuple):
            sample_inputs_for_export = sample_inputs
            # Try to get names from input_spec, fall back to generic
            if input_spec and isinstance(input_spec, dict):
                input_names = list(input_spec.keys())
            else:
                input_names = [f"input_{i}" for i in range(len(sample_inputs))]
        else:
            # Single tensor - try to get name from input_spec
            sample_inputs_for_export = sample_inputs
            if input_spec and isinstance(input_spec, dict):
                input_names = list(input_spec.keys())
            else:
                input_names = ["input"]

    # QNN requires static shapes (no dynamic_axes)
    dynamic_axes = None

    # Export the traced model to ONNX
    # Note: QNN requires static shapes, so we export with fixed batch size (typically 1)
    # For some models, the export might fail on first attempt due to
    # input format mismatch. We handle this gracefully.
    export_success = False
    export_error = None

    with torch.no_grad():
        try:
            torch.onnx.export(
                traced,
                sample_inputs_for_export,
                str(fp32_path),
                input_names=input_names,
                output_names=["output"],
                dynamic_axes=dynamic_axes,  # None for QNN (static shapes required)
                opset_version=13,
                do_constant_folding=True,
                export_params=True,
            )
            export_success = True
        except Exception as e:
            export_error = e
            # If single input failed, try wrapping in tuple (fallback)
            if isinstance(sample_inputs_for_export, torch.Tensor):
                print(
                    f"[INFO] First export attempt failed, trying with tuple wrapper..."
                )
                try:
                    torch.onnx.export(
                        traced,
                        (sample_inputs_for_export,),
                        str(fp32_path),
                        input_names=input_names,
                        output_names=["output"],
                        dynamic_axes=dynamic_axes,
                        opset_version=13,
                        do_constant_folding=True,
                        export_params=True,
                    )
                    export_success = True
                except Exception as e2:
                    export_error = e2

    if not export_success:
        raise RuntimeError(
            f"Failed to export ONNX model: {export_error}\n"
            f"Input format: {type(sample_inputs_for_export)}\n"
            f"Input names: {input_names}"
        )

    print(f"[INFO] FP32 ONNX saved to: {fp32_path}")

    # Validate exported ONNX
    try:
        onnx_model = onnx.load(str(fp32_path))
        onnx.checker.check_model(onnx_model)
        print(f"[INFO] ONNX model validation passed")
    except Exception as e:
        print(f"[WARNING] ONNX validation warning (non-fatal): {e}")

    return fp32_path


def _build_single_batch_loader(
    model: Any, dataset_name: str, input_spec: Dict, num_samples: int
):
    """
    Create a dataloader for QNN evaluation.

    QNN typically expects batch size of 1, so we create a loader that yields
    a single batch containing all samples with individual batch dims.

    Args:
        model: QAI Hub model instance
        dataset_name: Dataset name
        input_spec: Input specification
        num_samples: Number of samples to load

    Returns:
        Iterator yielding (batch_inputs, batch_labels) tuples
    """
    import numpy as np
    import torch
    from qai_hub_models.datasets import DatasetSplit, get_dataset_from_name
    from qai_hub_models.utils.evaluate import get_deterministic_sample

    def to_numpy(x):
        """Convert various types to numpy array."""
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # Load dataset
    dataset = get_dataset_from_name(dataset_name, DatasetSplit.VAL)
    sampler = get_deterministic_sample(
        dataset, num_samples=num_samples, samples_per_job=num_samples
    )

    # Collect samples
    inputs_list = []
    labels_list = []

    for sample in sampler:
        # Extract input and label
        if isinstance(sample, (list, tuple)) and len(sample) == 2:
            inputs, label = sample
        else:
            inputs, label = sample, None

        # Handle different input formats
        if isinstance(inputs, dict):
            x = next(iter(inputs.values()))
        elif isinstance(inputs, (list, tuple)):
            x = inputs[0]
        else:
            x = inputs

        # Convert to numpy and ensure 4D (NCHW)
        x = to_numpy(x)
        if x.ndim == 3:  # CHW -> NCHW
            x = np.expand_dims(x, 0)
        inputs_list.append(x)

        # Process label
        if label is not None:
            if isinstance(label, torch.Tensor):
                labels_list.append(int(label.flatten()[0].item()))
            elif isinstance(label, np.ndarray):
                labels_list.append(int(label.flatten()[0]))
            else:
                labels_list.append(int(label))

    if not inputs_list:
        return [(np.empty((0,)), np.empty((0,), dtype=np.int64))]

    # Concatenate into batches
    batch_inputs = np.concatenate(inputs_list, axis=0)
    batch_labels = (
        np.array(labels_list, dtype=np.int64)
        if labels_list
        else np.empty((0,), dtype=np.int64)
    )

    return [(batch_inputs, batch_labels)]


# ==================== Main Pipeline ====================


def run_single_config(config_path: str) -> Dict[str, Any]:
    """
    Execute the complete evaluation pipeline for a single configuration.

    This function orchestrates:
    1. Load configuration and model
    2. Export FP32 ONNX locally (no AI Hub)
    3. Apply AIMET quantization
    4. Evaluate accuracy at multiple stages
    5. Optionally run on-device QNN evaluation
    6. Generate reports

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary with all evaluation results

    Raises:
        RuntimeError: If configuration is invalid or pipeline fails
        FileNotFoundError: If config file doesn't exist
    """
    # ============ Configuration Loading ============
    print(f"\n{'=' * 60}")
    print(f"Running: {config_path}")
    print(f"{'=' * 60}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract configuration
    model_name = config.get("model_name")
    if not model_name:
        raise RuntimeError("'model_name' not specified in configuration")

    device_name = config.get("device", "Samsung Galaxy S24 (Family)")
    feature_name = str(config.get("feature", "quantsim")).strip().lower()

    if feature_name not in FEATURE_RUNNERS:
        raise RuntimeError(
            f"Unsupported feature: {feature_name}. "
            f"Supported: {list(FEATURE_RUNNERS.keys())}"
        )

    print(f"Model:    {model_name}")
    print(f"Feature:  {feature_name}")
    print(f"Device:   {device_name}")

    # ============ Step 1: Model Loading ============
    print(f"\n[Step 1] Loading model and dataset from QAI Hub Models...")
    model, _dataset, input_spec, _ = load_model_data(model_name)
    dataset_name = resolve_dataset_name(model)
    print(f"Dataset: {dataset_name}")

    # ============ Step 2: FP32 Baseline ============
    # Always use local export (no AI Hub compilation)
    print(f"\n[Step 2] Creating FP32 baseline via local ONNX export...")
    fp32_path = _export_torch_to_onnx_local(
        model, input_spec, ARTIFACTS_DIR, model_name
    )

    # Evaluate FP32 accuracy
    fp32_eval_samples = int(
        config.get("fp32_eval_samples", config.get("quant_eval_samples", 200))
    )
    print(f"[Step 2] Evaluating FP32 accuracy with {fp32_eval_samples} samples...")
    fp32_acc = eval_onnx_model(
        fp32_path, model, dataset_name, num_samples=fp32_eval_samples
    )
    print(f"[Step 2] FP32 Accuracy: {fp32_acc:.4f}")

    # ============ Step 3: AIMET Quantization ============
    print(f"\n[Step 3] Applying {feature_name} quantization...")
    runner = FEATURE_RUNNERS[feature_name]

    aimet_onnx_path, feature_acc, stats, aimet_bundle_dir = runner(
        fp32_onnx_path=str(fp32_path),
        model=model,
        dataset_name=dataset_name,
        config=config,
    )

    # Validate AIMET outputs
    if not aimet_onnx_path:
        raise RuntimeError(f"{feature_name} did not return an exported ONNX path")

    aimet_onnx_path = Path(aimet_onnx_path)
    if not aimet_onnx_path.exists():
        raise FileNotFoundError(f"AIMET-exported ONNX not found: {aimet_onnx_path}")

    print(
        f"[Step 3] AIMET Accuracy: {feature_acc:.4f}"
        if feature_acc
        else "[Step 3] AIMET Accuracy: N/A"
    )

    # ============ Step 4: ONNX Validation ============
    print(f"\n[Step 4] Validating exported quantized ONNX...")
    quant_eval_samples = int(
        config.get("quant_onnx_eval_samples", config.get("quant_eval_samples", 200))
    )
    onnx_acc = eval_onnx_model(
        str(aimet_onnx_path), model, dataset_name, num_samples=quant_eval_samples
    )
    print(f"[Step 4] ONNX Accuracy: {onnx_acc:.4f}")

    # ============ Step 5: QNN On-Device Evaluation (Optional) ============
    qnn_options = config.get("qnn_options")
    qnn_latency_ms = None
    qnn_acc = None
    qnn_job_urls = {}

    if qnn_options:
        print(f"\n[Step 5] Running QNN on-device evaluation...")
        print(f"[Step 5] QNN options: {qnn_options}")

        if not aimet_bundle_dir or not os.path.isdir(str(aimet_bundle_dir)):
            print(
                f"[Step 5] WARNING: AIMET bundle directory missing: {aimet_bundle_dir}"
            )
        else:
            # Compile and profile on device
            try:
                ret = compile_and_profile_aimet_bundle(
                    aimet_bundle_dir=str(aimet_bundle_dir),
                    device_name=device_name,
                    model_name=model_name,
                    export_dir=str(ARTIFACTS_DIR),
                    options=qnn_options,
                )

                if isinstance(ret, (list, tuple)) and len(ret) >= 4:
                    qnn_latency_ms, uploaded_model, _zip, qnn_job_urls = ret
                    print(
                        f"[Step 5] QNN Latency: {qnn_latency_ms:.3f} ms"
                        if qnn_latency_ms
                        else "[Step 5] QNN Latency: N/A"
                    )
                else:
                    uploaded_model = None

            except Exception as e:
                print(f"[Step 5] ERROR: QNN compilation failed: {e}")
                uploaded_model = None

            # Evaluate accuracy on device
            qnn_eval_samples = int(config.get("qnn_eval_samples", 0))
            if qnn_eval_samples > 0 and uploaded_model is not None:
                print(
                    f"[Step 5] Evaluating on-device accuracy with {qnn_eval_samples} samples..."
                )

                try:
                    qnn_loader = _build_single_batch_loader(
                        model, dataset_name, input_spec, qnn_eval_samples
                    )

                    ret_acc = eval_qnn_accuracy(
                        target_model=uploaded_model,
                        device_name=device_name,
                        input_spec=input_spec,
                        dataset_loader=qnn_loader,
                        debug_print_feeds=False,
                    )

                    if isinstance(ret_acc, (list, tuple)) and len(ret_acc) >= 2:
                        qnn_acc, inference_urls = ret_acc
                        if inference_urls and isinstance(inference_urls, dict):
                            qnn_job_urls.update(inference_urls)

                    if qnn_acc is not None:
                        print(f"[Step 5] QNN Accuracy: {qnn_acc:.4f}")

                except Exception as e:
                    print(f"[Step 5] ERROR: QNN accuracy evaluation failed: {e}")
    else:
        print(f"\n[Step 5] Skipping QNN evaluation (qnn_options not specified)")

    # ============ Compile Results ============
    result = {
        "Model": model_name,
        "Feature": feature_name,
        "Techniques": (stats or {}).get("techniques", feature_name),
        # Accuracy metrics
        "FP32_accuracy": float(fp32_acc) if fp32_acc is not None else None,
        "AIMET Accuracy": float(feature_acc) if feature_acc is not None else None,
        "ONNX Accuracy": float(onnx_acc) if onnx_acc is not None else None,
        "QNN Accuracy": float(qnn_acc) if qnn_acc is not None else None,
        # Performance metrics
        "QNN Latency": f"{qnn_latency_ms:.3f} ms"
        if qnn_latency_ms is not None
        else None,
        "AIMET Runtime": (stats or {}).get("runtime", ""),
        "AIMET Memory": (stats or {}).get("memory", ""),
        # AI Hub job URLs (only for QNN workflows)
        "AI Hub QNN Compile Job": qnn_job_urls.get("compile", ""),
        "AI Hub QNN Profile Job": qnn_job_urls.get("profile", ""),
        "AI Hub QNN Inference Job": qnn_job_urls.get("inference", ""),
    }

    # ============ Generate Reports ============
    print(f"\n[Step 6] Generating reports...")

    csv_path = str(REPORTS_DIR / "results.csv")
    html_path = str(REPORTS_DIR / "results.html")

    write_csv([result], csv_path)
    write_html([result], html_path)

    print(f"[Step 6] CSV:  {csv_path}")
    print(f"[Step 6] HTML: {html_path}")

    # ============ Summary ============
    print(f"\n{'=' * 60}")
    print("Pipeline completed successfully!")
    print(f"{'=' * 60}")

    print("\nResults Summary:")
    for key, value in result.items():
        if value is not None and value != "":
            print(f"  {key:30s}: {value}")

    return result


# ==================== Entry Point ====================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ONNXRegression/runner.py <config_file.yaml>")
        print("\nExample:")
        print(
            "  python ONNXRegression/runner.py ONNXRegression/configs/mobilenetv2_quantsim.yaml"
        )
        sys.exit(1)

    try:
        result = run_single_config(sys.argv[1])
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

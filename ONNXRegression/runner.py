# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# pylint: disable=missing-module-docstring

"""
ONNXRegression Pipeline Runner - Single Test Execution

This module orchestrates the complete AIMET quantization evaluation pipeline
for a single test configuration using the new hierarchical config system.

Usage:
    # Run a test with profile
    python runner.py --model resnet50 --test quantsim_int8 --profile nightly

    # Run without profile (use defaults only)
    python runner.py --model resnet50 --test quantsim_int8

    # Dry run to preview configuration
    python runner.py --model resnet50 --test quantsim_int8 --profile nightly --dry-run

Pipeline Steps:
1. Load configuration (merge defaults → profile → model → test)
2. Load model from QAI Hub Models
3. Export to FP32 ONNX locally (torch.jit.trace + ONNX conversion)
4. Apply AIMET quantization technique (QuantSim, Lite-MP, or AdaRound)
5. Evaluate accuracy at multiple stages
6. Optionally run on-device evaluation via QNN (AI Hub)
7. Generate comprehensive reports

Key Design Decisions:
- Local ONNX export to reduce AI Hub API usage
- AI Hub only used for on-device QNN evaluation when specified
- Models loaded from QAI Hub Models repository
- Configuration merged from: defaults → profile → model → test
"""

import os
import sys
import yaml
import onnx
import torch
import argparse
from pathlib import Path
from typing import Dict, Any

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

# ==================== Config Loader ====================
from ONNXRegression.config_loader import load_config, validate_config


# Registry of available feature runners
FEATURE_RUNNERS = {
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
    if isinstance(sample_inputs, dict):
        input_names = list(input_spec.keys())
        input_values = list(sample_inputs.values())

        # For single-input models, extract the tensor
        if len(input_values) == 1:
            sample_inputs_for_export = input_values[0]
        else:
            sample_inputs_for_export = tuple(input_values)
    else:
        # Non-dict inputs
        if isinstance(sample_inputs, tuple):
            sample_inputs_for_export = sample_inputs
            if input_spec and isinstance(input_spec, dict):
                input_names = list(input_spec.keys())
            else:
                input_names = [f"input_{i}" for i in range(len(sample_inputs))]
        else:
            sample_inputs_for_export = sample_inputs
            if input_spec and isinstance(input_spec, dict):
                input_names = list(input_spec.keys())
            else:
                input_names = ["input"]

    # QNN requires static shapes (no dynamic_axes)
    dynamic_axes = None

    # Export the traced model to ONNX
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
                dynamic_axes=dynamic_axes,
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


def run_single_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the complete evaluation pipeline for a single configuration.

    This function orchestrates:
    1. Load model from QAI Hub Models
    2. Export FP32 ONNX locally (no AI Hub)
    3. Apply AIMET quantization
    4. Evaluate accuracy at multiple stages
    5. Optionally run on-device QNN evaluation
    6. Generate reports

    Args:
        config: Configuration dictionary (already merged)

    Returns:
        Dictionary with all evaluation results

    Raises:
        RuntimeError: If configuration is invalid or pipeline fails
    """
    # Validate configuration
    validate_config(config)

    # Extract configuration
    model_name = config.get("model_name")
    device_name = config.get("device", "Samsung Galaxy S24 (Family)")
    feature_name = config.get("feature", "quantsim").strip().lower()

    if feature_name not in FEATURE_RUNNERS:
        raise RuntimeError(
            f"Unsupported feature: {feature_name}. "
            f"Supported: {list(FEATURE_RUNNERS.keys())}"
        )

    print(f"\n{'=' * 60}")
    print(f"Running Single Test")
    print(f"{'=' * 60}")
    print(f"Model:    {model_name}")
    print(f"Feature:  {feature_name}")
    print(f"Device:   {device_name}")
    print(f"{'=' * 60}\n")

    # ============ Step 1: Model Loading ============
    print(f"\n[Step 1] Loading model and dataset from QAI Hub Models...")
    model, _dataset, input_spec, _ = load_model_data(model_name)
    dataset_name = resolve_dataset_name(model)
    print(f"Dataset: {dataset_name}")

    # ============ Step 2: FP32 Baseline ============
    print(f"\n[Step 2] Creating FP32 baseline via local ONNX export...")
    fp32_path = _export_torch_to_onnx_local(
        model, input_spec, ARTIFACTS_DIR, model_name
    )

    # Evaluate FP32 accuracy
    fp32_eval_samples = int(config.get("fp32_eval_samples", 200))
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


def main():
    """
    Main entry point for single test execution.

    Parses command-line arguments, loads configuration using the new
    hierarchical system, and executes the test pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Run a single AIMET quantization test (NEW CONFIG SYSTEM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with nightly profile (fast, no QNN)
  python runner.py --model resnet50 --test quantsim_int8 --profile nightly

  # Run without profile (use defaults)
  python runner.py --model resnet50 --test quantsim_int8

  # Dry run to preview configuration
  python runner.py --model resnet50 --test quantsim_int8 --profile nightly --dry-run

Available models:
  resnet50, mobilenetv2, efficientnet_b0, densenet121
  (see configs/models/ for full list)

Available tests (per model):
  quantsim_int8, quantsim_int8_int16, lite_mp_25, adaround_500
  (see model YAML for available tests)

Available profiles:
  nightly (fast, no QNN)
  (see configs/_profiles/ for full list)
        """,
    )

    # Required arguments
    parser.add_argument(
        "--model",
        required=True,
        help="Model name (e.g., resnet50, mobilenetv2). Corresponds to configs/models/<model>.yaml",
    )

    parser.add_argument(
        "--test",
        required=True,
        help="Test name from model config (e.g., quantsim_int8, lite_mp_25)",
    )

    # Optional arguments
    parser.add_argument(
        "--profile",
        help="Profile name (e.g., nightly, smoke). If not specified, uses defaults only.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show merged configuration without executing the test",
    )

    args = parser.parse_args()

    try:
        # Load and merge configuration
        print(f"Loading configuration...")
        print(f"  Model: {args.model}")
        print(f"  Test: {args.test}")
        print(f"  Profile: {args.profile or '(none - using defaults)'}")

        model_yaml = f"models/{args.model}.yaml"
        config = load_config(model_yaml, args.test, args.profile)

        # Dry run - show configuration and exit
        if args.dry_run:
            print(f"\n{'=' * 60}")
            print("DRY RUN - Configuration Preview")
            print(f"{'=' * 60}\n")

            print("Merged configuration:")
            for key, value in sorted(config.items()):
                if not key.startswith("_"):
                    print(f"  {key:30s}: {value}")

            print(f"\n{'=' * 60}")
            print("Would execute with this configuration")
            print(f"{'=' * 60}")
            return 0

        # Execute test
        result = run_single_config(config)
        return 0

    except FileNotFoundError as e:
        print(f"\n❌ Configuration file not found: {e}")
        return 1

    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        return 1

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

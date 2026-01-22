# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring

"""
AIMET Torch QuantSim Feature Runner

Applies AIMET's PyTorch QuantizationSimModel to simulate quantization effects
on native PyTorch models without early ONNX conversion.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from ONNXRegression.evaluation.eval_torch import eval_pytorch_model
from ONNXRegression.evaluation.metrics_utils import measure_inference_metrics
from ONNXRegression.features.torch._common import (
    bitwidth_from_token,
    build_quantsim_torch,
    export_torch_bundle,
    create_dummy_input,
)

_ARTIFACTS_DIR = Path("./ONNXRegression/artifacts")
_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def run_quantsim(
    *,
    model: Any,
    input_spec: Dict,
    dataset_name: str,
    config: Dict[str, Any],
    export_dir: Path = None,
) -> Tuple[Path, float, Dict[str, str], str]:
    """
    Apply AIMET Torch QuantSim to simulate quantization on a PyTorch model.

    Args:
        model: QAI Hub model object
        input_spec: Input specification for dummy input creation
        dataset_name: Dataset name for evaluation
        config: Configuration dictionary
        export_dir: Optional export directory

    Returns:
        Tuple of (qdq_path, accuracy, stats, bundle_dir)
    """
    model_name = config["model_name"]

    if export_dir is None:
        export_dir = config.get("_export_dir")
        if export_dir:
            export_dir = Path(export_dir)
        else:
            export_dir = Path("./ONNXRegression/artifacts") / model_name
            export_dir.mkdir(parents=True, exist_ok=True)
    else:
        export_dir = Path(export_dir)

    # Extract quantization parameters directly using bitwidth_from_token
    quant_scheme = str(config.get("quant_scheme", "tf_enhanced"))
    default_param_bw = bitwidth_from_token(config.get("param_type", "int8"), 8)
    default_output_bw = bitwidth_from_token(config.get("activation_type", "int8"), 8)

    aimet_cfg_file = config.get("config_file", None)

    calib_samples = int(config.get("calib_samples", 256))
    eval_samples = int(config.get("eval_samples", 256))
    metrics_samples = int(config.get("metrics_samples", 64))
    metrics_runs = int(config.get("metrics_runs", 1))
    metrics_warmup = int(config.get("metrics_warmup", 0))
    apply_prepare_model = config.get("apply_prepare_model")
    use_cuda = torch.cuda.is_available()

    print(f"[AIMET Torch QuantSim] Configuration:")
    print(f"  Scheme: {quant_scheme}")
    print(f"  Precision: W{default_param_bw}/A{default_output_bw}")
    print(f"  CUDA: {'Yes' if use_cuda else 'No'}")

    print(f"[AIMET Torch QuantSim] Extracting PyTorch model...")

    if hasattr(model, "to_torch_model"):
        torch_model = model.to_torch_model()
    else:
        torch_model = model

    torch_model.eval()

    print(f"[AIMET Torch QuantSim] Creating dummy input...")
    device = torch.device("cuda" if use_cuda else "cpu")
    dummy_input = create_dummy_input(input_spec, device)

    print(f"[AIMET Torch QuantSim] Moving model to device...")
    torch_model = torch_model.to(device)

    for buffer in torch_model.buffers():
        buffer.data = buffer.data.to(device)

    dummy_input = dummy_input.to(device)

    print(f"[AIMET Torch QuantSim] Building QuantizationSimModel...")
    sim = build_quantsim_torch(
        model=torch_model,
        dummy_input=dummy_input,
        quant_scheme=quant_scheme,
        default_param_bw=default_param_bw,
        default_output_bw=default_output_bw,
        config_file=aimet_cfg_file,
        use_cuda=use_cuda,
        apply_prepare_model=apply_prepare_model,
    )

    print(
        f"[AIMET Torch QuantSim] Calibrating encodings with {calib_samples} samples..."
    )

    def calibration_callback(model_to_calibrate: torch.nn.Module, args):
        """Forward pass callback for encoding calibration."""
        model_to_calibrate.eval()
        with torch.no_grad():
            eval_pytorch_model(
                model_to_calibrate,
                model,
                dataset_name,
                num_samples=args,
            )

    sim.model.eval()
    sim.compute_encodings(
        forward_pass_callback=calibration_callback,
        forward_pass_callback_args=calib_samples,
    )

    print(f"[AIMET Torch QuantSim] Calibration complete")

    feature_acc = eval_pytorch_model(
        sim.model,
        model,
        dataset_name,
        num_samples=eval_samples,
    )

    print(f"[AIMET Torch QuantSim] Quantized accuracy: {feature_acc:.4f}")

    print(f"[AIMET Torch QuantSim] Measuring runtime and memory...")

    def eval_for_metrics():
        sim.model.eval()
        with torch.no_grad():
            return eval_pytorch_model(
                sim.model,
                model,
                dataset_name,
                num_samples=metrics_samples,
            )

    runtime_str, memory_str = measure_inference_metrics(
        eval_for_metrics,
        runs=metrics_runs,
        warmup=metrics_warmup,
    )

    print(f"[AIMET Torch QuantSim] Runtime: {runtime_str}, Memory: {memory_str}")

    print(f"[AIMET Torch QuantSim] Exporting QDQ ONNX and encodings...")
    print(f"[AIMET Torch QuantSim] Moving model and dummy_input to CPU for export...")

    sim.model.cpu()
    dummy_input_cpu = dummy_input.cpu()

    qdq_path, bundle_dir = export_torch_bundle(
        sim=sim,
        dummy_input=dummy_input_cpu,
        export_dir=export_dir,
        model_name=model_name,
        input_spec=input_spec,
    )

    print(f"[AIMET Torch QuantSim] Exported to: {bundle_dir}")

    technique_str = f"quantsim(W{default_param_bw}A{default_output_bw}, {quant_scheme})"
    stats = {
        "techniques": technique_str,
        "runtime": runtime_str,
        "memory": memory_str,
    }

    return qdq_path, float(feature_acc), stats, str(bundle_dir)

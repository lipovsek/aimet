# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring

"""
Lightweight QNN (AI Hub) integration utilities used by ONNXRegression.

This module does two things:

1) `compile_and_profile_aimet_bundle(...)`
   - Uploads the AIMET-exported bundle directory (<model>.onnx + <model>.encodings)
   - Compiles it for the target device/runtime via AI Hub (QNN)
   - Profiles the compiled model and returns the latency (if available)
   - Returns the compiled model handle/zip path and job URLs (compile, profile)

2) `eval_qnn_accuracy(...)`
   - Submits inference jobs to AI Hub for the compiled model
   - Each job packages inputs as **per-sample arrays with batch dim = 1** because most
     QNN-compiled graphs expect N=1 at runtime
   - Computes Top-1 accuracy on device
   - IMPORTANT: always returns the first inference job URL (if submission succeeds),
     even if accuracy cannot be computed (e.g., empty outputs / device error)
"""

import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from qai_hub import (
    Device,
    submit_compile_job,
    submit_profile_job,
    submit_inference_job,
)


# ------------------------------ helpers --------------------------------------
def _np_dtype_from_spec(spec_dtype):
    """
    Map Hub/tensor dtypes to a NumPy dtype. Defaults to float32 if unknown.
    """
    if isinstance(spec_dtype, np.dtype):
        return spec_dtype

    # ONNX proto numbers → numpy
    try:
        import onnx

        if spec_dtype == onnx.TensorProto.FLOAT:
            return np.float32
    except Exception:
        pass

    # Torch dtype objects → numpy
    try:
        import torch

        if spec_dtype == getattr(torch, "float32", None):
            return np.float32
        if spec_dtype == getattr(torch, "float16", None):
            return np.float16
        if spec_dtype == getattr(torch, "uint8", None):
            return np.uint8
        if spec_dtype == getattr(torch, "int64", None):
            return np.int64
    except Exception:
        pass

    # Fallback
    return np.float32


def _as_numpy(x) -> np.ndarray:
    """
    Convert common array/tensor types to a contiguous NumPy array.
    """
    if isinstance(x, np.ndarray):
        arr = x
    else:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                arr = x.detach().cpu().numpy()
            else:
                arr = np.asarray(x)
        except Exception:
            arr = np.asarray(x)
    return np.ascontiguousarray(arr)


def _make_inputs_dict(
    input_spec: Dict, batch_x: np.ndarray
) -> Dict[str, List[np.ndarray]]:
    """
    Build `{input_name: [sample0, sample1, ...]}` for AI Hub inference.

    CRITICAL: Use the actual input name from input_spec, not a generic name.
    This ensures QNN can match the input correctly (e.g., "image_tensor" not "input").
    """
    if not input_spec:
        raise ValueError("Empty input_spec")

    # Use the ACTUAL first input name from input_spec (don't hardcode "input")
    input_name = next(iter(input_spec.keys()))
    spec0 = input_spec[input_name]
    want_dtype = _np_dtype_from_spec(getattr(spec0, "dtype", None))

    x = _as_numpy(batch_x)
    samples: List[np.ndarray] = []

    # (N,C,H,W) → split into N arrays of shape (1,C,H,W)
    if x.ndim == 4:
        for i in range(x.shape[0]):
            xi = x[i]
            if xi.ndim == 3:
                xi = np.expand_dims(xi, 0)
            xi = xi.astype(want_dtype, copy=False)
            samples.append(np.ascontiguousarray(xi))

    # (C,H,W) → single sample (1,C,H,W)
    elif x.ndim == 3:
        xi = np.expand_dims(x, 0)
        xi = xi.astype(want_dtype, copy=False)
        samples.append(np.ascontiguousarray(xi))

    # Fallback: wrap 1D/2D with a batch dim; otherwise pass as-is
    else:
        if x.ndim in (1, 2):
            xi = np.expand_dims(x, 0)
        else:
            xi = x
        xi = xi.astype(want_dtype, copy=False)
        samples.append(np.ascontiguousarray(xi))

    return {input_name: samples}


def _stack_outputs(outputs: Dict) -> np.ndarray:
    """
    Hub returns {out_name: [np(1, K), np(1, K), ...]} or {out_name: np(N, K)}.
    Return a single (N, K) ndarray.
    """
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("Empty/invalid outputs from inference job")

    out_name = next(iter(outputs.keys()))
    val = outputs[out_name]

    if isinstance(val, list) and len(val) > 0:
        rows = []
        for v in val:
            a = _as_numpy(v)
            a = a.squeeze(0) if a.ndim >= 2 and a.shape[0] == 1 else a
            rows.append(a)
        return np.vstack(rows)
    else:
        arr = _as_numpy(val)
        return arr.squeeze(0) if arr.ndim >= 2 and arr.shape[0] == 1 else arr


def _validate_aimet_bundle_dir(bundle_dir: str, model_name: str):
    """
    Ensure AIMET export directory contains the two required files:
      • <model_name>.onnx
      • <model_name>.encodings
    """
    if not os.path.isdir(bundle_dir):
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")
    onnx_p = os.path.join(bundle_dir, f"{model_name}.onnx")
    enc_p = os.path.join(bundle_dir, f"{model_name}.encodings")
    missing = [p for p in (onnx_p, enc_p) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"AIMET bundle missing files: {missing}")


# --------------------------- public API ---------------------------------------
def compile_and_profile_aimet_bundle(
    aimet_bundle_dir: str,
    device_name: str,
    model_name: str,
    export_dir: str,
    options: Optional[str] = None,
) -> Tuple[Optional[float], object, str, Dict[str, str]]:
    """
    Upload an AIMET-exported bundle (ONNX + encodings), compile to QNN, and profile.

    Returns
    -------
    (latency_ms, compiled_model_handle, compiled_zip_path, job_urls)
    """
    _validate_aimet_bundle_dir(aimet_bundle_dir, model_name)
    dev = Device(device_name)
    os.makedirs(export_dir, exist_ok=True)

    print(f"[QNN] AIMET bundle dir: {aimet_bundle_dir}")
    print(f"[QNN] Using options: {options or '(Hub defaults)'}")

    job_urls: Dict[str, str] = {}

    # 1) Compile – options apply at compile time.
    compile_job = submit_compile_job(
        model=aimet_bundle_dir,
        device=dev,
        name=f"{model_name}.aimet",
        options=options,
    )
    job_urls["compile"] = getattr(compile_job, "url", "") or ""

    compiled_model = compile_job.get_target_model()
    compiled_zip = os.path.join(export_dir, f"{model_name}_qnn.zip")
    compiled_model.download(compiled_zip)  # save ZIP to disk
    print(f"[QNN] Compiled artifact saved to: {compiled_zip}")

    # 2) Profile – *do not* pass compile options again.
    prof_job = submit_profile_job(model=compiled_model, device=dev)
    job_urls["profile"] = getattr(prof_job, "url", "") or ""

    profile = prof_job.download_profile()
    latency_ms: Optional[float] = None

    # Try multiple ways to extract latency
    if isinstance(profile, dict):
        us = (
            profile.get("execution_summary", {}).get("estimated_inference_time")
            or profile.get("inference_summary", {}).get("estimated_inference_time_us")
            or profile.get("latency_us")
        )
        if us is not None:
            try:
                latency_ms = float(us) / 1000.0
            except Exception:
                pass

    # Try attribute access if dict access didn't work
    if latency_ms is None and hasattr(profile, "summary"):
        try:
            if hasattr(profile.summary, "latency_ms"):
                latency_ms = float(profile.summary.latency_ms)
            elif hasattr(profile.summary, "estimated_inference_time_us"):
                latency_ms = float(profile.summary.estimated_inference_time_us) / 1000.0
        except Exception:
            pass

    print(
        f"[QNN] Profile latency: {f'{latency_ms:.3f} ms' if latency_ms is not None else 'N/A'}"
    )

    return latency_ms, compiled_model, compiled_zip, job_urls


def eval_qnn_accuracy(
    *,
    target_model,
    device_name: str,
    input_spec: Dict[str, object],
    dataset_loader: Iterable,  # yields (batch_x, batch_y)
    debug_print_feeds: bool = False,
) -> Tuple[Optional[float], Dict[str, str]]:
    """
    Submit inference to AI Hub (QNN) and compute Top-1 accuracy.

    Returns
    -------
    (accuracy, job_urls)
    """
    dev = Device(device_name)
    total_correct = 0
    total = 0
    first_infer_url: str = ""

    for batch_x, batch_y in dataset_loader:
        # Build per-sample input list ({name: [arr(N=1), ...]}).
        # CRITICAL: Uses actual input name from input_spec
        inputs_dict = _make_inputs_dict(input_spec, batch_x)

        if debug_print_feeds:
            for k, lst in inputs_dict.items():
                print(f"[DBG] feed '{k}': {len(lst)} samples")
                for i, a in enumerate(lst[:2]):
                    print(
                        f"  sample{i}: shape={a.shape}, dtype={a.dtype}, contiguous={a.flags['C_CONTIGUOUS']}"
                    )

        # Submit an inference job for this batch
        try:
            inf_job = submit_inference_job(
                model=target_model, device=dev, inputs=inputs_dict
            )
            if not first_infer_url:
                first_infer_url = getattr(inf_job, "url", "") or ""
        except Exception as e:
            print(f"[QNN] WARNING: Failed to submit inference job: {e}")
            continue

        # Download and validate outputs
        try:
            outputs = inf_job.download_output_data()
            if not outputs:
                print(
                    "[QNN] WARNING: inference returned no outputs; skipping this batch"
                )
                continue
        except Exception as e:
            print(f"[QNN] WARNING: Failed to download inference outputs: {e}")
            continue

        # Assume the first output is logits/probs for classification
        out_name = next(iter(outputs.keys()))
        out = outputs[out_name]

        # Handle both list and direct array outputs
        if isinstance(out, list):
            if len(out) == 0:
                print(
                    "[QNN] WARNING: inference outputs empty list; skipping this batch"
                )
                continue
            # Convert each sample's output to a Top-1 prediction
            preds = []
            for sample in out:
                arr = _as_numpy(sample)
                if arr.ndim == 1:
                    pred = int(np.argmax(arr))
                elif arr.ndim >= 2:
                    # Some backends return shapes like (1, C). Flatten the per-sample row.
                    pred = int(np.argmax(arr.reshape(arr.shape[0], -1), axis=1).item())
                else:
                    continue
                preds.append(pred)
        else:
            # Direct array output
            arr = _stack_outputs(outputs)
            preds = np.argmax(arr, axis=1)

        # Normalize labels to a 1D integer array
        y = _as_numpy(batch_y).reshape(-1)
        if y.dtype.kind not in {"i", "u"}:
            y = y.astype(np.int64, copy=False)

        # Tally only the aligned portion (defensive against length mismatches)
        n = min(len(preds), len(y))
        if n > 0:
            preds_arr = np.array(preds[:n]) if isinstance(preds, list) else preds[:n]
            total_correct += int(np.sum(preds_arr == y[:n]))
            total += n

    if total == 0:
        print("[QNN] Could not compute accuracy (no valid predictions).")
        # Return the link even when accuracy is None (lets the report link to Hub logs).
        return None, ({"inference": first_infer_url} if first_infer_url else {})

    acc = total_correct / float(total)
    print(f"[QNN] Accuracy over {total} samples: {acc * 100:.2f}%")
    return acc, ({"inference": first_infer_url} if first_infer_url else {})

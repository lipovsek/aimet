# AIMET Regression Framework

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Directory Structure](#directory-structure)
4. [Configuration System](#configuration-system)
5. [Usage](#usage)
6. [Feature Runners](#feature-runners)
7. [Adding a New Feature](#adding-a-new-feature)
8. [Workflow Integration](#workflow-integration)
9. [Reports and Baselines](#reports-and-baselines)

## Overview

The AIMET Regression Framework provides automated testing and evaluation of AIMET (AI Model Efficiency Toolkit) quantization techniques. It supports both **ONNX** and **PyTorch (Torch)** frameworks, testing quantization accuracy and performance both locally and on-device via Qualcomm AI Hub.

### Features

- **Dual Framework Support**: ONNX and Torch quantization pipelines
- **Multiple Quantization Techniques**: QuantSim, Lite Mixed-Precision, AdaRound, Automatic Mixed-Precision (AMP)
- **Hierarchical Configuration**: 4-level config merging (defaults → profile → model → test)
- **Suite-Based Testing**: Predefined test suites (nightly, weekly, smoke) with filtering
- **GitHub Actions Integration**: Nightly/weekly regression with baseline tracking
- **Reporting**: HTML and CSV reports with baseline comparison

### Supported Frameworks

| Framework | Features | GPU Required |
|-----------|----------|-------------|
| **ONNX** | QuantSim, Lite-MP, AdaRound, AMP | Yes (CUDA) |
| **Torch** | QuantSim, AdaRound, AMP | Yes (CUDA) |

### Supported Models

All models from the `qai_hub_models` package, including ResNet50, MobileNetV2, EfficientNet-B0, DenseNet121, YOLOv5, YOLOv8, ViT, Swin, and more.

## Architecture

### ONNX Pipeline

```
Config → Load Model → Export to ONNX (FP32) → AIMET ONNX QuantSim → QDQ ONNX → [QNN on-device]
                                                                           ↓
                                                                     HTML/CSV Report
```

### Torch Pipeline

```
Config → Load Model → FP32 PyTorch Eval → AIMET Torch QuantSim → Export QDQ ONNX → [QNN on-device]
                                                                            ↓
                                                                      HTML/CSV Report
```

### Execution Flow

1. **suite_runner.py** loads a suite YAML (e.g., `nightly-torch.yaml`)
2. For each model in the suite, it discovers available tests and applies filters
3. For each test, **config_loader.py** merges: `_defaults.yaml` → profile → model → test
4. **runner.py** dispatches to the appropriate feature runner based on `framework` (onnx/torch) and `feature` (quantsim/adaround/etc.)
5. The feature runner (e.g., `features/torch/quantsim_runner.py`) performs quantization, evaluation, and ONNX export
6. Results are collected and written to HTML/CSV reports

## Directory Structure

```
AIMETRegression/
├── configs/
│   ├── _defaults.yaml              # Base settings (framework, samples, thresholds)
│   ├── _profiles/                  # Runtime profiles
│   │   ├── nightly.yaml            # Reduced samples, no QNN
│   │   ├── smoke.yaml              # Minimal samples for PR validation
│   │   └── weekly.yaml             # Full samples, QNN on-device
│   └── models/                     # Per-model test definitions
│       ├── resnet50.yaml
│       ├── mobilenetv2.yaml
│       ├── yolov5.yaml
│       └── ...
├── features/
│   ├── onnx/                       # ONNX feature runners
│   │   ├── _common.py              # Shared ONNX utilities (build_quantsim, export)
│   │   ├── quantsim_runner.py
│   │   ├── lite_mp_runner.py
│   │   ├── adaround_runner.py
│   │   └── mixed_precision_runner.py
│   └── torch/                      # Torch feature runners
│       ├── _common.py              # Shared Torch utilities
│       ├── quantsim_runner.py
│       ├── adaround_runner.py
│       ├── mixed_precision_runner.py
│       └── utils.py                # Device patching for QAI Hub models
├── evaluation/
│   ├── eval_onnx.py                # ONNXRuntime evaluation
│   ├── eval_torch.py               # PyTorch evaluation
│   ├── eval_qnn.py                 # QNN on-device evaluation via AI Hub
│   └── metrics_utils.py            # Runtime and memory measurement
├── models/
│   └── ai_hub_loader.py            # Load models from qai_hub_models
├── report/
│   └── report_writer.py            # HTML and CSV report generation
├── suites/                         # Test suite definitions
│   ├── nightly-onnx.yaml           # Nightly ONNX regression
│   ├── nightly-torch.yaml          # Nightly Torch regression
│   ├── smoke-torch.yaml            # PR-level smoke test (Torch)
│   ├── weekly-onnx.yaml            # Weekly ONNX regression
│   └── weekly-torch.yaml           # Weekly Torch regression
├── workflow/
│   ├── artifacts.py                # GitHub artifact management
│   └── utils.py                    # Baseline setup, model extras install
├── baseline_comparison.py          # Baseline tracking and comparison
├── config_loader.py                # Hierarchical config merging
├── runner.py                       # Single test execution pipeline
└── suite_runner.py                 # Suite execution with filtering
```

## Configuration System

### 4-Level Hierarchy

```
_defaults.yaml          →  Base settings for all tests
  ↓ overridden by
_profiles/nightly.yaml  →  Runtime profile (sample counts, QNN toggle)
  ↓ overridden by
models/resnet50.yaml    →  Model-level settings (model_name, apply_bn_fold)
  ↓ overridden by
test definition         →  Per-test settings (feature, precision, scheme)
```

Later levels override earlier ones via `dict.update()`.

### Suite Files

A suite selects which models and tests to run under a given profile:

```yaml
suite_name: nightly-torch
profile: nightly
framework: torch

models:
  - models/resnet50.yaml
  - models/mobilenetv2.yaml
  - models/yolov5.yaml

# Optional: only run these tests from each model
test_filter:
  - quantsim_int8
  - quantsim_int8_int16
```

The `framework` field at suite level is a shorthand that overrides `_defaults.yaml`'s framework for all tests in the suite.

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `framework` | onnx | Framework: `onnx` or `torch` |
| `model_name` | (required) | Model identifier from qai_hub_models |
| `feature` | (required) | Quantization technique: quantsim, lite_mp, adaround, mixed_precision |
| `quant_scheme` | min_max | Quantization scheme |
| `param_type` | int8 | Weight precision: int4, int8, int16 |
| `activation_type` | int8 | Activation precision: int8, int16 |
| `config_file` | htp_v79 | HTP config: htp_v79, htp_v81 |
| `calib_samples` | 256 | Calibration samples |
| `eval_samples` | 200 | Evaluation samples |
| `qnn_eval_samples` | 50 | On-device evaluation samples (0 to disable) |
| `qnn_options` | "--target_runtime qnn_dlc" | QNN compilation options (null to disable) |
| `max_accuracy_drop` | 1.0 | Max acceptable accuracy drop (percentage points) |
| `apply_prepare_model` | false | Apply AIMET prepare_model before quantsim (Torch) |
| `apply_bn_fold` | true | Apply batch norm folding (Torch) |

## Usage

### Running a Single Test

```bash
# Run with profile
python -m AIMETRegression.runner --model resnet50 --test quantsim_int8 --profile nightly

# Dry run to preview merged configuration
python -m AIMETRegression.runner --model resnet50 --test quantsim_int8 --dry-run
```

### Running a Test Suite

```bash
# Run a predefined suite
python -m AIMETRegression.suite_runner --suite nightly-torch

# Filter by model name (substring match)
python -m AIMETRegression.suite_runner --suite nightly-torch --filter-model resnet

# Filter by test name (substring match)
python -m AIMETRegression.suite_runner --suite nightly-torch --filter-test quantsim

# Combine filters
python -m AIMETRegression.suite_runner --suite nightly-onnx --filter-model resnet --filter-test int8

# Dry run to preview test matrix
python -m AIMETRegression.suite_runner --suite nightly-torch --dry-run
```

### Available Suites

| Suite | Framework | Profile | Purpose |
|-------|-----------|---------|---------|
| `nightly-onnx` | ONNX | nightly | Daily ONNX regression |
| `nightly-torch` | Torch | nightly | Daily Torch regression |
| `smoke-torch` | Torch | smoke | PR-level validation |
| `weekly-onnx` | ONNX | weekly | Weekly comprehensive + QNN |
| `weekly-torch` | Torch | weekly | Weekly comprehensive + QNN |

## Feature Runners

### Framework-Specific Features

| Feature | ONNX | Torch | Description |
|---------|------|-------|-------------|
| `quantsim` | Yes | Yes | Standard INT8/INT16 quantization simulation |
| `lite_mp` | Yes | No | Mixed precision with sensitive layer promotion |
| `adaround` | Yes | Yes | Adaptive rounding optimization |
| `mixed_precision` | Yes | Yes | Automatic mixed precision (AMP) |

When `framework: torch` is set, `lite_mp` tests are automatically skipped.

### Feature Runner Interface

**ONNX runners** (`features/onnx/`) receive an FP32 ONNX model path:

```python
def run_quantsim(*, fp32_onnx_path, model, dataset_name, config) -> Tuple[str, float, dict, str]:
```

**Torch runners** (`features/torch/`) receive the PyTorch model and input spec:

```python
def run_quantsim(*, model, input_spec, dataset_name, config) -> Tuple[str, float, dict, str]:
```

Both return: `(exported_onnx_path, accuracy, stats_dict, bundle_dir)`

## Adding a New Feature

1. Create a runner in the appropriate directory:
   - ONNX: `features/onnx/my_feature_runner.py`
   - Torch: `features/torch/my_feature_runner.py`

2. Import shared utilities from the framework's `_common.py`

3. Implement the `run_<feature>()` function with the standard interface

4. Register the runner in `runner.py` under the appropriate `FEATURE_RUNNERS` dict

5. Add test definitions in model config files:
   ```yaml
   tests:
     - name: my_feature_test
       feature: my_feature
       # feature-specific parameters...
   ```

6. Test locally:
   ```bash
   python -m AIMETRegression.runner --model resnet50 --test my_feature_test --dry-run
   ```

## Workflow Integration

### Nightly Regression (`nightly-regression.yaml`)

Runs daily on a schedule. Two parallel jobs:
- **ONNX regression**: Builds AIMET ONNX wheel, runs `nightly-onnx` suite
- **Torch regression**: Builds AIMET Torch wheel, runs `nightly-torch` suite

Each job:
1. Builds AIMET from the branch via `build-wheels.yml`
2. Sets up the environment (installs AIMET + qai-hub-models)
3. Downloads previous baseline artifact
4. Runs the suite via `suite_runner.py`
5. Compares results against baseline
6. Uploads new baseline and reports as artifacts

### Manual Dispatch

The nightly workflow supports manual triggering with a custom suite name and branch.

### Baseline Management

- Baselines are stored as GitHub Actions artifacts (30-day retention)
- Each suite produces its own baseline (e.g., `baseline-nightly-torch`)
- `baseline_comparison.py` handles comparison and GitHub summary generation

## Reports and Baselines

### Report Output

Reports are generated in `AIMETRegression/reports/`:
- `results_<suite_name>.csv` — Machine-readable results
- `results_<suite_name>.html` — Interactive HTML table

### Key Metrics

| Metric | Description |
|--------|-------------|
| **FP32 Accuracy** | Baseline accuracy without quantization |
| **AIMET Accuracy** | Accuracy after AIMET quantization |
| **QDQ Accuracy** | Accuracy of exported QDQ ONNX model |
| **QNN Accuracy** | On-device accuracy via AI Hub (if enabled) |
| **QNN Latency** | On-device inference time (if enabled) |
| **AIMET Runtime** | Host inference time |
| **AIMET Memory** | Peak GPU memory during inference |

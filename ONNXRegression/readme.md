# AIMET ONNX Regression Framework Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [API Reference](#api-reference)
7. [Troubleshooting](#troubleshooting)

## Overview

The AIMET ONNX Regression Framework provides automated testing and evaluation of AIMET (AI Model Efficiency Toolkit) quantization techniques on ONNX models using Qualcomm AI Hub for on-device profiling.

### Key Features

- **Multiple Quantization Techniques**: QuantSim, Lite-MP, AdaRound
- **Automated Pipeline**: FP32 → AIMET → ONNX → QNN (on-device)
- **Comprehensive Evaluation**: Accuracy, latency, and memory metrics
- **Flexible Configuration**: YAML-based with suite support
- **Rich Reporting**: Interactive HTML and CSV reports

### Supported Models

All models from `qai_hub_models` package are supported

## Architecture

```
┌─────────────────┐
│  Config (YAML)  │
└────────┬────────┘
         │
    ┌────▼─────┐
    │  Runner  │──────► Load Model from AI Hub Models
    └────┬─────┘
         │
    ┌────▼────────────┐
    │ Compile to ONNX  │──────► FP32 Baseline
    │   (AI Hub)       │
    └────┬─────────────┘
         │
    ┌────▼────────────┐
    │ AIMET Feature   │──────► Apply Quantization
    │    Runner        │        (QuantSim/Lite-MP/etc.)
    └────┬─────────────┘
         │
    ┌────▼────────────┐
    │   Evaluation    │──────► AIMET Accuracy
    │  (ORT + QNN)    │──────► ONNX Accuracy
    └────┬────────────┘        On-Target Accuracy
         │
    ┌────▼────────────┐
    │ Report Writer   │──────► HTML + CSV Reports
    └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.x (for GPU acceleration)
- Qualcomm AI Hub account

### Step 1: Clone Repository

```bash
git clone https://github.qualcomm.com/qualcomm-ai/aimet
```

### Step 2: Install Dependencies

```bash
# Core dependencies
pip install onnx onnxruntime-gpu  # or onnxruntime for CPU
pip install torch torchvision

# Qualcomm packages
pip install qai-hub qai-hub-models

# AIMET
pip install aimet-onnx

# Utilities
pip install pyyaml jinja2 markupsafe pynvml
```

### Step 3: Configure AI Hub

```bash
qai-hub configure
# Enter your API token when prompted
```

## Configuration

### Model Configuration (YAML)

Create a YAML file in `ONNXRegression/configs/`:

```yaml
# Basic configuration
model_name: resnet50 # Model from qai_hub_models
feature: quantsim # Quantization technique
framework: onnx # Framework (always onnx)

# Quantization parameters
quant_scheme: tf_enhanced # Quantization scheme
param_type: int8 # Weight precision
activation_type: int8 # Activation precision
config_file: htp_v79 # AIMET config file (optional)

# Evaluation parameters
calib_samples: 256 # Calibration samples
eval_samples: 256 # Evaluation samples
fp32_eval_samples: 200 # FP32 baseline samples
quant_eval_samples: 200 # ONNX evaluation samples
qnn_eval_samples: 50 # On-device samples (0 to skip)

# Performance measurement
metrics_samples: 64 # Samples for timing
metrics_runs: 3 # Number of timing runs
metrics_warmup: 1 # Warmup runs

# Device configuration
device: Samsung Galaxy S24 (Family) # Target device
qnn_options: "--target_runtime qnn_lib_aarch64_android" # QNN options
```

### Suite Configuration

Create suite files in `ONNXRegression/suites/`:

```yaml
suite_name: comprehensive_test
description: Test all techniques on multiple models
include:
  - resnet50_quantsim.yaml
  - resnet50_lite_mp.yaml
  - mobilenetv2_quantsim.yaml

overrides: # Apply to all configs
  eval_samples: 100
  qnn_eval_samples: 25
```

## Usage

### Running a Single Configuration

```bash
python ONNXRegression/runner.py ONNXRegression/configs/resnet50_quantsim.yaml
```

### Running a Test Suite

```bash
# Predefined suites
python ONNXRegression/suite_runner.py --suite aimet_only
python ONNXRegression/suite_runner.py --suite aimet_plus_ontarget

# Custom suite
python ONNXRegression/suite_runner.py --suite-file my_suite.yaml

# With filtering
python ONNXRegression/suite_runner.py --suite aimet_only --filter quantsim

```

### Interpreting Reports

Reports are generated in `ONNXRegression/reports/`:

- **HTML Report**: Interactive table with filtering and sorting
- **CSV Report**: For further analysis in Excel/Python

Key metrics:

- **FP32 Accuracy**: Baseline accuracy without quantization
- **AIMET Accuracy**: Accuracy after AIMET quantization (simulated)
- **ONNX Accuracy**: Accuracy of exported ONNX model
- **QNN Accuracy**: On-device accuracy (if applicable)
- **QNN Latency**: On-device inference time (ms)

## API Reference

### Core Modules

#### `runner.py`

Main pipeline orchestrator.

```python
def run_single_config(config_path: str) -> Dict[str, Any]:
    """
    Execute full pipeline for one model configuration.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary with results (accuracy, latency, job URLs)
    """
```

#### `suite_runner.py`

Batch execution for multiple configs.

```python
def main():
    """
    Run a suite of configurations with optional filtering.
    Command-line interface for batch testing.
    """
```

### Feature Runners

All feature runners follow the same interface:

```python
def run_<feature>(
    *,
    fp32_onnx_path: str,
    model: Any,
    dataset_name: str,
    config: Dict[str, Any],
) -> Tuple[str, float, Dict[str, str], str]:
    """
    Apply <feature> quantization technique.

    Args:
        fp32_onnx_path: Path to FP32 ONNX model
        model: QAI Hub model object
        dataset_name: Dataset identifier
        config: Configuration dictionary

    Returns:
        Tuple of:
        - exported_onnx_path: Path to quantized ONNX
        - feature_accuracy: Accuracy after quantization
        - stats: Dictionary with runtime/memory stats
        - aimet_bundle_dir: Directory with ONNX + encodings
    """
```

### Evaluation Modules

#### `eval_onnx.py`

```python
def eval_onnx_model(
    session_or_path,
    model,
    dataset_name: str,
    num_samples: int = 200
) -> float:
    """Evaluate ONNX model accuracy using ONNXRuntime."""
```

#### `eval_qnn.py`

```python
def compile_and_profile_aimet_bundle(
    aimet_bundle_dir: str,
    device_name: str,
    model_name: str,
    export_dir: str,
    options: Optional[str] = None
) -> Tuple[Optional[float], object, str, Dict[str, str]]:
    """Compile AIMET bundle to QNN and profile on device."""

def eval_qnn_accuracy(
    *,
    target_model,
    device_name: str,
    input_spec: Dict[str, object],
    dataset_loader: Iterable,
    debug_print_feeds: bool = False
) -> Tuple[Optional[float], Dict[str, str]]:
    """Evaluate accuracy on target device via AI Hub."""
```

## Troubleshooting

### Common Issues

#### 1. AI Hub Connection Failed

```
Error: AI Hub connection failed
```

**Solution**: Run `qai-hub configure` and ensure your API token is valid.

#### 2. CUDA Out of Memory

```
Error: CUDA out of memory
```

**Solution**: Reduce batch sizes in config:

```yaml
calib_samples: 128 # Reduce from 256
eval_samples: 128 # Reduce from 256
```

#### 3. QNN Compilation Failed

```
Error: QNN compilation failed
```

**Solution**: Check `qnn_options` in config. Common options:

- Android: `--target_runtime qnn_lib_aarch64_android`
- Windows ARM: `--target_runtime qnn_lib_aarch64_windows`

#### 4. Missing Job URLs

```
Issue: Job URLs showing as empty in report
```

**Solution**: Check AI Hub API version. Update qai-hub package:

```bash
pip install --upgrade qai-hub
```

#### 5. Low On-Device Accuracy

```
Issue: QNN accuracy much lower than ONNX accuracy
```

**Possible causes**:

- Too few evaluation samples (`qnn_eval_samples`)
- Input preprocessing mismatch
- Quantization too aggressive

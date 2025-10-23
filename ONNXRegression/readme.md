# AIMET ONNX Regression Framework

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Configuration System](#configuration-system)
5. [Usage](#usage)
6. [Feature Runners](#feature-runners)
7. [Creating Feature Files](#creating-feature-files)
8. [Workflow Integration](#workflow-integration)
9. [Reports and Baselines](#reports-and-baselines)
10. [API Reference](#api-reference)

## Overview

The AIMET ONNX Regression Framework provides automated testing and evaluation of AIMET (AI Model Efficiency Toolkit) quantization techniques on ONNX models. The framework tests quantization accuracy and performance both locally and on-device via Qualcomm AI Hub.

### Key Features

- **Multiple Quantization Techniques**: QuantSim, Lite Mixed-Precision (Lite-MP), and AdaRound
- **Hierarchical Configuration System**: 4-level config merging (defaults → profile → model → test)
- **Local ONNX Export**: Reduces AI Hub API usage by exporting models locally
- **Automated Pipeline**: FP32 → AIMET → ONNX → QNN (optional on-device)
- **Comprehensive Evaluation**: Accuracy, latency, and memory metrics
- **GitHub Actions Integration**: Nightly regression with baseline tracking
- **Rich Reporting**: Interactive HTML and CSV reports with baseline comparison

### Supported Models

All models from the `qai_hub_models` package are supported, including:
- ResNet50
- MobileNetV2
- EfficientNet-B0
- DenseNet121
- And many more...

## Architecture

```
┌─────────────────┐
│  Configuration  │ (YAML: defaults → profile → model → test)
└────────┬────────┘
         │
    ┌────▼──────┐
    │  Runner   │────────► Load Model from QAI Hub Models
    └────┬──────┘
         │
    ┌────▼────────────────┐
    │ Local ONNX Export   │────────► FP32 Baseline (torch.jit.trace)
    │  (No AI Hub API)    │
    └────┬────────────────┘
         │
    ┌────▼────────────────┐
    │  AIMET Feature      │────────► Apply Quantization
    │     Runner          │         (QuantSim/Lite-MP/AdaRound)
    └────┬────────────────┘
         │
    ┌────▼────────────────┐
    │   Evaluation        │────────► AIMET Accuracy (simulated)
    │   (ONNXRuntime)     │────────► ONNX Accuracy (exported)
    └────┬────────────────┘
         │
    ┌────▼────────────────┐
    │  QNN (Optional)     │────────► On-Device Accuracy
    │   via AI Hub        │────────► Latency Measurement
    └────┬────────────────┘
         │
    ┌────▼────────────────┐
    │  Report Writer      │────────► HTML + CSV Reports
    └─────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8-3.10
- CUDA 11.x or 12.x (for GPU acceleration)
- Qualcomm AI Hub account (for on-device testing)

### Step 1: Clone Repository

```bash
git clone https://github.qualcomm.com/qualcomm-ai/aimet
cd aimet
```

### Step 2: Install Dependencies

```bash
# Core dependencies
pip install onnxruntime-gpu==1.19.2  # or onnxruntime for CPU-only
pip install torch torchvision

# Qualcomm packages
pip install qai-hub qai-hub-models

# AIMET (version 2.15+)
pip install aimet-onnx

# Additional utilities
pip install pynvml pytest
```

### Step 3: Configure AI Hub (Optional, for on-device testing)

```bash
qai-hub configure
# Enter your API token when prompted
```

## Configuration System

The framework uses a hierarchical YAML configuration system with 4 levels of precedence:

### Configuration Hierarchy

1. **_defaults.yaml** - Base settings for all tests
2. **Profile** (e.g., _profiles/nightly.yaml) - Runtime scenario settings
3. **Model** (e.g., models/resnet50.yaml) - Model-specific settings
4. **Test** - Individual test settings within model file

Later configurations override earlier ones using simple dict.update().

### Configuration Structure

```
ONNXRegression/
├── configs/
│   ├── _defaults.yaml          # Base configuration
│   ├── _profiles/              # Runtime profiles
│   │   ├── nightly.yaml        # Fast, reduced samples, no QNN
│   │   ├── smoke.yaml          # Quick validation
│   │   └── weekly.yaml         # Comprehensive testing
│   └── models/                 # Model configurations
│       ├── resnet50.yaml       # Model + test definitions
│       ├── mobilenetv2.yaml
│       └── ...
└── suites/                     # Test suite definitions
    └── nightly.yaml           # Nightly regression suite
```

### Example Model Configuration

```yaml
# models/resnet50.yaml
model_name: resnet50

tests:
  - name: quantsim_int8
    feature: quantsim
    quant_scheme: tf_enhanced
    param_type: int8
    activation_type: int8
    config_file: htp_v79

  - name: lite_mp_25
    feature: lite_mp
    percent_flip: 25
    override_precision: float16

  - name: adaround_500
    feature: adaround
    adaround_iters: 500
```

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | (required) | Model identifier from qai_hub_models |
| `feature` | (required) | Quantization technique: quantsim, lite_mp, or adaround |
| `quant_scheme` | tf_enhanced | Quantization scheme: tf_enhanced, tf, percentile, entropy |
| `param_type` | int8 | Weight precision: int4, int8, int16 |
| `activation_type` | int8 | Activation precision: int4, int8, int16 |
| `calib_samples` | 256 | Samples for calibration |
| `eval_samples` | 256 | Samples for evaluation |
| `qnn_eval_samples` | 50 | On-device evaluation samples (0 to disable) |
| `qnn_options` | "--target_runtime qnn_dlc" | QNN compilation options |

## Usage

### Running a Single Test

```bash
# Run with profile (e.g., nightly for fast execution)
python ONNXRegression/runner.py --model resnet50 --test quantsim_int8 --profile nightly

# Run without profile (uses defaults only)
python ONNXRegression/runner.py --model resnet50 --test quantsim_int8

# Dry run to preview configuration
python ONNXRegression/runner.py --model resnet50 --test quantsim_int8 --dry-run
```

### Running a Test Suite

```bash
# Run predefined suite
python ONNXRegression/suite_runner.py --suite nightly

# Filter by model
python ONNXRegression/suite_runner.py --suite nightly --filter-model resnet

# Filter by test
python ONNXRegression/suite_runner.py --suite nightly --filter-test quantsim

# Combine filters
python ONNXRegression/suite_runner.py --suite nightly --filter-model resnet --filter-test int8

# Dry run to preview test matrix
python ONNXRegression/suite_runner.py --suite nightly --dry-run
```

## Feature Runners

### QuantSim (Basic Quantization)

Simulates INT8/INT16 quantization effects without actually quantizing weights:

```python
# Configuration
feature: quantsim
quant_scheme: tf_enhanced  # or tf, percentile, entropy
param_type: int8           # Weight precision
activation_type: int8      # Activation precision
```

### Lite-MP (Mixed Precision)

Intelligently applies mixed precision by promoting sensitive layers to FP16:

```python
# Configuration
feature: lite_mp
percent_flip: 25           # Percentage of layers to promote
override_precision: float16  # Target precision for sensitive layers
```

### AdaRound (Adaptive Rounding)

Optimizes weight rounding to minimize quantization error:

```python
# Configuration
feature: adaround
adaround_iters: 15000      # Optimization iterations
adaround_samples: 64       # Samples for optimization
```

## Creating Feature Files

Feature files are Python modules that implement quantization techniques. Each feature follows a standard interface and structure.

### Feature File Structure

Each feature file must:
1. Import common utilities from `_common.py`
2. Define a `run_<feature>()` function with the standard interface
3. Handle configuration, execution, and result reporting

### Standard Feature Interface

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
        dataset_name: Dataset name for evaluation
        config: Configuration dictionary

    Returns:
        Tuple of:
        - exported_onnx_path: Path to quantized ONNX
        - feature_accuracy: Accuracy after quantization
        - stats: Dictionary with runtime/memory stats
        - aimet_bundle_dir: Directory with ONNX + encodings
    """
```

### Example Feature Implementation

Here's a simplified example of creating a new feature:

```python
# features/my_feature_runner.py
from pathlib import Path
from typing import Any, Dict, Tuple

import onnxruntime as ort
from qai_hub_models.utils.evaluate import evaluate_session_on_dataset

from ONNXRegression.evaluation.metrics_utils import measure_inference_metrics
from ONNXRegression.features._common import (
    build_quantsim,
    export_aimet,
    build_bundle,
)

_ARTIFACTS_DIR = Path("./ONNXRegression/artifacts")

def run_my_feature(
    *,
    fp32_onnx_path: str,
    model: Any,
    dataset_name: str,
    config: Dict[str, Any],
) -> Tuple[str, float, Dict[str, str], str]:
    """Apply my custom quantization technique."""

    # 1. Extract configuration
    model_name = config["model_name"]
    param_type = config.get("param_type", "int8")
    # ... other parameters

    # 2. Build QuantSim
    sim = build_quantsim(
        fp32_or_fpN_onnx_path=fp32_onnx_path,
        scheme="tf_enhanced",
        param_type=param_type,
        activation_type="int8",
        config_file=None,
        use_cuda=True,
    )

    # 3. Calibrate
    def calibration_callback(sess, _):
        evaluate_session_on_dataset(
            sess, model, dataset_name, num_samples=256
        )

    sim.compute_encodings(
        forward_pass_callback=calibration_callback,
        forward_pass_callback_args=None
    )

    # 4. Apply your custom technique
    # ... custom logic here ...

    # 5. Evaluate
    accuracy, *_ = evaluate_session_on_dataset(
        sim.session, model, dataset_name, num_samples=256
    )

    # 6. Measure performance
    runtime, memory = measure_inference_metrics(
        lambda: evaluate_session_on_dataset(
            sim.session, model, dataset_name, num_samples=64
        ),
        runs=1,
        warmup=0,
    )

    # 7. Export
    exported_onnx_path, enc_path = export_aimet(sim, _ARTIFACTS_DIR, model_name)
    bundle_dir = build_bundle(exported_onnx_path, enc_path, _ARTIFACTS_DIR, model_name)

    # 8. Return results
    stats = {
        "techniques": f"my_feature({param_type})",
        "runtime": runtime,
        "memory": memory,
    }

    return str(exported_onnx_path), float(accuracy), stats, str(bundle_dir)
```

### Common Utilities (_common.py)

The `_common.py` file provides shared functionality:

- **`pick_providers()`**: Select ORT execution providers
- **`make_session()`**: Create ORT InferenceSession
- **`build_quantsim()`**: Construct AIMET QuantSim model
- **`export_aimet()`**: Export QDQ ONNX and encodings
- **`build_bundle()`**: Create standardized bundle for QNN
- **`clean_dir()`**: Clean temporary files

### Best Practices for Feature Files

1. **Use Common Utilities**: Leverage `_common.py` for consistency
2. **Handle Errors Gracefully**: Validate inputs and handle edge cases
3. **Document Thoroughly**: Include docstrings with algorithm details
4. **Follow Naming Conventions**: `run_<feature>()` function name
5. **Log Progress**: Use print statements for user feedback
6. **Clean Up**: Remove temporary files after processing

### Testing Your Feature

1. Create a test configuration:
```yaml
# configs/models/test_model.yaml
model_name: resnet50

tests:
  - name: my_feature_test
    feature: my_feature
    param_type: int8
    # ... other parameters
```

2. Run the test:
```bash
python ONNXRegression/runner.py --model test_model --test my_feature_test --dry-run
```

## Workflow Integration

The framework integrates with GitHub Actions for automated nightly regression testing using artifact-based baseline storage.

### Workflow Overview

The ONNX Nightly workflow provides:
- **Automated Testing**: Runs daily at 6 AM UTC
- **Baseline Tracking**: Compares against previous run's results
- **Artifact Storage**: Uses GitHub Actions artifacts (30-day retention)
- **Flexible Execution**: Three testing modes for different use cases

### Testing Modes

1. **Suite Mode** (Default)
   - Runs all tests in a suite (nightly, weekly, etc.)
   - Most common mode for regression testing
   - Example: All nightly tests across multiple models

2. **Single Test Mode**
   - Runs one specific model+test combination
   - Useful for debugging specific failures
   - Includes optional profiling

3. **Suite with Filters**
   - Runs suite with model or test name filters
   - Targeted testing for specific components
   - Example: All ResNet tests or all INT8 tests

### Manual Workflow Dispatch

The workflow can be triggered manually via GitHub Actions UI:

```yaml
# Example inputs for different scenarios:

# Run nightly suite (default)
run_mode: suite
suite: nightly

# Debug specific test
run_mode: single_test
model: resnet50
test: quantsim_int8
profile: nightly

# Run filtered tests
run_mode: suite_with_filter
suite: nightly
filter_model: resnet
filter_test: int8
```

### Baseline Management

The workflow uses artifact-based baseline storage:

1. **First Run**:
   - No baseline exists
   - Creates initial baseline from results
   - Shows "First run - no baseline to compare" in summary

2. **Subsequent Runs**:
   - Downloads previous baseline from artifacts
   - Compares current results against baseline
   - Reports regressions, improvements, and stable results
   - Saves new baseline for next run

3. **Baseline Expiration**:
   - Artifacts expire after 30 days
   - After expiration, treated as first run again

### GitHub Summary Output

The workflow generates a comprehensive summary in the GitHub Actions UI:

- **Test Run Information**: Mode, branch, suite details
- **Available Artifacts**:
  - Test reports (CSV, HTML, logs)
  - Baseline for future runs

### Workflow Configuration

Key environment variables and settings:

```yaml
env:
  CUDA_VISIBLE_DEVICES: "0"      # Use first GPU
  QAI_HUB_ACCEPT_LICENSE: "1"    # Auto-accept AI Hub license
  PYTHONUNBUFFERED: "1"          # Show output immediately
  MPLBACKEND: "Agg"              # Headless matplotlib
  QT_QPA_PLATFORM: "offscreen"   # Headless Qt
```

### Integration with baseline_comparison.py

The workflow uses `baseline_comparison.py` for intelligent baseline tracking:

```bash
# Workflow command
python ONNXRegression/baseline_comparison.py run \
  --results results_nightly.csv \
  --suite-name nightly \
  --baselines-dir ONNXRegression/baselines \
  --github-summary \
  --no-fail-on-regression
```

This script:
- Stores current results as new baseline
- Compares with previous baseline (if exists)
- Generates GitHub-formatted markdown report
- Handles first-run scenario gracefully

## Reports and Baselines

### Report Generation

Reports are generated in `ONNXRegression/reports/`:

- **HTML Report** (`results_<suite>.html`): Interactive table with sorting/filtering
- **CSV Report** (`results_<suite>.csv`): For further analysis

### Key Metrics

| Metric | Description |
|--------|-------------|
| **FP32 Accuracy** | Baseline accuracy without quantization |
| **AIMET Accuracy** | Accuracy after AIMET quantization (simulated) |
| **ONNX Accuracy** | Accuracy of exported ONNX model |
| **QNN Accuracy** | On-device accuracy (if enabled) |
| **QNN Latency** | On-device inference time in milliseconds |
| **AIMET Runtime** | Host inference time |
| **AIMET Memory** | Peak memory usage during inference |

### Baseline Tracking

With artifact-based storage, baselines are:
- Stored as GitHub Actions artifacts
- Named by suite (e.g., `baseline-nightly`)
- Retained for 30 days
- Automatically compared in each run

The baseline comparison shows:
- Accuracy changes over time
- Performance regressions
- Test stability trends

## API Reference

### Core Modules

#### `runner.py`

Single test execution with hierarchical config:

```python
def run_single_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute full pipeline for one test configuration.

    Args:
        config: Merged configuration dictionary

    Returns:
        Dictionary with results (accuracy, latency, job URLs)
    """
```

#### `suite_runner.py`

Batch execution with filtering:

```python
# Command-line interface
python suite_runner.py --suite <suite_name> [--filter-model <pattern>] [--filter-test <pattern>]
```

#### `config_loader.py`

Configuration loading and merging:

```python
def load_config(model_yaml: str, test_name: str, profile: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and merge configuration from multiple sources.
    Precedence: defaults → profile → model → test
    """

def list_tests(model_yaml: str) -> List[str]:
    """List all test names defined in a model configuration."""
```

### Feature Runner Interface

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

    Returns:
        Tuple of:
        - exported_onnx_path: Path to quantized ONNX
        - feature_accuracy: Accuracy after quantization
        - stats: Dictionary with runtime/memory stats
        - aimet_bundle_dir: Directory with ONNX + encodings
    """
```

## Contributing

When adding new features or models:

1. **Add Model Config**: Create YAML in `configs/models/`
2. **Define Tests**: Add test definitions within model YAML
3. **Update Suite**: Include in appropriate suite file
4. **Test Locally**: Run with `--dry-run` first
5. **Check Baselines**: Ensure baseline comparison works
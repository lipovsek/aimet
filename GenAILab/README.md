# AIMET GenAI Laboratory

AIMET's GenAI laboratory provides an easy way to evaluate large models against quantization techniques provided by both AIMET-ONNX and AIMET-Torch. Define your experiment in a YAML config and the framework handles model loading, quantization, evaluation, and optional ONNX export.

## Prerequisites

- Either AIMET-Torch or AIMET-ONNX
- pytest
- HuggingFace transformers and datasets
- See `requirements.txt` for full list

## Quick Start

```bash
# Run with Torch
python -m GenAILab --framework torch --config GenAILab/example_config.yaml

# Run with ONNX
python -m GenAILab --framework onnx --config GenAILab/example_config.yaml

# Run both Torch and ONNX sequentially
python -m GenAILab --framework both --config GenAILab/example_config.yaml
```

### Runner Options

```bash
# Cache control
python -m GenAILab --framework torch --config cfg.yaml \
    --fp-cache-dir /path/to/fp/cache       # Custom FP results cache directory
    --clear-fp-cache                        # Clear FP cache before running
    --model-cache-dir /path/to/model/cache  # Custom ONNX model cache directory
    --clear-model-cache                     # Clear model cache before running
    --force-export                          # Force re-export of all artifacts

# Custom output directories
python -m GenAILab --framework torch --config cfg.yaml \
    --export-dir /path/to/exports \
    --results-dir /path/to/results

# Remote execution via GitHub Actions
python -m GenAILab --framework torch --config cfg.yaml --online
python -m GenAILab --framework torch --config cfg.yaml --online --wait
python -m GenAILab --framework torch --config cfg.yaml --download <run_id>
```

All extra arguments are forwarded to pytest for local runs (e.g. `-v`, `-k`).

## Config File Format

Configs use YAML, and a single file can contain multiple experiments separated by `---`. The full schema — every field, accepted values, defaults, validation rules, and the list of currently-registered recipes/datasets/metrics/adaptations — is documented in [CONFIG.md](CONFIG.md).

## Artifacts and Caching

All artifacts are stored under `GenAILab/artifacts/` by default. Each directory can be overridden via CLI flags.

```
GenAILab/artifacts/
├── results/                   # Profiling output
│   ├── profiling_data.json    # Hierarchical results (model_type → entries)
│   └── profiling_data.csv     # Flat format for database ingestion
├── exports/                   # ONNX export output
│   └── {model_slug}_{timestamp}/
│       ├── config.yaml        # Copy of the input config
│       ├── backbone/
│       │   ├── model.onnx     # Exported ONNX model
│       │   ├── model.data     # External tensor data
│       │   └── model.encodings # Quantization encodings
│       ├── visual/            # (VLMs only)
│       │   ├── model.onnx
│       │   └── model.data
│       └── embedding.pth      # (VLMs only)
└── cache/
    ├── fp/                    # FP model output cache
    ├── recipe/                # Recipe chain checkpoint cache
    └── model/                 # ONNX model export cache
```

### Results

Each test run appends to `profiling_data.json` and `profiling_data.csv` in the results directory. The JSON format is hierarchical (keyed by model type), while the CSV is a flat table with JSON-encoded columns suitable for PostgreSQL ingestion. Both files use file locking for process-safe concurrent writes.

Results include model configuration, precision settings, per-step resource utilization (GPU memory, elapsed time), metric scores, and environment metadata (CUDA version, GPU name, pip freeze, git SHA).

### FP Cache

Caches full-precision model outputs (e.g., MMLU logits) so distance metrics can compare quantized vs. FP without re-running the FP model. Entries are stored as `.pt` files (torch tensors) keyed by a hash of the model configuration and metric name. The cache is loaded lazily on first access.

| Flag | Effect |
|------|--------|
| `--fp-cache-dir <path>` | Custom FP cache directory |
| `--clear-fp-cache` | Wipe cache before running |

### Recipe Cache

Caches intermediate quantization state after cacheable recipe steps (currently `SeqMSE` and `AdaScale`). The cache key is a Merkle chain hash: each step's hash extends the previous one, incorporating the recipe name, kwargs, dataset config, model ID, precision, and an environment hash (library versions + AIMET source hashes). This means shared recipe prefixes across different configs hit the same cache entry.

On a cache hit, the recipe chain skips already-computed steps and loads the saved checkpoint directly into the quantsim model. After each new cacheable step, the result is saved for future runs.

| Flag | Effect |
|------|--------|
| `--recipe-cache-dir <path>` | Custom recipe cache directory |
| `--no-recipe-cache` | Disable recipe caching entirely |
| `--clear-recipe-cache` | Wipe cache before running |

### Model Cache (ONNX)

Caches exported ONNX models so that repeated ONNX evaluation runs don't require re-export from Torch. The cache key is a hash of model ID, sequence/context length, adaptations, and precision config. Includes staleness detection: if the HuggingFace hub config changes (e.g., model updated upstream), the cached entry is invalidated.

| Flag | Effect |
|------|--------|
| `--model-cache-dir <path>` | Custom model cache directory |
| `--clear-model-cache` | Wipe cache before running |

## Architecture

```
GenAILab/
├── __main__.py                # CLI entry point
├── conftest.py                # Pytest fixtures (caches, directories)
├── shared/                    # Framework-agnostic code
│   ├── helpers/
│   │   ├── yaml_config_parser.py  # Config parsing and plugin registry
│   │   ├── precision_config.py    # Precision configuration
│   │   ├── recipe_chain.py        # Multi-step recipe execution
│   │   ├── datasets.py            # Dataset implementations
│   │   ├── metrics.py             # Evaluation metrics
│   │   ├── eval_context.py        # FP/quant result caching
│   │   ├── profiler.py            # GPU profiling and result output
│   │   └── export.py              # Export utilities
│   └── models/
│       ├── generator.py           # Generator class for inference
│       ├── base.py                # LLM/VLM base classes, SimCollection
│       └── adaptations/           # Model adaptations (SHA, FastExportable, etc.)
├── torch/                     # PyTorch-specific
│   ├── models/                # Torch model classes (LLM_Torch, VLMs)
│   ├── helpers/
│   │   └── quant_recipes.py   # Torch quantization recipes
│   └── test_genai.py          # Torch test entry point
├── onnx/                      # ONNX-specific
│   ├── models/                # ONNX model classes
│   ├── helpers/
│   │   └── quant_recipes.py   # ONNX quantization recipes
│   └── test_genai.py          # ONNX test entry point
└── configs/                   # Regression test configs
```

## How It Works

1. **Config Parsing**: YAML config is parsed into model class, precision, recipe steps, and metrics
2. **Model Instantiation**: Load the HuggingFace model, apply adaptations, wrap in QuantizationSimModel
3. **Precision Configuration**: Apply quantization bitwidths and granularity settings
4. **Recipe Application**: Execute recipe steps in order (e.g., SeqMSE then Calibration), with caching
5. **Evaluation**: Run specified metrics on the quantized model
6. **Export** (optional): Export to ONNX format
7. **ONNX Evaluation** (optional): Re-evaluate the exported ONNX model
8. **Results**: Write profiling data (JSON + CSV) for dashboard ingestion

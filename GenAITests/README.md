# AIMET GenAI Test Framework

AIMET's GenAI framework provides an easy way to evaluate large models against quantization techniques provided by both AIMET-ONNX and AIMET-Torch. Define your experiment in a YAML config and the framework handles model loading, quantization, evaluation, and optional ONNX export.

## Prerequisites

- Either AIMET-Torch or AIMET-ONNX
- pytest
- HuggingFace transformers and datasets
- See `requirements.txt` for full list

## Quick Start

```bash
# Set up PYTHONPATH
source GenAITests/update_pythonpath.sh

# Run with Torch
python -m GenAITests --framework torch --config GenAITests/example_config.yaml

# Run with ONNX
python -m GenAITests --framework onnx --config GenAITests/example_config.yaml

# Run both Torch and ONNX sequentially
python -m GenAITests --framework both --config GenAITests/example_config.yaml
```

### Runner Options

```bash
# Cache control
python -m GenAITests --framework torch --config cfg.yaml \
    --fp-cache-dir /path/to/fp/cache       # Custom FP results cache directory
    --clear-fp-cache                        # Clear FP cache before running
    --model-cache-dir /path/to/model/cache  # Custom ONNX model cache directory
    --clear-model-cache                     # Clear model cache before running
    --force-export                          # Force re-export of all artifacts

# Custom output directories
python -m GenAITests --framework torch --config cfg.yaml \
    --export-dir /path/to/exports \
    --results-dir /path/to/results

# Remote execution via GitHub Actions
python -m GenAITests --framework torch --config cfg.yaml --online
python -m GenAITests --framework torch --config cfg.yaml --online --wait
python -m GenAITests --framework torch --config cfg.yaml --download <run_id>
```

All extra arguments are forwarded to pytest for local runs (e.g. `-v`, `-k`).

## Config File Format

Configs use YAML format. A single file can contain multiple experiments separated by `---`.

### Minimal LLM Config

```yaml
model:
  model_id: meta-llama/Llama-3.2-1B-Instruct
  sequence_length: 2048
  context_length: 4096
recipe:
  backbone:
    - name: Calibration
      dataset:
        name: Wikitext
        split: train
metrics:
  - name: PPL
  - name: TinyMMLU
```

### VLM Config with Multi-Step Recipe

```yaml
model:
  model_id: Qwen/Qwen2.5-VL-7B-Instruct
  sequence_length: 2048
  context_length: 4096
  image_size: [504, 336]
  attention_mask_min: -100
  adaptations:
    - FastExportable
precision:
  blocks:
    qtype: int4
  activations: int16
  kv_cache: int8
  lm_head:
    qtype: int8
  visual:
    weight:
      qtype: int8
    activations: int16
recipe:
  backbone:
    - name: SeqMSE
      dataset:
        name: Wikitext
        split: train
    - name: Calibration
      num_iterations: 128
      dataset:
        name: Interleaved
        source_datasets:
          - name: Wikitext
            split: train
          - name: AOKVQA
            split: train
  visual:
    - name: Calibration
      num_iterations: 128
      dataset:
        name: AOKVQA
        split: train
metrics:
  - name: PPL
  - name: MMMU
  - name: MultimodalPrompts
```

### FP Baseline (No Quantization)

```yaml
model:
  model_id: meta-llama/Llama-3.2-1B-Instruct
  sequence_length: 2048
  context_length: 4096
recipe:
  backbone:
    - name: RemoveQuantization
metrics:
  - name: PPL
```

## Config Sections

### `model` (required)

| Field | Required | Description |
|-------|----------|-------------|
| `model_id` | Yes | HuggingFace model ID or local checkpoint path |
| `sequence_length` | Yes | Number of tokens per inference pass |
| `context_length` | Yes | Maximum token context the model retains |
| `adaptations` | No | List of adaptations to apply (see [Adaptations](#adaptations)) |
| `image_size` | No | `[height, width]` for VLMs |
| `attention_mask_min` | No | Minimum attention mask value (default: large negative) |
| `encodings` | No | Path to pre-computed quantization encodings |
| `dtype` | No | Model dtype override (e.g. `float16`) |

### `precision` (optional)

Controls quantization bitwidths and granularity. When omitted, defaults to W4A16 with int8 lm_head and KV cache.

```yaml
precision:
  blocks:                    # Decoder block weight precision
    qtype: int4              # int2, int4, int8
    granularity: LPBQ        # PCQ (default), BQ, LPBQ
    block_size: 64           # Required for BQ/LPBQ
  activations: int16         # Activation quantization type
  kv_cache: int8             # KV cache quantization type
  embedding: int16           # Embedding quantization type
  lm_head:                   # LM head weight precision
    qtype: int8
    granularity: PCQ
  visual:                    # Visual encoder (VLMs only)
    weight:
      qtype: int8
    activations: int16
```

The `blocks` field also accepts shorthand:

```yaml
precision:
  blocks: int4       # Equivalent to blocks: { qtype: int4 }
```

### `recipe` (required)

Specifies quantization recipes per component. Each component accepts a list of steps executed in order.

**Component format (recommended):**

```yaml
recipe:
  backbone:
    - name: SeqMSE
      dataset:
        name: Wikitext
        split: train
    - name: Calibration
      num_iterations: 128
      dataset:
        name: Wikitext
        split: train
  visual:                     # Optional, for VLMs
    - name: Calibration
      dataset:
        name: AOKVQA
        split: train
```

If a recipe chain does not end with `Calibration`, `RemoveQuantization`, or `Skip`, a `Calibration` step is automatically appended.

**Simple format (single recipe, backbone only):**

```yaml
recipe:
  backbone:
    - name: Calibration
      dataset:
        name: Wikitext
        split: train
```

When the `recipe` section is omitted entirely, `RemoveQuantization` is applied (FP baseline).

#### Available Recipes

| Recipe | Torch | ONNX | Description |
|--------|-------|------|-------------|
| `RemoveQuantization` | Yes | Yes | Remove all quantization (FP baseline) |
| `Skip` | Yes | Yes | No-op (for pre-computed encodings) |
| `Calibration` | Yes | Yes | Calibrate quantization encodings from data |
| `SeqMSE` | Yes | Yes | Sequential MSE weight optimization |
| `AdaScale` | Yes | Yes | Adaptive scaling optimization |
| `SpinQuant` | Yes | Yes | SpinQuant rotation |

Recipe-specific parameters are passed directly in the step config:

```yaml
- name: AdaScale
  num_batches: 128
  num_iterations: 1024
  dataset:
    name: Wikitext
    split: train
```

### `metrics` (required)

List of evaluation metrics to run after quantization.

```yaml
metrics:
  - name: PPL
  - name: MMLU
    num_fewshot: 5
```

#### Available Metrics

**Accuracy metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `PPL` | Text | Perplexity on Wikitext test set |
| `TinyMMLU` | Text | TinyMMLU benchmark (fast) |
| `MMLU` | Text | Full MMLU benchmark (5-shot by default) |
| `MMLU1000` | Text | First 1000 MMLU samples |
| `MMMLU` | Text | Multilingual MMLU |
| `MMMU` | Multimodal | Multimodal Understanding (for VLMs) |

**Distance metrics** (compare quantized vs. FP outputs):

| Metric | Type | Description |
|--------|------|-------------|
| `MMLUFlips` | Text | % of predictions that differ from FP |
| `MMLUKLDivergence` | Text | KL divergence from FP logits |
| `MMLUReverseKLDivergence` | Text | Reverse KL divergence |
| `MMLUJSDivergence` | Text | Jensen-Shannon divergence |
| `MMMUFlips` | Multimodal | MMMU prediction disagreement |
| `MMMUKLDivergence` | Multimodal | MMMU KL divergence |
| `MMMUReverseKLDivergence` | Multimodal | MMMU reverse KL divergence |
| `MMMUJSDivergence` | Multimodal | MMMU Jensen-Shannon divergence |

**Interactive metrics:**

| Metric | Description |
|--------|-------------|
| `Interactive` | Live chat interface |
| `Prompts` | Run predefined prompts |
| `TrickyPrompts` | Model-specific edge case prompts |
| `MultimodalPrompts` | VLM image+text prompts |

### `export` (optional)

Export the quantized model to ONNX format.

```yaml
export: true                    # Export to default artifacts directory
export: /path/to/output         # Export to specific directory
```

### `eval_in_onnx` (optional)

Evaluate the exported ONNX model (automatically enables `export`):

```yaml
export: true
eval_in_onnx: true
```

### `run_group` (optional)

Logical grouping identifier for linking related results (e.g., Torch and ONNX runs of the same config):

```yaml
run_group: my-experiment-2024-01-15
```

## Adaptations

Adaptations modify how models are loaded or structured. They are applied via the `adaptations` field in the model config.

```yaml
model:
  adaptations:
    - FastExportable
    - SHA
```

### Available Adaptations

| Adaptation | Model Types | Description                                                        |
|------------|-------------|--------------------------------------------------------------------|
| `SHA` | llama, qwen3 | Split-Head Attention for per-head quantization                     |
| `SHA_Conv` | llama, qwen3 | SHA + replaces Linear layers with Conv2d                           |
| `FastExportable` | qwen2_5_vl | Attention mask-based export for cleaner ONNX graphs                |
| `Expert_Selection` | qwen3_moe | Mixture of Experts support                                         |
| `AIHM` | all (ONNX only) | Load AI Hub Models checkpoints (requires qai_hub_models installed) |

`SHA`, `SHA_Conv`, and `AIHM` are exclusive adaptations (cannot be combined with others).

`FastExportable` is automatically required for ONNX export of its respective model type.

## Datasets

Datasets are specified within recipe configurations.

| Dataset | Type | Description |
|---------|------|-------------|
| `Wikitext` | Text | Wikitext-2 (split: `train` or `test`) |
| `C4` | Text | HuggingFace C4 dataset |
| `TinyMMLU` | Text | TinyMMLU benchmark dataset |
| `MMLU` | Text | Full MMLU dataset |
| `MMMLU` | Text | Multilingual MMLU dataset |
| `MMMU` | Multimodal | MMMU dataset (split: `validation`) |
| `AOKVQA` | Multimodal | A-OKVQA visual QA dataset |
| `Interleaved` | Meta | Round-robin interleaving of multiple datasets |

### Interleaved Dataset

Combines multiple datasets by interleaving entries in round-robin order. Useful for mixed calibration:

```yaml
dataset:
  name: Interleaved
  source_datasets:
    - name: Wikitext
      split: train
    - name: AOKVQA
      split: train
```

## Supported Models

Model type is auto-detected from the HuggingFace model ID. Any transformer-based model supported by AIMET should work with the default LLM handler.

**LLMs (default handler):** Llama, Qwen3, Phi, and other HuggingFace causal LMs.

**VLMs (specialized handlers):**

| Model Type | Models |
|------------|--------|
| `qwen2_5_vl` | Qwen2.5-VL series |

## Artifacts and Caching

All artifacts are stored under `GenAITests/artifacts/` by default. Each directory can be overridden via CLI flags.

```
GenAITests/artifacts/
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
GenAITests/
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

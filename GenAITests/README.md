# AIMET GenAI Test Framework

AIMET's GenAI framework provides an easy way to evaluate large models against quantization techniques provided by both AIMET-ONNX and AIMET-Torch. You can use a config file and have the framework take care of running it all for you, or you can use the utilities provided to write an ad-hoc script.

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
pytest -s GenAITests/torch/test_genai.py --config GenAITests/example_config.yaml

# Run with ONNX
pytest -s GenAITests/onnx/test_genai.py --config GenAITests/example_config.yaml
```

## Config File Format

Config files use YAML format. Here's a basic example:

```yaml
model:
  model_id: meta-llama/Llama-3.2-1B-Instruct
  sequence_length: 2048
  context_length: 4096
recipe:
  name: PCQ
  dataset:
    name: Wikitext
    split: train
metrics:
  - name: TinyMMLU
  - name: PPL
```

### Multi-Document Configs

A single config file can contain multiple test configurations separated by `---`:

```yaml
model:
  model_id: meta-llama/Llama-3.2-1B-Instruct
  sequence_length: 2048
  context_length: 4096
recipe:
  name: PCQ
  dataset:
    name: Wikitext
    split: train
metrics:
  - name: PPL
---
model:
  model_id: meta-llama/Llama-3.2-1B-Instruct
  sequence_length: 2048
  context_length: 4096
recipe:
  name: SeqMSE
  dataset:
    name: Wikitext
    split: train
metrics:
  - name: PPL
```

## Config Sections

### `model` (required)

Specifies which model to load and how to configure it.

| Field | Required | Description |
|-------|----------|-------------|
| `model_id` | Yes | HuggingFace model ID (e.g., `meta-llama/Llama-3.2-1B-Instruct`) or local checkpoint path |
| `sequence_length` | Yes | Number of tokens the model processes in a single inference |
| `context_length` | Yes | Maximum number of tokens the model can retain |
| `model_type` | No | HuggingFace model type (auto-detected from `model_id` if not specified) |
| `adaptations` | No | List of adaptations to apply (see Adaptations section) |

Example with adaptations:

```yaml
model:
  model_id: meta-llama/Llama-3.2-1B-Instruct
  adaptations:
    - SHA
  sequence_length: 2048
  context_length: 4096
```

### `recipe` (required)

Specifies the quantization technique to apply.

**Simple format:**

```yaml
recipe:
  name: PCQ
  dataset:
    name: Wikitext
    split: train
```

**Component format (for advanced recipes):**

```yaml
recipe:
  backbone:
    name: AdaScale
    num_batches: 128
    num_iterations: 1024
    dataset:
      name: Wikitext
      split: train
```

#### Available Recipes

| Recipe | Torch | ONNX | Description |
|--------|-------|------|-------------|
| `RemoveQuantization` | Yes | Yes | Remove all quantization (FP baseline) |
| `Skip` | Yes | Yes | Do nothing (for precomputed encodings) |
| `PCQ` | Yes | Yes | Per-channel quantization |
| `LPBQ` | Yes | Yes | Low-precision blockwise quantization |
| `SeqMSE` | Yes | Yes | Sequential MSE optimization |
| `LPBQ_SeqMSE` | Yes | Yes | LPBQ + SeqMSE |
| `AdaScale` | Yes | Yes | AdaScale optimization |
| `OmniQuant` | Yes | No | OmniQuant optimization |
| `SpinQuant` | Yes | No | SpinQuant rotation |
| `SpinQuant_AdaScale` | Yes | No | SpinQuant + AdaScale |

Recipe-specific parameters can be passed directly in the config:

```yaml
recipe:
  backbone:
    name: AdaScale
    num_batches: 20
    num_iterations: 1500
    dataset:
      name: Wikitext
      split: train
```

### `metrics` (required)

List of evaluation metrics to run after quantization.

```yaml
metrics:
  - name: PPL
  - name: TinyMMLU
  - name: MMLU
    num_fewshot: 5
```

#### Available Metrics

| Metric | Description |
|--------|-------------|
| `PPL` | Perplexity on Wikitext test set |
| `TinyMMLU` | TinyMMLU benchmark (fast) |
| `MMLU` | Full MMLU benchmark (5-shot by default) |
| `MMLU1000` | First 1000 samples of MMLU |
| `MMMLU` | Multilingual MMLU |
| `Interactive` | Interactive chat mode |
| `Prompts` | Run predefined prompts |
| `TrickyPrompts` | Model-specific edge case prompts |

### `export` (optional)

Export the quantized model to ONNX format.

```yaml
export: true  # Export to default artifacts directory
# or
export: /path/to/output  # Export to specific directory
```

### `eval_in_onnx` (optional)

Evaluate the exported ONNX model (requires `export` to be enabled):

```yaml
export: true
eval_in_onnx: true
```

## Adaptations

Adaptations modify how models are loaded or structured. They are applied via the `adaptations` field in the model config.

### Available Adaptations

| Adaptation | Model Types | Description |
|------------|-------------|-------------|
| `SHA` | llama, qwen3 | Split-Head Attention - splits projection layers per head |
| `SHA_Conv` | llama, qwen3 | SHA + replaces Linear layers with Conv2d |
| `FastExportable` | qwen2_vl | Uses attention masks for cleaner ONNX export |
| `AIHM` | * (ONNX only) | Load AI Hub Models checkpoints |

### SHA Example

```yaml
model:
  model_id: meta-llama/Llama-3.2-1B-Instruct
  adaptations:
    - SHA
  sequence_length: 2048
  context_length: 4096
```

### AIHM Example (ONNX)

```yaml
model:
  model_id: meta-llama/Llama-3.2-1B-Instruct
  adaptations:
    - AIHM
  sequence_length: 2048
  context_length: 4096
```

Note: `AIHM` is exclusive and cannot be combined with other adaptations.

## Datasets

Datasets are specified within recipe configurations.

| Dataset | Description |
|---------|-------------|
| `Wikitext` | Wikitext-2 dataset |
| `TinyMMLU` | TinyMMLU benchmark dataset |
| `MMLU` | Full MMLU dataset |
| `MMMLU` | Multilingual MMLU dataset |

```yaml
recipe:
  name: PCQ
  dataset:
    name: Wikitext
    split: train  # or test
```

## Supported Models

The framework automatically detects model type from the HuggingFace model ID. Supported model families include:

**LLMs (via default LLM class):**
- Llama (llama)
- Qwen3 (qwen3)
- Phi (phi3)
- And other HuggingFace transformer models.

Any LLM with standard IO supported by aimet-torch or aimet-onnx should be
supported by GenAITests out of the box. Note: some techniques and model components may require AIMET updates.

**VLMs (specialized classes):**
- Qwen2-VL (qwen2_vl)
- Qwen3-VL (qwen3_vl)

## Running Tests

### Basic Usage

```bash
# Torch
pytest -s GenAITests/torch/test_genai.py --config <config.yaml>

# ONNX
pytest -s GenAITests/onnx/test_genai.py --config <config.yaml>
```

### Example Configs

See the `configs/` directory for regression test configs and `example_config*.yaml` files for usage examples.

## Writing Custom Scripts

For more control, you can use the framework utilities directly:

```python
from GenAITests.torch.models import LLM_Torch
from GenAITests.shared.helpers.datasets import Wikitext
from GenAITests.shared.helpers.metrics import PPL

# Load model
model_cls = LLM_Torch
model = model_cls.instantiate_model("meta-llama/Llama-3.2-1B-Instruct")
tokenizer = model_cls.get_tokenizer("meta-llama/Llama-3.2-1B-Instruct")

# Create quantsim and generator
quantsim = model_cls.get_quantsim(model, ...)
generator = model_cls.get_generator(quantsim.model, tokenizer, ...)

# Run evaluation
ppl = PPL.evaluate(generator, tokenizer, context_length=4096)
```

See `GenAITests/torch/example_custom_script.py` for a complete example.

## Architecture

```
GenAITests/
├── shared/                    # Shared utilities
│   ├── helpers/
│   │   ├── datasets.py       # Dataset implementations
│   │   ├── metrics.py        # Evaluation metrics
│   │   ├── yaml_config_parser.py  # Config parsing
│   │   └── export.py         # Export utilities
│   └── models/
│       ├── generator.py      # Generator class for inference
│       ├── base.py           # Base model class
│       └── adaptations/      # Model adaptations
├── torch/                     # PyTorch-specific
│   ├── models/               # Torch model classes
│   ├── helpers/
│   │   └── quant_recipes.py  # Torch quantization recipes
│   └── test_genai.py         # Test entry point
├── onnx/                      # ONNX-specific
│   ├── models/
│   │   └── adaptations/
│   │       └── hub_models.py # AIHM adaptation
│   ├── helpers/
│   │   └── quant_recipes.py  # ONNX quantization recipes
│   └── test_genai.py         # Test entry point
└── configs/                   # Regression test configs
```

## How It Works

1. **Model Instantiation**: Load the model from HuggingFace, optionally applying adaptations
2. **Tokenizer Setup**: Load the corresponding tokenizer
3. **QuantizationSimModel**: Wrap the model for quantization simulation
4. **Generator**: Create a Generator object for inference with static shapes
5. **Dataset Loading**: Load and tokenize the calibration dataset
6. **Recipe Application**: Apply the specified quantization technique
7. **Evaluation**: Run the specified metrics on the quantized model
8. **Export** (optional): Export to ONNX format

# LM Driver (`models/`)

This package is the inference driver for LLMs and VLMs that operate under
static-shape constraints (fixed sequence length, fixed KV cache size). It is
designed to be synced as a standalone subtree into consumer repos
(e.g. ai-hub-models-internal) via `git subtree`.

## What's in this package

```
models/
├── base.py            # LLM / VLM abstract base classes, SimCollection dataclass
├── generator.py       # Generator (core), VLM_Generator, mixins
├── qwen2_vl.py        # Qwen 2.5 VL model class
├── qwen3_vl.py        # Qwen 3 VL model class
└── utils/
    ├── attention_mask.py   # 2D → 4D mask conversion
    ├── compat.py           # HuggingFace compatibility patches
    ├── exportable.py       # ONNXExportableModuleWithCache wrapper
    ├── layer_cache.py      # LayerCacheDescriptor, config → cache shape logic
    └── rope_embedding.py   # RopeEmbedding (HF-coupled, used by backends)
```

## Key classes

| Class | Role |
|-------|------|
| `Generator` | Pads inputs to static shapes, manages KV cache, slices long sequences into multiple forward passes. Exposes HF-compatible `forward()` and `generate()`. |
| `VLM_Generator` | Extends Generator with vision encoding, embedding merge, position ID processing. |
| `ONNXExportableModuleWithCache` | Wraps a HuggingFace model so it accepts/returns flat tensors instead of Cache objects. Required when the model is a raw `PreTrainedModel`. |
| `LLM` / `VLM` | Abstract base classes defining the interface for model instantiation, input/output naming, and quantsim setup. |

## Integration guide

### Minimal LLM example

```python
from models.generator import Generator
from models.utils.exportable import ONNXExportableModuleWithCache

# 1. Wrap your model so it satisfies the flat-tensor I/O contract
model = ONNXExportableModuleWithCache(hf_model)

# 2. Create a Generator
gen = Generator(model, tokenizer, sequence_length=2048, context_length=4096)

# 3. Use it
output = gen(input_ids=tokens)          # forward pass
output = gen.generate(input_ids=tokens) # autoregressive generation
```

### Generator mixins

The package provides composable mixins for common checkpoint conventions:

- **`PrecomputedCosSinGeneratorMixin`** — replaces `position_ids` with
  precomputed RoPE cos/sin tensors. Expects a `rope_provider` kwarg when
  calling `prepare_inputs`. The provider must implement:

  ```python
  def get_embedding(position_ids: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
      """Return (cos, sin) embeddings for the given position IDs."""
      ...
  ```

  It's the caller's responsibility to build and pass the provider.

- **`TransposedKVGeneratorMixin`** — permutes KV cache between HF layout
  `(B, H, S, D)` and transposed layout `(H, B, D, S)` for keys /
  `(H, B, S, D)` for values.

## Model contract

The model passed to `Generator.__init__` must satisfy the contract documented
in the `Generator` class docstring. The key requirements are:

1. Callable as `model(*tensors) -> tuple[Tensor, ...]`
2. Exposes `.config` (PretrainedConfig), `.device`, `.dtype`
3. Accepts inputs padded to fixed `sequence_length`
4. KV cache shape: `(batch, num_kv_heads, seq_len, head_dim)`
5. 4D attention mask: `(batch, 1, seq_len, context_len)`, float
6. Returns flat tuple: `(logits, *kv_states)` — not HF ModelOutput

If your model returns HF Cache objects, wrap it in
`ONNXExportableModuleWithCache` which handles the flattening.

## Constraints

- `context_length > sequence_length` — the KV cache holds
  `context_length - sequence_length` tokens of history.
- Input ordering is fixed and doubles as ONNX input names:
  `input_ids | inputs_embeds`, `attention_mask`, `position_ids`,
  `past_key_0_in`, `past_value_0_in`, ...
- The Generator handles multi-slice prefill automatically when input length
  exceeds `sequence_length`.

## Dependencies

This package depends on:
- `torch`
- `transformers` (for `PretrainedConfig`, `GenerationMixin`, `DynamicCache`)

It does **not** depend on AIMET, ONNX runtime, or any quantization framework.
Those concerns live in the `backends/` layer which imports from here.

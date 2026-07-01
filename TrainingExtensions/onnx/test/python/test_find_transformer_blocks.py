# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import io
import re
import shutil
import pytest
import sys
import platform
import onnx
import torch
import transformers
from transformers import AutoModelForCausalLM
import transformers.masking_utils as mu

from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.experimental.adascale.find_blocks import (
    get_decoder_blocks_end_points,
    get_conv_linear_layers_decoder_block,
)
from aimet_onnx.experimental.spinquant.model_analysis.block_identifier import (
    get_decoder_block_boundaries,
)
from .utils import add_genai_tests_path
from .conftest import skip_module_on_windows_arm64

skip_module_on_windows_arm64(
    "transformers and onnx_sim is not available on Windows ARM64"
)

_NUM_LAYERS = 2
_BOTH = ("torchscript", "dynamo")
_TS_ONLY = ("torchscript",)
_DYNAMO_ONLY = ("dynamo",)

# (test id, transformers config-class, export backends, detect kwargs, homogeneous).
# - Backends differ because some architectures only export under one exporter.
# - homogeneous=True means every decoder block has the same number of weighted
#   linear (MatMul/Gemm/Conv) ops; False for architectures whose layers genuinely
#   differ (MoE dense-replace / shared experts, Mamba-hybrid, per-layer attention
#   variation) so the uniformity assertion is skipped.
_BLOCK_TOPOLOGY_MODELS = [
    # dense pre-norm
    ("llama", "LlamaConfig", _BOTH, {}, True),
    ("qwen2", "Qwen2Config", _BOTH, {}, True),
    ("mistral", "MistralConfig", _BOTH, {}, True),
    ("ministral", "MinistralConfig", _BOTH, {}, True),
    ("ministral3", "Ministral3Config", _BOTH, {}, True),
    ("phi3", "Phi3Config", _BOTH, {}, True),
    ("gemma", "GemmaConfig", _BOTH, {}, True),
    ("granite", "GraniteConfig", _BOTH, {}, True),
    ("smollm3", "SmolLM3Config", _BOTH, {}, True),
    ("arcee", "ArceeConfig", _BOTH, {}, True),
    ("seed_oss", "SeedOssConfig", _BOTH, {}, True),
    # hybrid internal-norm
    ("qwen3", "Qwen3Config", _BOTH, {}, True),
    ("gemma2", "Gemma2Config", _BOTH, {}, True),
    ("gemma3_text", "Gemma3TextConfig", _BOTH, {}, True),
    ("gemma4_text", "Gemma4TextConfig", _BOTH, {}, False),
    # hybrid linear-attention
    ("lfm2", "Lfm2Config", _BOTH, {}, True),
    # parallel attn+MLP: single norm per block -> active_norms_per_block=1
    ("cohere", "CohereConfig", _DYNAMO_ONLY, {"active_norms_per_block": 1}, True),
    ("cohere2", "Cohere2Config", _DYNAMO_ONLY, {"active_norms_per_block": 1}, True),
    # dynamo-only dense
    ("glm4", "Glm4Config", _DYNAMO_ONLY, {}, True),
    ("ernie4_5", "Ernie4_5Config", _DYNAMO_ONLY, {}, True),
    # MoE
    ("qwen2_moe", "Qwen2MoeConfig", _TS_ONLY, {}, False),
    ("qwen3_moe", "Qwen3MoeConfig", _TS_ONLY, {}, True),
    ("glm4_moe", "Glm4MoeConfig", _TS_ONLY, {}, False),
    ("gpt_oss", "GptOssConfig", _TS_ONLY, {}, False),
    ("olmoe", "OlmoeConfig", _TS_ONLY, {}, True),
    ("nemotron_h", "NemotronHConfig", _TS_ONLY, {}, False),
    ("qwen3_next", "Qwen3NextConfig", _TS_ONLY, {}, True),
]

# VLM language backbones (text decoder + lm_head extracted from the full VLM).
_VLM_BACKBONE_MODELS = ["llava", "internvl", "qwen2_5_vl", "qwen3_vl"]


def _patch_sdpa_mask():
    """Make HF mask construction traceable under torch.jit.trace."""
    orig = mu.sdpa_mask

    def patched(*args, **kwargs):
        if (
            "q_length" in kwargs
            and isinstance(kwargs["q_length"], torch.Tensor)
            and kwargs["q_length"].ndim == 0
        ):
            kwargs["q_length"] = kwargs["q_length"].item()
        elif len(args) >= 2 and isinstance(args[1], torch.Tensor) and args[1].ndim == 0:
            args = (args[0], args[1].item(), *args[2:])
        return orig(*args, **kwargs)

    mu.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = patched
    mu.sdpa_mask = patched


def _make_config(config_attr, **extra):
    """Build a tiny decoder config, or skip if the architecture is unavailable."""
    cfg_cls = getattr(transformers, config_attr, None)
    if cfg_cls is None:
        pytest.skip(f"{config_attr} not available in this transformers version")

    cfg = cfg_cls(**extra)
    overrides = dict(
        num_hidden_layers=_NUM_LAYERS,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=1024,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        # MoE knobs (best-effort; ignored by dense configs)
        num_experts=4,
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        decoder_sparse_step=1,
    )
    for key, value in overrides.items():
        if hasattr(cfg, key):
            try:
                setattr(cfg, key, value)
            except (AttributeError, TypeError):
                pass
    return cfg


class _PrefillWrapper(torch.nn.Module):
    """input_ids -> logits prefill wrapper for ONNX export.

    NOTE: ``use_cache=False`` keeps the graph to a prefill pass,
    block detection needs no KV-cache I/O.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, position_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).logits


def _build_block_topology_model(config_attr):
    """Instantiate ``AutoModelForCausalLM`` with random, distinct norm weights."""
    cfg = _make_config(config_attr)
    torch.manual_seed(0)
    try:
        model = AutoModelForCausalLM.from_config(
            cfg, experts_implementation="eager"
        ).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_config(cfg).eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.copy_(torch.rand_like(param) * 0.5 + 0.75)
    return _PrefillWrapper(model).eval(), cfg


def _sample_inputs(cfg, seq_len=8):
    """Build dummy (input_ids, position_ids, attention_mask) for a prefill pass."""
    input_ids = torch.randint(3, cfg.vocab_size, (1, seq_len), dtype=torch.long)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    return input_ids, position_ids, attention_mask


def _build_vlm(name):
    """Build a tiny full VLM (text + vision)."""
    if name == "llava":
        text = _make_config("LlamaConfig")
        vision = transformers.CLIPVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=_NUM_LAYERS,
            num_attention_heads=4,
            image_size=32,
            patch_size=16,
        )
        cfg = transformers.LlavaConfig(text_config=text, vision_config=vision)
        return transformers.LlavaForConditionalGeneration(cfg).eval()
    if name == "internvl":
        text = _make_config("Qwen2Config")
        vision = transformers.InternVLVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=_NUM_LAYERS,
            num_attention_heads=4,
            image_size=32,
            patch_size=16,
        )
        cfg = transformers.InternVLConfig(text_config=text, vision_config=vision)
        return transformers.InternVLForConditionalGeneration(cfg).eval()
    if name == "qwen2_5_vl":
        text = _make_config(
            "Qwen2_5_VLTextConfig",
            rope_scaling={"rope_type": "mrope", "mrope_section": [2, 2, 4]},
        )
        vision = transformers.Qwen2_5_VLVisionConfig(
            hidden_size=32, intermediate_size=64, depth=2, num_heads=4
        )
        cfg = transformers.Qwen2_5_VLConfig(text_config=text, vision_config=vision)
        return transformers.Qwen2_5_VLForConditionalGeneration(cfg).eval()
    if name == "qwen3_vl":
        text = _make_config(
            "Qwen3VLTextConfig",
            rope_scaling={"rope_type": "default", "mrope_section": [2, 2, 4]},
        )
        vision = transformers.Qwen3VLVisionConfig(
            hidden_size=32, intermediate_size=64, depth=2, num_heads=4
        )
        cfg = transformers.Qwen3VLConfig(text_config=text, vision_config=vision)
        return transformers.Qwen3VLForConditionalGeneration(cfg).eval()
    raise ValueError(f"unknown VLM '{name}'")


class _BackboneEmbedsWrapper(torch.nn.Module):
    """inputs_embeds -> logits prefill wrapper for a VLM language backbone."""

    def __init__(self, language_model, lm_head):
        super().__init__()
        self.language_model = language_model
        self.lm_head = lm_head

    def forward(self, inputs_embeds, position_ids, attention_mask):
        hidden = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        return self.lm_head(hidden)


def _build_vlm_backbone(name):
    """Build a VLM, extract its language backbone + lm_head, randomize norms."""
    torch.manual_seed(0)
    vlm = _build_vlm(name)
    language_model = vlm.model.language_model
    with torch.no_grad():
        for pname, param in language_model.named_parameters():
            if pname.endswith("norm.weight"):
                param.copy_(torch.rand_like(param) * 0.5 + 0.75)
    return _BackboneEmbedsWrapper(
        language_model, vlm.lm_head
    ).eval(), language_model.config


def _vlm_sample_inputs(cfg, seq_len=8):
    """VLM backbones consume inputs_embeds (no token-embedding Gather)."""
    inputs_embeds = torch.randn(1, seq_len, cfg.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    return inputs_embeds, position_ids, attention_mask


def _export_onnx(model, inputs, backend, input_names=None):
    """Export the wrapped model to ONNX via the torchscript or dynamo backend."""
    if input_names is None:
        input_names = ["input_ids", "position_ids", "attention_mask"]
    buf = io.BytesIO()
    with torch.no_grad():
        if backend == "dynamo":
            program = torch.export.draft_export(model, inputs, strict=False)
            torch.onnx.export(program, (), buf, input_names=input_names, dynamo=True)
        else:
            torch.onnx.export(
                model,
                inputs,
                buf,
                input_names=input_names,
                opset_version=17,
                dynamo=False,
            )
    buf.seek(0)
    return onnx.load_model(buf)


def _detect_from_config(config_attr, backend, detect_kwargs):
    """Build, export, and run block detection for one LLM (model, backend) case."""
    _patch_sdpa_mask()
    model, cfg = _build_block_topology_model(config_attr)
    onnx_model = _export_onnx(model, _sample_inputs(cfg), backend)
    connected_graph = ConnectedGraph(onnx_model)
    blocks, active_norms = get_decoder_block_boundaries(
        onnx_model, connected_graph, **detect_kwargs
    )
    return blocks, active_norms, connected_graph


def _detect_vlm_backbone(vlm_key, backend, detect_kwargs):
    """Build, export, and run block detection for one VLM-backbone (model, backend) case."""
    _patch_sdpa_mask()
    model, cfg = _build_vlm_backbone(vlm_key)
    onnx_model = _export_onnx(
        model,
        _vlm_sample_inputs(cfg),
        backend,
        input_names=["inputs_embeds", "position_ids", "attention_mask"],
    )
    connected_graph = ConnectedGraph(onnx_model)
    blocks, active_norms = get_decoder_block_boundaries(
        onnx_model, connected_graph, **detect_kwargs
    )
    return blocks, active_norms, connected_graph


def _layer_index(op):
    """Decoder layer index.

    NOTE: For now, this is only meaningful under torchscript.
          For dynamo, we will need a pass to fix op names to preserve scope hierarchy.
    """
    match = re.search(r"layers\.(\d+)\b", op.name)
    return int(match.group(1)) if match else None


def _layer_weighted_linear_count(connected_graph, start_op, end_op):
    """Count per-layer weighted linears in the topological span [start_op, end_op)."""
    topo = {id(op): i for i, op in enumerate(connected_graph.ordered_ops)}
    start, end = topo[id(start_op)], topo[id(end_op)]
    return sum(
        1
        for op in connected_graph.ordered_ops[start:end]
        if op.type in ("MatMul", "Gemm", "Conv") and "layers." in op.name
    )


def _assert_block_detection(
    blocks,
    active_norms,
    connected_graph,
    backend,
    *,
    active_norms_per_block=2,
    homogeneous=True,
):
    """Assert detected blocks describe a correct decoder stack."""
    # counts
    assert len(blocks) == _NUM_LAYERS
    assert len(active_norms) == active_norms_per_block * _NUM_LAYERS + 1

    # contiguity: block i end is block i+1 start
    for i in range(len(blocks) - 1):
        assert id(blocks[i][1]) == id(blocks[i + 1][0])

    # every active norm feeds a downstream weight linear
    assert active_norms
    for active_norm in active_norms:
        assert active_norm.downstream_linears

    # Name-based structural checks only for torchscript backend
    if backend == "torchscript":
        start_indices = [_layer_index(start) for start, _ in blocks]
        assert all(idx is not None for idx in start_indices)
        assert start_indices == sorted(set(start_indices))
        for i, (_, end) in enumerate(blocks):
            if i < len(blocks) - 1:
                assert _layer_index(end) is not None
            else:
                assert _layer_index(end) is None

        # homogeneous stacks: identical per-layer weighted-linear count per block.
        if homogeneous:
            counts = [
                _layer_weighted_linear_count(connected_graph, s, e) for s, e in blocks
            ]
            assert len(set(counts)) == 1


def _block_topology_params():
    """Flatten the model matrix into (config_attr, backend, kwargs, homogeneous) params."""
    for (
        test_id,
        config_attr,
        backends,
        detect_kwargs,
        homogeneous,
    ) in _BLOCK_TOPOLOGY_MODELS:
        backend = backends[0]
        yield pytest.param(
            config_attr, backend, detect_kwargs, homogeneous, id=f"{test_id}-{backend}"
        )


def _strip_matmul_suffix(name):
    """Normalize op names by stripping trailing /MatMul for projection layers.

    Different transformers versions (or import-order side effects) may export
    nn.Linear projections as either ``v_proj/MatMul`` or just ``v_proj``.
    Stripping the suffix lets us assert on the logical layer identity.
    """
    if name.endswith("/MatMul") and name.count("/") > 3:
        prefix = name[: -len("/MatMul")]
        # Only strip if it looks like a named projection (not bare self_attn/MatMul)
        last_part = prefix.rsplit("/", 1)[-1]
        if last_part.endswith("_proj"):
            return prefix
    return name


def verify_find_blocks(sim, model_type):
    end_points = get_decoder_blocks_end_points(sim, model_type)
    end_points_names = [(op1.name, op2.name) for op1, op2 in end_points]
    assert end_points_names == [
        (
            "/model/model/layers.0/input_layernorm",
            "/model/model/layers.1/input_layernorm",
        ),
        ("/model/model/layers.1/input_layernorm", "/model/model/norm"),
    ]
    conv_linear_blocks = get_conv_linear_layers_decoder_block(sim, end_points)
    conv_linear_blocks_names = []
    for ops in conv_linear_blocks:
        conv_linear_blocks_names.append([_strip_matmul_suffix(op.name) for op in ops])

    assert conv_linear_blocks_names == [
        [
            "/model/model/layers.0/self_attn/v_proj",
            "/model/model/layers.0/self_attn/k_proj",
            "/model/model/layers.0/self_attn/q_proj",
            "/model/model/layers.0/self_attn/MatMul",
            "/model/model/layers.0/self_attn/MatMul_1",
            "/model/model/layers.0/self_attn/o_proj",
            "/model/model/layers.0/mlp/up_proj",
            "/model/model/layers.0/mlp/gate_proj",
            "/model/model/layers.0/mlp/down_proj",
        ],
        [
            "/model/model/layers.1/self_attn/v_proj",
            "/model/model/layers.1/self_attn/k_proj",
            "/model/model/layers.1/self_attn/q_proj",
            "/model/model/layers.1/self_attn/MatMul",
            "/model/model/layers.1/self_attn/MatMul_1",
            "/model/model/layers.1/self_attn/o_proj",
            "/model/model/layers.1/mlp/up_proj",
            "/model/model/layers.1/mlp/gate_proj",
            "/model/model/layers.1/mlp/down_proj",
        ],
    ]


def test_get_decoder_blocks(add_genai_tests_path):
    from GenAILab.qai_hub_lm.backends.onnx.llm import LLM_ONNX
    from GenAILab.qai_hub_lm.backends.onnx.export_utils import (
        get_model_checkpoint_path,
    )

    model_id = "Qwen/Qwen2-0.5B"
    cache_dir = get_model_checkpoint_path(model_id)
    try:
        entry = LLM_ONNX.instantiate_float_model(model_id, 32, 16, small_model=True)
        collection = LLM_ONNX.instantiate_quantsim(entry)
        verify_find_blocks(collection.backbone, "qwen2")
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_get_decoder_blocks_qwen3(add_genai_tests_path):
    from GenAILab.qai_hub_lm.backends.onnx.llm import LLM_ONNX
    from GenAILab.qai_hub_lm.backends.onnx.export_utils import (
        get_model_checkpoint_path,
    )

    model_id = "Qwen/Qwen3-0.6B"
    cache_dir = get_model_checkpoint_path(model_id)
    try:
        entry = LLM_ONNX.instantiate_float_model(model_id, 32, 16, small_model=True)
        collection = LLM_ONNX.instantiate_quantsim(entry)
        verify_find_blocks(collection.backbone, "qwen3")
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_get_decoder_blocks_qwen3_5(add_genai_tests_path):
    import GenAILab.qai_hub_lm.transforms.exportable_linear_attention  # noqa: F401
    from GenAILab.bench.yaml_config_parser import YAMLConfigParser
    from GenAILab.qai_hub_lm.backends.onnx.export_utils import (
        get_model_checkpoint_path,
    )

    model_id = "Qwen/Qwen3.5-0.8B"
    cache_dir = get_model_checkpoint_path(model_id)
    try:
        model_cls = YAMLConfigParser.get_model_class(
            "qwen3_5", ["ExportableLinearAttention"]
        )
        entry = model_cls.instantiate_float_model(model_id, 32, 16, small_model=True)
        collection = model_cls.instantiate_quantsim(entry)

        blocks, active_norms = get_decoder_block_boundaries(
            collection.backbone.model.model,
            collection.backbone.connected_graph,
        )
        assert len(blocks) == 2
        assert len(active_norms) == 2 * len(blocks) + 1
        for i in range(len(blocks) - 1):
            assert id(blocks[i][1]) == id(blocks[i + 1][0])
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.mark.skip_on_windows_arm64("transformers is not available on Windows ARM64")
@pytest.mark.skip_on_windows_amd64(
    "transformers ONNX export unreliable on Windows AMD64"
)
class TestDecoderBlockBoundaries:
    """Block detection across architectures."""

    @pytest.mark.parametrize(
        "config_attr, backend, detect_kwargs, homogeneous",
        list(_block_topology_params()),
    )
    def test_block_detection(self, config_attr, backend, detect_kwargs, homogeneous):
        """Detect block boundaries on a LLM decoder."""
        blocks, active_norms, cg = _detect_from_config(
            config_attr, backend, detect_kwargs
        )
        _assert_block_detection(
            blocks,
            active_norms,
            cg,
            backend,
            active_norms_per_block=detect_kwargs.get("active_norms_per_block", 2),
            homogeneous=homogeneous,
        )

    @pytest.mark.parametrize("vlm_key", _VLM_BACKBONE_MODELS)
    def test_vlm_backbone_block_detection(self, vlm_key):
        """Detect blocks on a VLM language backbone (torchscript export)."""
        blocks, active_norms, cg = _detect_vlm_backbone(vlm_key, "torchscript", {})
        _assert_block_detection(blocks, active_norms, cg, "torchscript")

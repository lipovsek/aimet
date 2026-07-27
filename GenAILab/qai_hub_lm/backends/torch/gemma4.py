# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Gemma4 Torch backend: W4A8 quantsim + packed-QAT checkpoint loading.

The framework-agnostic model definition lives in
``GenAILab.qai_hub_lm.models.gemma4`` (shared with the ONNX backend). All the
torch/aimet-specific QAT machinery -- dequantizing packed weights, building the
sim, and loading the trained QAT scales -- lives here.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn
from transformers import AutoConfig
from huggingface_hub import hf_hub_download

from transformers.models.gemma4 import modeling_gemma4

from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.qai_hub_lm.backends.torch.vlm import VLM_Torch
from GenAILab.qai_hub_lm.models.gemma4 import Gemma4_VLM


def _quant_config_from_config(config) -> dict | None:
    """Return the quantization_config dict if it is a 'gemma' packed config."""
    quant_config = getattr(config, "quantization_config", None)
    if quant_config is None:
        return None
    if not isinstance(quant_config, dict):
        quant_config = getattr(quant_config, "to_dict", lambda: None)() or vars(
            quant_config
        )
    if quant_config.get("quant_method") != "gemma":
        return None
    return quant_config


def _dequantize_packed_weights(model, dtype: torch.dtype = torch.float32) -> None:
    """In-place replace packed Gemma quant modules with plain float modules.

    QuantizedLinear -> nn.Linear, QuantizedEmbedding -> Gemma4TextScaledWordEmbedding.
    Uses transformers' own unpacking routines.
    """
    from transformers.integrations.gemma_quant import (
        QuantizedEmbedding,
        QuantizedLinear,
    )

    scaled_embedding_cls = modeling_gemma4.Gemma4TextScaledWordEmbedding
    pad_idx = getattr(model.config.text_config, "pad_token_id", None)

    for name, module in list(model.named_modules()):
        if isinstance(module, QuantizedLinear):
            new_module = nn.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
            )
            new_module.weight = nn.Parameter(
                module._dequantize_weights().to(dtype), requires_grad=False
            )
            if module.bias is not None:
                new_module.bias = nn.Parameter(
                    module.bias.detach().to(dtype), requires_grad=False
                )
            model.set_submodule(name, new_module)
        elif isinstance(module, QuantizedEmbedding):
            new_module = scaled_embedding_cls(
                module.num_embeddings,
                module.embedding_dim,
                padding_idx=pad_idx,
                embed_scale=module.scalar_embed_scale,
            )
            new_module.weight = nn.Parameter(
                module.weight.detach().to(dtype), requires_grad=False
            )
            model.set_submodule(name, new_module)


# ---------------------------------------------------------------------------
# QAT checkpoint helpers (Gemma-only for now). Build/apply the symmetric-signed
# grids the packed Gemma QAT checkpoint was trained with. Kept module-level so
# load_qat_encodings can call them without duplicating the math.
# ---------------------------------------------------------------------------
def _symmetric_qdq(scale, num_bits):
    """Build a frozen-grid symmetric-signed QuantizeDequantize from a raw scale.

    Matches the QAT grid: symmetric, zero offset, qmin=-2^(b-1), qmax=2^(b-1)-1.

    No device handling needed: load_qat_encodings runs inside instantiate_quantsim,
    BEFORE generator.on_device(...) — the sim (and the safetensors scales) are on
    CPU, and the whole sim is moved to the eval device later.
    """
    from aimet_torch.quantization.affine import AffineEncoding, QuantizeDequantize

    qmax = 2 ** (num_bits - 1) - 1
    qmin = -qmax - 1
    scale = scale.to(torch.float32)
    encoding = AffineEncoding(
        scale=scale,
        offset=torch.zeros_like(scale),
        qmin=qmin,
        qmax=qmax,
        symmetry=True,
    )
    return QuantizeDequantize.from_encodings(encoding)


def _fake_quant_weight(w, scale, num_bits):
    """Symmetric-signed fake quantization of ``w`` (returns a new fp32 tensor).

    Handles per-channel scales (rows, 1) and group-wise scales (rows, num_groups).
    """
    w = w.to(torch.float32)
    # Match scale to w's device (safetensors scales are CPU; w may be on GPU).
    scale = scale.to(device=w.device, dtype=torch.float32)
    _, cols = w.shape
    num_groups = scale.shape[-1]
    if num_groups == 1:
        scale_full = scale
    else:
        if cols % num_groups != 0:
            raise ValueError(
                f"Scale groups ({num_groups}) do not divide column count ({cols})."
            )
        scale_full = scale.repeat_interleave(cols // num_groups, dim=1)
    qmax = 2 ** (num_bits - 1) - 1
    qmin = -qmax - 1
    return torch.clamp(torch.round(w / scale_full), qmin, qmax) * scale_full


@torch.no_grad()
def _fake_quant_embedding(embedding: torch.nn.Module, scale, num_bits) -> None:
    """Apply symmetric-signed fake quantization to an embedding weight in place."""
    fq = _fake_quant_weight(embedding.weight.data, scale, num_bits)
    embedding.weight.data.copy_(fq.to(embedding.weight.dtype))


@YAMLConfigParser.register_model("gemma4")
class Gemma4_Torch(VLM_Torch, Gemma4_VLM):
    """Gemma4 VLM Torch backend with packed-QAT checkpoint support."""

    @staticmethod
    def is_qat_checkpoint(config) -> bool:
        """True if the model config is a packed Gemma QAT checkpoint."""
        return _quant_config_from_config(config) is not None

    @classmethod
    def dequantize_packed_weights(cls, model) -> None:
        """In-place dequantize a packed Gemma4 QAT model to plain float modules.

        Replaces QuantizedLinear/QuantizedEmbedding with float modules so aimet
        can wrap them. Call only when is_qat_checkpoint(model.config) is True.
        """
        _dequantize_packed_weights(model)

    @staticmethod
    @torch.no_grad()
    def disable_uninitialized_quantizers(sim_model: torch.nn.Module) -> int:
        """Null out quantizers with no computed encoding so the forward doesn't
        crash on an unset QuantizeDequantize. Returns the count nulled."""
        from aimet_torch.v2.nn import BaseQuantizationMixin

        n = 0
        for module in sim_model.modules():
            if not isinstance(module, BaseQuantizationMixin):
                continue
            for container, keys in (
                (module.input_quantizers, range(len(module.input_quantizers))),
                (module.output_quantizers, range(len(module.output_quantizers))),
                (module.param_quantizers, list(module.param_quantizers.keys())),
            ):
                for key in keys:
                    q = container[key]
                    if q is not None and not q.is_initialized():
                        container[key] = None
                        n += 1
        return n

    @classmethod
    def load_qat_encodings(
        cls, sim_collection, model_id: str, *, freeze: bool = True
    ) -> dict:
        """Load Gemma4 QAT weight + activation scales from the packed checkpoint.

        Reads scales directly from model_id's model.safetensors via hf_hub_download.
        Does NOT load weights (the sim is already built from the dequantized model).

        Steps (order matters):
          1. Resolve weight quantizer bitwidths from quantization_config.
          2. Load weight_scale -> param_quantizers["weight"], frozen.
          3. Load q/k/v input scales -> input_layernorm.output_quantizers[0], frozen.
          4. Load gate/up input scales -> pre_feedforward_layernorm.output_quantizers[0].
          5. o_proj input: re-enable input_quantizers[0] and load scale, frozen.
          6. Other activation scales (down_proj, per_layer_projection, o_proj out, etc.).
          7. Embedding fake-quant in-place (Option B: precision.embedding = float32).
        """
        from safetensors import safe_open
        from aimet_torch.v2.nn import QuantizationMixin

        # SRQ activations are always 8-bit, independent of a layer's weight bitwidth.
        SRQ_ACTIVATION_BITS = 8

        # --- constants ---
        _MIN_BW = 4  # minimum exportable weight bitwidth (2-bit -> 4-bit clamp)
        _WEIGHT_SCALE = "weight_scale"
        _EMBED_SCALE = "embedding_scale"
        _IN_ACT = "input_activation_scale"
        _OUT_ACT = "output_activation_scale"
        _EMBED_TOKENS = "model.language_model.embed_tokens"
        _EMBED_PER_LAYER = "model.language_model.embed_tokens_per_layer"
        _EMBED_BITS = {_EMBED_TOKENS: 2, _EMBED_PER_LAYER: 4}

        # input_activation_scale keys whose input edge is a SHARED residual.
        # Loading these downgrades the residual stream.
        _SKIP_INPUT_SCALE_MODULES = {"per_layer_input_gate"}

        # Activation scale keys to skip because scale == 0.0 (sentinel).
        # lm_head act scales are intentionally 0.0 and will be calibrated later.

        def _backbone_module_name(ck_name: str) -> str:
            """Map a checkpoint path to its sim path in the backbone QuantSim."""
            if ck_name == "lm_head":
                return "lm_head.lm_head"
            prefix = "model.language_model."
            if ck_name.startswith(prefix):
                return "model." + ck_name[len(prefix) :]
            return ck_name

        def _sim_module_name_to_ckpt(sim_name: str) -> str:
            """Reverse of _backbone_module_name (for bitwidth-from-config lookup)."""
            if sim_name == "lm_head.lm_head":
                return "lm_head"
            if sim_name.startswith("model."):
                return "model.language_model." + sim_name[len("model.") :]
            return sim_name

        # --- resolve quantization_config ---
        quant_config = _quant_config_from_config(sim_collection.config)
        if quant_config is None:
            quant_config = _quant_config_from_config(
                AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            )

        def _resolve_weight_num_bits(ckpt_path: str) -> int | None:
            if quant_config is None:
                return None
            path = ckpt_path.removeprefix("model.")
            for pattern, cfg in (
                quant_config.get("module_quant_configs") or {}
            ).items():
                if re.search(pattern, path):
                    return cfg.get("num_bits", quant_config.get("num_bits"))
            return quant_config.get("num_bits")

        backbone_modules = dict(sim_collection.backbone.model.named_modules())

        def _resolve_backbone(ck_module: str):
            return backbone_modules.get(_backbone_module_name(ck_module))

        counts = {
            "weight_scales": 0,
            "input_act_scales": 0,
            "output_act_scales": 0,
            "input_act_scales_reenabled": 0,
            "embedding_scales": 0,
            "skipped_zero": 0,
            "skipped_no_target": 0,
            "groups_unequal": 0,
        }

        def _resolve_quantized(ck_module: str):
            """Return the quantized-mixin sim module for a checkpoint path, or None
            (bumping skipped_no_target)."""
            module = _resolve_backbone(ck_module)
            if module is None or not isinstance(module, QuantizationMixin):
                counts["skipped_no_target"] += 1
                return None
            return module

        def _set_qdq(owner, kind: str, idx: int, scale, num_bits: int) -> None:
            """Build a frozen symmetric QDQ from scale and set it on
            owner.<kind>_quantizers[idx]."""
            qdq = _symmetric_qdq(scale, num_bits)
            if freeze:
                qdq.allow_overwrite(False)
            getattr(owner, f"{kind}_quantizers")[idx] = qdq

        # q/k/v and gate/up each share one input scale (F1); load it onto the
        # producer RMSNorm's output quantizer. Map: consumer short names ->
        # (producer norm suffix, the member we load from).
        _PRODUCER_GROUPS = {
            ("q_proj", "k_proj", "v_proj"): ("input_layernorm", "q_proj"),
            ("gate_proj", "up_proj"): ("pre_feedforward_layernorm", "gate_proj"),
        }
        # Track group scales per (layer, producer) to verify q/k/v (and gate/up)
        # really share a scale before collapsing to the producer (F1 safety net).
        _group_scale_seen: dict[tuple[str, str], float] = {}

        safetensors_path = hf_hub_download(model_id, "model.safetensors")

        # --- Step 1: set weight quantizer bitwidths from quantization_config ---
        if quant_config is not None:
            for sim_name, module in backbone_modules.items():
                pq = getattr(module, "param_quantizers", None)
                wq = pq["weight"] if pq is not None and "weight" in pq else None
                if wq is None:
                    continue
                ckpt_path = _sim_module_name_to_ckpt(sim_name)
                num_bits = _resolve_weight_num_bits(ckpt_path)
                if num_bits is None:
                    continue
                applied = max(num_bits, _MIN_BW)
                if wq.bitwidth != applied:
                    wq.bitwidth = applied

        with safe_open(safetensors_path, framework="pt") as f:
            all_keys = list(f.keys())

            # --- Step 2: weight scales ---
            for key in all_keys:
                if not key.endswith("." + _WEIGHT_SCALE):
                    continue
                ck_module = key[: -(len(_WEIGHT_SCALE) + 1)]
                module = _resolve_quantized(ck_module)
                if module is None:
                    continue
                pq = getattr(module, "param_quantizers", None)
                wq = pq["weight"] if pq is not None and "weight" in pq else None
                if wq is None:
                    counts["skipped_no_target"] += 1
                    continue
                scale = f.get_tensor(key)
                target_shape = getattr(wq, "shape", None)
                if target_shape is not None:
                    scale = scale.reshape(tuple(target_shape))
                _set_qdq(module, "param", "weight", scale, wq.bitwidth)
                counts["weight_scales"] += 1

            # --- Steps 3-6: activation scales ---
            # q/k/v & gate/up -> producer RMSNorm output quantizer (shared scale);
            # per_layer_input_gate -> skipped (shared residual, V2); o_proj input ->
            # re-enable; everything else -> its own input/output quantizer.
            for key in all_keys:
                for suffix, role in ((_IN_ACT, "input"), (_OUT_ACT, "output")):
                    if not key.endswith("." + suffix):
                        continue
                    ck_module = key[: -(len(suffix) + 1)]
                    # Short module name relative to a layer (e.g. "q_proj")
                    short = ck_module.split(".")[-1]

                    scale = f.get_tensor(key)
                    if float(scale) == 0.0:
                        counts["skipped_zero"] += 1
                        break

                    # SRQ activations are always 8-bit regardless of the sim's
                    # default activation bitwidth (int16, so non-QAT / residual
                    # activations calibrate at 16-bit). Build every QAT activation
                    # quantizer at SRQ_ACTIVATION_BITS and REPLACE the sim default --
                    # do NOT read the existing quantizer's bitwidth.
                    s0 = scale.reshape(())

                    if role == "output":
                        module = _resolve_quantized(ck_module)  # counts None case
                        if module is None:
                            break
                        if not getattr(module, "output_quantizers", None):
                            counts["skipped_no_target"] += 1
                            break
                        _set_qdq(module, "output", 0, s0, SRQ_ACTIVATION_BITS)
                        counts["output_act_scales"] += 1
                        break

                    # role == "input"
                    producer_spec = next(
                        (v for grp, v in _PRODUCER_GROUPS.items() if short in grp), None
                    )
                    if producer_spec is not None:
                        # q/k/v or gate/up: load the shared scale onto the producer
                        # RMSNorm's output quantizer. Load once (from the designated
                        # member); verify the group actually shares a scale (F1).
                        norm_suffix, load_from = producer_spec
                        layer_ck = ck_module.rsplit(".", 2)[
                            0
                        ]  # strip "self_attn.q_proj"
                        norm_sim_name = _backbone_module_name(
                            layer_ck + "." + norm_suffix
                        )
                        producer = backbone_modules.get(norm_sim_name)
                        if (
                            producer is None
                            or not getattr(producer, "output_quantizers", None)
                            or producer.output_quantizers[0] is None
                        ):
                            counts["skipped_no_target"] += 1
                            break
                        gkey = (norm_sim_name, norm_suffix)
                        prev = _group_scale_seen.get(gkey)
                        if prev is None:
                            _group_scale_seen[gkey] = float(s0)
                        elif prev != float(s0):
                            counts["groups_unequal"] += 1
                        if short == load_from:
                            _set_qdq(producer, "output", 0, s0, SRQ_ACTIVATION_BITS)
                            counts["input_act_scales"] += 1
                        break

                    # per_layer_input_gate: shared residual edge -- skip (V2).
                    if short in _SKIP_INPUT_SCALE_MODULES:
                        counts["skipped_no_target"] += 1
                        break

                    # o_proj input (re-enable disabled quantizer) + any other input.
                    module = _resolve_quantized(ck_module)  # counts None case
                    if module is None:
                        break
                    if not getattr(module, "input_quantizers", None):
                        counts["skipped_no_target"] += 1
                        break
                    was_disabled = module.input_quantizers[0] is None
                    _set_qdq(module, "input", 0, s0, SRQ_ACTIVATION_BITS)
                    counts[
                        "input_act_scales_reenabled"
                        if was_disabled
                        else "input_act_scales"
                    ] += 1
                    break

            # --- Step 7: embedding fake-quant (Option B) ---
            embedding = sim_collection.embedding
            embed_per_layer = (
                sim_collection.extras.get("embed_tokens_per_layer")
                if getattr(sim_collection, "extras", None)
                else None
            )
            for ck_name, emb_mod in (
                (_EMBED_TOKENS, embedding),
                (_EMBED_PER_LAYER, embed_per_layer),
            ):
                key = ck_name + "." + _EMBED_SCALE
                if key not in all_keys or emb_mod is None:
                    continue
                _fake_quant_embedding(emb_mod, f.get_tensor(key), _EMBED_BITS[ck_name])
                counts["embedding_scales"] += 1

        print("[load_qat_encodings] loaded:")
        for k, v in counts.items():
            print(f"    {k}: {v}")
        return counts

    @classmethod
    def instantiate_quantsim(
        cls,
        model,
        context_length,
        sequence_length,
        precision=None,
        image_size=None,
        model_id=None,
        **kwargs,
    ):
        """Build the W4A8 sim; if the checkpoint is a packed Gemma QAT model,
        dequantize it, then load its trained QAT scales onto the sim and
        disable the quantizers the load left uninitialized.

        The returned SimCollection is what Stage 2 produces: QAT scales loaded
        (frozen) and uninitialized quantizers nulled -- no recipe needed. A
        standard float gemma4 model_id skips all QAT work (no-op).
        """
        is_qat = cls.is_qat_checkpoint(model.config)
        if is_qat:
            print(f"[Gemma4] Dequantizing packed QAT weights for {model_id}")
            cls.dequantize_packed_weights(model)

        sim_collection = super().instantiate_quantsim(
            model,
            context_length,
            sequence_length,
            precision=precision,
            image_size=image_size,
            **kwargs,
        )

        if is_qat:
            if model_id is None:
                raise RuntimeError(
                    "Gemma4 QAT checkpoint requires model_id to locate "
                    "model.safetensors; pass model_id to instantiate_quantsim."
                )
            counts = cls.load_qat_encodings(sim_collection, model_id)
            n = cls.disable_uninitialized_quantizers(sim_collection.backbone.model)
            print(f"[Gemma4 QAT] loaded {counts}; nulled {n} uninitialized quantizers")

        return sim_collection

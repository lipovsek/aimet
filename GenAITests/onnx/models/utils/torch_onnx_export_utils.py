# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for exporting models from ONNX to Torch"""

import os
from pathlib import Path
import torch
import onnx
import glob
from transformers import AutoConfig
from huggingface_hub import HfApi

ONNX_OPSET_VERSION = 17


def is_huggingface_ckpt(model_id: str) -> bool:
    hf_api = HfApi()
    try:
        _ = hf_api.model_info(model_id)
        return True
    except Exception:
        return False


def get_model_checkpoint_path(model_id: str, classname: str | None = None) -> str:
    if is_huggingface_ckpt(model_id):
        # user has passed in a huggingface checkpoint, use default framework cache path
        if classname is not None:
            return f"onnx_checkpoints/{model_id}/{classname}"
        return f"onnx_checkpoints/{model_id}"
    else:
        # user has passed in a local path, verify that .onnx file and .config files exist and just return the path
        if not os.path.isdir(model_id):
            raise RuntimeError(
                f"Provided model_id '{model_id}' is not a valid HuggingFace model ID or a local directory."
            )

        if not any(
            filename.name.endswith(".onnx") for filename in Path(model_id).rglob("*")
        ):
            raise RuntimeError(
                f"No .onnx file found in the provided local directory '{model_id}'.'"
            )

        if not any(
            filename.name == "config.json" for filename in Path(model_id).rglob("*")
        ):
            raise RuntimeError(
                f"No config.json file found in the provided local directory '{model_id}'.'"
            )

        return model_id


def equivalent_configs(config_a, config_b) -> bool:
    config_dict_a = config_a.to_dict()
    config_dict_b = config_b.to_dict()
    del config_dict_a["_name_or_path"]
    del config_dict_b["_name_or_path"]
    return config_dict_a == config_dict_b


def get_opset(filepath):
    if not os.path.exists(filepath):
        raise RuntimeError(f"File `{filepath}` does not exist.")

    model = onnx.ModelProto()
    with open(filepath, "rb") as f:
        # Only parse specific fields (field number 8 = opset_import)
        model.MergeFromString(f.read())

    return {op.domain or "ai.onnx": op.version for op in model.opset_import}


def check_opset_equal_to(filepath, opset_version: int) -> bool:
    opset = get_opset(filepath)
    return opset.get("ai.onnx", -1) == opset_version


def load_model_components_from_disk(
    checkpoint: str | os.PathLike,
    context_length: int,
    sequence_length: int,
) -> tuple[onnx.ModelProto, onnx.ModelProto | None, torch.nn.Embedding | None]:
    aihm_format_backbone_path = os.path.join(
        checkpoint, f"model_seqlen{sequence_length}_cl{context_length}.onnx"
    )
    genaitests_format_backbone_path = os.path.join(
        checkpoint, "backbone", f"model_sl{sequence_length}_cl{context_length}.onnx"
    )

    backbone = onnx.load(
        aihm_format_backbone_path
        if os.path.exists(aihm_format_backbone_path)
        else genaitests_format_backbone_path
    )

    visual_path = os.path.join(checkpoint, "visual", "model.onnx")
    visual = onnx.load(visual_path) if os.path.exists(visual_path) else None

    embedding_path = os.path.join(checkpoint, "embedding.pth")
    if os.path.exists(embedding_path):
        weights = torch.load(
            embedding_path, map_location="cpu"
        )  # -> torch.Tensor of shape [V, D]
        if not isinstance(weights, torch.Tensor) or weights.ndim != 2:
            raise ValueError("Expected a 2D embedding tensor in embedding.pth")

        embedding = torch.nn.Embedding.from_pretrained(weights, freeze=False)
        embedding = embedding.to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        embedding = None

    return backbone, visual, embedding


def get_onnx_model(
    checkpoint: str | os.PathLike,
    fp_backbone_model: torch.nn.Module,
    context_length: int,
    sequence_length: int,
    sample_input: tuple[torch.Tensor, ...],
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    fp_visual_model: torch.nn.Module | None = None,
    sample_visual_input: tuple[torch.Tensor, ...] | None = None,
    visual_input_names: tuple[str, ...] | None = None,
    visual_output_names: tuple[str, ...] | None = None,
) -> tuple[onnx.ModelProto, onnx.ModelProto | None]:
    # Create the checkpoint directory if it does not exist.
    os.makedirs(checkpoint, exist_ok=True)
    onnx_backbone_path = os.path.join(
        checkpoint, "backbone", f"model_sl{sequence_length}_cl{context_length}.onnx"
    )
    onnx_visual_path = os.path.join(checkpoint, "visual", "model.onnx")
    config_path = os.path.join(checkpoint, "config.json")

    visual_model_exists = fp_visual_model is not None

    fp_backbone_model.eval()
    fp_backbone_model.train(False)

    # re-export model if model/config is not found on disk OR if config on disk does not match model config
    if (
        not os.path.exists(onnx_backbone_path)
        or not os.path.exists(config_path)
        or not equivalent_configs(
            AutoConfig.from_pretrained(config_path), fp_backbone_model.config
        )
        or (visual_model_exists and not os.path.exists(onnx_visual_path))
        or not check_opset_equal_to(onnx_backbone_path, ONNX_OPSET_VERSION)
        or (
            visual_model_exists
            and not check_opset_equal_to(onnx_visual_path, ONNX_OPSET_VERSION)
        )
    ):
        print("Exporting model(s) to ONNX...")
        fp_backbone_model.to(torch.device("cpu"))

        fp_backbone_model.config.save_pretrained(checkpoint)
        with torch.no_grad():
            os.makedirs(os.path.join(checkpoint, "backbone"), exist_ok=True)
            torch.onnx.export(
                fp_backbone_model,
                sample_input,
                onnx_backbone_path,
                input_names=input_names,
                output_names=output_names,
                opset_version=ONNX_OPSET_VERSION,
                dynamo=False,
            )
            if visual_model_exists:
                os.makedirs(os.path.join(checkpoint, "visual"), exist_ok=True)
                torch.onnx.export(
                    fp_visual_model,
                    sample_visual_input,
                    onnx_visual_path,
                    input_names=visual_input_names,
                    output_names=visual_output_names,
                    opset_version=ONNX_OPSET_VERSION,
                    dynamo=False,
                )

        print("Loading ONNX model(s)...")
        model = onnx.load(onnx_backbone_path)
        if visual_model_exists:
            visual_model = onnx.load(onnx_visual_path)

        # Clean up multiple weights files
        for model_path in [onnx_backbone_path, onnx_visual_path]:
            for extension in ["*.weight", "*.bias", "onnx__*", "*__value"]:
                for file in glob.glob(
                    os.path.join(os.path.dirname(model_path), extension)
                ):
                    os.remove(file)

        onnx.save_model(
            model,
            onnx_backbone_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.data",
        )
        if visual_model_exists:
            onnx.save_model(
                visual_model,
                onnx_visual_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="model.data",
            )

        onnx.external_data_helper.load_external_data_for_model(
            model, os.path.dirname(onnx_backbone_path)
        )
        if visual_model_exists:
            onnx.external_data_helper.load_external_data_for_model(
                visual_model, os.path.dirname(onnx_visual_path)
            )
        return model, visual_model if visual_model_exists else None
    else:
        print("Loading cached ONNX model...")
        backbone, visual, *_ = load_model_components_from_disk(
            checkpoint, context_length=context_length, sequence_length=sequence_length
        )
        return backbone, visual

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for exporting models from ONNX to Torch"""

import os
import torch
import onnx
import glob
from transformers import AutoConfig
from huggingface_hub import HfApi


def is_huggingface_ckpt(model_id: str) -> bool:
    hf_api = HfApi()
    try:
        _ = hf_api.model_info(model_id)
        return True
    except Exception:
        return False


def get_model_checkpoint_path(model_id: str) -> str:
    if is_huggingface_ckpt(model_id):
        # user has passed in a huggingface checkpoint, use default framework cache path
        return f"onnx_checkpoints/{model_id}"
    else:
        # user has passed in a local path, verify that .onnx file and .config files exist and just return the path
        if not os.path.isdir(model_id):
            raise RuntimeError(
                f"Provided model_id '{model_id}' is not a valid HuggingFace model ID or a local directory."
            )

        for filename in os.listdir(model_id):
            if filename.endswith(".onnx"):
                break
        else:
            raise RuntimeError(
                f"No .onnx file found in the provided local directory '{model_id}'.'"
            )

        if not os.path.exists(os.path.join(model_id, "config.json")):
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


def get_onnx_model(
    checkpoint: str | os.PathLike,
    fp_model: torch.nn.Module,
    context_length: int,
    sequence_length: int,
    sample_input: tuple[torch.Tensor, ...],
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
) -> onnx.ModelProto:
    # Create the checkpoint directory if it does not exist.
    os.makedirs(checkpoint, exist_ok=True)
    onnx_model_path = os.path.join(
        checkpoint, f"model_sl{sequence_length}_cl{context_length}.onnx"
    )
    config_path = os.path.join(checkpoint, "config.json")

    fp_model.eval()
    fp_model.train(False)

    # re-export model if model/config is not found on disk OR if config on disk does not match model config
    if (
        not os.path.exists(onnx_model_path)
        or not os.path.exists(config_path)
        or not equivalent_configs(
            AutoConfig.from_pretrained(config_path), fp_model.config
        )
    ):
        print("Exporting model to ONNX...")
        fp_model.to(torch.device("cpu"))

        fp_model.config.save_pretrained(checkpoint)
        with torch.no_grad():
            torch.onnx.export(
                fp_model,
                sample_input,
                onnx_model_path,
                input_names=input_names,
                output_names=output_names,
                opset_version=17,
                dynamo=False,
            )

        print("Loading ONNX model...")
        model = onnx.load(onnx_model_path)

        # Clean up multiple weights files
        for file in glob.glob(
            os.path.join(os.path.dirname(onnx_model_path), "*.weight")
        ):
            os.remove(file)
        for file in glob.glob(
            os.path.join(os.path.dirname(onnx_model_path), "onnx__*")
        ):
            os.remove(file)
        for file in glob.glob(
            os.path.join(os.path.dirname(onnx_model_path), "*__value")
        ):
            os.remove(file)

        onnx.save_model(
            model,
            onnx_model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.data",
        )

        onnx.external_data_helper.load_external_data_for_model(
            model, os.path.dirname(onnx_model_path)
        )
    else:
        print("Loading cached ONNX model...")
        model = onnx.load(onnx_model_path)

    return model

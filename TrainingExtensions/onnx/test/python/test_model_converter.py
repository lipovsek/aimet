# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
from onnx.utils import extract_model
import onnxruntime as ort

from aimet_onnx.experimental.adascale.model_converter import ModelConverter

from aimet_onnx.quantsim import QuantizationSimModel
import pytest
import os
import torch
from onnx import numpy_helper
import numpy as np
from dataclasses import dataclass
import copy
from GenAITests.shared.models.generator import Generator
from GenAITests.onnx.models.utils.torch_onnx_interface import TorchONNXInterface
from GenAITests.onnx.helpers.quant_recipes import _prefill_inputs
from GenAITests.shared.helpers.datasets import Wikitext
from aimet_common.utils import compute_psnr
from aimet_onnx.experimental.adascale.find_blocks import (
    get_decoder_blocks_end_points,
)


def test_model_round_trip(monkeypatch):
    path = os.path.abspath(os.path.join("../../../../GenAITests"))
    monkeypatch.syspath_prepend(path)
    from GenAITests.onnx.models.qwen import Qwen_25_ONNX
    from transformers import AutoConfig

    small_model = True
    context_length = 32
    sequence_length = 16
    model_id = "Qwen/Qwen2.5-0.5B"
    sim = Qwen_25_ONNX.instantiate_quantsim(
        "Qwen/Qwen2.5-0.5B", 32, 16, small_model=small_model
    )
    llm_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if small_model:
        llm_config.num_hidden_layers = 2
    ################ Input for qwen2.5
    tokenizer = Qwen_25_ONNX.instantiate_tokenizer(model_id)

    train_dataset = Wikitext.load_encoded_dataset(tokenizer, context_length, "train")
    quantsim_with_torch_interface = TorchONNXInterface(sim, llm_config)
    generator = Generator(
        quantsim_with_torch_interface, tokenizer, sequence_length, context_length
    )

    inputs = _prefill_inputs(sim, generator, train_dataset, num_iterations=5)
    ################ fp32 onnx model
    CHECKPOINT_DIR = "onnx_checkpoints_debugging"
    CHECKPOINT_FP_DIR = "onnx_checkpoints_debugging/fp_models"
    os.makedirs(CHECKPOINT_FP_DIR, exist_ok=True)
    path = os.path.abspath(os.path.join("../../../../GenAITests"))

    fp32_model = copy.deepcopy(sim.model.model)
    fp32_model = QuantizationSimModel.remove_quantizers(fp32_model)
    common_inputs = ["attention_mask", "position_ids"]
    adascale_blocks_end_points = get_decoder_blocks_end_points(sim)
    block_inputs = [adascale_blocks_end_points[0][0].inputs[0].name]
    converter = ModelConverter(fp32_model, CHECKPOINT_DIR)
    model_before_block = os.path.join(CHECKPOINT_FP_DIR, "before_decoder_block.onnx")
    fp_model_path = converter._get_onnx_fp_model()
    extract_model(
        fp_model_path, model_before_block, list(inputs[0].keys()), block_inputs
    )
    before_session = ort.InferenceSession(
        model_before_block, providers=["CPUExecutionProvider"]
    )
    block_input_tensor = before_session.run(block_inputs, inputs[0])
    for block_id, (block_start, block_end) in enumerate(
        get_decoder_blocks_end_points(sim)
    ):
        block_inputs = [block_start.inputs[0].name]
        block_input_names = (
            block_inputs
            + common_inputs
            + [f"past_key_{block_id}_in", f"past_value_{block_id}_in"]
        )
        block_output_names = [block_end.inputs[0].name]
        block_input_output_names = (block_input_names, block_output_names)
        pt_block, block_model_path = converter.get_pt_block(block_input_output_names)
        ################ run forward pass 1 through onnx block
        onnx_fp_block_sess = ort.InferenceSession(
            block_model_path, providers=["CPUExecutionProvider"]
        )
        block_test_inputs = inputs[0].copy()
        block_test_inputs[block_inputs[0]] = block_input_tensor[0]
        for name in inputs[0].keys():
            if name not in block_input_names:
                del block_test_inputs[name]
        onnx_fp_out = onnx_fp_block_sess.run(None, block_test_inputs)
        ################ run forward pass 2 through converted pytorch(assert 1==2)
        torch_out = (
            pt_block(
                torch.from_numpy(block_input_tensor[0]).float(),
                torch.from_numpy(inputs[0]["attention_mask"]).long(),
                torch.from_numpy(inputs[0]["position_ids"]).long(),
                torch.from_numpy(inputs[0][f"past_key_{block_id}_in"]).float(),
                torch.from_numpy(inputs[0][f"past_value_{block_id}_in"]).float(),
            )
            .detach()
            .numpy()
        )
        assert compute_psnr(onnx_fp_out[0], torch_out) == 100

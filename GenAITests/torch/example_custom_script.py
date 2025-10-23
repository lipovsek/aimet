# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Example flattened script showcasing LPBQ on llama"""

import os
import argparse
import itertools
import torch
from tqdm import tqdm

from transformers import AutoTokenizer
from transformers.models.llama import modeling_llama

from aimet_common.defs import QuantScheme
from aimet_torch import QuantizationSimModel
from aimet_torch.v2.nn.transformers.models.llama.modeling_llama import (
    QuantizedLlamaRMSNorm,
)
from aimet_torch.utils import place_model, remove_all_quantizers

from GenAITests.shared.models.base import LLM
from GenAITests.shared.models.generator import Generator
from GenAITests.shared.models.utils.model_utils import ONNXExportableModuleWithCache
from GenAITests.shared.helpers.datasets import Wikitext
from GenAITests.shared.helpers.metrics import PPL

SEQUENCE_LENGTH = 2048
CONTEXT_LENGTH = 4096

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        help="Huggingface model id of desired Llama 3, 3.1, or 3.2 variant",
        default="meta-llama/Llama-3.2-1B-Instruct",
    )
    parser.add_argument(
        "--skip-quantization",
        action="store_true",
        help="Skip quantizing model",
    )
    parser.add_argument(
        "--export-path",
        help="Path to export model. Export will be skipped if no path is provided.",
        default=None,
    )
    args = parser.parse_args()

    # Fetch specified model and tokenizer from huggingface
    hf_model = modeling_llama.LlamaForCausalLM.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, use_fast=True, trust_remote_code=True
    )
    # Need to wrap model in this in order to enable JIT trace
    traceable_model = ONNXExportableModuleWithCache(hf_model)

    # Create dummy inputs used to initialize QuantizationSimModel
    dummy_input_ids = torch.zeros((1, SEQUENCE_LENGTH), dtype=torch.int)
    dummy_attention_mask = torch.ones((1, SEQUENCE_LENGTH), dtype=torch.int)
    assembled_dummy_inputs = Generator.prepare_inputs(
        model=traceable_model,
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
        past_key_values=[],
        context_length=CONTEXT_LENGTH,
        sequence_length=SEQUENCE_LENGTH,
    )

    # Create QuantizationSimModel with 4-bit integer weights and 16-bit integer activations
    quantsim = QuantizationSimModel(
        model=traceable_model,
        quant_scheme=QuantScheme.post_training_tf,
        dummy_input=assembled_dummy_inputs,
        default_output_bw=16,
        default_param_bw=4,
        in_place=True,
        config_file=LLM.get_quantsim_config(),
    )

    # Create a generator object to accurately simulate inference with static graph constraints while maintaining the
    # same interface. Use the generator object to do all forward passes through the model, including calibration, eval
    generator = Generator(quantsim.model, tokenizer, SEQUENCE_LENGTH, CONTEXT_LENGTH)

    # Apply mixed precision to model
    quantsim.model.model.lm_head.param_quantizers["weight"].bitwidth = 8
    for _, module in quantsim.model.named_modules():
        if isinstance(module, QuantizedLlamaRMSNorm):
            module.param_quantizers["weight"].bitwidth = 16

    # Use CUDA if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with place_model(quantsim.model, device):
        if args.skip_quantization:
            remove_all_quantizers(quantsim.model)
        else:
            # Load WikiText dataset from Huggingface
            train_dataset = Wikitext.load_encoded_dataset(
                tokenizer, CONTEXT_LENGTH, "train"
            )

            num_iterations = 20

            def calibration_callback(model: torch.nn.Module):
                sliced_dataloader = itertools.islice(train_dataset, num_iterations)
                for batch in tqdm(
                    sliced_dataloader, total=num_iterations, desc="Calibrating"
                ):
                    # Use generator for forward passes
                    generator(input_ids=batch["input_ids"].to(device=model.device))

            quantsim.compute_encodings(calibration_callback)

        ppl_score = PPL.evaluate(generator, tokenizer, CONTEXT_LENGTH)
        print(f"PPL: {ppl_score}")

        if args.export_path is not None:
            os.makedirs(args.export_path, exist_ok=True)
            quantsim.export(
                path=args.export_path,
                filename_prefix="model",
                dummy_input=assembled_dummy_inputs,
            )

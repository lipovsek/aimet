# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=missing-docstring
import torch
from torch.utils.data import Dataset, DataLoader
from aimet_torch.experimental.omniquant import apply_omniquant
from transformers import AutoTokenizer, AutoConfig
from transformers import LlamaForCausalLM, default_data_collator
from transformers.models.llama import modeling_llama
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from itertools import chain


# [setup]
# Load the model
# General setup that can be changed as needed
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "meta-llama/Llama-3.2-1B-Instruct"
model_config = AutoConfig.from_pretrained(model_id)
model_config.return_dict=False
model_config.use_cache = False

model = modeling_llama.LlamaForCausalLM.from_pretrained(model_id, config=model_config)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)

# End of [setup]

# [prepare-dataloader]
def tokenize(examples):
    seq_length = 2048
    examples = tokenizer(examples["text"])
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= seq_length:
        total_length = (total_length // seq_length) * seq_length
    result = {
        k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

train_dataset = load_dataset(path='wikitext', name='wikitext-2-raw-v1', split='train').map(tokenize, batched=True, remove_columns=['text'])
test_dataset = load_dataset(path='wikitext', name='wikitext-2-raw-v1', split='test').map(tokenize, batched=True, remove_columns=['text'])
train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=1, collate_fn=default_data_collator)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn=default_data_collator)

# Custom class to use limited samples from dataloader
dataloader_wrapper_len = 40
class LimitedBatchDataLoader(DataLoader):
    def __init__(self, data_loader):
        self.data_loader = data_loader
 
    def __len__(self):
        return dataloader_wrapper_len
 
    def __iter__(self):
        return iter(self.data_loader)

# End of [prepare-dataloader]

# [create-sim]
from aimet_torch.common.defs import QuantScheme
from aimet_torch import QuantizationSimModel

seq_length = 2048
input_ids = torch.randint(0, model_config.vocab_size, (1, seq_length), device=device)
attention_mask = torch.ones((1, seq_length), dtype=torch.long, device=device)
dummy_input = (input_ids, attention_mask)
sim = QuantizationSimModel(model,
                           dummy_input=dummy_input,
                           quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                           default_param_bw=4,
                           default_output_bw=16,
                           in_place=True)
# End of [create-sim]

# [apply-omniquant]
# Find and freeze optimal encodings candidate for weight parameters of supported layers

apply_omniquant(quant_sim=sim,
               dataloader=train_dataloader,
               forward_fn=lambda model, input: model.forward(**input),
               num_iterations=800)

# End of [apply-omniquant]

# [compute_encodings]
def calibration_wrapper(model, dataloader, max_iterations: int):
    for batch_id, batch in enumerate(dataloader):
        if batch_id < max_iterations:
            batch = tuple((d.to(device) for d in batch.values()))
            model.to(device)(*batch)
        else:
            break

# Compute the Quantization Encodings
# compute encodings for all activations and parameters of uninitialized layer(s)/operations(s)
sim.compute_encodings(calibration_wrapper, dataloader = train_dataloader, max_iterations=40)
# End of [compute_encodings]

# [evaluation]
# Determine simulated quantized accuracy
...
# End of [evaluation]

# [export]
# Export the model for on-target inference
path = './'
filename = 'dummy_model'
sim.export(path=path, filename_prefix="quantized_" + filename, dummy_input=dummy_input.cpu())
# End of [export]
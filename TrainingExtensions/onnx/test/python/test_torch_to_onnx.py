# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os

import onnx
import onnxruntime as ort
import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM
from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.models.mistral.modeling_mistral import MistralForCausalLM
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

import aimet_onnx
from aimet_onnx.utils import make_dummy_input
import aimet_torch

from GenAITests.shared.models.utils.model_utils import ONNXExportableModuleWithCache
from .utils import tmp_dir


@pytest.mark.parametrize(
    "model_cls, config_cls",
    [
        (Gemma3ForCausalLM, Gemma3TextConfig),
        (LlamaForCausalLM, LlamaConfig),
        (MistralForCausalLM, MistralConfig),
        (Phi3ForCausalLM, Phi3Config),
        (Qwen2ForCausalLM, Qwen2Config),
        (Qwen3ForCausalLM, Qwen3Config),
    ],
)
def test_hf_torch_to_onnx_workflow(tmp_dir, model_cls, config_cls):
    """
    Given: HF model quantized / exported as onnx QDQ from aimet-torch
    When: Import onnx QDQ into aimet-onnx
    Then: aimet-onnx sim should produce same output as aimet-torch sim
    """
    config = config_cls(
        vocab_size=1000,
        hidden_size=32,
        intermediate_size=32,
        num_attention_heads=32,
        num_hidden_layers=1,
        pad_token_id=999,
        return_dict=False,
    )
    model = ONNXExportableModuleWithCache(model_cls(config))
    input_ids = torch.randint(0, config.vocab_size, (1, 128))

    torch_sim = aimet_torch.QuantizationSimModel(
        model,
        dummy_input=input_ids,
        config_file="htp_quantsim_config_v81_per_channel_linear.json",
        in_place=True,
    )
    torch_sim.compute_encodings(lambda model: model(input_ids))
    onnx_qdq_model_path = os.path.join(tmp_dir, "model_qdq.onnx")
    aimet_torch.onnx.export(
        torch_sim.model,
        input_ids,
        onnx_qdq_model_path,
        opset_version=21,
        input_names=["input_ids"],
        output_names=["output"],
        dynamic_axes={"input_ids": {0: "batch_size"}},
        dynamo=False,
    )

    onnx_sim = aimet_onnx.QuantizationSimModel.from_onnx_qdq(
        onnx.load(onnx_qdq_model_path),
        config_file="htp_quantsim_config_v81_per_channel_linear.json",
    )
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    onnx_qdq_sess = ort.InferenceSession(onnx_qdq_model_path, sess_options=sess_options)

    # Allow off-by-three due to inevitable numerical errors
    atol = 3 * torch_sim.model.model.lm_head.output_quantizers[0].get_scale().item()

    for i in range(10):
        np.random.seed(i)
        input = make_dummy_input(onnx_sim.model.model)
        onnx_sim_out, *_ = onnx_sim.session.run(None, input)
        onnx_qdq_out, *_ = onnx_qdq_sess.run(None, input)
        assert np.allclose(onnx_sim_out, onnx_qdq_out, atol=atol)

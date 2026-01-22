# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import json

import torch

from aimet_onnx.utils import make_dummy_input

from aimet_onnx.common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx

from aimet_onnx.experimental.llm_configurator.llm_configurator import (
    _apply_int8_kv_cache_tying_and_lm_head,
    _set_matmul_second_input_to_8b,
    _get_quantizer_no_split_slice,
)

import onnx
import onnxsim
import os

from aimet_onnx.common.onnx._utils import _is_grid_preserving_op
from aimet_onnx.qc_quantize_op import QcQuantizeOp

from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaConfig

from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM, Phi3Config

from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Config

from transformers.cache_utils import DynamicCache

from .models import models_for_tests

from aimet_onnx.quantsim import QuantizationSimModel


def _get_enabled_quantizer_name(quant_sim, tensor_name: str) -> QcQuantizeOp:
    """
    Returns closest enabled quantizer to tensor traversing upwards only through invariant ops

    :param tensor_name: Name of tensor for which to find quantizer
    """
    quantizer = quant_sim.qc_quantize_op_dict.get(tensor_name, None)
    if quantizer and quantizer.enabled:
        return tensor_name

    prod_dict = quant_sim.connected_graph.get_all_products()
    product = prod_dict.get(tensor_name, None)

    if product == None:
        if tensor_name.endswith(("_updated", "_qdq")):
            raise KeyError(
                f"Could not find quantizer for tensor {tensor_name}. Input tensor_name must be the name of a tensor in the original (unquantized) graph"
            )
        else:
            raise KeyError(
                f"Could not find quantizer for tensor {tensor_name}. Tensor name does not exist in the graph"
            )

    producer = product.producer

    if producer == None:
        return None

    if not (_is_grid_preserving_op(producer.type)):
        return None

    if len(producer.inputs) == 0:
        return None

    upstream_tensor = producer.inputs[0]
    return _get_enabled_quantizer_name(quant_sim, upstream_tensor.name)


def check_config(
    quant_sim: QuantSimOnnx,
    encodings_path: str,
    kv_io_map: dict,
    lm_head_tensor_name: str,
    bw: int,
    is_sym: bool,
    dtype: str,
):
    with open(encodings_path, "r") as f:
        contents = json.load(f)

    activations = contents["activation_encodings"]
    params = contents["param_encodings"]

    for input, output in kv_io_map.items():
        kv_io_map[input] = _get_enabled_quantizer_name(quant_sim, output)

    names = set(list(kv_io_map.keys()) + list(kv_io_map.values()))

    quantizer_map = dict()

    assert len(activations) != 0, f"Activation Encodings are empty!"

    for act in activations:
        if act["name"] in names:
            assert act["bw"] == bw, (
                f"{act['name']} does not have bit width {bw}, has {act['bw']}!"
            )
            assert act["is_sym"] == is_sym, (
                f"{act['name']} does not have symmetry {is_sym}!"
            )
            assert act["dtype"] == dtype, (
                f"{act['name']} does not have data type {dtype}!"
            )

            quantizer_map[act["name"]] = (act["offset"], act["scale"])

    for param in params:
        if param["name"] == lm_head_tensor_name:
            assert param["bw"] == bw, (
                f"LM head {param['name']} does not have bit width {bw}!"
            )

    for input, output in kv_io_map.items():
        assert quantizer_map[input] == quantizer_map[output], (
            f"{input} and {output} quantizers are not tied!"
        )


class ExportableBase(torch.nn.Module):
    N_KEY_VALUE_HEADS = 32

    def base_forward(
        self,
        model_forward,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *past_key_values: torch.Tensor,
    ):
        kv_cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(
            zip(past_key_values[::2], past_key_values[1::2])
        ):
            k_split = [k[i : i + 1] for i in range(32)]
            v_split = [v[i : i + 1] for i in range(self.N_KEY_VALUE_HEADS)]
            k = torch.cat(k_split, axis=1).permute(0, 1, 3, 2)
            v = torch.cat(v_split, axis=1)

            kv_cache.update(k, v, layer_idx, {})  # pyright: ignore [reportArgumentType]

        out = model_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=kv_cache,
        )

        out_cache = out["past_key_values"]
        flat_output_past_key_values = []
        for layer in range(len(out_cache)):
            k = out_cache.key_cache[layer][:, :, -128:, :].permute(1, 0, 3, 2)
            v = out_cache.value_cache[layer][:, :, -128:, :].permute(1, 0, 2, 3)
            flat_output_past_key_values += [k, v]

        return [out["logits"]] + flat_output_past_key_values

    def get_output_names(self, num_hidden_layers: int):
        output_names = ["logits"]
        for layer in range(num_hidden_layers):
            output_names.append(f"past_key_{layer}_out")
            output_names.append(f"past_value_{layer}_out")
        return output_names

    def get_input_names(self, num_hidden_layers: int):
        output_names = ["input_ids", "attention_mask"]
        for layer in range(num_hidden_layers):
            output_names.append(f"past_key_{layer}_in")
            output_names.append(f"past_value_{layer}_in")
        return output_names


class ExportableLlama(LlamaForCausalLM, ExportableBase):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *past_key_values: torch.Tensor,
    ):
        return self.base_forward(
            super().forward, input_ids, attention_mask, *past_key_values
        )


class ExportableQwen(Qwen2ForCausalLM, ExportableBase):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *past_key_values: torch.Tensor,
    ):
        return self.base_forward(
            super().forward, input_ids, attention_mask, *past_key_values
        )


class ExportablePhi(Phi3ForCausalLM, ExportableBase):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *past_key_values: torch.Tensor,
    ):
        return self.base_forward(
            super().forward, input_ids, attention_mask, *past_key_values
        )


def apply_to_model(model_id, tmp_path):
    vocab_size = 8
    num_hidden_layers = 2
    hidden_size = 64
    num_attention_heads = 32
    num_key_value_heads = 32
    embed_dim = hidden_size // num_attention_heads // 2
    intermediate_size = 2
    sequence_length = 16
    context_length = 32

    if model_id == "llama":
        llm_config = LlamaConfig(
            vocab_size=vocab_size,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
        )

        model = ExportableLlama(config=llm_config)

    elif model_id == "qwen":
        llm_config = Qwen2Config(
            vocab_size=vocab_size,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
        )

        model = ExportableQwen(config=llm_config)

    elif model_id == "phi":
        llm_config = Phi3Config(
            vocab_size=vocab_size,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            pad_token_id=4,
        )

        model = ExportablePhi(config=llm_config)

    checkpoint = tmp_path / str(model_id)
    checkpoint.mkdir()

    onnx_model_path = os.path.join(checkpoint, f"model_cl{context_length}.onnx")

    dummy_input_ids = torch.zeros((1, sequence_length), dtype=torch.int32)
    dummy_attention_mask = torch.ones(
        (1, 1, sequence_length, context_length), dtype=torch.float32
    )

    past_key_values = []
    for _ in range(num_hidden_layers):
        past_key = torch.zeros(
            (num_key_value_heads, 1, embed_dim * 2, context_length - sequence_length),
            dtype=torch.float32,
        )
        past_value = torch.zeros(
            (num_key_value_heads, 1, context_length - sequence_length, embed_dim * 2),
            dtype=torch.float32,
        )
        past_key_values.append(past_key)
        past_key_values.append(past_value)

    example_input = [dummy_input_ids, dummy_attention_mask] + past_key_values

    with torch.no_grad():
        torch.onnx.export(
            model.eval(),
            tuple(example_input),
            onnx_model_path,
            input_names=model.get_input_names(2),
            output_names=model.get_output_names(2),
            opset_version=17,
            dynamo=False,
        )

        onnx_model = onnx.load(onnx_model_path)
        onnx_model, _ = onnxsim.simplify(onnx_model)

    model_name = f"{model_id}_2HL_simplified"

    bw = 8
    is_sym = True
    dtype = "INT"

    kv_io_map = {
        "past_key_0_in": "past_key_0_out",
        "past_key_1_in": "past_key_1_out",
        "past_value_0_in": "past_value_0_out",
        "past_value_1_in": "past_value_1_out",
    }

    host_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if host_device.type == "cuda" and host_device.index is not None:
        providers = [
            ("CUDAExecutionProvider", {"device_id": host_device.index}),
            "CPUExecutionProvider",
        ]
    elif host_device.type == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    quant_sim = QuantSimOnnx(
        model=onnx_model,
        quant_scheme=QuantScheme.post_training_tf,
        default_activation_bw=16,
        default_param_bw=4,
        config_file="htp_v73",
        providers=providers,
    )

    lm_head_tensor_name = None
    for weight in quant_sim.model.model.graph.initializer:
        if any(dim == vocab_size for dim in weight.dims):
            dimensions = list(weight.dims)
            if dimensions[-1] == vocab_size:
                lm_head_tensor_name = weight.name

    configured_quant_sim = _apply_int8_kv_cache_tying_and_lm_head(
        quant_sim, kv_io_map, lm_head_tensor_name
    )

    configured_quant_sim.compute_encodings(
        lambda session: session.run(
            None, make_dummy_input(configured_quant_sim.model.model)
        )
    )

    export_dir = checkpoint / f"configured_{model_name}"
    export_dir.mkdir(exist_ok=True)

    configured_quant_sim.export(str(export_dir), f"{model_name}_model")

    encodings_path = export_dir / f"{model_name}_model.encodings"
    check_config(
        configured_quant_sim,
        encodings_path,
        kv_io_map,
        lm_head_tensor_name,
        bw,
        is_sym,
        dtype,
    )


class TestLLMConfigurator:
    """Tests for applying quantsim configuration for LLMs"""

    def test_llm_configurator(self, tmp_path):
        apply_to_model("llama", tmp_path)

        apply_to_model("qwen", tmp_path)

        apply_to_model("phi", tmp_path)

    def test_set_matmul_second_input_to_8b(self):
        model = models_for_tests.model_with_split_matmul()
        sim = QuantizationSimModel(model)

        quantizer = _get_quantizer_no_split_slice(sim, "reshape_output")

        _set_matmul_second_input_to_8b(sim)

        quantizer = sim.qc_quantize_op_dict["reshape_output"]
        assert quantizer.enabled == True
        assert quantizer.bitwidth == 8

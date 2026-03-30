# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""AIMET ONNX LoRA: Multi-adapter quantization support for ONNX models.

Composable standalone functions for LoRA quantization workflows::

    from aimet_onnx.experimental.lora import (
        export_peft_to_onnx, set_lora_bitwidth,
        freeze_base_model, unfreeze_lora_quantizers,
        get_lora_encodings, set_lora_encodings, get_zero_weights,
    )
    from safetensors.numpy import load_file

    # 1. Export
    model, lora_names = export_peft_to_onnx(peft_model, sample_inputs, adapter_paths, output_dir)

    # 2. Create QuantSim
    sim = QuantizationSimModel(model, ...)
    set_lora_bitwidth(sim, lora_names, param_type=int16, activation_type=int8)

    # 3. Base calibration (zero weights = LoRA disabled)
    zero_weights = get_zero_weights(model, lora_names)
    sim.compute_encodings(lambda sess: calibrate(sess, zero_weights))

    # 4. Freeze base, per-adapter calibration
    freeze_base_model(sim, lora_names)
    adapter_encodings = {}
    for name in adapters:
        unfreeze_lora_quantizers(sim, lora_names)
        weights = load_file(f"{output_dir}/{name}.safetensors")
        sim.compute_encodings(lambda sess, w=weights: calibrate(sess, w))
        adapter_encodings[name] = get_lora_encodings(sim, lora_names)

    # 5. Inference with specific adapter
    set_lora_encodings(sim, adapter_encodings["code"])
    weights = load_file(f"{output_dir}/code.safetensors")
    output = sim.session.run(None, {**input_data, **weights})
"""

from aimet_onnx.experimental.lora.lora_adapter_quantization import (
    freeze_base_activation_quantizers,
    freeze_base_model,
    freeze_base_param_quantizers,
    get_lora_encodings,
    get_zero_weights,
    set_lora_bitwidth,
    set_lora_encodings,
    unfreeze_lora_quantizers,
    write_adaptor_list,
    write_lora_config,
    write_lora_weight_list,
)
from aimet_onnx.experimental.lora.peft_to_onnx import export_peft_to_onnx

__all__ = [
    "export_peft_to_onnx",
    "set_lora_bitwidth",
    "freeze_base_param_quantizers",
    "freeze_base_activation_quantizers",
    "freeze_base_model",
    "unfreeze_lora_quantizers",
    "get_lora_encodings",
    "set_lora_encodings",
    "get_zero_weights",
    "write_lora_weight_list",
    "write_lora_config",
    "write_adaptor_list",
]

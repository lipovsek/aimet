# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""AIMET ONNX LoRA: Multi-adapter quantization support for ONNX models.

Composable standalone functions for LoRA quantization workflows::

    from aimet_onnx.experimental.lora import (
        configure_lora_onnx, add_lora_branches, set_lora_bitwidth,
        freeze_base_model, unfreeze_lora_quantizers,
        get_lora_encodings, set_lora_encodings,
        get_zero_weights, get_adapter_scale_weights,
    )
    from safetensors.numpy import load_file

    # 1. User-owned export + configure
    #    Export with dynamo=True, optimize=False to preserve LoRA names.
    torch.onnx.export(
        peft_model.base_model.model, sample_inputs, "model.onnx",
        dynamo=True, optimize=False,
    )
    peft_keys = get_peft_model_state_dict(peft_model).keys()
    model, lora_names = configure_lora_onnx(
        "model.onnx", peft_keys, "prepared/model.onnx",
        adapter_paths=["adapters/code", "adapters/medical"],
    )

    # For legacy workflows, ``export_peft_to_onnx`` handles both export
    # and configuration in one call.

    # 2. Create QuantSim
    sim = QuantizationSimModel(model, ...)
    set_lora_bitwidth(sim, lora_names, param_type=int16, activation_type=int8)

    # 3. Base calibration (zero weights = LoRA disabled)
    zero_weights = get_zero_weights(model, lora_names)
    sim.compute_encodings(lambda sess: calibrate(sess, zero_weights))

    # 4. Freeze base, per-adapter calibration
    freeze_base_model(sim, lora_names)
    adapter_encodings = {}
    for name, path in adapters.items():
        unfreeze_lora_quantizers(sim, lora_names)
        weights = load_file(os.path.join(path, "adapter_model.safetensors"))
        feed = {**zero_weights, **weights}
        scales = get_adapter_scale_weights(lora_names, path)
        sim.compute_encodings(lambda sess, f=feed, s=scales: calibrate(sess, {**f, **s}))
        adapter_encodings[name] = get_lora_encodings(sim, lora_names)

    # 5. Inference with specific adapter
    set_lora_encodings(sim, adapter_encodings["code"])
    weights = load_file(os.path.join(adapters["code"], "adapter_model.safetensors"))
    feed = {**zero_weights, **weights}
    scales = get_adapter_scale_weights(lora_names, adapters["code"])
    output = sim.session.run(None, {**input_data, **feed, **scales})
"""

from aimet_onnx.experimental.lora.lora_adapter_quantization import (
    freeze_base_activation_quantizers,
    freeze_base_model,
    freeze_base_param_quantizers,
    get_adapter_scale_weights,
    get_lora_encodings,
    get_zero_weights,
    set_lora_bitwidth,
    set_lora_encodings,
    unfreeze_lora_quantizers,
    write_adapter_list,
    write_lora_config,
    write_lora_weight_list,
)
from aimet_onnx.experimental.lora.lora_configure import (
    add_lora_branches,
    configure_lora_onnx,
)
from aimet_onnx.experimental.lora.peft_to_onnx import export_peft_to_onnx

__all__ = [
    "configure_lora_onnx",
    "export_peft_to_onnx",
    "add_lora_branches",
    "set_lora_bitwidth",
    "freeze_base_param_quantizers",
    "freeze_base_activation_quantizers",
    "freeze_base_model",
    "unfreeze_lora_quantizers",
    "get_lora_encodings",
    "set_lora_encodings",
    "get_zero_weights",
    "get_adapter_scale_weights",
    "write_lora_weight_list",
    "write_lora_config",
    "write_adapter_list",
]

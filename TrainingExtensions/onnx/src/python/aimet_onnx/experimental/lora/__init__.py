# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""AIMET ONNX LoRA: Multi-adapter quantization support for ONNX models.

Composable standalone functions for LoRA quantization workflows.

**Recommended workflow** (``prepare_lora_onnx`` — works with ``dynamo=False``)::

    from aimet_onnx.experimental.lora import (
        prepare_lora_onnx, disable_lora_quantizers,
        enable_lora_calibration, set_lora_weights,
        freeze_base_model, unfreeze_lora_quantizers,
        get_lora_encodings, set_lora_encodings,
        get_zero_weights, get_adapter_scale_weights,
        get_adapter_names, get_adapter_lora_names,
        build_concurrent_feed_dict, adapt_base_encodings_for_lora,
    )
    from safetensors.numpy import load_file

    # 1. Configure PeftModel's dynamo=False ONNX export
    model, lora_names = prepare_lora_onnx(
        "peft_model.onnx",
        adapter_config="adapter_A/adapter_config.json",
        output_path="configured_model.onnx",
    )

    # 2. Create QuantSim with LoRA quantizers disabled (recipe-compatible)
    sim = QuantizationSimModel(model, ...)
    disable_lora_quantizers(sim, lora_names)

    # 3. Apply recipes — only base quantizers active
    # AdaScale.apply_adascale(sim, inputs, config)

    # 4. Base calibration (zero weights = LoRA disabled)
    sim.compute_encodings(lambda sess: calibrate(sess))
    freeze_base_model(sim, lora_names)

    # 5. Enable LoRA quantizers for calibration
    enable_lora_calibration(sim, lora_names, param_type="int16",
                            activation_type="int8")

    # 6. Per-adapter calibration
    adapter_encodings = {}
    for name, path in adapters.items():
        unfreeze_lora_quantizers(sim, lora_names)
        weights = load_file(os.path.join(path, "adapter_model.safetensors"))
        set_lora_weights(sim, lora_names, name, weights)
        sim.compute_encodings(lambda sess: calibrate(sess))
        adapter_encodings[name] = get_lora_encodings(sim, lora_names)

    # 7. Load specific adapter for inference
    load_adapter(sim, lora_names, "code", code_weights, code_encodings)

**Legacy workflow** (``configure_lora_onnx`` — requires ``dynamo=True``)::

    model, lora_names = configure_lora_onnx(
        "model.onnx", peft_keys, "prepared/model.onnx",
        adapter_paths=["adapters/code", "adapters/medical"],
    )
"""

from aimet_onnx.experimental.lora.lora_adapter_quantization import (
    adapt_base_encodings_for_lora,
    build_concurrent_feed_dict,
    disable_lora_quantizers,
    get_lora_node_names_for_exclusion,
    enable_lora_calibration,
    export_for_deployment,
    freeze_base_activation_quantizers,
    freeze_base_model,
    freeze_base_param_quantizers,
    get_adapter_lora_names,
    get_adapter_names,
    get_adapter_scale_weights,
    get_lora_encodings,
    get_zero_weights,
    load_adapter,
    set_lora_bitwidth,
    set_lora_encodings,
    set_lora_weights,
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
from aimet_onnx.experimental.lora.prepare_lora import prepare_lora_onnx

__all__ = [
    # Recommended API (dynamo=False compatible, single + multi-adapter)
    "prepare_lora_onnx",
    "disable_lora_quantizers",
    "enable_lora_calibration",
    "load_adapter",
    "set_lora_weights",
    # Multi-adapter helpers
    "get_adapter_names",
    "get_adapter_lora_names",
    "build_concurrent_feed_dict",
    "adapt_base_encodings_for_lora",
    # Legacy API (dynamo=True required)
    "configure_lora_onnx",
    "export_peft_to_onnx",
    "export_for_deployment",
    "add_lora_branches",
    # Quantizer helpers
    "set_lora_bitwidth",
    "freeze_base_param_quantizers",
    "freeze_base_activation_quantizers",
    "freeze_base_model",
    "unfreeze_lora_quantizers",
    "get_lora_encodings",
    "set_lora_encodings",
    # Weight / scale helpers
    "get_zero_weights",
    "get_adapter_scale_weights",
    # QAIRT artifact writers
    "write_lora_weight_list",
    "write_lora_config",
    "write_adapter_list",
]

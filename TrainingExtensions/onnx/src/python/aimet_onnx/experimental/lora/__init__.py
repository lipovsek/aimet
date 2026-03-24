# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""AIMET ONNX LoRA: Multi-adapter quantization support for ONNX models.

Workflow
--------

**Quick path** (recommended — 3 calls)::

    import aimet_onnx
    from aimet_onnx.experimental.lora import export_peft_to_onnx, calibrate_lora, export_lora

    model, result = export_peft_to_onnx(peft_model, sample_inputs, adapter_paths, output_dir)
    sim = QuantizationSimModel(model, ...)
    calibrate_lora(sim, result, dataloader, lora_param_type=aimet_onnx.int16)
    export_lora(sim, result, export_dir, target="qairt")  # or target="ort"

**Composable path** (for custom calibration strategies)::

    model, result = export_peft_to_onnx(peft_model, sample_inputs, adapter_paths, output_dir)
    sim = QuantizationSimModel(model, ...)
    configure_lora_quantizers(sim, result, lora_param_type=aimet_onnx.int16)

    # Base calibration
    sim.compute_encodings(callback)
    freeze_base_param_quantizers(sim, result)

    # Per-adapter calibration
    for adapter_name in adapters:
        unfreeze_lora_quantizers(sim, result)
        sim.compute_encodings(callback)
        encodings = get_lora_encodings(sim, result)
        set_lora_encodings(sim, result, encodings)  # restore later

    # Export
    export_lora(sim, result, export_dir, target="qairt")
"""

from aimet_onnx.experimental.lora.lora_adapter_quantization import (
    LoRAResult,
    configure_lora_quantizers,
    freeze_base_param_quantizers,
    freeze_base_activation_quantizers,
    freeze_base_model,
    unfreeze_lora_quantizers,
    export_lora_weights,
    get_lora_encodings,
    set_lora_encodings,
    calibrate_lora,
    export_lora,
)
from aimet_onnx.experimental.lora.peft_to_onnx import export_peft_to_onnx

__all__ = [
    "LoRAResult",
    "export_peft_to_onnx",
    "configure_lora_quantizers",
    "freeze_base_param_quantizers",
    "freeze_base_activation_quantizers",
    "freeze_base_model",
    "unfreeze_lora_quantizers",
    "export_lora_weights",
    "get_lora_encodings",
    "set_lora_encodings",
    "calibrate_lora",
    "export_lora",
]

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""FastExportable adaptations for VLM models."""

from transformers import PreTrainedModel

from GenAILab.shared.helpers.yaml_config_parser import YAMLConfigParser


@YAMLConfigParser.register_adaptation("FastExportable", model_type="qwen2_5_vl")
class Qwen2VLFastExportableAdaptation:
    """FastExportable adaptation for Qwen2 VL models.

    Uses attention masks instead of loop-based splitting for cleaner ONNX export.
    """

    @classmethod
    def instantiate_model(cls, *args, **kwargs) -> PreTrainedModel:
        from GenAILab.shared.models.qwen2_vl import (
            enable_fast_exportable_vision_attention,
        )

        with enable_fast_exportable_vision_attention():
            model = super().instantiate_model(*args, **kwargs)
        return model

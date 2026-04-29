# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""FastExportable adaptations for VLM models."""

import types

from transformers import PreTrainedModel

from GenAILab.bench.yaml_config_parser import YAMLConfigParser


@YAMLConfigParser.register_adaptation(
    "FastExportable", model_type="qwen2_5_vl", required_for_export=True
)
class Qwen2VLFastExportableAdaptation:
    """FastExportable adaptation for Qwen2 VL models.

    Uses attention masks instead of loop-based splitting for cleaner ONNX export.
    """

    @classmethod
    def instantiate_model(cls, *args, **kwargs) -> PreTrainedModel:
        from GenAILab.qai_hub_lm.models.qwen2_vl import (
            enable_fast_exportable_vision_attention,
        )

        with enable_fast_exportable_vision_attention():
            model = super().instantiate_model(*args, **kwargs)
        return model


@YAMLConfigParser.register_adaptation(
    "FastExportable", model_type="qwen3_vl", required_for_export=True
)
class Qwen3VLFastExportableAdaptation:
    """FastExportable adaptation for Qwen3 VL models.

    Uses attention masks instead of loop-based splitting for cleaner ONNX export.
    """

    @classmethod
    def instantiate_model(cls, *args, **kwargs) -> PreTrainedModel:
        from GenAILab.qai_hub_lm.models.qwen3_vl import (
            _exportable_deepstack_process,
            enable_fast_exportable_vision_attention,
        )

        with enable_fast_exportable_vision_attention():
            model = super().instantiate_model(*args, **kwargs)

        # Patch _deepstack_process on the model instance so it persists
        # beyond the context manager. The class-level patch reverts when
        # the CM exits, but instance methods take precedence.
        # Hierarchy: Qwen3VLForConditionalGeneration.model (Qwen3VLModel)
        #            .language_model (Qwen3VLTextModel — owns _deepstack_process)
        text_model = model.model.language_model
        text_model._deepstack_process = types.MethodType(
            _exportable_deepstack_process, text_model
        )
        return model

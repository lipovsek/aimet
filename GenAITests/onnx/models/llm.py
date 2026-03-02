# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Generic ONNX LLM class"""

import tempfile
import onnx
import torch
from transformers import AutoConfig

from aimet_onnx import quantsim
from aimet_onnx.quantsim import QuantizationSimModel

from GenAITests.shared.helpers.model_cache import DiskBackedModelCache, ModelCacheEntry
from GenAITests.shared.helpers.yaml_config_parser import YAMLConfigParser
from GenAITests.shared.models.base import LLM, SimCollection
from GenAITests.shared.models.utils.model_utils import ONNXExportableModuleWithCache

from GenAITests.onnx.models.utils.torch_onnx_export_utils import (
    get_onnx_model,
    load_model_components_from_disk,
    get_model_checkpoint_path,
    is_huggingface_ckpt,
)
from GenAITests.onnx.models.utils.quantsim_utils import (
    _set_tensors_to_output_n_bit_symmmetric,
    _tie_quantizers_for_kv_cache,
    _set_lm_head_to_8b,
    get_ort_providers,
    AttributePatch,
)


@YAMLConfigParser.register_default_llm
class LLM_ONNX(LLM):
    """Generic LLM for AIMET-ONNX quantization."""

    @classmethod
    def instantiate_quantsim(
        cls,
        model_id: str,
        context_length: int,
        sequence_length: int,
        small_model: bool = False,
        kv_bits: int = 8,
        model_cache: DiskBackedModelCache | None = None,
        *args,
        **kwargs,
    ) -> SimCollection:
        is_hf = is_huggingface_ckpt(model_id)

        # Use model cache for HuggingFace checkpoints when a cache is provided
        if is_hf:
            if model_cache is not None:
                with tempfile.TemporaryDirectory() as tmpdir:
                    params = {
                        "model_id": model_id,
                        "class": cls.__name__,
                        "sequence_length": sequence_length,
                        "context_length": context_length,
                        "small_model": small_model,
                    }
                    key = DiskBackedModelCache.build_key(params)
                    entry = model_cache.get_or_export(
                        key,
                        lambda: cls._export_to_cache_entry(
                            model_id,
                            context_length,
                            sequence_length,
                            small_model,
                            tmpdir,
                        ),
                        metadata=params,
                    )
            else:
                entry = cls._export_to_cache_entry(
                    model_id,
                    context_length,
                    sequence_length,
                    small_model,
                    get_model_checkpoint_path(model_id),
                )
            onnx_model = entry.backbone
            config = entry.config
        else:
            onnx_model, *_ = load_model_components_from_disk(
                model_id,
                context_length=context_length,
                sequence_length=sequence_length,
            )
            config = AutoConfig.from_pretrained(model_id)

        with (
            AttributePatch(quantsim, "op_types_to_tie_qtzrs", ["Concat"]),
            AttributePatch(quantsim, "_tie_qtzrs", True),
            AttributePatch(
                quantsim,
                "op_outputs_to_ignore",
                quantsim.op_outputs_to_ignore + ["Slice", "Constant"],
            ),
        ):
            quant_sim = QuantizationSimModel(
                model=onnx_model,
                quant_scheme="min_max",
                default_activation_bw=16,
                default_param_bw=4,
                config_file=cls.get_quantsim_config(),
                providers=get_ort_providers(
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                ),
            )

        # Setting kv_cache and some other layers to 8-bit
        _set_tensors_to_output_n_bit_symmmetric(quant_sim, kv_bits)
        # Setting the LM head weights to 8-bit
        _set_lm_head_to_8b(quant_sim)
        # Tie kv_cache
        _tie_quantizers_for_kv_cache(quant_sim)

        return SimCollection(quant_sim, config=config)

    @classmethod
    def _export_to_cache_entry(
        cls,
        model_id: str,
        context_length: int,
        sequence_length: int,
        small_model: bool,
        directory: str,
    ) -> ModelCacheEntry:
        """Export the torch model to ONNX in a temp dir and return a :class:`ModelCacheEntry`."""
        instantiated_model = cls.instantiate_model(model_id, small_model)
        if isinstance(instantiated_model, tuple):
            if context_length != 4096 or sequence_length != 2048:
                raise ValueError(
                    "Context length and sequence length must be 4096 and 2048 for AIHM adapted models."
                )
            assert isinstance(instantiated_model[0], onnx.ModelProto)
            return ModelCacheEntry(
                backbone=instantiated_model[0],
                config=instantiated_model[1],
            )

        assert isinstance(instantiated_model, torch.nn.Module)
        instantiated_model = instantiated_model.to(dtype=torch.float32)
        exportable_model = ONNXExportableModuleWithCache(instantiated_model)

        onnx_model, *_ = get_onnx_model(
            checkpoint=directory,
            fp_backbone_model=exportable_model,
            context_length=context_length,
            sequence_length=sequence_length,
            sample_input=cls.get_sample_backbone_inputs(
                exportable_model, context_length, sequence_length
            ),
            input_names=cls.get_backbone_input_names(
                instantiated_model.config.num_hidden_layers
            ),
            output_names=cls.get_backbone_output_names(
                instantiated_model.config.num_hidden_layers
            ),
        )

        return ModelCacheEntry(
            backbone=onnx_model,
            config=instantiated_model.config,
        )

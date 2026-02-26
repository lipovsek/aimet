# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""GenAI test runner"""

import warnings
import pytest
import torch
import gc
import os
from pathlib import Path
from transformers.processing_utils import ProcessorMixin

from aimet_onnx.quantsim import load_encodings_to_sim

from GenAITests.shared.helpers.profiler import (
    GPUMeter,
    MetricResult,
    ComponentRecipeStats,
    write_stats_to_disk,
)
from GenAITests.shared.helpers.determinism import set_seed
from GenAITests.shared.models.base import LLM, VLM
from GenAITests.shared.helpers import datasets, metrics
from GenAITests.onnx import models
from GenAITests.onnx.helpers import quant_recipes
from GenAITests.onnx.models.utils.generator_utils import generator_factory


def _extract_recipe_config(recipe_dict):
    """Extract recipe class, dataset config, and kwargs from a recipe config dict."""
    recipe_dict = recipe_dict.copy()
    recipe_cls = recipe_dict.pop("class")
    dataset_config = recipe_dict.pop("dataset", {}).copy()
    dataset_cls = dataset_config.pop("class", None)
    dataset_kwargs = dataset_config
    recipe_kwargs = recipe_dict
    return recipe_cls, recipe_kwargs, dataset_cls, dataset_kwargs


def test_llm_quantization(test_parameters):
    if test_parameters is None:
        pytest.skip("No GenAI test parameters provided.")
    set_seed(42)

    print(test_parameters)
    model_kwargs = test_parameters.pop("model")
    model_cls: type[LLM] = model_kwargs.pop("class")
    context_length = model_kwargs.pop("context_length")
    sequence_length = model_kwargs.pop("sequence_length")
    model_id = model_kwargs.pop("model_id")
    model_type = model_kwargs.pop("model_type")
    model_dtype = model_kwargs.pop("dtype", None)
    precomputed_encodings = model_kwargs.pop("encodings", None)

    if model_dtype is not None:
        warnings.warn(
            "User-specified dtypes are not yet supported in ONNX GenAITests. All models are FP32 by default."
        )

    if test_parameters["eval_in_onnx"]:
        warnings.warn("eval_in_onnx is ignored for ONNX GenAI tests.")

    all_recipes = test_parameters.pop("recipe")

    profiler_kwargs = test_parameters.pop("profiler")
    profiler_capture_intermediate_data = profiler_kwargs.pop(
        "capture_intermediate_data", False
    )

    metrics = test_parameters.pop("metrics")

    gc.collect()
    torch.cuda.empty_cache()

    sim_collection = model_cls.instantiate_quantsim(
        model_id, context_length, sequence_length, **model_kwargs
    )
    tokenizer = model_cls.instantiate_tokenizer(model_id)
    generator = generator_factory(
        sim_collection,
        model_cls.get_generator_cls(),
        tokenizer,
        sequence_length,
        context_length,
        visual_output_names=model_cls.get_visual_output_names()
        if issubclass(model_cls, VLM)
        else None,
        **model_kwargs,
    )

    if precomputed_encodings is not None:
        backbone_encodings = os.path.join(
            precomputed_encodings,
            "backbone",
            f"model_sl{sequence_length}_cl{context_length}.encodings",
        )
        if os.path.exists(backbone_encodings):
            print(f"Loading precomputed backbone encodings from {backbone_encodings}.")
            load_encodings_to_sim(
                sim_collection.backbone,
                backbone_encodings,
                strict=False,
                allow_overwrite=False,
                disable_missing_quantizers=False,
            )
        else:
            warnings.warn(
                f"Precomputed backbone encodings not found  at {backbone_encodings}. Proceeding without loading."
            )

        if sim_collection.visual is not None:
            visual_encodings = os.path.join(
                precomputed_encodings, "visual", "model.encodings"
            )
            if os.path.exists(visual_encodings):
                print(f"Loading precomputed visual encodings from {visual_encodings}.")
                load_encodings_to_sim(
                    sim_collection.visual,
                    visual_encodings,
                    strict=False,
                    allow_overwrite=False,
                    disable_missing_quantizers=False,
                )
            else:
                warnings.warn(
                    f"Precomputed visual encodings not found  at {visual_encodings}. Proceeding without loading."
                )

    # Apply backbone recipe with its own profiler
    (
        backbone_recipe_cls,
        backbone_recipe_kwargs,
        backbone_dataset_cls,
        backbone_dataset_kwargs,
    ) = _extract_recipe_config(all_recipes["backbone"])

    backbone_train_dataset = (
        backbone_dataset_cls.load_encoded_dataset(
            tokenizer.tokenizer if isinstance(tokenizer, ProcessorMixin) else tokenizer,
            context_length,
            **backbone_dataset_kwargs,
        )
        if backbone_dataset_cls
        else None
    )
    with GPUMeter(
        **profiler_kwargs, capture_intermediate_data=profiler_capture_intermediate_data
    ) as backbone_profiler:
        backbone_recipe_cls.apply(
            sim_collection.backbone,
            generator,
            backbone_train_dataset,
            **backbone_recipe_kwargs,
        )

    # Apply visual recipe if specified, with its own profiler
    visual_profiler = None
    if "visual" in all_recipes and sim_collection.visual is not None:
        (
            visual_recipe_cls,
            visual_recipe_kwargs,
            visual_dataset_cls,
            visual_dataset_kwargs,
        ) = _extract_recipe_config(all_recipes["visual"])
        visual_train_dataset = (
            visual_dataset_cls.load_encoded_dataset(
                tokenizer,
                context_length,
                **visual_dataset_kwargs,
            )
            if visual_dataset_cls
            else None
        )
        with GPUMeter(
            **profiler_kwargs,
            capture_intermediate_data=profiler_capture_intermediate_data,
        ) as visual_profiler:
            visual_recipe_cls.apply(
                sim_collection.visual,
                generator,
                visual_train_dataset,
                **visual_recipe_kwargs,
            )

    gc.collect()
    torch.cuda.empty_cache()

    if test_parameters["export"]:
        sim_collection.config.save_pretrained(test_parameters["export"])
        tokenizer.save_pretrained(test_parameters["export"])
        os.mkdir(os.path.join(test_parameters["export"], "backbone"))
        sim_collection.backbone.export(
            os.path.join(test_parameters["export"], "backbone"),
            f"model_sl{sequence_length}_cl{context_length}",
            export_model=True,
        )
        if sim_collection.visual is not None:
            os.mkdir(os.path.join(test_parameters["export"], "visual"))
            sim_collection.visual.export(
                os.path.join(test_parameters["export"], "visual"),
                "model",
                export_model=True,
            )
        if sim_collection.embedding is not None:
            torch.save(
                sim_collection.embedding.state_dict(),
                os.path.join(test_parameters["export"], "embedding.pth"),
            )

    evaluation_results = []
    with torch.no_grad():
        for metric_kwargs in metrics:
            metric_cls = metric_kwargs.pop("class")
            with GPUMeter(
                capture_intermediate_data=False, **profiler_kwargs
            ) as profiler:
                result = metric_cls.evaluate(
                    generator,
                    tokenizer.tokenizer
                    if isinstance(tokenizer, ProcessorMixin)
                    else tokenizer,
                    context_length,
                    **metric_kwargs,
                )
                print(f"{metric_cls.__name__} result: {result}")

            evaluation_results.append(
                MetricResult(
                    metric_name=metric_cls.__name__,
                    result=result,
                    profiler=profiler if profiler_capture_intermediate_data else None,
                )
            )

    model_kwargs["context_length"] = context_length
    model_kwargs["sequence_length"] = sequence_length
    if precomputed_encodings is not None:
        model_kwargs["encodings"] = precomputed_encodings

    output_folder = Path(os.getcwd()) / "genai_test_artifacts"
    output_folder.mkdir(parents=True, exist_ok=True)

    components = {
        "backbone": ComponentRecipeStats(
            recipe_name=backbone_recipe_cls.__name__,
            recipe_kwargs=backbone_recipe_kwargs,
            dataset_name=backbone_dataset_cls.__name__ if backbone_dataset_cls else "",
            dataset_kwargs=backbone_dataset_kwargs,
            profiler=backbone_profiler,
        )
    }
    if "visual" in all_recipes:
        components |= {
            "visual": ComponentRecipeStats(
                recipe_name=visual_recipe_cls.__name__,
                recipe_kwargs=visual_recipe_kwargs,
                dataset_name=visual_dataset_cls.__name__ if visual_dataset_cls else "",
                dataset_kwargs=visual_dataset_kwargs,
                profiler=visual_profiler,
            )
        }

    write_stats_to_disk(
        output_folder=str(output_folder),
        filename="profiling_data",
        model_type=model_type,
        model_id=model_id,
        model_modifiers=model_kwargs,
        components=components,
        accuracy_results=evaluation_results,
        export_location=test_parameters["export"]
        if test_parameters["export"]
        else None,
    )

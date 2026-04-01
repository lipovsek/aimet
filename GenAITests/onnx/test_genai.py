# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""GenAI test runner"""

import inspect
import warnings
import pytest
import torch
import gc
import os
from pathlib import Path
from transformers.processing_utils import ProcessorMixin

from aimet_onnx.quantsim import load_encodings_to_sim

from GenAITests.shared.helpers.yaml_config_parser import YAMLConfigParser
from GenAITests.shared.helpers.profiler import (
    GPUMeter,
    MetricResult,
    RecipeStepStats,
    ComponentRecipeStats,
    write_stats_to_disk,
)
from GenAITests.shared.helpers.determinism import set_seed
from GenAITests.shared.helpers.eval_context import EvaluationContext
from GenAITests.shared.helpers.fp_cache import DiskBackedFPCache
from GenAITests.shared.helpers.metrics import TextEvaluationMetric
from GenAITests.shared.helpers.model_cache import DiskBackedModelCache
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


def _apply_recipe_chain(
    recipe_list,
    sim_component,
    generator,
    tokenizer,
    context_length,
    image_size,
    profiler_kwargs,
    profiler_capture_intermediate_data,
):
    """Apply a chain of recipe steps sequentially, returning per-step stats."""
    step_stats = []
    for recipe_config in recipe_list:
        recipe_cls, recipe_kwargs, dataset_cls, dataset_kwargs = _extract_recipe_config(
            recipe_config
        )
        if image_size is not None and dataset_cls is not None:
            sig = inspect.signature(dataset_cls.load_encoded_dataset)
            if "image_size" in sig.parameters:
                dataset_kwargs.setdefault("image_size", image_size)
        train_dataset = (
            dataset_cls.load_encoded_dataset(
                tokenizer,
                context_length,
                **dataset_kwargs,
            )
            if dataset_cls
            else None
        )
        with GPUMeter(
            **profiler_kwargs,
            capture_intermediate_data=profiler_capture_intermediate_data,
        ) as profiler:
            recipe_cls.apply(sim_component, generator, train_dataset, **recipe_kwargs)
        step_stats.append(
            RecipeStepStats(
                recipe_name=recipe_cls.__name__,
                recipe_kwargs=recipe_kwargs,
                dataset_name=dataset_cls.__name__ if dataset_cls else "",
                dataset_kwargs=dataset_kwargs,
                profiler=profiler,
            )
        )
        gc.collect()
        torch.cuda.empty_cache()
    return step_stats


def test_llm_quantization(
    test_config,
    model_cache: DiskBackedModelCache,
    fp_cache: DiskBackedFPCache,
    export_dir,
    results_dir,
):
    if test_config is None:
        pytest.skip("No GenAI test parameters provided.")
    set_seed(42)

    test_parameters = YAMLConfigParser.parse_document(
        test_config, export_base_dir=export_dir
    )
    print(test_parameters)

    model_kwargs = test_parameters.pop("model")

    # Snapshot model args before destructive pops for the EvaluationContext hash
    eval_ctx = EvaluationContext(fp_cache=fp_cache, model_args=model_kwargs.copy())

    model_cls: type[LLM] = model_kwargs.pop("class")
    context_length = model_kwargs.pop("context_length")
    sequence_length = model_kwargs.pop("sequence_length")
    model_id = model_kwargs.pop("model_id")
    model_type = model_kwargs.pop("model_type")
    model_dtype = model_kwargs.pop("dtype", None)
    image_size = model_kwargs.pop("image_size", None)
    precomputed_encodings = model_kwargs.pop("encodings", None)

    if model_dtype is not None:
        warnings.warn(
            "User-specified dtypes are not yet supported in ONNX GenAITests. All models are FP32 by default."
        )

    if test_parameters["eval_in_onnx"]:
        warnings.warn("eval_in_onnx is ignored for ONNX GenAI tests.")

    precision = test_parameters.pop("precision")
    run_group = test_parameters.pop("run_group", None)
    all_recipes = test_parameters.pop("recipe")

    profiler_kwargs = test_parameters.pop("profiler")
    profiler_capture_intermediate_data = profiler_kwargs.pop(
        "capture_intermediate_data", False
    )

    metrics = test_parameters.pop("metrics")

    gc.collect()
    torch.cuda.empty_cache()

    sim_collection = model_cls.instantiate_quantsim(
        model_id,
        context_length,
        sequence_length,
        precision=precision,
        model_cache=model_cache,
        image_size=image_size,
        **model_kwargs,
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

    # Apply recipe chains
    backbone_steps = _apply_recipe_chain(
        all_recipes["backbone"],
        sim_collection.backbone,
        generator,
        tokenizer.tokenizer if isinstance(tokenizer, ProcessorMixin) else tokenizer,
        context_length,
        image_size,
        profiler_kwargs,
        profiler_capture_intermediate_data,
    )

    visual_steps = []
    if "visual" in all_recipes and sim_collection.visual is not None:
        visual_steps = _apply_recipe_chain(
            all_recipes["visual"],
            sim_collection.visual,
            generator,
            tokenizer,
            context_length,
            image_size,
            profiler_kwargs,
            profiler_capture_intermediate_data,
        )

    gc.collect()
    torch.cuda.empty_cache()

    export_dir = test_parameters["export"] if test_parameters["export"] else None
    if export_dir:
        sim_collection.config.save_pretrained(export_dir)
        tokenizer.save_pretrained(export_dir)

        os.mkdir(os.path.join(export_dir, "backbone"))
        sim_collection.backbone.export(
            os.path.join(export_dir, "backbone"),
            f"model_sl{sequence_length}_cl{context_length}",
            export_model=True,
        )

        if sim_collection.visual is not None:
            os.mkdir(os.path.join(export_dir, "visual"))
            sim_collection.visual.export(
                os.path.join(export_dir, "visual"),
                "model",
                export_model=True,
            )

        if sim_collection.embedding is not None:
            torch.save(
                sim_collection.embedding.state_dict(),
                os.path.join(export_dir, "embedding.pth"),
            )

    evaluation_results = []
    with torch.no_grad():
        for metric_kwargs in metrics:
            metric_cls = metric_kwargs.pop("class")
            with GPUMeter(
                capture_intermediate_data=False, **profiler_kwargs
            ) as profiler:
                extra_metric_kwargs = {}
                if not issubclass(metric_cls, TextEvaluationMetric):
                    extra_metric_kwargs["image_size"] = image_size
                tokenizer_arg = (
                    tokenizer.tokenizer
                    if isinstance(tokenizer, ProcessorMixin)
                    and issubclass(metric_cls, TextEvaluationMetric)
                    else tokenizer
                )
                result = metric_cls.evaluate(
                    generator,
                    tokenizer_arg,
                    context_length,
                    eval_ctx=eval_ctx,
                    **extra_metric_kwargs,
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
    if image_size is not None:
        model_kwargs["image_size"] = list(image_size)
    if precomputed_encodings is not None:
        model_kwargs["encodings"] = precomputed_encodings

    components = {
        "backbone": ComponentRecipeStats(steps=backbone_steps),
    }
    if visual_steps:
        components["visual"] = ComponentRecipeStats(steps=visual_steps)

    results_folder = Path(results_dir)
    results_folder.mkdir(parents=True, exist_ok=True)
    precision_dict = precision.to_dict()
    write_stats_to_disk(
        output_folder=str(results_folder),
        filename="profiling_data",
        model_type=model_type,
        model_id=model_id,
        model_modifiers=model_kwargs,
        components=components,
        accuracy_results=evaluation_results,
        export_location=export_dir,
        precision=precision_dict,
        run_group=run_group,
    )

    if export_dir:
        write_stats_to_disk(
            output_folder=export_dir,
            filename="profiling_data",
            model_type=model_type,
            model_id=model_id,
            model_modifiers=model_kwargs,
            components=components,
            accuracy_results=evaluation_results,
            precision=precision_dict,
            run_group=run_group,
        )

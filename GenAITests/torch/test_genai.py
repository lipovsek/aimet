# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""GenAI test runner"""

import pytest
import torch
import gc
import os
import yaml
from pathlib import Path
from transformers.processing_utils import ProcessorMixin

from aimet_torch.v2.nn import QuantizationMixin

from GenAITests.shared.models.base import LLM, VLM
from GenAITests.shared.helpers.yaml_config_parser import YAMLConfigParser
from GenAITests.shared.helpers.profiler import (
    GPUMeter,
    MetricResult,
    ComponentRecipeStats,
    write_stats_to_disk,
)
from GenAITests.shared.helpers.determinism import set_seed
from GenAITests.shared.helpers.eval_context import EvaluationContext
from GenAITests.shared.helpers.fp_cache import DiskBackedFPCache
from GenAITests.shared.helpers.metrics import TextEvaluationMetric
from GenAITests.shared.helpers import datasets, metrics
from GenAITests.torch import models
from GenAITests.torch.helpers import quant_recipes
from GenAITests.torch.models.utils.generator_utils import (
    place_collection,
    generator_factory,
)


def _extract_recipe_config(recipe_dict):
    """Extract recipe class, dataset config, and kwargs from a recipe config dict."""
    recipe_dict = recipe_dict.copy()
    recipe_cls = recipe_dict.pop("class")
    dataset_config = recipe_dict.pop("dataset", {}).copy()
    dataset_cls = dataset_config.pop("class", None)
    dataset_kwargs = dataset_config
    recipe_kwargs = recipe_dict
    return recipe_cls, recipe_kwargs, dataset_cls, dataset_kwargs


def test_llm_quantization(
    test_config, fp_cache: DiskBackedFPCache, export_dir, results_dir
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
    precomputed_encodings = model_kwargs.pop("encodings", None)

    if "dtype" in model_kwargs:
        model_kwargs["dtype"] = getattr(torch, model_kwargs["dtype"])

    precision = test_parameters.pop("precision")

    all_recipes = test_parameters.pop("recipe")
    profiler_kwargs = test_parameters.pop("profiler")
    profiler_capture_intermediate_data = profiler_kwargs.pop(
        "capture_intermediate_data", False
    )
    metrics = test_parameters.pop("metrics")

    gc.collect()
    torch.cuda.empty_cache()

    sim_collection = model_cls.instantiate_quantsim(
        model_id, context_length, sequence_length, precision=precision, **model_kwargs
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
        print(f"Loading precomputed encodings from {precomputed_encodings}.")
        sim_collection.backbone.load_encodings(
            precomputed_encodings,
            partial=True,
            strict=False,
            allow_overwrite=False,
        )
        if sim_collection.visual is not None:
            sim_collection.visual.load_encodings(
                precomputed_encodings,
                partial=True,
                strict=False,
                allow_overwrite=False,
            )
        if sim_collection.embedding is not None:
            pass
            # todo: need to update this to intelligently load encodings for embedding table if it exists

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with place_collection(sim_collection, device):
        # Apply backbone recipe with its own profiler
        (
            backbone_recipe_cls,
            backbone_recipe_kwargs,
            backbone_dataset_cls,
            backbone_dataset_kwargs,
        ) = _extract_recipe_config(all_recipes["backbone"])
        backbone_train_dataset = (
            backbone_dataset_cls.load_encoded_dataset(
                tokenizer.tokenizer
                if isinstance(tokenizer, ProcessorMixin)
                else tokenizer,
                context_length,
                **backbone_dataset_kwargs,
            )
            if backbone_dataset_cls
            else None
        )
        with GPUMeter(
            **profiler_kwargs,
            capture_intermediate_data=profiler_capture_intermediate_data,
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

    export_dir = test_parameters["export"] if test_parameters["export"] else None
    if export_dir:
        tokenizer.save_pretrained(export_dir)
        sim_collection.backbone.model.config.save_pretrained(export_dir)

        backbone_onnx_name = f"model_sl{sequence_length}_cl{context_length}.onnx"
        os.mkdir(os.path.join(export_dir, "backbone"))
        sim_collection.backbone.onnx.export(
            f=os.path.join(export_dir, "backbone", backbone_onnx_name),
            args=model_cls.get_sample_backbone_inputs(
                model=sim_collection.backbone.model,
                context_length=context_length,
                sequence_length=sequence_length,
            ),
            input_names=model_cls.get_backbone_input_names(
                sim_collection.backbone.model.model.config.num_hidden_layers
            ),
            output_names=model_cls.get_backbone_output_names(
                sim_collection.backbone.model.model.config.num_hidden_layers
            ),
            opset_version=17,
            dynamo=False,
            export_int32_bias=False,
        )

        if sim_collection.visual is not None:
            assert isinstance(model_cls, VLM)
            os.mkdir(os.path.join(export_dir, "visual"))
            sim_collection.visual.model.config.save_pretrained(export_dir)
            sim_collection.visual.onnx.export(
                f=os.path.join(export_dir, "visual", "model.onnx"),
                args=model_cls.get_sample_vision_inputs(sim_collection.config),
                input_names=model_cls.get_visual_input_names(),
                output_names=model_cls.get_visual_output_names(),
                opset_version=17,
                dynamo=False,
                export_int32_bias=False,
            )

        if sim_collection.embedding is not None:
            if isinstance(sim_collection.embedding, QuantizationMixin):
                sim_collection.embedding.fold_param_quantizers()

            torch.save(
                sim_collection.embedding.state_dict(),
                os.path.join(export_dir, "embedding.pth"),
            )

        if test_parameters["eval_in_onnx"]:
            data = {
                "model": {
                    "model_id": export_dir,
                    "encodings": export_dir,
                    "sequence_length": sequence_length,
                    "context_length": context_length,
                    **model_kwargs,
                },
                "precision": precision,
                "recipe": {
                    "backbone": {
                        "name": quant_recipes.Calibration.__name__,
                        "dataset": {
                            "name": backbone_dataset_cls.__name__,
                            **backbone_dataset_kwargs,
                        },
                    },
                }
                | {
                    "visual": {
                        "name": quant_recipes.Calibration.__name__,
                        "dataset": {
                            "name": visual_dataset_cls.__name__,
                            **visual_dataset_kwargs,
                        },
                    }
                }
                if "visual" in all_recipes and sim_collection.visual is not None
                else {},
                "metrics": [
                    {
                        "name": metric["class"].__name__,
                        **{k: v for k, v in metric.items() if k != "class"},
                    }
                    for metric in metrics
                ],
            }

            with open(os.path.join(export_dir, "onnx_eval_config.yaml"), "w") as file:
                yaml.dump(data, file, default_flow_style=False)

    with place_collection(sim_collection, device):
        evaluation_results = []
        with torch.no_grad():
            for metric_kwargs in metrics:
                metric_cls = metric_kwargs.pop("class")
                tokenizer_arg = (
                    tokenizer.tokenizer
                    if isinstance(tokenizer, ProcessorMixin)
                    and issubclass(metric_cls, TextEvaluationMetric)
                    else tokenizer
                )
                with GPUMeter(
                    capture_intermediate_data=False, **profiler_kwargs
                ) as profiler:
                    result = metric_cls.evaluate(
                        generator,
                        tokenizer_arg,
                        context_length,
                        eval_ctx=eval_ctx,
                        **metric_kwargs,
                    )
                    print(f"{metric_cls.__name__} result: {result}")

                evaluation_results.append(
                    MetricResult(
                        metric_name=metric_cls.__name__,
                        result=result,
                        profiler=profiler
                        if profiler_capture_intermediate_data
                        else None,
                    )
                )

    model_kwargs["context_length"] = context_length
    model_kwargs["sequence_length"] = sequence_length
    if precomputed_encodings is not None:
        model_kwargs["encodings"] = precomputed_encodings

    components = {
        "backbone": ComponentRecipeStats(
            recipe_name=backbone_recipe_cls.__name__,
            recipe_kwargs=backbone_recipe_kwargs,
            dataset_name=backbone_dataset_cls.__name__ if backbone_dataset_cls else "",
            dataset_kwargs=backbone_dataset_kwargs,
            profiler=backbone_profiler,
        )
    }
    if "visual" in all_recipes and sim_collection.visual is not None:
        components |= {
            "visual": ComponentRecipeStats(
                recipe_name=visual_recipe_cls.__name__,
                recipe_kwargs=visual_recipe_kwargs,
                dataset_name=visual_dataset_cls.__name__ if visual_dataset_cls else "",
                dataset_kwargs=visual_dataset_kwargs,
                profiler=visual_profiler,
            )
        }

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
        )

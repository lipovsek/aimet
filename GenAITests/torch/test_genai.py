# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""GenAI test runner"""

import pytest
import torch
import gc
import os
import yaml
from pathlib import Path

from transformers import PreTrainedTokenizer, ProcessorMixin

from GenAITests.shared.models.base import LLM, VLM
from GenAITests.shared.helpers.profiler import (
    GPUMeter,
    MetricResult,
    ComponentRecipeStats,
    write_stats_to_disk,
)
from GenAITests.shared.helpers.determinism import set_seed
from GenAITests.shared.helpers.metrics import TextEvaluationMetric

from GenAITests.shared.helpers import datasets, metrics
from GenAITests.torch import models
from GenAITests.torch.helpers import quant_recipes

from GenAITests.torch.models.utils import place_collection, generator_factory
from aimet_torch.v2.nn import QuantizationMixin


def _extract_recipe_config(recipe_dict):
    """Extract recipe class, dataset config, and kwargs from a recipe config dict."""
    recipe_dict = recipe_dict.copy()
    recipe_cls = recipe_dict.pop("class")
    dataset_config = recipe_dict.pop("dataset", {}).copy()
    dataset_cls = dataset_config.pop("class", None)
    dataset_kwargs = dataset_config
    recipe_kwargs = recipe_dict
    return recipe_cls, recipe_kwargs, dataset_cls, dataset_kwargs


def _build_recipe_output_config(recipe_cls, dataset_cls, dataset_kwargs):
    """Build recipe config dict for ONNX eval config output."""
    recipe_name = recipe_cls.__name__
    if "RemoveQuantization" in recipe_name:
        output_name = "RemoveQuantization"
    elif "LPBQ" in recipe_name:
        output_name = "LPBQ"
    else:
        output_name = "PCQ"
    return {
        "name": output_name,
        "dataset": {
            "name": dataset_cls.__name__,
            **dataset_kwargs,
        },
    }


def test_llm_quantization(test_parameters):
    if test_parameters is None:
        pytest.skip("No GenAI test parameters provided.")
    set_seed(42)

    print(test_parameters)
    model_kwargs = test_parameters.pop("model")
    model_cls: type[LLM] = model_kwargs.pop("class")
    context_length = model_kwargs.pop("context_length")
    sequence_length = model_kwargs.pop("sequence_length")
    model_id = model_kwargs.pop("model_id", None)
    precomputed_encodings = model_kwargs.pop("encodings", None)

    if "dtype" in model_kwargs:
        model_kwargs["dtype"] = getattr(torch, model_kwargs["dtype"])

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
        with GPUMeter(
            **profiler_kwargs,
            capture_intermediate_data=profiler_capture_intermediate_data,
        ) as quantization_profiler:
            # Apply backbone recipe
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
            backbone_recipe_cls.apply(
                sim_collection.backbone,
                generator,
                backbone_train_dataset,
                **backbone_recipe_kwargs,
            )

            # Apply visual recipe if specified
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
                visual_recipe_cls.apply(
                    sim_collection.visual,
                    generator,
                    visual_train_dataset,
                    **visual_recipe_kwargs,
                )

        gc.collect()
        torch.cuda.empty_cache()

    if test_parameters["export"]:
        tokenizer.save_pretrained(test_parameters["export"])

        sim_collection.backbone.model.config.save_pretrained(test_parameters["export"])
        sim_collection.backbone.onnx.export(
            f=os.path.join(
                test_parameters["export"],
                "backbone",
                f"model_sl{sequence_length}_cl{context_length}.onnx",
            ),
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
            sim_collection.visual.model.config.save_pretrained(
                test_parameters["export"]
            )
            sim_collection.visual.onnx.export(
                f=os.path.join(test_parameters["export"], "visual", "model.onnx"),
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
                os.path.join(test_parameters["export"], "embedding.pth"),
            )

        if test_parameters["eval_in_onnx"]:
            data = {
                "model": {
                    "name": model_cls.__name__.removesuffix("_Torch"),
                    "model_id": test_parameters["export"],
                    "encodings": test_parameters["export"],
                    "sequence_length": sequence_length,
                    "context_length": context_length,
                    **model_kwargs,
                },
                "recipe": {
                    "backbone": _build_recipe_output_config(
                        backbone_recipe_cls,
                        backbone_dataset_cls,
                        backbone_dataset_kwargs,
                    ),
                }
                | {
                    "visual": _build_recipe_output_config(
                        visual_recipe_cls, visual_dataset_cls, visual_dataset_kwargs
                    )
                }
                if "visual" in all_recipes
                else {},
                "metrics": [
                    {
                        "name": metric["class"].__name__,
                        **{k: v for k, v in metric.items() if k != "class"},
                    }
                    for metric in metrics
                ],
            }

            with open(
                os.path.join(test_parameters["export"], "onnx_eval_config.yaml"), "w"
            ) as file:
                yaml.dump(data, file, default_flow_style=False)

    with place_collection(sim_collection, device):
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
                        and issubclass(metric_cls, TextEvaluationMetric)
                        else tokenizer,
                        context_length,
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

    output_folder = Path(os.getcwd()) / "genai_test_artifacts"
    output_folder.mkdir(parents=True, exist_ok=True)

    components = (
        {
            "backbone": ComponentRecipeStats(
                recipe_name=backbone_recipe_cls.__name__,
                recipe_kwargs=backbone_recipe_kwargs,
                dataset_name=backbone_dataset_cls.__name__
                if backbone_dataset_cls
                else "",
                dataset_kwargs=backbone_dataset_kwargs,
            )
        }
        | {
            "visual": ComponentRecipeStats(
                recipe_name=visual_recipe_cls.__name__,
                recipe_kwargs=visual_recipe_kwargs,
                dataset_name=visual_dataset_cls.__name__ if visual_dataset_cls else "",
                dataset_kwargs=visual_dataset_kwargs,
            )
        }
        if "visual" in all_recipes
        else {}
    )

    write_stats_to_disk(
        output_folder=output_folder,
        filename="profiling_data",
        model_family=model_cls.__name__,
        model_id=model_id if model_id is not None else model_cls.DEFAULT_MODEL_ID,
        model_modifiers=model_kwargs,
        components=components,
        quantization_results=quantization_profiler,
        accuracy_results=evaluation_results,
        export_location=test_parameters["export"]
        if test_parameters["export"]
        else None,
    )

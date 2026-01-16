# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""GenAI test runner"""

import pytest
import torch
import gc
import os
import yaml
from pathlib import Path

from aimet_torch.utils import place_model

from GenAITests.shared.models.generator import Generator
from GenAITests.shared.helpers.profiler import (
    GPUMeter,
    MetricResult,
    write_stats_to_disk,
)

from GenAITests.shared.helpers import datasets, metrics
from GenAITests.torch import models
from GenAITests.torch.helpers import quant_recipes


def test_llm_quantization(test_parameters):
    if test_parameters is None:
        pytest.skip("No GenAI test parameters provided.")

    print(test_parameters)
    model_kwargs = test_parameters.pop("model")
    model_cls = model_kwargs.pop("class")
    context_length = model_kwargs.pop("context_length")
    sequence_length = model_kwargs.pop("sequence_length")
    model_id = model_kwargs.pop("model_id", None)
    precomputed_encodings = model_kwargs.pop("encodings", None)

    if "dtype" in model_kwargs:
        model_kwargs["dtype"] = getattr(torch, model_kwargs["dtype"])

    dataset_kwargs = test_parameters.pop("dataset")
    dataset_cls = dataset_kwargs.pop("class")

    recipe_kwargs = test_parameters.pop("recipe")
    recipe_cls = recipe_kwargs.pop("class")

    profiler_kwargs = test_parameters.pop("profiler")
    profiler_capture_intermediate_data = profiler_kwargs.pop(
        "capture_intermediate_data", False
    )

    metrics = test_parameters.pop("metrics")

    gc.collect()
    torch.cuda.empty_cache()

    quantsim = model_cls.instantiate_quantsim(
        model_id, context_length, sequence_length, **model_kwargs
    )
    tokenizer = model_cls.instantiate_tokenizer(model_id)
    generator = Generator(
        quantsim.model, tokenizer, sequence_length, context_length, **model_kwargs
    )

    if precomputed_encodings is not None:
        print(f"Loading precomputed encodings from {precomputed_encodings}.")
        quantsim.load_encodings(
            precomputed_encodings,
            partial=True,
            strict=False,
            allow_overwrite=False,
        )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with place_model(quantsim.model, device):
        with GPUMeter(
            **profiler_kwargs,
            capture_intermediate_data=profiler_capture_intermediate_data,
        ) as quantization_profiler:
            train_dataset = dataset_cls.load_encoded_dataset(
                tokenizer, context_length, **dataset_kwargs
            )
            recipe_cls.apply(quantsim, generator, train_dataset, **recipe_kwargs)

        gc.collect()
        torch.cuda.empty_cache()

    if test_parameters["export"]:
        quantsim.model.config.save_pretrained(test_parameters["export"])
        tokenizer.save_pretrained(test_parameters["export"])

        dummy_input_ids = torch.zeros((1, sequence_length), dtype=torch.int)
        dummy_attention_mask = torch.ones((1, sequence_length), dtype=torch.int)

        assembled_dummy_inputs = Generator.prepare_inputs(
            model=quantsim.model.model,
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            past_key_values=[],
            sequence_length=sequence_length,
            context_length=context_length,
        )

        quantsim.onnx.export(
            f=os.path.join(test_parameters["export"], f"model_cl{context_length}.onnx"),
            args=assembled_dummy_inputs,
            input_names=Generator.get_input_names(
                quantsim.model.model.config.num_hidden_layers
            ),
            output_names=Generator.get_output_names(
                quantsim.model.model.config.num_hidden_layers
            ),
            opset_version=17,
            dynamo=False,
            export_int32_bias=False,
        )

        if test_parameters["eval_in_onnx"]:
            data = {
                "model": {
                    "name": model_cls.__name__.removesuffix("_Torch"),
                    "model_id": test_parameters["export"],
                    "encodings": test_parameters["export"]
                    + f"/model_cl{context_length}.encodings",
                    "sequence_length": sequence_length,
                    "context_length": context_length,
                    **model_kwargs,
                },
                "dataset": {
                    "name": dataset_cls.__name__,
                    **dataset_kwargs,
                },
                "recipe": {"name": "LPBQ" if "LPBQ" in recipe_cls.__name__ else "PCQ"},
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

    with place_model(quantsim.model, device):
        evaluation_results = []
        with torch.no_grad():
            for metric_kwargs in metrics:
                metric_cls = metric_kwargs.pop("class")
                with GPUMeter(
                    capture_intermediate_data=False, **profiler_kwargs
                ) as profiler:
                    result = metric_cls.evaluate(
                        generator, tokenizer, context_length, **metric_kwargs
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

    write_stats_to_disk(
        output_folder=output_folder,
        filename="profiling_data",
        model_family=model_cls.__name__,
        model_id=model_id if model_id is not None else model_cls.DEFAULT_MODEL_ID,
        model_modifiers=model_kwargs,
        recipe_name=recipe_cls.__name__,
        recipe_modifiers=recipe_kwargs,
        dataset_name=dataset_cls.__name__,
        dataset_modifiers=dataset_kwargs,
        quantization_results=quantization_profiler,
        accuracy_results=evaluation_results,
        export_location=test_parameters["export"]
        if test_parameters["export"]
        else None,
    )

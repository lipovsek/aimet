# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""GenAI test runner"""

import warnings
import pytest
import torch
import gc
import os
from pathlib import Path
from transformers import AutoConfig

from aimet_onnx.quantsim import load_encodings_to_sim

from GenAITests.shared.helpers.profiler import (
    GPUMeter,
    MetricResult,
    write_stats_to_disk,
)
from GenAITests.shared.models.generator import Generator
from GenAITests.shared.helpers.determinism_utils import set_seed

from GenAITests.shared.helpers import datasets, metrics
from GenAITests.onnx import models
from GenAITests.onnx.helpers import quant_recipes

from GenAITests.onnx.models.utils.torch_onnx_interface import TorchONNXInterface
from GenAITests.onnx.models.utils.torch_onnx_export_utils import (
    get_model_checkpoint_path,
)


def test_llm_quantization(test_parameters):
    if test_parameters is None:
        pytest.skip("No GenAI test parameters provided.")
    set_seed(42)

    print(test_parameters)
    model_kwargs = test_parameters.pop("model")
    model_cls = model_kwargs.pop("class")
    context_length = model_kwargs.pop("context_length")
    sequence_length = model_kwargs.pop("sequence_length")
    model_id = model_kwargs.pop("model_id", None)
    model_dtype = model_kwargs.pop("dtype", None)
    precomputed_encodings = model_kwargs.pop("encodings", None)

    if model_dtype is not None:
        warnings.warn(
            "User-specified dtypes are not yet supported in ONNX GenAITests. All models are FP32 by default."
        )

    if test_parameters["eval_in_onnx"]:
        warnings.warn("eval_in_onnx is ignored for ONNX GenAI tests.")

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
    config = AutoConfig.from_pretrained(
        get_model_checkpoint_path(
            model_id if model_id is not None else model_cls.DEFAULT_MODEL_ID
        )
    )

    if precomputed_encodings is not None:
        print(f"Loading precomputed encodings from {precomputed_encodings}.")
        load_encodings_to_sim(
            quantsim,
            precomputed_encodings,
            strict=False,
            allow_overwrite=False,
            disable_missing_quantizers=False,
        )

    quantsim_with_torch_interface = TorchONNXInterface(quantsim, config)
    generator = Generator(
        quantsim_with_torch_interface,
        tokenizer,
        sequence_length,
        context_length,
        **model_kwargs,
    )

    with GPUMeter(
        **profiler_kwargs, capture_intermediate_data=profiler_capture_intermediate_data
    ) as quantization_profiler:
        train_dataset = dataset_cls.load_encoded_dataset(
            tokenizer, context_length, **dataset_kwargs
        )
        recipe_cls.apply(quantsim, generator, train_dataset, **recipe_kwargs)

    gc.collect()
    torch.cuda.empty_cache()

    if test_parameters["export"]:
        config.save_pretrained(test_parameters["export"])
        tokenizer.save_pretrained(test_parameters["export"])
        quantsim.export(test_parameters["export"], "model", export_model=True)

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
                    profiler=profiler if profiler_capture_intermediate_data else None,
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

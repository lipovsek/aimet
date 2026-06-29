# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared recipe chain application logic for Torch and ONNX test runners."""

import contextlib
import gc

import torch

from transformers.processing_utils import ProcessorMixin

from GenAILab.bench.datasets import GeneratedDataset, Interleaved, TextDataset
from GenAILab.bench.profiler import GPUMeter, RecipeStepStats


def extract_recipe_config(recipe_dict):
    """Extract recipe class, dataset config, and kwargs from a recipe config dict."""
    recipe_dict = recipe_dict.copy()
    recipe_cls = recipe_dict.pop("class")
    dataset_config = recipe_dict.pop("dataset", {}).copy()
    dataset_cls = dataset_config.pop("class", None)
    dataset_kwargs = dataset_config
    recipe_kwargs = recipe_dict
    return recipe_cls, recipe_kwargs, dataset_cls, dataset_kwargs


def apply_recipe_chain(
    recipe_list,
    sim_component,
    generator,
    tokenizer,
    context_length,
    image_size,
    profiler_kwargs,
    profiler_capture_intermediate_data,
    framework,
    model_id,
    precision,
    model_kwargs,
    component="backbone",
    recipe_cache=None,
    spinquant_config=None,
):
    """Apply a chain of recipe steps, with automatic cache lookup and save.

    If recipe_cache is provided, performs a cache lookup first and skips
    already-cached steps. After each cacheable step, saves to the cache.
    Returns the full list of RecipeStepStats (cached + newly computed).

    ``spinquant_config`` carries the pre-sim SpinQuant rotation flags (aimet-onnx
    only) so the cache base hash distinguishes rotated from non-rotated graphs;
    SpinQuant is not a chain step.
    """
    skip_to = 0
    cached_step_stats = []
    chain_hashes = None

    if recipe_cache:
        skip_to, cached_step_stats, chain_hashes = recipe_cache.lookup(
            recipe_list,
            sim_component,
            model_id,
            precision,
            model_kwargs,
            framework,
            component,
            spinquant_config,
        )

    step_stats = list(cached_step_stats)

    for i, recipe_config in enumerate(recipe_list[skip_to:], start=skip_to):
        recipe_cls, recipe_kwargs, dataset_cls, dataset_kwargs = extract_recipe_config(
            recipe_config
        )

        if dataset_cls is not None:
            # Text datasets need just the tokenizer; multimodal datasets
            # need the full processor. Interleaved is text-or-multimodal
            # depending on its sub-datasets, so it only requires a processor
            # when the model actually provides one.
            if not issubclass(dataset_cls, TextDataset):
                if image_size is not None:
                    dataset_kwargs.setdefault("image_size", image_size)
                if not issubclass(dataset_cls, Interleaved):
                    assert isinstance(tokenizer, ProcessorMixin), (
                        f"Multimodal dataset {dataset_cls.__name__} requires a "
                        f"ProcessorMixin, got {type(tokenizer).__name__}"
                    )
            # Interleaved handles per-sub tokenizer/processor selection
            # internally, so hand it the full tokenizer/processor object.
            dataset_tokenizer = (
                getattr(tokenizer, "tokenizer", tokenizer)
                if issubclass(dataset_cls, TextDataset)
                else tokenizer
            )
            # Move model to CPU if the dataset is generated, to avoid GPU memory issues during generation.
            if issubclass(dataset_cls, GeneratedDataset):
                dataset_kwargs.setdefault("model_id", model_id)
                gen_ctx = generator.on_device(torch.device("cpu"))
            else:
                gen_ctx = contextlib.nullcontext()
            with gen_ctx:
                train_dataset = dataset_cls.load_encoded_dataset(
                    dataset_tokenizer,
                    context_length,
                    **dataset_kwargs,
                )
        else:
            train_dataset = None

        with GPUMeter(
            **profiler_kwargs,
            capture_intermediate_data=profiler_capture_intermediate_data,
        ) as profiler:
            recipe_cls.apply(
                sim_component,
                generator,
                train_dataset,
                component=component,
                **recipe_kwargs,
            )
        step_stats.append(
            RecipeStepStats(
                recipe_name=recipe_cls.__name__,
                recipe_kwargs=recipe_kwargs,
                dataset_name=dataset_cls.__name__ if dataset_cls else "",
                dataset_kwargs=dataset_kwargs,
                profiler=profiler,
            )
        )

        # Save to cache after cacheable steps
        if recipe_cache and chain_hashes and recipe_cls.cacheable():
            recipe_cache.save(
                chain_hash=chain_hashes[i + 1],
                parent_hash=chain_hashes[i],
                step_stats=step_stats,
                sim_component=sim_component,
                framework=framework,
                component=component,
                model_id=model_id,
            )

        gc.collect()
        torch.cuda.empty_cache()
    return step_stats

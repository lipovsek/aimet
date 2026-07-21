# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""GenAI test runner"""

import contextlib
import uuid
import pytest
import torch
import gc
import os
import yaml
from pathlib import Path
from transformers.processing_utils import ProcessorMixin

from aimet_torch.v2.nn import QuantizationMixin, compute_param_encodings
from aimet_torch.v2.utils import remove_all_quantizers

from GenAILab.qai_hub_lm.models.base import LLM, VLM
from GenAILab.qai_hub_lm.models.utils.layer_cache import build_layer_cache_descriptors
from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.bench.profiler import (
    GPUMeter,
    MetricResult,
    ComponentRecipeStats,
    RecipeStepStats,
    write_stats_to_disk,
)
from GenAILab.bench.determinism import set_seed
from GenAILab.bench.eval_context import EvaluationContext
from GenAILab.bench.fp_cache import DiskBackedFPCache
from GenAILab.bench.metrics import TextEvaluationMetric
from GenAILab.bench.recipe_chain import apply_recipe_chain
from GenAILab.bench import datasets, metrics
from GenAILab.qai_hub_lm.backends import torch as models  # noqa: F401 — triggers registration
from GenAILab.bench.torch import quant_recipes
from GenAILab.qai_hub_lm.backends.torch.generator_utils import generator_factory
from GenAILab.qai_hub_lm.backends.torch.quantsim_utils import apply_spinquant_pre_sim


def test_llm_quantization(
    test_config, fp_cache: DiskBackedFPCache, recipe_cache, export_dir, results_dir
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
    image_size = model_kwargs.pop("image_size", None)
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

    # SpinQuant rotates the float model before the sim is built. Load the raw
    # model first, rotate it, then build the sim on the rotated graph. The
    # parser pulls the SpinQuant flags out of the recipe chain so the chain
    # never contains a SpinQuant step.
    spinquant_config = test_parameters.pop("spinquant", None)

    gc.collect()
    torch.cuda.empty_cache()

    model = model_cls.instantiate_float_model(
        model_id,
        **model_kwargs,
    )

    # SpinQuant rotates the float model before the sim is built, so it is not a
    # recipe-chain step (the parser strips it out). Profile it here and re-attach
    # it as a synthetic leading step below so the recorded recipe still reflects
    # that SpinQuant was applied (and with which rotations).
    spinquant_profiler = None
    if spinquant_config is not None:
        with GPUMeter(
            **profiler_kwargs,
            capture_intermediate_data=profiler_capture_intermediate_data,
        ) as spinquant_profiler:
            apply_spinquant_pre_sim(model, spinquant_config)
    else:
        apply_spinquant_pre_sim(model, spinquant_config)

    # Pass model_id so a QAT-aware instantiate_quantsim (e.g. Gemma4_Torch) can
    # locate the packed checkpoint's scales. Other backends absorb it via **kwargs.
    sim_collection = model_cls.instantiate_quantsim(
        model,
        context_length,
        sequence_length,
        precision=precision,
        image_size=image_size,
        model_id=model_id,
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
        image_size=image_size,
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
    with generator.on_device(device):
        # Disable visual quantizers during backbone recipes so the vision
        # encoder runs in FP mode (its quantizers aren't calibrated yet).
        visual_ctx = (
            remove_all_quantizers(sim_collection.visual.model)
            if sim_collection.visual is not None
            else contextlib.nullcontext()
        )
        with visual_ctx:
            backbone_steps = apply_recipe_chain(
                all_recipes["backbone"],
                sim_collection.backbone,
                generator,
                tokenizer,
                context_length,
                image_size,
                profiler_kwargs,
                profiler_capture_intermediate_data,
                framework="torch",
                model_id=model_id,
                precision=precision,
                model_kwargs=model_kwargs,
                component="backbone",
                recipe_cache=recipe_cache,
                spinquant_config=spinquant_config,
            )

        visual_steps = []
        if "visual" in all_recipes and sim_collection.visual is not None:
            # Disable backbone quantizers during visual recipes and switch
            # the generator to yield vision model inputs from prefill().
            backbone_ctx = remove_all_quantizers(sim_collection.backbone.model)
            with backbone_ctx, generator.visual_quantization_mode():
                visual_steps = apply_recipe_chain(
                    all_recipes["visual"],
                    sim_collection.visual,
                    generator,
                    tokenizer,
                    context_length,
                    image_size,
                    profiler_kwargs,
                    profiler_capture_intermediate_data,
                    framework="torch",
                    model_id=model_id,
                    precision=precision,
                    model_kwargs=model_kwargs,
                    component="visual",
                    recipe_cache=recipe_cache,
                    spinquant_config=spinquant_config,
                )

        # Finalize embedding quantization after recipes have had a chance to
        # transform the weights. The embedding is already rotated when SpinQuant
        # ran on the float model before sim creation.
        # Skip if RemoveQuantization was applied to the backbone — the
        # embedding should stay in FP to match.
        backbone_removed_quant = any(
            s.recipe_name == "RemoveQuantization" for s in backbone_steps
        )
        if (
            sim_collection.embedding is not None
            and isinstance(sim_collection.embedding, QuantizationMixin)
            and not backbone_removed_quant
        ):
            compute_param_encodings(sim_collection.embedding)

        gc.collect()
        torch.cuda.empty_cache()

    run_group = None
    export_dir = test_parameters["export"] if test_parameters["export"] else None
    # TODO: remove skip exports for models that require Dynamo export
    if export_dir and not model_cls.use_dynamo_export():
        tokenizer.save_pretrained(export_dir)
        sim_collection.config.save_pretrained(export_dir)

        os.mkdir(os.path.join(export_dir, "backbone"))
        use_dynamic = isinstance(sequence_length, list) and len(sequence_length) > 1
        max_sl = (
            max(sequence_length)
            if isinstance(sequence_length, list)
            else sequence_length
        )
        sl_tag = "dynamic" if use_dynamic else str(max_sl)
        layer_cache_descs = build_layer_cache_descriptors(
            sim_collection.backbone.model.model.config
        )
        sim_collection.backbone.onnx.export(
            f=os.path.join(
                export_dir, "backbone", f"model_sl{sl_tag}_cl{context_length}.onnx"
            ),
            args=model_cls.get_sample_backbone_inputs(
                model=sim_collection.backbone.model,
                context_length=context_length,
                sequence_length=max_sl,
                layer_cache_descriptors=generator.layer_cache_descriptors,
                image_size=image_size,
                config=sim_collection.config,
            ),
            input_names=model_cls.get_backbone_input_names(layer_cache_descs),
            output_names=model_cls.get_backbone_output_names(layer_cache_descs),
            opset_version=17,
            dynamo=model_cls.use_dynamo_export(),
            dynamic_axes=model_cls.get_backbone_dynamic_axes(layer_cache_descs)
            if use_dynamic
            else None,
            export_int32_bias=False,
        )

        if sim_collection.visual is not None:
            assert issubclass(model_cls, VLM)
            os.mkdir(os.path.join(export_dir, "visual"))
            sim_collection.visual.onnx.export(
                f=os.path.join(export_dir, "visual", "model.onnx"),
                args=model_cls.get_sample_vision_inputs(
                    sim_collection.config, image_size=image_size
                ),
                input_names=model_cls.get_visual_input_names(),
                output_names=model_cls.get_visual_output_names(),
                opset_version=17,
                dynamo=model_cls.use_dynamo_export(),
                export_int32_bias=False,
            )

        if sim_collection.embedding is not None:
            if isinstance(sim_collection.embedding, QuantizationMixin):
                sim_collection.embedding.fold_param_quantizers()

            torch.save(
                sim_collection.embedding.state_dict()["weight"].as_subclass(
                    torch.Tensor
                ),
                os.path.join(export_dir, "embedding.pth"),
            )

        # Save any extra auxiliary modules (e.g. Gemma4's per-layer embedding
        # embed_tokens_per_layer) under extras/<name>.pth so the ONNX phase can
        # restore them into the generator's extras.
        if sim_collection.extras:
            extras_dir = os.path.join(export_dir, "extras")
            os.makedirs(extras_dir, exist_ok=True)
            for extra_name, extra_mod in sim_collection.extras.items():
                if extra_mod is None:
                    continue
                if isinstance(extra_mod, QuantizationMixin):
                    extra_mod.fold_param_quantizers()
                weight = getattr(extra_mod, "weight", None)
                if weight is None and hasattr(extra_mod, "state_dict"):
                    weight = extra_mod.state_dict().get("weight")
                if weight is not None:
                    torch.save(
                        weight.as_subclass(torch.Tensor),
                        os.path.join(extras_dir, f"{extra_name}.pth"),
                    )

        if test_parameters["eval_in_onnx"]:
            run_group = uuid.uuid4().hex[:16]

            # Use the last Calibration step's dataset for ONNX re-calibration
            def _last_calibration_for_onnx(steps):
                for s in reversed(steps):
                    if s.recipe_name == quant_recipes.Calibration.__name__:
                        return {
                            "name": quant_recipes.Calibration.__name__,
                            "dataset": {"name": s.dataset_name, **s.dataset_kwargs},
                            **s.recipe_kwargs,
                        }
                return {"name": quant_recipes.RemoveQuantization.__name__}

            onnx_recipe = {
                "backbone": _last_calibration_for_onnx(backbone_steps),
            }
            if visual_steps:
                onnx_recipe["visual"] = _last_calibration_for_onnx(visual_steps)

            data = {
                "model": {
                    "model_id": export_dir,
                    "encodings": export_dir,
                    "sequence_length": sequence_length,
                    "context_length": context_length,
                    **({"image_size": list(image_size)} if image_size else {}),
                    **model_kwargs,
                },
                "precision": precision.to_dict(),
                "run_group": run_group,
                "recipe": onnx_recipe,
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

    with generator.on_device(device):
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
                    extra_metric_kwargs = {}
                    if not issubclass(metric_cls, TextEvaluationMetric):
                        extra_metric_kwargs["image_size"] = image_size
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
                        profiler=profiler
                        if profiler_capture_intermediate_data
                        else None,
                        scoring_version=metric_cls.SCORING_VERSION,
                    )
                )

    model_kwargs["context_length"] = context_length
    model_kwargs["sequence_length"] = sequence_length
    if image_size is not None:
        model_kwargs["image_size"] = list(image_size)
    if precomputed_encodings is not None:
        model_kwargs["encodings"] = precomputed_encodings

    # Re-attach SpinQuant as a synthetic leading step so the recorded recipe
    # reflects the pre-sim rotation. The single apply_spinquant_pre_sim call
    # rotates both backbone and visual graphs, so the profiler is attached to the
    # backbone step only; the visual marker carries no profiler to avoid
    # double-counting the rotation time in aggregated utilization.
    if spinquant_config is not None:
        backbone_steps = [
            RecipeStepStats(
                recipe_name="SpinQuant",
                recipe_kwargs=spinquant_config,
                dataset_name="",
                dataset_kwargs={},
                profiler=spinquant_profiler,
            ),
            *backbone_steps,
        ]
        if visual_steps:
            visual_steps = [
                RecipeStepStats(
                    recipe_name="SpinQuant",
                    recipe_kwargs=spinquant_config,
                    dataset_name="",
                    dataset_kwargs={},
                    profiler=None,
                ),
                *visual_steps,
            ]

    components = {
        "backbone": ComponentRecipeStats(steps=backbone_steps),
    }
    if visual_steps:
        components["visual"] = ComponentRecipeStats(steps=visual_steps)

    results_folder = Path(results_dir)
    results_folder.mkdir(parents=True, exist_ok=True)
    precision_dict = precision.to_dict()
    if "dtype" in model_kwargs:
        model_kwargs["dtype"] = str(model_kwargs["dtype"])
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

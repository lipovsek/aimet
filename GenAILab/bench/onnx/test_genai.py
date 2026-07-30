# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""GenAI test runner"""

import contextlib
import warnings
import pytest
import torch
import gc
import os
from pathlib import Path
from transformers.processing_utils import ProcessorMixin

from aimet_onnx.quantsim import load_encodings_to_sim

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
from GenAILab.bench.model_cache import DiskBackedModelCache
from GenAILab.bench.recipe_chain import (
    apply_pre_quantization_chain,
    apply_quantization_chain,
)
from GenAILab.bench.precision import float16, float32
from GenAILab.qai_hub_lm.models.base import LLM, VLM
from GenAILab.bench import datasets, metrics  # noqa: F401 — triggers registration
from GenAILab.qai_hub_lm.backends import onnx as models  # noqa: F401 — triggers registration
from GenAILab.bench.onnx import quant_recipes  # noqa: F401 — triggers registration
from GenAILab.qai_hub_lm.backends.onnx.generator_utils import generator_factory
from GenAILab.qai_hub_lm.backends.onnx.quantsim_utils import quantize_embedding_weights


@contextlib.contextmanager
def _disable_onnx_quantizers(quantsim):
    """Temporarily disable all quantize ops on an ONNX quantsim."""
    ops = {name: op for name, op in quantsim.qc_quantize_op_dict.items() if op.enabled}
    for op in ops.values():
        op.enabled = False
    try:
        yield
    finally:
        for op in ops.values():
            op.enabled = True


def test_llm_quantization(
    test_config,
    model_cache: DiskBackedModelCache,
    fp_cache: DiskBackedFPCache,
    recipe_cache,
    export_dir,
    results_dir,
):
    if test_config is None:
        pytest.skip("No GenAI test parameters provided.")
    set_seed(42)

    config = YAMLConfigParser.parse_document(test_config, export_base_dir=export_dir)
    print(config)

    eval_ctx = EvaluationContext(fp_cache=fp_cache, model_config=config.model)

    model_cls = config.model.model_cls
    context_length = config.model.context_length
    sequence_length = config.model.sequence_length
    model_id = config.model.model_id
    model_type = config.model.model_type
    model_dtype = config.model.dtype
    image_size = config.model.image_size
    precomputed_encodings = config.model.encodings

    sl_tag = (
        "dynamic"
        if isinstance(sequence_length, list) and len(sequence_length) > 1
        else str(
            max(sequence_length)
            if isinstance(sequence_length, list)
            else sequence_length
        )
    )

    model_kwargs = config.model.extra_kwargs.copy()

    dtype_name = model_dtype or "float32"
    if dtype_name not in ("float32", "float16"):
        raise ValueError(
            f"Unsupported model dtype {dtype_name!r}; expected 'float32' or 'float16'."
        )
    model_kwargs["dtype"] = getattr(torch, dtype_name)

    if config.eval_in_onnx:
        warnings.warn("eval_in_onnx is ignored for ONNX GenAI tests.")

    precision = config.precision
    run_group = config.run_group

    # Pre-sim techniques (e.g. SpinQuant) modify the float ONNX graph before the
    # sim is built. The parser splits them into a flat pre_sim tuple; export the
    # raw model(s), apply the pre-sim chain, then build the sim on the updated graph.

    gc.collect()
    torch.cuda.empty_cache()

    entry = model_cls.instantiate_float_model(
        model_id,
        context_length,
        sequence_length,
        model_cache=model_cache,
        image_size=image_size,
        **model_kwargs,
    )

    # Apply the pre-sim chain generically (name -> registered
    # PreQuantizationTechnique -> apply(float_model, **flags)); the onnx float
    # model is the exported ``entry`` bundle (backbone/visual/embedding), which
    # the technique unpacks. Each pre-sim technique rotates the whole model once.
    # ``pre_sim_profilers`` maps technique -> profiler for re-attachment below.
    pre_sim_profilers = apply_pre_quantization_chain(
        config.recipe.pre_sim,
        entry,
        profiler_kwargs=config.profiler.gpu_meter_kwargs,
        profiler_capture_intermediate_data=config.profiler.capture_intermediate_data,
    )

    sim_collection = model_cls.instantiate_quantsim(
        entry,
        precision=precision,
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
        image_size=image_size,
        **model_kwargs,
    )

    if precomputed_encodings is not None:
        backbone_encodings = os.path.join(
            precomputed_encodings,
            "backbone",
            f"model_sl{sl_tag}_cl{context_length}.encodings",
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

    # Disable visual quantizers during backbone recipes so the vision
    # encoder runs in FP mode (its quantizers aren't calibrated yet).
    visual_ctx = (
        _disable_onnx_quantizers(sim_collection.visual)
        if sim_collection.visual is not None
        else contextlib.nullcontext()
    )
    with visual_ctx:
        backbone_steps = apply_quantization_chain(
            config.recipe.backbone,
            sim_collection.backbone,
            generator,
            tokenizer,
            context_length,
            image_size,
            config.profiler.gpu_meter_kwargs,
            config.profiler.capture_intermediate_data,
            framework="onnx",
            model_id=model_id,
            precision=precision,
            model_kwargs=model_kwargs,
            component="backbone",
            recipe_cache=recipe_cache,
            pre_sim=config.recipe.pre_sim,
        )

    visual_steps = []
    if config.recipe.visual is not None and sim_collection.visual is not None:
        # Disable backbone quantizers during visual recipes and switch
        # the generator to yield vision model inputs from prefill().
        backbone_ctx = _disable_onnx_quantizers(sim_collection.backbone)
        with backbone_ctx, generator.visual_quantization_mode():
            visual_steps = apply_quantization_chain(
                config.recipe.visual,
                sim_collection.visual,
                generator,
                tokenizer,
                context_length,
                image_size,
                config.profiler.gpu_meter_kwargs,
                config.profiler.capture_intermediate_data,
                framework="onnx",
                model_id=model_id,
                precision=precision,
                model_kwargs=model_kwargs,
                component="visual",
                recipe_cache=recipe_cache,
                pre_sim=config.recipe.pre_sim,
            )

    # Finalize embedding quantization after recipes have had a chance to
    # transform the weights (e.g. SpinQuant rotation).
    # Skip if RemoveQuantization was applied to the backbone — the
    # embedding should stay in FP to match.
    backbone_removed_quant = any(
        s.recipe_name == "RemoveQuantization" for s in backbone_steps
    )
    if (
        sim_collection.embedding is not None
        and precision.embedding not in (float16, float32)
        and not backbone_removed_quant
    ):
        quantize_embedding_weights(sim_collection.embedding, precision.embedding.bits)

    gc.collect()
    torch.cuda.empty_cache()

    export_dir = config.export
    if export_dir:
        sim_collection.config.save_pretrained(export_dir)
        tokenizer.save_pretrained(export_dir)

        os.mkdir(os.path.join(export_dir, "backbone"))
        sim_collection.backbone.export(
            os.path.join(export_dir, "backbone"),
            f"model_sl{sl_tag}_cl{context_length}",
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
        for metric in config.metrics:
            metric_cls = metric.metric_cls
            with GPUMeter(
                capture_intermediate_data=False, **config.profiler.gpu_meter_kwargs
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
                    **metric.metric_kwargs,
                )
                print(f"{metric_cls.__name__} result: {result}")

            evaluation_results.append(
                MetricResult(
                    metric_name=metric_cls.__name__,
                    result=result,
                    profiler=profiler
                    if config.profiler.capture_intermediate_data
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

    # Re-attach pre-sim steps (e.g. SpinQuant) as synthetic leading steps so the
    # recorded recipe reflects the pre-sim rotations. A single pre-sim pass
    # rotates the whole model, so the same markers are prepended to both
    # backbone and visual component recipes.
    pre_markers = [
        RecipeStepStats(
            recipe_name=step.name,
            recipe_kwargs=step.recipe_kwargs,
            dataset_name="",
            dataset_kwargs={},
            profiler=pre_sim_profilers.get(step.name),
        )
        for step in config.recipe.pre_sim
    ]
    backbone_steps = [*pre_markers, *backbone_steps]
    if visual_steps:
        visual_steps = [*pre_markers, *visual_steps]

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

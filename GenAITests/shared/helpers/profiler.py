# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""General utils for GenAI model testing"""

import os
import json
import csv
import collections
import sys
from dataclasses import dataclass

# Import GPUMeter from ONNXRegression evaluation module
sys.path.append(
    os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "../../../ONNXRegression/evaluation"
    )
)
from metrics_utils import GPUMeter


def convert_gpu_meter_to_dict(
    profiler: GPUMeter, remove_finegrained: bool = False
) -> dict[str, str]:
    if profiler is None:
        return {}
    results = {
        "elapsed_ms": profiler.elapsed_ms,
        "cuda_starting_mb": profiler.cuda_first_mb,
        "cuda_peak_mb": profiler.cuda_peak_mb,
        "cuda_running_mb": profiler.cuda_running_mb if not remove_finegrained else None,
        "cpu_starting_mb": profiler.cpu_first_mb,
        "cpu_peak_mb": profiler.cpu_peak_mb,
        "cpu_running_mb": profiler.cpu_running_mb if not remove_finegrained else None,
    }
    return {key: value for key, value in results.items() if value is not None}


@dataclass
class MetricResult:
    """Dataclass to hold accuracy and profiling results from running a metric"""

    metric_name: str
    result: float | list[str]
    profiler: GPUMeter


@dataclass
class ComponentRecipeStats:
    """Dataclass to hold recipe and dataset info for a model component"""

    recipe_name: str
    recipe_kwargs: dict
    dataset_name: str
    dataset_kwargs: dict


def recursive_update(d, u):
    """Internal helper function to update nested dict"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def write_stats_to_disk(
    output_folder: str,
    filename: str,
    model_family: str,
    model_id: str,
    model_modifiers: dict[str, str],
    components: dict[str, ComponentRecipeStats],
    quantization_results: GPUMeter,
    accuracy_results: list[MetricResult],
    export_location: str | None = None,
):
    _write_stats_to_json(
        str(os.path.join(output_folder, filename + ".json")),
        model_family,
        model_id,
        model_modifiers,
        components,
        quantization_results,
        accuracy_results,
        export_location,
    )

    _write_stats_to_csv(
        str(os.path.join(output_folder, filename + ".csv")),
        model_family,
        model_id,
        model_modifiers,
        components,
        quantization_results,
        accuracy_results,
        export_location,
    )


def _write_stats_to_csv(
    filename: str,
    model_cls: str,
    model_id: str,
    model_modifiers: dict[str, str],
    components: dict[str, ComponentRecipeStats],
    quantization_results: GPUMeter,
    accuracy_results: list[MetricResult],
    export_location: str | None = None,
):
    def dict_to_postgres_csv_json_field(d):
        json_str = json.dumps(d, separators=(",", ":"))  # Compact JSON
        escaped = json_str.replace('"', '""')
        return f'"{escaped}"'

    accuracy_table = {
        result.metric_name: {"result": result.result}
        | convert_gpu_meter_to_dict(result.profiler, remove_finegrained=True)
        for result in accuracy_results
    }

    # Convert components to serializable dict
    components_dict = {
        name: {
            "recipe_name": stats.recipe_name,
            "recipe_kwargs": stats.recipe_kwargs,
            "dataset_name": stats.dataset_name,
            "dataset_kwargs": stats.dataset_kwargs,
        }
        for name, stats in components.items()
    }

    stats = [
        model_cls,
        model_id,
        dict_to_postgres_csv_json_field(model_modifiers),
        dict_to_postgres_csv_json_field(components_dict),
        dict_to_postgres_csv_json_field(
            convert_gpu_meter_to_dict(quantization_results, remove_finegrained=True)
        ),
        dict_to_postgres_csv_json_field(accuracy_table),
        export_location if export_location is not None else "",
    ]

    if not os.path.exists(filename):
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "model_family",
                    "model_id",
                    "model_modifiers",
                    "components",
                    "quantization_results",
                    "accuracy_results",
                    "export",
                ]
            )

    with open(filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(stats)


def _write_stats_to_json(
    filename: str,
    model_family: str,
    model_id: str,
    model_modifiers: dict[str, str],
    components: dict[str, ComponentRecipeStats],
    quantization_results: GPUMeter,
    accuracy_results: list[MetricResult],
    export_location: str | None = None,
):
    """Helper function to write collected stats to disk, only overwriting newly collected fields"""

    # Check if the file exists
    if os.path.exists(filename):
        # Open the file and load the existing data
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        # If the file does not exist, create an empty dictionary
        data = {}

    model_params_string_formatted = ", ".join(
        [f"{key}={value}" for key, value in model_modifiers.items()]
        + [f"model_id={model_id}"]
    )

    # Build component key string (e.g., "backbone:AdaScale+Wikitext, visual:PCQ+Wikitext")
    component_strings = []
    for comp_name, comp_stats in components.items():
        component_strings.append(
            f"{comp_name}:{comp_stats.recipe_name}+{comp_stats.dataset_name}"
        )
    components_key = ", ".join(component_strings)

    # Build recipe params string from all components
    recipe_params = []
    for comp_name, comp_stats in components.items():
        for key, value in comp_stats.recipe_kwargs.items():
            recipe_params.append(f"{comp_name}.{key}={value}")
    recipe_params_string_formatted = (
        ", ".join(recipe_params) if recipe_params else "default"
    )

    stats = {
        "quantization": convert_gpu_meter_to_dict(quantization_results),
        "components": {
            comp_name: {
                "recipe": comp_stats.recipe_name,
                "recipe_kwargs": comp_stats.recipe_kwargs,
                "dataset": comp_stats.dataset_name,
                "dataset_kwargs": comp_stats.dataset_kwargs,
            }
            for comp_name, comp_stats in components.items()
        },
    } | {
        result.metric_name: {"result": result.result}
        | convert_gpu_meter_to_dict(result.profiler)
        for result in accuracy_results
    }

    if export_location is not None:
        stats["export"] = export_location

    # Update the dictionary with nested structure
    x = {recipe_params_string_formatted: stats}
    x = {components_key: x}
    x = {model_params_string_formatted: x}
    x = {model_family: x}
    recursive_update(data, x)

    # Write the updated dictionary back to the file
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

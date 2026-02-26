# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""General utils for GenAI model testing"""

import os
import json
import csv
import collections
import sys
import fcntl
from contextlib import contextmanager
from dataclasses import dataclass

# Import GPUMeter from AIMETRegression evaluation module
sys.path.append(
    os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        "../../../AIMETRegression/evaluation",
    )
)
from metrics_utils import GPUMeter


@contextmanager
def _file_lock(filepath: str):
    """Context manager for process-safe file locking.

    Uses a separate .lock file to coordinate access between processes.
    """
    lock_path = filepath + ".lock"
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


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
    """Dataclass to hold recipe, dataset, and profiling info for a model component"""

    recipe_name: str
    recipe_kwargs: dict
    dataset_name: str
    dataset_kwargs: dict
    profiler: GPUMeter = None


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
    model_type: str,
    model_id: str,
    model_modifiers: dict[str, str],
    components: dict[str, ComponentRecipeStats],
    accuracy_results: list[MetricResult],
    export_location: str | None = None,
):
    _write_stats_to_json(
        str(os.path.join(output_folder, filename + ".json")),
        model_type,
        model_id,
        model_modifiers,
        components,
        accuracy_results,
        export_location,
    )

    _write_stats_to_csv(
        str(os.path.join(output_folder, filename + ".csv")),
        model_type,
        model_id,
        model_modifiers,
        components,
        accuracy_results,
        export_location,
    )


def _write_stats_to_csv(
    filename: str,
    model_type: str,
    model_id: str,
    model_modifiers: dict[str, str],
    components: dict[str, ComponentRecipeStats],
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

    # Convert components to serializable dict with resource_utilization
    components_dict = {
        name: {
            "recipe_name": stats.recipe_name,
            "recipe_kwargs": stats.recipe_kwargs,
            "dataset_name": stats.dataset_name,
            "dataset_kwargs": stats.dataset_kwargs,
            "resource_utilization": convert_gpu_meter_to_dict(
                stats.profiler, remove_finegrained=True
            ),
        }
        for name, stats in components.items()
    }

    stats = [
        model_type,
        model_id,
        dict_to_postgres_csv_json_field(model_modifiers),
        dict_to_postgres_csv_json_field(components_dict),
        dict_to_postgres_csv_json_field(accuracy_table),
        export_location if export_location is not None else "",
    ]

    # Use file lock to ensure process-safe writes
    with _file_lock(filename):
        if not os.path.exists(filename):
            with open(filename, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        "model_type",
                        "model_id",
                        "model_modifiers",
                        "components",
                        "accuracy_results",
                        "export",
                    ]
                )

        with open(filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(stats)


def _write_stats_to_json(
    filename: str,
    model_type: str,
    model_id: str,
    model_modifiers: dict[str, str],
    components: dict[str, ComponentRecipeStats],
    accuracy_results: list[MetricResult],
    export_location: str | None = None,
):
    """Helper function to write collected stats to disk, only overwriting newly collected fields.

    This function is process-safe and uses file locking to prevent race conditions
    when multiple processes write to the same file.
    """
    stats = {
        "model_id": model_id,
        "model_modifiers": model_modifiers,
        "components": {
            comp_name: {
                "recipe": comp_stats.recipe_name,
                "recipe_kwargs": comp_stats.recipe_kwargs,
                "dataset": comp_stats.dataset_name,
                "dataset_kwargs": comp_stats.dataset_kwargs,
                "resource_utilization": convert_gpu_meter_to_dict(comp_stats.profiler),
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

    # Use file lock to ensure process-safe read-modify-write
    with _file_lock(filename):
        # Check if the file exists
        if os.path.exists(filename):
            # Open the file and load the existing data
            with open(filename, "r") as f:
                data = json.load(f)
        else:
            # If the file does not exist, create an empty dictionary
            data = {}

        # Append to list of results for this model_type
        if model_type not in data:
            data[model_type] = []
        data[model_type].append(stats)

        # Write the updated dictionary back to the file
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""General utils for GenAI model testing"""

import os
import json
import csv
import collections
import datetime
import platform
import subprocess
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


_cached_environment = None


def _collect_environment():
    """Collect environment info: Python version, GPU, platform, key packages.

    Results are cached after the first call since the environment doesn't change
    within a single process.
    """
    global _cached_environment
    if _cached_environment is not None:
        return _cached_environment

    env = {
        "run_type": "local",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    try:
        import torch

        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            env["pip_freeze"] = result.stdout.strip().split("\n")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    _cached_environment = env
    return env


def write_stats_to_disk(
    output_folder: str,
    filename: str,
    model_type: str,
    model_id: str,
    model_modifiers: dict[str, str],
    components: dict[str, ComponentRecipeStats],
    accuracy_results: list[MetricResult],
    export_location: str | None = None,
    precision: dict | None = None,
):
    _write_stats_to_json(
        str(os.path.join(output_folder, filename + ".json")),
        model_type,
        model_id,
        model_modifiers,
        components,
        accuracy_results,
        export_location,
        precision,
    )

    _write_stats_to_csv(
        str(os.path.join(output_folder, filename + ".csv")),
        model_type,
        model_id,
        model_modifiers,
        components,
        accuracy_results,
        export_location,
        precision,
    )


def _write_stats_to_csv(
    filename: str,
    model_type: str,
    model_id: str,
    model_modifiers: dict[str, str],
    components: dict[str, ComponentRecipeStats],
    accuracy_results: list[MetricResult],
    export_location: str | None = None,
    precision: dict | None = None,
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
        dict_to_postgres_csv_json_field(precision or {}),
        dict_to_postgres_csv_json_field(components_dict),
        dict_to_postgres_csv_json_field(accuracy_table),
        export_location if export_location is not None else "",
        dict_to_postgres_csv_json_field(_collect_environment()),
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
                        "precision",
                        "components",
                        "accuracy_results",
                        "export",
                        "environment",
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
    precision: dict | None = None,
):
    """Helper function to write collected stats to disk, only overwriting newly collected fields.

    This function is process-safe and uses file locking to prevent race conditions
    when multiple processes write to the same file.
    """
    stats = {
        "model_id": model_id,
        "model_modifiers": model_modifiers,
        "precision": precision or {},
        "environment": _collect_environment(),
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


def _read_json_data(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return {}


def _append_json_entry(data, model_type, entry):
    if model_type not in data:
        data[model_type] = []
    data[model_type].append(entry)


def merge_json_results(source_path, dest_path):
    """Merge entries from a source profiling_data.json into a destination file.

    Each model_type key in the source has its entries appended to the
    corresponding list in the destination.
    """
    if not os.path.exists(source_path):
        return 0

    with open(source_path, "r") as f:
        source_data = json.load(f)

    dest_data = _read_json_data(dest_path)

    count = 0
    for model_type, entries in source_data.items():
        for entry in entries:
            _append_json_entry(dest_data, model_type, entry)
            count += 1

    with open(dest_path, "w") as f:
        json.dump(dest_data, f, indent=4)

    return count


def merge_csv_results(source_path, dest_path):
    """Append rows from a source profiling_data.csv into a destination file.

    Creates the destination with headers if it doesn't exist.
    """
    if not os.path.exists(source_path):
        return 0

    with open(source_path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) <= 1:
        return 0

    header = rows[0]
    data_rows = rows[1:]

    if not os.path.exists(dest_path):
        with open(dest_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    with open(dest_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data_rows)

    return len(data_rows)

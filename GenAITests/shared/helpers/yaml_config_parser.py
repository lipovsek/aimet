# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Config parser for GenAI model testing"""

import os
import warnings
from pathlib import Path
import yaml
from .export import get_test_artifacts_path


class YAMLConfigParser:
    recipe_lookup: dict = {}
    model_lookup: dict = {}
    dataset_lookup: dict = {}
    metrics_lookup: dict = {}

    @classmethod
    def register_model(cls, model_cls):
        model_name = model_cls.__name__
        if not model_name.endswith("_ONNX") and not model_name.endswith("_Torch"):
            return
        model_name = model_name.removesuffix("_ONNX").removesuffix("_Torch")
        cls.model_lookup[model_name] = model_cls
        return model_cls

    @classmethod
    def register_recipe(cls, recipe_cls):
        cls.recipe_lookup[recipe_cls.__name__] = recipe_cls
        return recipe_cls

    @classmethod
    def register_dataset(cls, dataset_cls):
        cls.dataset_lookup[dataset_cls.__name__] = dataset_cls
        return dataset_cls

    @classmethod
    def register_metric(cls, metric_cls):
        cls.metrics_lookup[metric_cls.__name__] = metric_cls
        return metric_cls

    @classmethod
    def get_model(cls, model_name):
        return cls.model_lookup[model_name]

    @classmethod
    def get_recipe(cls, recipe_name):
        return cls.recipe_lookup[recipe_name]

    @classmethod
    def get_dataset(cls, dataset_name):
        return cls.dataset_lookup[dataset_name]

    @classmethod
    def get_metric(cls, metrics_name):
        return cls.metrics_lookup[metrics_name]

    @classmethod
    def validate_config(cls, doc):
        if "model" not in doc:
            raise RuntimeError("Model not specified.")
        if "recipe" not in doc:
            raise RuntimeError("Recipe not specified.")
        if "metrics" not in doc:
            raise RuntimeError("Metrics not specified.")

        if not isinstance(doc["model"], dict):
            raise RuntimeError(
                "Multiple models cannot be specified in a single document."
            )
        if not isinstance(doc["recipe"], dict):
            raise RuntimeError(
                "Multiple recipes cannot be specified in a single document."
            )

        if "name" not in doc["model"]:
            raise RuntimeError("Model name not specified.")

        if "sequence_length" not in doc["model"]:
            raise RuntimeError("Sequence length not specified.")
        if "context_length" not in doc["model"]:
            raise RuntimeError("Context length not specified.")

        # Normalize single recipe to component format
        has_recipe_name = "name" in doc["recipe"]
        has_backbone = "backbone" in doc["recipe"]
        if not has_recipe_name and not has_backbone:
            raise RuntimeError(
                "Recipe must have either 'name' or 'backbone' specified."
            )
        elif has_recipe_name and has_backbone:
            raise RuntimeError(
                "Recipe cannot have both 'name' and 'backbone' specified."
            )
        elif has_recipe_name:
            doc["recipe"] = {"backbone": doc["recipe"]}

        # Backward compatibility: migrate top-level dataset into backbone component
        if "dataset" in doc:
            if "dataset" not in doc["recipe"]["backbone"]:
                doc["recipe"]["backbone"]["dataset"] = doc.pop("dataset")
            else:
                doc.pop("dataset")  # Component has its own dataset, discard top-level

        metrics = (
            doc["metrics"] if isinstance(doc["metrics"], list) else [doc["metrics"]]
        )
        for metric in metrics:
            if "name" not in metric:
                raise RuntimeError("Metric name not specified.")

    @classmethod
    def parse_document(cls, doc):
        cls.validate_config(doc)
        task_params = {}

        task_params["export"] = doc.pop("export", False)
        if not isinstance(task_params["export"], bool):
            raise ValueError("Export field must be a boolean value.")

        task_params["eval_in_onnx"] = doc.pop("eval_in_onnx", False)
        if not isinstance(task_params["export"], bool):
            raise ValueError("Export field must be a boolean value.")

        if task_params["eval_in_onnx"] and not task_params["export"]:
            warnings.warn(
                "eval_in_onnx is enabled, but export is disabled. Overriding export to True."
            )
            task_params["export"] = True

        if task_params["export"]:
            task_params["export"] = get_test_artifacts_path(doc)
            Path(task_params["export"]).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(task_params["export"], "config.yaml"), "w") as file:
                yaml.dump(doc, file)

        model_name = doc["model"]["name"]
        try:
            model_cls = cls.get_model(model_name)
            task_params["model"] = doc.pop("model")
            task_params["model"]["class"] = model_cls
            del task_params["model"]["name"]
        except LookupError as exc:
            raise LookupError(
                f"Specified model name ({model_name}) not found."
            ) from exc

        task_params["recipe"] = {}
        for component_name, component_config in doc["recipe"].items():
            recipe_name = component_config["name"]
            try:
                recipe_cls = cls.get_recipe(recipe_name)
                task_params["recipe"][component_name] = component_config.copy()
                task_params["recipe"][component_name]["class"] = recipe_cls
                del task_params["recipe"][component_name]["name"]
            except LookupError as exc:
                raise LookupError(
                    f"Specified quantization recipe name ({recipe_name}) not found."
                ) from exc

            # Parse dataset within component
            if "dataset" in component_config:
                dataset_config = task_params["recipe"][component_name]["dataset"]
                dataset_name = dataset_config["name"]
                try:
                    dataset_cls = cls.get_dataset(dataset_name)
                    dataset_config["class"] = dataset_cls
                    del dataset_config["name"]
                except LookupError as exc:
                    raise LookupError(
                        f"Specified dataset name ({dataset_name}) not found."
                    ) from exc
        del doc["recipe"]

        metrics = (
            doc["metrics"] if isinstance(doc["metrics"], list) else [doc["metrics"]]
        )
        task_params["metrics"] = []
        for metric in metrics:
            metric_name = metric["name"]
            try:
                metric_cls = cls.get_metric(metric_name)
                task_params["metrics"].append(metric)
                task_params["metrics"][-1]["class"] = metric_cls
                del task_params["metrics"][-1]["name"]
            except LookupError as exc:
                raise LookupError(
                    f"Specified metric name ({metric_name}) not found."
                ) from exc
        del doc["metrics"]

        task_params["profiler"] = doc.pop("profiler", {})

        if len(doc) > 0:
            raise ValueError(f"Unrecognized sections in config: {doc.keys()}")

        return task_params

    @classmethod
    def parse(cls, filename):
        print(filename)
        with open(filename, "r") as file:
            docs = yaml.safe_load_all(file)
            for doc in docs:
                yield cls.parse_document(doc)

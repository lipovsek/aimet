# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Config parser for GenAI model testing"""

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from transformers import AutoConfig

from .export import get_test_artifacts_path
from GenAILab.qai_hub_lm.precision import PrecisionConfig


@dataclass
class AdaptationInfo:
    """Metadata about an adaptation."""

    mixin_cls: type
    exclusive: bool = False  # If True, cannot combine with other adaptations
    required_for_export: bool = False  # If True, auto-enforced when exporting


class YAMLConfigParser:
    recipe_lookup: dict = {}
    model_lookup: dict = {}  # {model_type: model_cls} for VLMs and other special models
    dataset_lookup: dict = {}
    metrics_lookup: dict = {}
    adaptation_lookup: dict = {}  # {(model_type, adaptation_name): AdaptationInfo}

    _default_llm_cls: type = None

    # ========================
    # Default LLM registration
    # ========================

    @classmethod
    def register_default_llm(cls, llm_cls: type):
        """Register the default LLM class.

        Called by framework-specific modules on import.
        Fails if a different framework has already registered.

        Usage (in torch/models/llm.py):
            @YAMLConfigParser.register_default_llm
            class LLM_Torch(LLM):
                ...

        Usage (in onnx/models/llm.py):
            @YAMLConfigParser.register_default_llm
            class LLM_ONNX(LLM):
                ...
        """
        if cls._default_llm_cls is not None and cls._default_llm_cls is not llm_cls:
            raise RuntimeError(
                f"LLM class ({cls._default_llm_cls.__name__}) is already registered. "
                f"Only one framework can be active at a time."
            )
        cls._default_llm_cls = llm_cls
        return llm_cls

    @classmethod
    def get_default_llm(cls) -> type:
        """Get the registered default LLM class."""
        if cls._default_llm_cls is None:
            raise RuntimeError(
                "No default LLM class registered. "
                "Ensure you import the framework module before parsing configs."
            )
        return cls._default_llm_cls

    # ========================
    # Adaptation registration
    # ========================

    @classmethod
    def register_adaptation(
        cls,
        adaptation_name: str,
        model_type: str = "*",
        *,
        exclusive: bool = False,
        required_for_export: bool = False,
    ):
        """Register an adaptation mixin.

        Args:
            adaptation_name: Name used in config (e.g., "SHA", "AIHM")
            model_type: HuggingFace model_type this applies to, or "*" for all
            exclusive: If True, cannot combine with other adaptations
            required_for_export: If True, this adaptation is enforced when the
                model will be exported. Skipped when an exclusive adaptation
                (which owns the full pipeline) is already selected.

        Usage:
            @YAMLConfigParser.register_adaptation("SHA", model_type="llama")
            class LlamaSHAMixin:
                ...

            @YAMLConfigParser.register_adaptation("AIHM", model_type="*", exclusive=True)
            class AIHMMixin:
                ...
        """

        def decorator(mixin_cls):
            info = AdaptationInfo(
                mixin_cls=mixin_cls,
                exclusive=exclusive,
                required_for_export=required_for_export,
            )
            cls.adaptation_lookup[(model_type, adaptation_name)] = info
            return mixin_cls

        return decorator

    # ========================
    # Model registration (for VLMs and special models)
    # ========================

    @classmethod
    def register_model(cls, model_type: str):
        """Register a model class by its model_type.

        Used for VLMs and other special models that need custom handling.
        Regular LLMs use the default LLM class instead.

        Args:
            model_type: HuggingFace model_type this class handles (e.g., "qwen2_vl")

        Usage:
            @YAMLConfigParser.register_model("qwen2_vl")
            class Qwen_25_VL_Torch(VLM):
                ...
        """

        def decorator(model_cls):
            if model_type in cls.model_lookup:
                existing_cls = cls.model_lookup[model_type]
                raise RuntimeError(
                    f"Model type '{model_type}' is already registered by {existing_cls.__name__}. "
                    f"Cannot register {model_cls.__name__} under the same model_type."
                )
            cls.model_lookup[model_type] = model_cls
            return model_cls

        return decorator

    # ========================
    # Other registrations
    # ========================

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

    # ========================
    # Lookups
    # ========================

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
    def detect_model_type(cls, model_id: str) -> str:
        """Detect model type from HuggingFace model_id or local checkpoint."""
        try:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            return config.model_type
        except Exception as e:
            raise ValueError(
                f"Could not detect model type from '{model_id}'. "
                f"Ensure the model_id is valid or config.json is present in checkpoint. "
                f"Error: {e}"
            )

    # ========================
    # Adaptation handling
    # ========================

    @classmethod
    def _get_adaptation_info(
        cls, model_type: str, adaptation_name: str
    ) -> AdaptationInfo:
        """Find the AdaptationInfo for this model type and adaptation."""
        # Try model-specific first, then universal
        if (model_type, adaptation_name) in cls.adaptation_lookup:
            return cls.adaptation_lookup[(model_type, adaptation_name)]
        if ("*", adaptation_name) in cls.adaptation_lookup:
            return cls.adaptation_lookup[("*", adaptation_name)]

        raise LookupError(
            f"No '{adaptation_name}' adaptation registered for model_type='{model_type}'."
        )

    @staticmethod
    def _normalize_adaptations(
        adaptations_raw: list,
    ) -> tuple[list[str], dict[str, dict]]:
        """Normalize a mixed list of adaptation entries.

        Each entry can be:
        - A string:  ``"SHA"``
        - A single-key dict:  ``{"AttentionMaskScale": {"layer_multipliers": {0: 0.8}}}``

        Returns:
            (adaptation_names, adaptation_kwargs) where *adaptation_kwargs*
            maps adaptation name → dict of keyword arguments.
        """
        names: list[str] = []
        kwargs: dict[str, dict] = {}
        for entry in adaptations_raw:
            if isinstance(entry, str):
                names.append(entry)
            elif isinstance(entry, dict):
                if len(entry) != 1:
                    raise ValueError(
                        "Dict adaptation entry must have exactly one key "
                        f"(the adaptation name), got {list(entry.keys())}"
                    )
                name, params = next(iter(entry.items()))
                names.append(name)
                kwargs[name] = params if isinstance(params, dict) else {}
            else:
                raise ValueError(
                    f"Each adaptation must be a string or single-key dict, "
                    f"got {type(entry)}"
                )
        return names, kwargs

    @classmethod
    def _validate_adaptations(cls, model_type: str, adaptations: list[str]):
        """Validate that the requested adaptations are compatible."""
        if not adaptations:
            return

        for name in adaptations:
            info = cls._get_adaptation_info(model_type, name)
            if info.exclusive and len(adaptations) > 1:
                raise ValueError(
                    f"Adaptation '{name}' is exclusive and cannot be combined with "
                    f"other adaptations: {[a for a in adaptations if a != name]}"
                )

    @classmethod
    def get_required_export_adaptations(cls, model_type: str) -> list[str]:
        """Return non-exclusive adaptation names required for export for a model_type."""
        return [
            name
            for (mt, name), info in cls.adaptation_lookup.items()
            if mt == model_type and info.required_for_export and not info.exclusive
        ]

    @classmethod
    def get_model_class(
        cls,
        model_type: str,
        adaptations: Optional[list[str]] = None,
        adaptation_kwargs: Optional[dict[str, dict]] = None,
    ) -> type:
        """Get the final model class with adaptations applied.

        First checks model_lookup for a registered model (e.g., VLMs),
        then falls back to the default LLM class.

        Args:
            model_type: HuggingFace model_type string.
            adaptations: List of adaptation names.
            adaptation_kwargs: Per-adaptation keyword arguments to set as
                class attributes on the dynamically-created class.  Keyed
                by adaptation name.
        """
        if adaptation_kwargs is None:
            adaptation_kwargs = {}

        # Check for registered model first (VLMs, special models)
        if model_type in cls.model_lookup:
            base_cls = cls.model_lookup[model_type]
        else:
            # Fall back to default LLM
            base_cls = cls.get_default_llm()

        if not adaptations:
            return base_cls

        cls._validate_adaptations(model_type, adaptations)

        # Apply each adaptation
        result_cls = base_cls
        for adaptation_name in adaptations:
            info = cls._get_adaptation_info(model_type, adaptation_name)
            new_name = f"{result_cls.__name__}_{adaptation_name}"
            cls_dict = adaptation_kwargs.get(adaptation_name, {})
            result_cls = type(new_name, (info.mixin_cls, result_cls), cls_dict)

        return result_cls

    # ========================
    # Config validation & parsing
    # ========================

    @classmethod
    def validate_config(cls, doc):
        if "model" not in doc:
            raise RuntimeError("Model section not specified.")
        if "metrics" not in doc:
            raise RuntimeError("Metrics not specified.")

        if not isinstance(doc["model"], dict):
            raise RuntimeError(
                "Multiple models cannot be specified in a single document."
            )
        if "recipe" in doc and not isinstance(doc["recipe"], (dict, list)):
            raise RuntimeError("Recipe must be a dict or list.")

        if "model_id" not in doc["model"]:
            raise RuntimeError("Model 'model_id' not specified.")
        if "sequence_length" not in doc["model"]:
            raise RuntimeError("Sequence length not specified.")
        sl = doc["model"]["sequence_length"]
        if not isinstance(sl, (int, list)):
            raise RuntimeError("sequence_length must be an int or list of ints.")
        if isinstance(sl, list) and not all(isinstance(x, int) for x in sl):
            raise RuntimeError("sequence_length must be an int or list of ints.")
        if "context_length" not in doc["model"]:
            raise RuntimeError("Context length not specified.")

        # Normalize single recipe to component format
        if "recipe" in doc:
            if isinstance(doc["recipe"], list):
                # Top-level list of recipe steps → treat as backbone chain
                doc["recipe"] = {"backbone": doc["recipe"]}
            elif isinstance(doc["recipe"], dict):
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

            # Normalize each component's value: dict → [dict], list stays as-is
            for comp_name in list(doc["recipe"].keys()):
                comp_val = doc["recipe"][comp_name]
                if isinstance(comp_val, dict):
                    doc["recipe"][comp_name] = [comp_val]
                elif isinstance(comp_val, list):
                    for step in comp_val:
                        if not isinstance(step, dict) or "name" not in step:
                            raise RuntimeError(
                                f"Each recipe step in '{comp_name}' must be a dict with a 'name' key."
                            )
                else:
                    raise RuntimeError(
                        f"Recipe component '{comp_name}' must be a dict or list."
                    )

            # Auto-insert Calibration if no Calibration or RemoveQuantization is present
            _TERMINAL_RECIPES = {"Calibration", "RemoveQuantization", "Skip"}
            for comp_name, recipe_list in doc["recipe"].items():
                step_names = {step["name"] for step in recipe_list}
                if not step_names & _TERMINAL_RECIPES:
                    recipe_list.append(
                        {
                            "name": "Calibration",
                            "dataset": {"name": "Wikitext", "split": "train"},
                        }
                    )
                    warnings.warn(
                        f"\n"
                        f"{'=' * 70}\n"
                        f"  AUTO-INSERTED Calibration step for '{comp_name}'\n"
                        f"\n"
                        f"  No Calibration, RemoveQuantization, or Skip recipe was found\n"
                        f"  in the '{comp_name}' recipe chain. A Calibration step using\n"
                        f"  the Wikitext dataset (split=train) has been automatically\n"
                        f"  appended to ensure activation encodings are computed.\n"
                        f"\n"
                        f"  To suppress this, add an explicit Calibration or\n"
                        f"  RemoveQuantization step to your config.\n"
                        f"{'=' * 70}",
                        stacklevel=2,
                    )

        # Backward compatibility: migrate top-level dataset into backbone component
        if "dataset" in doc:
            first_backbone = doc["recipe"]["backbone"][0]
            if "dataset" not in first_backbone:
                first_backbone["dataset"] = doc.pop("dataset")
            else:
                doc.pop("dataset")  # Component has its own dataset, discard top-level

        metrics = (
            doc["metrics"] if isinstance(doc["metrics"], list) else [doc["metrics"]]
        )
        for metric in metrics:
            if "name" not in metric:
                raise RuntimeError("Metric name not specified.")

    @classmethod
    def parse_document(cls, doc, export_base_dir="GenAILab/artifacts/exports"):
        cls.validate_config(doc)
        task_params = {}

        task_params["export"] = doc.pop("export", False)
        if not isinstance(task_params["export"], (bool, str)):
            raise ValueError("Export field must be a boolean or a string path.")

        task_params["eval_in_onnx"] = doc.pop("eval_in_onnx", False)
        if not isinstance(task_params["eval_in_onnx"], bool):
            raise ValueError("eval_in_onnx field must be a boolean value.")

        task_params["run_group"] = doc.pop("run_group", None)

        if task_params["eval_in_onnx"] and not task_params["export"]:
            warnings.warn(
                "eval_in_onnx is enabled, but export is disabled. Overriding export to True."
            )

        if task_params["export"] or task_params["eval_in_onnx"]:
            task_params["export"] = (
                get_test_artifacts_path(doc, base_dir=export_base_dir)
                if isinstance(task_params["export"], bool)
                else task_params["export"]
            )
            Path(task_params["export"]).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(task_params["export"], "config.yaml"), "w") as file:
                yaml.dump(doc, file)

        # Model setup
        model_id = doc["model"]["model_id"]
        adaptations_raw = doc["model"].pop("adaptations", [])
        adaptation_names, adaptation_kwargs = cls._normalize_adaptations(
            adaptations_raw
        )
        model_type = cls.detect_model_type(model_id)

        try:
            model_cls = cls.get_model_class(
                model_type, adaptation_names, adaptation_kwargs
            )
            task_params["model"] = doc.pop("model")
            task_params["model"]["class"] = model_cls
            task_params["model"]["model_type"] = model_type
            task_params["model"]["adaptations"] = (
                adaptations_raw  # Preserve original form for profiler output
            )
        except LookupError as exc:
            raise LookupError(
                f"Failed to configure model for model_id='{model_id}', "
                f"adaptations={adaptation_names}."
            ) from exc

        # Check that required export adaptations are present.
        # Skip when an exclusive adaptation (e.g. AIHM) is selected — it owns
        # the full pipeline including export.
        # A local directory as model_id means the model is already exported.
        is_local_checkpoint = os.path.isdir(model_id)
        will_export = task_params["export"] or (
            "ONNX" in cls.get_default_llm().__name__ and not is_local_checkpoint
        )
        has_exclusive = any(
            cls._get_adaptation_info(model_type, a).exclusive for a in adaptation_names
        )
        if will_export and not has_exclusive:
            required = cls.get_required_export_adaptations(model_type)
            missing = [a for a in required if a not in adaptation_names]
            if missing:
                raise ValueError(
                    f"ONNX export for model_type '{model_type}' requires the "
                    f"following adaptation(s): {missing}.\n"
                    f"Add them under 'model.adaptations' in your YAML config:\n"
                    f"  model:\n"
                    f"    model_id: {model_id}\n"
                    f"    adaptations:\n" + "".join(f"      - {a}\n" for a in missing)
                )

        # Precision config
        precision = PrecisionConfig.from_dict(doc.pop("precision", None))
        task_params["precision"] = precision

        # Recipe parsing — each component is a list of recipe steps
        task_params["recipe"] = {}
        if "recipe" in doc:
            for component_name, recipe_list in doc["recipe"].items():
                parsed_steps = []
                for step_config in recipe_list:
                    recipe_name = step_config["name"]
                    try:
                        recipe_cls = cls.get_recipe(recipe_name)
                    except LookupError as exc:
                        raise LookupError(
                            f"Specified quantization recipe name ({recipe_name}) not found."
                        ) from exc

                    parsed = step_config.copy()
                    parsed["class"] = recipe_cls
                    del parsed["name"]

                    # Parse dataset within step
                    if "dataset" in step_config:
                        dataset_config = parsed["dataset"]
                        dataset_name = dataset_config["name"]
                        try:
                            dataset_cls = cls.get_dataset(dataset_name)
                            dataset_config["class"] = dataset_cls
                            del dataset_config["name"]
                        except LookupError as exc:
                            raise LookupError(
                                f"Specified dataset name ({dataset_name}) not found."
                            ) from exc

                    parsed_steps.append(parsed)
                task_params["recipe"][component_name] = parsed_steps
            del doc["recipe"]
        else:
            has_encodings = "encodings" in task_params.get("model", {})
            default_recipe = "Skip" if has_encodings else "RemoveQuantization"
            task_params["recipe"] = {
                "backbone": [{"class": cls.get_recipe(default_recipe)}],
                "visual": [{"class": cls.get_recipe(default_recipe)}],
            }

        # Validate: if SpinQuant is in backbone recipes and a visual component
        # exists, SpinQuant must also be in the visual recipes. The R1 rotation
        # on the decoder stack changes the expected input distribution, so the
        # merger's post-MLP Hadamard rotation must also be applied.
        backbone_recipe_names = {
            step["class"].__name__ for step in task_params["recipe"].get("backbone", [])
        }
        if "SpinQuant" in backbone_recipe_names and "visual" in task_params["recipe"]:
            visual_steps = task_params["recipe"]["visual"]
            visual_recipe_names = {step["class"].__name__ for step in visual_steps}
            if "SpinQuant" not in visual_recipe_names:
                raise RuntimeError(
                    "SpinQuant is specified for the backbone but not the visual component. "
                    "When using SpinQuant on a VLM, it must be applied to both the backbone "
                    "and visual components to maintain consistency between the decoder stack "
                    "and the vision encoder merger layers."
                )

            # For aimet-onnx, the backbone SpinQuant step applies the visual rotation as a side effect,
            # and the entire backbone runs before any visual steps execute.
            # TODO: Remove this check once individual APIs are invoked for each recipe in aimet-onnx.
            first_spinquant_idx = next(
                i
                for i, step in enumerate(visual_steps)
                if step["class"].__name__ == "SpinQuant"
            )
            if first_spinquant_idx > 0:
                steps_before = [
                    step["class"].__name__
                    for step in visual_steps[:first_spinquant_idx]
                ]
                raise RuntimeError(
                    f"SpinQuant must be the first step in the visual recipe, but found "
                    f"{steps_before} before it."
                )

        # Metrics parsing
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
    def parse(cls, filename, export_base_dir="GenAILab/artifacts/exports"):
        print(filename)
        with open(filename, "r") as file:
            docs = yaml.safe_load_all(file)
            for doc in docs:
                yield cls.parse_document(doc, export_base_dir=export_base_dir)

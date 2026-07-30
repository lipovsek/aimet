# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Declarative schema for GenAI quant configs (recipe + precision)."""

from __future__ import annotations

from .dataset import (
    AOKVQASpec,
    C4Spec,
    DatasetSpec,
    GeneratedDatasetSpec,
    InterleavedSpec,
    MMLUProSpec,
    MMLUSpec,
    MMMLUSpec,
    MMMUSpec,
    TinyMMLUSpec,
    WikitextSpec,
    dataset_name_of,
    dataset_names,
    spec_for_dataset,
)
from .precision import (
    Granularity,
    PrecisionSchema,
    QType,
    QTypeRef,
    VisualPrecisionSchema,
    WeightPrecisionSchema,
)
from .recipe import (
    FP_WEIGHT_ALLOWED_TECHNIQUES,
    ON_SIM_TECHNIQUES,
    PRE_SIM_TECHNIQUES,
    TERMINAL_TECHNIQUES,
    AdaScaleSpec,
    CalibrationSpec,
    ClipSpec,
    Phase,
    Recipe,
    RemoveQuantizationSpec,
    SeqMSESpec,
    SkipSpec,
    SpinQuantSpec,
    TechniqueSpec,
    contract_mismatch,
    has_pre_sim,
    pre_sim_flags,
    pre_sim_identity,
    spec_for_technique,
    spec_kwargs,
    split_recipe,
    technique_name_of,
    technique_names,
)

__all__ = [
    # precision
    "PrecisionSchema",
    "WeightPrecisionSchema",
    "VisualPrecisionSchema",
    "Granularity",
    "QType",
    "QTypeRef",
    # dataset specs + vocabulary
    "DatasetSpec",
    "WikitextSpec",
    "TinyMMLUSpec",
    "MMLUSpec",
    "MMLUProSpec",
    "MMMLUSpec",
    "MMMUSpec",
    "C4Spec",
    "AOKVQASpec",
    "GeneratedDatasetSpec",
    "InterleavedSpec",
    "dataset_names",
    "dataset_name_of",
    "spec_for_dataset",
    # technique specs + vocabulary
    "TechniqueSpec",
    "SpinQuantSpec",
    "RemoveQuantizationSpec",
    "SkipSpec",
    "ClipSpec",
    "CalibrationSpec",
    "SeqMSESpec",
    "AdaScaleSpec",
    "technique_names",
    "technique_name_of",
    "spec_for_technique",
    "spec_kwargs",
    "contract_mismatch",
    # phase + derived sets
    "Phase",
    "PRE_SIM_TECHNIQUES",
    "ON_SIM_TECHNIQUES",
    "TERMINAL_TECHNIQUES",
    "FP_WEIGHT_ALLOWED_TECHNIQUES",
    # recipe
    "Recipe",
    # pre-sim split / cache identity
    "split_recipe",
    "pre_sim_identity",
    "pre_sim_flags",
    "has_pre_sim",
]

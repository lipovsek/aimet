# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring

"""
Baseline Comparison and Reporting

This script:
1. Stores current results as baseline for next run
2. Compares current results with previous baseline
3. Validates quantization accuracy (FP32 vs AIMET)
4. Validates QDQ export correctness (AIMET vs QDQ)
5. Generates GitHub-style markdown report

Usage:
    python baseline_comparison.py store --results reports/results.csv
    python baseline_comparison.py compare --results reports/results.csv --github-summary
    python baseline_comparison.py run --results reports/results.csv --github-summary
    python baseline_comparison.py run --suite-name nightly --github-summary
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TestResult:
    """Single test result."""

    model: str
    feature: str
    fp32_accuracy: float
    aimet_accuracy: float
    qdq_accuracy: float
    qnn_latency_ms: Optional[float] = None
    qnn_accuracy: Optional[float] = None
    techniques: Optional[str] = None
    max_accuracy_drop: float = 1.0
    aimet_runtime_ms: Optional[float] = None
    aimet_memory_mb: Optional[float] = None


@dataclass
class QualityCheck:
    """Quality validation for FP32 → AIMET quantization."""

    model: str
    feature: str
    fp32_acc: float
    aimet_acc: float
    drop_abs: float
    drop_pct: float
    max_accuracy_drop: float = 1.0

    @property
    def status_emoji(self) -> str:
        """Get status indicator based on quantization quality."""
        if self.is_acceptable:
            return "✅"
        return "❌"

    @property
    def is_acceptable(self) -> bool:
        """Check if quantization quality is within allowed threshold.

        Only fails if accuracy dropped beyond the threshold.
        Improvements (positive drop_abs) always pass.

        drop_abs = aimet_accuracy - fp32_accuracy
          - Negative: accuracy dropped (bad)
          - Positive: accuracy improved (good)
        """
        return self.drop_abs >= -self.max_accuracy_drop

    @property
    def formatted_drop(self) -> str:
        """
        Format accuracy as side-by-side percentages with difference.

        Returns:
            Formatted string like: "84.535% / 84.331% (-0.204%) ✅"
            Shows FP32 accuracy / AIMET accuracy (difference) with status emoji

        Note: Accuracy values are already in percentage format (0-100 range),
              so we display them directly without multiplying by 100.
        """

        return f"{self.fp32_acc:.3f}% / {self.aimet_acc:.3f}% ({self.drop_abs:+.3f}%) {self.status_emoji}"


@dataclass
class ExportValidation:
    """Validation for AIMET → QDQ export correctness."""

    model: str
    feature: str
    aimet_acc: float
    qdq_acc: float
    diff_abs: float
    diff_pct: float

    @property
    def status_emoji(self) -> str:
        """Get emoji based on export validation."""
        abs_diff = abs(self.diff_abs)
        if abs_diff < 0.5:  # Less than 0.5 percentage points
            return "✅"
        elif abs_diff < 1.0:  # Less than 1 percentage point
            return "⚠️"
        else:
            return "❌"

    @property
    def is_valid(self) -> bool:
        """Check if export is valid."""
        return abs(self.diff_abs) < 0.5  # Less than 0.5 percentage points


@dataclass
class Comparison:
    """Comparison between baseline and current."""

    model: str
    feature: str
    baseline: float
    current: float
    diff: float
    diff_pct: float
    techniques: str = ""

    @property
    def is_regression(self) -> bool:
        """Check if this is a regression."""
        return self.diff < -1.0  # More than 1 percentage point drop

    @property
    def is_improvement(self) -> bool:
        """Check if this is an improvement."""
        return self.diff > 1.0  # More than 1 percentage point gain

    @property
    def emoji(self) -> str:
        """Get emoji based on severity."""
        if self.diff < -5.0:
            return "🔴"
        elif self.diff < -1.0:
            return "⚠️"
        elif self.diff > 1.0:
            return "✅"
        else:
            return "➖"

    @property
    def formatted_change(self) -> str:
        """Format change with emoji."""
        if abs(self.diff) < 0.1:
            return "stable ✅"
        return f"{self.diff:+.2f}% {self.emoji}"


@dataclass
class MetricComparison:
    """Comparison for performance metrics (runtime, memory)."""

    model: str
    feature: str
    metric_name: str
    baseline: float
    current: float
    diff_pct: float
    threshold_pct: float

    @property
    def is_regression(self) -> bool:
        """Check if metric increased beyond threshold."""
        return self.diff_pct > self.threshold_pct

    @property
    def emoji(self) -> str:
        """Get emoji based on change severity."""
        if self.diff_pct > self.threshold_pct * 2:
            return "🔴"
        elif self.diff_pct > self.threshold_pct:
            return "⚠️"
        elif self.diff_pct < -self.threshold_pct:
            return "📈"
        else:
            return "✅"

    @property
    def formatted_change(self) -> str:
        """Format the change with baseline and current values."""
        if self.metric_name == "runtime":
            return (
                f"{self.baseline:.1f}ms → {self.current:.1f}ms ({self.diff_pct:+.1f}%)"
            )
        else:
            return (
                f"{self.baseline:.1f}MB → {self.current:.1f}MB ({self.diff_pct:+.1f}%)"
            )


def validate_quantization_quality(result: TestResult) -> QualityCheck:
    """
    Validate FP32 → AIMET quantization quality.

    Args:
        result: Test result with FP32 and AIMET accuracies (in percentage format 0-100)

    Returns:
        QualityCheck with drop metrics and status
    """
    drop_abs = result.aimet_accuracy - result.fp32_accuracy

    drop_pct = (
        (drop_abs / result.fp32_accuracy * 100) if result.fp32_accuracy > 0 else 0
    )

    return QualityCheck(
        model=result.model,
        feature=result.feature,
        fp32_acc=result.fp32_accuracy,
        aimet_acc=result.aimet_accuracy,
        drop_abs=drop_abs,
        drop_pct=drop_pct,
        max_accuracy_drop=result.max_accuracy_drop,
    )


def validate_qdq_export(result: TestResult) -> ExportValidation:
    """
    Validate AIMET → QDQ export correctness.

    Args:
        result: Test result with AIMET and QDQ accuracies (in percentage format 0-100)

    Returns:
        ExportValidation with difference metrics and status
    """
    # Values are already percentages, so difference is in percentage points
    diff_abs = result.qdq_accuracy - result.aimet_accuracy

    # Calculate percentage change
    diff_pct = (
        (diff_abs / result.aimet_accuracy * 100) if result.aimet_accuracy > 0 else 0
    )

    return ExportValidation(
        model=result.model,
        feature=result.feature,
        aimet_acc=result.aimet_accuracy,
        qdq_acc=result.qdq_accuracy,
        diff_abs=diff_abs,
        diff_pct=diff_pct,
    )


def compute_overall_status(
    quality: QualityCheck,
    export_val: ExportValidation,
    baseline_comp: Optional[Comparison] = None,
) -> str:
    """
    Compute overall test status based on quantization quality threshold.

    Status is determined solely by whether the accuracy drop from FP32 to AIMET
    is within the configured max_accuracy_drop threshold.

    Args:
        quality: Quantization quality check
        export_val: QDQ export validation (unused, kept for API compatibility)
        baseline_comp: Optional baseline comparison (unused, kept for API compatibility)

    Returns:
        Status emoji: ✅ PASS (within threshold) / ❌ FAIL (exceeds threshold)
    """
    return quality.status_emoji


class BaselineManager:
    """Manage baseline storage and comparison."""

    def __init__(
        self,
        results_csv: str = "AIMETRegression/reports/results.csv",
        baselines_dir: str = "AIMETRegression/baselines",
    ):
        self.results_csv = Path(results_csv)
        self.baselines_dir = Path(baselines_dir)
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_file = self.baselines_dir / "latest.json"

    def load_current_results(self) -> Dict[str, TestResult]:
        """Load current test results from CSV."""
        if not self.results_csv.exists():
            print(f"❌ Results CSV not found: {self.results_csv}")
            return {}

        print(f"📊 Loading results from: {self.results_csv}")

        results = {}
        with open(self.results_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                techniques = row.get("Techniques", "")
                key = f"{row['Model']}_{row['Feature']}_{techniques}"

                def safe_float(value, default=0.0):
                    try:
                        return float(value or default)
                    except (ValueError, TypeError):
                        return default

                qnn_latency_str = row.get("QNN Latency", "")
                if qnn_latency_str and qnn_latency_str != "None":
                    qnn_latency = safe_float(qnn_latency_str.replace(" ms", ""), None)
                else:
                    qnn_latency = None

                qnn_acc_str = row.get("QNN Accuracy", "")
                if qnn_acc_str and qnn_acc_str != "None":
                    qnn_accuracy = safe_float(qnn_acc_str, None)
                else:
                    qnn_accuracy = None

                qdq_acc = safe_float(
                    row.get("QDQ Accuracy") or row.get("ONNX Accuracy", 0)
                )

                # Parse runtime (format: "123.45 ms" or "1.23 s")
                runtime_str = row.get("AIMET Runtime", "")
                aimet_runtime = None
                if runtime_str and runtime_str != "None":
                    if " ms" in runtime_str:
                        aimet_runtime = safe_float(runtime_str.replace(" ms", ""), None)
                    elif " s" in runtime_str:
                        val = safe_float(runtime_str.replace(" s", ""), None)
                        if val is not None:
                            aimet_runtime = val * 1000

                # Parse memory (format: "456.7 MB")
                memory_str = row.get("AIMET Memory", "")
                aimet_memory = None
                if memory_str and memory_str != "None":
                    aimet_memory = safe_float(memory_str.replace(" MB", ""), None)

                results[key] = TestResult(
                    model=row["Model"],
                    feature=row["Feature"],
                    fp32_accuracy=safe_float(row.get("FP32_accuracy")),
                    aimet_accuracy=safe_float(row.get("AIMET Accuracy")),
                    qdq_accuracy=qdq_acc,
                    qnn_latency_ms=qnn_latency,
                    qnn_accuracy=qnn_accuracy,
                    techniques=techniques,
                    max_accuracy_drop=safe_float(row.get("Max_Accuracy_Drop"), 1.0),
                    aimet_runtime_ms=aimet_runtime,
                    aimet_memory_mb=aimet_memory,
                )

        print(f"✔ Loaded {len(results)} test results from CSV")
        return results

    def save_baseline(self, results: Dict[str, TestResult]) -> None:
        """Save current results as baseline."""
        baseline_data = {}
        for key, result in results.items():
            baseline_data[key] = {
                "model": result.model,
                "feature": result.feature,
                "fp32_accuracy": result.fp32_accuracy,
                "aimet_accuracy": result.aimet_accuracy,
                "qdq_accuracy": result.qdq_accuracy,
                "qnn_latency_ms": result.qnn_latency_ms,
                "qnn_accuracy": result.qnn_accuracy,
                "techniques": result.techniques,
                "aimet_runtime_ms": result.aimet_runtime_ms,
                "aimet_memory_mb": result.aimet_memory_mb,
            }

        with open(self.baseline_file, "w") as f:
            json.dump(baseline_data, f, indent=2)

        print(f"✔ Baseline saved to: {self.baseline_file}")

    def load_baseline(self) -> Dict[str, Dict]:
        """Load baseline results."""
        if not self.baseline_file.exists():
            return {}

        with open(self.baseline_file, "r") as f:
            return json.load(f)

    def compare(
        self,
        current: Dict[str, TestResult],
        baseline: Dict[str, Dict],
    ) -> Tuple[List[Comparison], List[Comparison], List[Comparison]]:
        """
        Compare current results with baseline.

        Returns:
            Tuple of (regressions, improvements, unchanged)
        """
        regressions = []
        improvements = []
        unchanged = []

        for key, curr_result in current.items():
            if key not in baseline:
                continue

            base_acc = baseline[key]["aimet_accuracy"]
            curr_acc = curr_result.aimet_accuracy

            diff = curr_acc - base_acc
            diff_pct = (diff / base_acc * 100) if base_acc > 0 else 0

            comp = Comparison(
                model=curr_result.model,
                feature=curr_result.feature,
                baseline=base_acc,
                current=curr_acc,
                diff=diff,
                diff_pct=diff_pct,
                techniques=curr_result.techniques or "",
            )

            if comp.is_regression:
                regressions.append(comp)
            elif comp.is_improvement:
                improvements.append(comp)
            else:
                unchanged.append(comp)

        return regressions, improvements, unchanged

    def compare_metrics(
        self,
        current: Dict[str, TestResult],
        baseline: Dict[str, Dict],
        runtime_threshold_pct: float = 20.0,
        memory_threshold_pct: float = 15.0,
    ) -> Tuple[List[MetricComparison], List[MetricComparison]]:
        """
        Compare runtime and memory metrics with baseline.

        Args:
            current: Current test results
            baseline: Baseline data
            runtime_threshold_pct: Percentage increase threshold for runtime warnings
            memory_threshold_pct: Percentage increase threshold for memory warnings

        Returns:
            Tuple of (runtime_regressions, memory_regressions)
        """
        runtime_regressions = []
        memory_regressions = []

        for key, curr_result in current.items():
            if key not in baseline:
                continue

            base_data = baseline[key]

            # Compare runtime
            base_runtime = base_data.get("aimet_runtime_ms")
            curr_runtime = curr_result.aimet_runtime_ms
            if base_runtime and curr_runtime and base_runtime > 0:
                diff_pct = ((curr_runtime - base_runtime) / base_runtime) * 100
                comp = MetricComparison(
                    model=curr_result.model,
                    feature=curr_result.feature,
                    metric_name="runtime",
                    baseline=base_runtime,
                    current=curr_runtime,
                    diff_pct=diff_pct,
                    threshold_pct=runtime_threshold_pct,
                )
                if comp.is_regression:
                    runtime_regressions.append(comp)

            # Compare memory
            base_memory = base_data.get("aimet_memory_mb")
            curr_memory = curr_result.aimet_memory_mb
            if base_memory and curr_memory and base_memory > 0:
                diff_pct = ((curr_memory - base_memory) / base_memory) * 100
                comp = MetricComparison(
                    model=curr_result.model,
                    feature=curr_result.feature,
                    metric_name="memory",
                    baseline=base_memory,
                    current=curr_memory,
                    diff_pct=diff_pct,
                    threshold_pct=memory_threshold_pct,
                )
                if comp.is_regression:
                    memory_regressions.append(comp)

        return runtime_regressions, memory_regressions


class ReportGenerator:
    """Generate markdown reports for baseline comparison."""

    @staticmethod
    def generate_markdown(
        current: Dict[str, TestResult],
        baseline: Dict[str, Dict],
        regressions: List[Comparison],
        improvements: List[Comparison],
        unchanged: List[Comparison],
        runtime_regressions: Optional[List[MetricComparison]] = None,
        memory_regressions: Optional[List[MetricComparison]] = None,
        baseline_source: Optional[str] = None,
    ) -> str:
        """Generate markdown report with quality and export validations."""
        lines = []

        has_qnn_data = any(r.qnn_latency_ms is not None for r in current.values())

        lines.append("## 📊 Results Comparison\n")

        if baseline_source:
            lines.append(f"**Baseline source:** `{baseline_source}`\n")

        if not baseline:
            lines.append("### ℹ️  First Run - No Baseline\n")
            lines.append("Showing quantization accuracy checks:\n")

            if has_qnn_data:
                lines.append(
                    "| Model | Technique | FP32 vs AIMET | Max Allowed Drop | Status | QNN Acc | Latency |"
                )
                lines.append(
                    "|-------|-----------|---------------|------------------|--------|---------|---------|"
                )
            else:
                lines.append(
                    "| Model | Technique | FP32 vs AIMET | Max Allowed Drop | Status |"
                )
                lines.append(
                    "|-------|-----------|---------------|------------------|--------|"
                )

            for result in current.values():
                quality = validate_quantization_quality(result)
                technique = result.techniques or ""

                row = (
                    f"| {result.model} | {technique} | "
                    f"{result.fp32_accuracy:.2f}% / {result.aimet_accuracy:.2f}% ({quality.drop_abs:+.2f}%) | "
                    f"{result.max_accuracy_drop:.2f}% | "
                    f"{quality.status_emoji} |"
                )

                if has_qnn_data:
                    qnn_acc = (
                        f"{result.qnn_accuracy:.2f}%" if result.qnn_accuracy else "N/A"
                    )
                    qnn_lat = (
                        f"{result.qnn_latency_ms:.1f} ms"
                        if result.qnn_latency_ms
                        else "N/A"
                    )
                    row = row[:-1] + f" | {qnn_acc} | {qnn_lat} |"

                lines.append(row)

            lines.append("")

            legend = (
                "\n**Legend:**\n"
                "- **Technique**: Quantization method and parameters\n"
                "- **FP32 vs AIMET**: Original / quantized accuracy (drop)\n"
                "- **Max Allowed Drop**: Maximum allowed drop from FP32 to AIMET\n"
                "- **Status**: ✅ within threshold | ❌ exceeds threshold\n"
            )
            if has_qnn_data:
                legend += (
                    "- **QNN Acc**: On-device accuracy via AI Hub\n"
                    "- **Latency**: On-device inference latency\n"
                )
            lines.append(legend)

        else:
            # Calculate quality status counts
            passed_count = 0
            warning_count = 0
            failed_count = 0

            for result in current.values():
                quality = validate_quantization_quality(result)
                if quality.is_acceptable:
                    passed_count += 1
                else:
                    failed_count += 1

            comparison_header = (
                f"**Baseline Comparison** (vs `{baseline_source}` AIMET accuracy):\n"
                if baseline_source
                else f"**Baseline Comparison** (vs previous run's AIMET accuracy):\n"
            )
            lines.append(
                f"### Summary\n\n"
                + comparison_header
                + f"- ✅ Stable: {len(unchanged)}\n"
                f"- 📈 Improvements: {len(improvements)}\n"
                f"- ⚠️ Regressions: {len(regressions)}\n\n"
                f"**Quantization Status** (AIMET quantization vs FP32 original):\n"
                f"- ✅ Passed: {passed_count} tests (<1% loss)\n"
                f"- ⚠️ Warnings: {warning_count} tests\n"
                f"- ❌ Failed: {failed_count} tests (>1% loss)\n"
            )

            if regressions:
                lines.append("\n### ⚠️ Regressions\n")

                legend = (
                    "**Legend:**\n"
                    "- **Technique**: Quantization method and parameters\n"
                    "- **vs Baseline**: Accuracy change compared to baseline\n"
                    "- **FP32 vs AIMET**: Original / quantized accuracy (drop)\n"
                    "- **Max Allowed Drop**: Maximum allowed drop from FP32 to AIMET\n"
                    "- **Status**: ✅ within threshold | ❌ exceeds threshold\n"
                )
                if has_qnn_data:
                    legend += (
                        "- **QNN Acc**: On-device accuracy via AI Hub\n"
                        "- **Latency**: On-device inference latency\n"
                    )
                lines.append(legend + "\n")

                if has_qnn_data:
                    lines.append(
                        "| Model | Technique | Baseline | Current | vs Baseline | FP32 vs AIMET | Max Allowed Drop | Status | QNN Acc | Latency |"
                    )
                    lines.append(
                        "|-------|-----------|----------|---------|-------------|---------------|------------------|--------|---------|---------|"
                    )
                else:
                    lines.append(
                        "| Model | Technique | Baseline | Current | vs Baseline | FP32 vs AIMET | Max Allowed Drop | Status |"
                    )
                    lines.append(
                        "|-------|-----------|----------|---------|-------------|---------------|------------------|--------|"
                    )

                for r in sorted(regressions, key=lambda x: x.diff):
                    key = f"{r.model}_{r.feature}_{r.techniques}"
                    curr_result = current.get(key)
                    if curr_result:
                        quality = validate_quantization_quality(curr_result)
                        export_val = validate_qdq_export(curr_result)
                        overall_status = compute_overall_status(quality, export_val, r)
                        technique = curr_result.techniques or ""
                        threshold = curr_result.max_accuracy_drop
                        fp32_acc = curr_result.fp32_accuracy
                        aimet_acc = curr_result.aimet_accuracy
                        drop = quality.drop_abs
                        qnn_acc = curr_result.qnn_accuracy
                        qnn_lat = curr_result.qnn_latency_ms
                    else:
                        quality = None
                        overall_status = "⚠️"
                        technique = r.techniques or ""
                        threshold = 1.0
                        fp32_acc = 0.0
                        aimet_acc = r.current
                        drop = 0.0
                        qnn_acc = None
                        qnn_lat = None

                    row = (
                        f"| {r.emoji} {r.model} | {technique} | "
                        f"{r.baseline:.2f}% | {r.current:.2f}% | "
                        f"{r.diff:+.2f}% | "
                        f"{fp32_acc:.2f}% / {aimet_acc:.2f}% ({drop:+.2f}%) | "
                        f"{threshold:.2f}% | {overall_status} |"
                    )

                    if has_qnn_data:
                        qnn_acc_str = f"{qnn_acc:.2f}%" if qnn_acc else "N/A"
                        qnn_lat_str = f"{qnn_lat:.1f} ms" if qnn_lat else "N/A"
                        row = row[:-1] + f" | {qnn_acc_str} | {qnn_lat_str} |"

                    lines.append(row)
                lines.append("")

            if improvements:
                lines.append("### 📈 Improvements\n")

                legend = (
                    "**Legend:**\n"
                    "- **Technique**: Quantization method and parameters\n"
                    "- **vs Baseline**: Accuracy change compared to baseline\n"
                    "- **FP32 vs AIMET**: Original / quantized accuracy (drop)\n"
                    "- **Max Allowed Drop**: Maximum allowed drop from FP32 to AIMET\n"
                    "- **Status**: ✅ within threshold | ❌ exceeds threshold\n"
                )
                if has_qnn_data:
                    legend += (
                        "- **QNN Acc**: On-device accuracy via AI Hub\n"
                        "- **Latency**: On-device inference latency\n"
                    )
                lines.append(legend + "\n")

                if has_qnn_data:
                    lines.append(
                        "| Model | Technique | Baseline | Current | vs Baseline | FP32 vs AIMET | Max Allowed Drop | Status | QNN Acc | Latency |"
                    )
                    lines.append(
                        "|-------|-----------|----------|---------|-------------|---------------|------------------|--------|---------|---------|"
                    )
                else:
                    lines.append(
                        "| Model | Technique | Baseline | Current | vs Baseline | FP32 vs AIMET | Max Allowed Drop | Status |"
                    )
                    lines.append(
                        "|-------|-----------|----------|---------|-------------|---------------|------------------|--------|"
                    )

                for r in sorted(improvements, key=lambda x: x.diff, reverse=True):
                    key = f"{r.model}_{r.feature}_{r.techniques}"
                    curr_result = current.get(key)
                    if curr_result:
                        quality = validate_quantization_quality(curr_result)
                        export_val = validate_qdq_export(curr_result)
                        overall_status = compute_overall_status(quality, export_val, r)
                        technique = curr_result.techniques or ""
                        threshold = curr_result.max_accuracy_drop
                        fp32_acc = curr_result.fp32_accuracy
                        aimet_acc = curr_result.aimet_accuracy
                        drop = quality.drop_abs
                        qnn_acc = curr_result.qnn_accuracy
                        qnn_lat = curr_result.qnn_latency_ms
                    else:
                        quality = None
                        overall_status = "✅"
                        technique = r.techniques or ""
                        threshold = 1.0
                        fp32_acc = 0.0
                        aimet_acc = r.current
                        drop = 0.0
                        qnn_acc = None
                        qnn_lat = None

                    row = (
                        f"| {r.emoji} {r.model} | {technique} | "
                        f"{r.baseline:.2f}% | {r.current:.2f}% | "
                        f"{r.diff:+.2f}% | "
                        f"{fp32_acc:.2f}% / {aimet_acc:.2f}% ({drop:+.2f}%) | "
                        f"{threshold:.2f}% | {overall_status} |"
                    )

                    if has_qnn_data:
                        qnn_acc_str = f"{qnn_acc:.2f}%" if qnn_acc else "N/A"
                        qnn_lat_str = f"{qnn_lat:.1f} ms" if qnn_lat else "N/A"
                        row = row[:-1] + f" | {qnn_acc_str} | {qnn_lat_str} |"

                    lines.append(row)
                lines.append("")

            if unchanged:
                lines.append("<details>")
                lines.append("<summary>✅ Stable Tests (click to expand)</summary>\n")

                legend = (
                    "**Legend:**\n"
                    "- **Technique**: Quantization method and parameters\n"
                    "- **vs Baseline**: Accuracy change compared to baseline\n"
                    "- **FP32 vs AIMET**: Original / quantized accuracy (drop)\n"
                    "- **Max Allowed Drop**: Maximum allowed drop from FP32 to AIMET\n"
                    "- **Status**: ✅ within threshold | ❌ exceeds threshold\n"
                )
                if has_qnn_data:
                    legend += (
                        "- **QNN Acc**: On-device accuracy via AI Hub\n"
                        "- **Latency**: On-device inference latency\n"
                    )
                lines.append(legend + "\n")

                if has_qnn_data:
                    lines.append(
                        "| Model | Technique | Baseline | Current | vs Baseline | FP32 vs AIMET | Max Allowed Drop | Status | QNN Acc | Latency |"
                    )
                    lines.append(
                        "|-------|-----------|----------|---------|-------------|---------------|------------------|--------|---------|---------|"
                    )
                else:
                    lines.append(
                        "| Model | Technique | Baseline | Current | vs Baseline | FP32 vs AIMET | Max Allowed Drop | Status |"
                    )
                    lines.append(
                        "|-------|-----------|----------|---------|-------------|---------------|------------------|--------|"
                    )

                for r in unchanged:
                    key = f"{r.model}_{r.feature}_{r.techniques}"
                    curr_result = current.get(key)
                    if curr_result:
                        quality = validate_quantization_quality(curr_result)
                        export_val = validate_qdq_export(curr_result)
                        overall_status = compute_overall_status(quality, export_val, r)
                        technique = curr_result.techniques or ""
                        threshold = curr_result.max_accuracy_drop
                        fp32_acc = curr_result.fp32_accuracy
                        aimet_acc = curr_result.aimet_accuracy
                        drop = quality.drop_abs
                        qnn_acc = curr_result.qnn_accuracy
                        qnn_lat = curr_result.qnn_latency_ms
                    else:
                        quality = None
                        overall_status = "✅"
                        technique = r.techniques or ""
                        threshold = 1.0
                        fp32_acc = 0.0
                        aimet_acc = r.current
                        drop = 0.0
                        qnn_acc = None
                        qnn_lat = None

                    row = (
                        f"| {r.model} | {technique} | "
                        f"{r.baseline:.2f}% | {r.current:.2f}% | "
                        f"{r.diff:+.2f}% | "
                        f"{fp32_acc:.2f}% / {aimet_acc:.2f}% ({drop:+.2f}%) | "
                        f"{threshold:.2f}% | {overall_status} |"
                    )

                    if has_qnn_data:
                        qnn_acc_str = f"{qnn_acc:.2f}%" if qnn_acc else "N/A"
                        qnn_lat_str = f"{qnn_lat:.1f} ms" if qnn_lat else "N/A"
                        row = row[:-1] + f" | {qnn_acc_str} | {qnn_lat_str} |"

                    lines.append(row)
                lines.append("</details>\n")

        # Performance metrics section - collapsible table showing all tests
        if baseline and current:
            lines.extend(
                ReportGenerator._generate_performance_metrics_section(current, baseline)
            )

        return "\n".join(lines)

    @staticmethod
    def _generate_performance_metrics_section(
        current: Dict[str, TestResult],
        baseline: Dict[str, Dict],
    ) -> List[str]:
        """Generate performance metrics section with runtime and memory comparisons."""
        lines = []
        lines.append("<details>")
        lines.append("<summary>⏱️ Performance Metrics (click to expand)</summary>\n")
        lines.append(
            "**Legend:**\n"
            "- **Runtime**: AIMET quantization time\n"
            "- **Memory**: Peak GPU memory during quantization\n"
            "- **vs Baseline**: Change from previous run (⚠️ if >20% runtime or >15% memory increase)\n\n"
        )
        lines.append(
            "| Model | Technique | Runtime | vs Baseline | Memory | vs Baseline |"
        )
        lines.append(
            "|-------|-----------|---------|-------------|--------|-------------|"
        )

        for key, result in current.items():
            if key not in baseline:
                continue

            base_data = baseline[key]
            technique = result.techniques or ""

            # Format runtime
            curr_runtime = result.aimet_runtime_ms
            base_runtime = base_data.get("aimet_runtime_ms")
            if curr_runtime is not None:
                runtime_str = f"{curr_runtime:.1f} ms"
                if base_runtime and base_runtime > 0:
                    runtime_diff = ((curr_runtime - base_runtime) / base_runtime) * 100
                    runtime_emoji = "⚠️" if runtime_diff > 20 else "✅"
                    runtime_vs = f"{runtime_diff:+.1f}% {runtime_emoji}"
                else:
                    runtime_vs = "—"
            else:
                runtime_str = "—"
                runtime_vs = "—"

            # Format memory
            curr_memory = result.aimet_memory_mb
            base_memory = base_data.get("aimet_memory_mb")
            if curr_memory is not None:
                memory_str = f"{curr_memory:.1f} MB"
                if base_memory and base_memory > 0:
                    memory_diff = ((curr_memory - base_memory) / base_memory) * 100
                    memory_emoji = "⚠️" if memory_diff > 15 else "✅"
                    memory_vs = f"{memory_diff:+.1f}% {memory_emoji}"
                else:
                    memory_vs = "—"
            else:
                memory_str = "—"
                memory_vs = "—"

            lines.append(
                f"| {result.model} | {technique} | "
                f"{runtime_str} | {runtime_vs} | "
                f"{memory_str} | {memory_vs} |"
            )

        lines.append("</details>\n")
        return lines

    @staticmethod
    def write_github_summary(markdown: str) -> None:
        """Write to GitHub step summary."""
        summary_file = os.getenv("GITHUB_STEP_SUMMARY")
        if not summary_file:
            print("⚠️  GITHUB_STEP_SUMMARY not set (not in GitHub Actions)")
            print("\n" + markdown)
            return

        with open(summary_file, "a") as f:
            f.write("\n" + markdown)

        print("✔ Report written to GitHub step summary")


def _write_summary_json(
    output_path: str,
    current: Dict[str, "TestResult"],
    regressions: list,
    improvements: list,
    unchanged: list,
) -> None:
    """Write a compact JSON summary for the workflow results job."""
    import json

    passed, warnings, failed, failed_tests = 0, 0, 0, []
    for key, result in current.items():
        quality = validate_quantization_quality(result)
        if quality.is_acceptable:
            passed += 1
        else:
            failed += 1
            failed_tests.append(f"{result.model}/{result.techniques}")

    regression_tests = [f"{r.model}/{r.techniques}" for r in regressions]

    summary = {
        "total": len(current),
        "stable": len(unchanged),
        "improvements": len(improvements),
        "regressions": len(regressions),
        "passed": passed,
        "warnings": warnings,
        "failed": failed,
        "failed_tests": failed_tests,
        "regression_tests": regression_tests,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"✔ Summary JSON written to: {out}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare test results with baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "action",
        choices=["store", "compare", "run"],
        help="Action: store baseline, compare with baseline, or both",
    )

    parser.add_argument(
        "--results",
        default=None,
        help="Path to results CSV",
    )

    parser.add_argument(
        "--suite-name",
        dest="suite_name",
        default=None,
        help="Suite name (looks for results_<suite>.csv)",
    )

    parser.add_argument(
        "--baselines-dir",
        dest="baselines_dir",
        default="AIMETRegression/baselines",
        help="Directory for baseline files",
    )

    parser.add_argument(
        "--github-summary",
        dest="github_summary",
        action="store_true",
        help="Write report to GitHub step summary",
    )

    parser.add_argument(
        "--output-file",
        dest="output_file",
        default=None,
        help="Write markdown report to this file (for cross-job sharing)",
    )

    parser.add_argument(
        "--baseline-source",
        dest="baseline_source",
        default=None,
        help="Branch the baseline was downloaded from (shown in report header)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("AIMET Baseline Comparison")
    print("=" * 60)

    if not args.results:
        reports_dir = Path("AIMETRegression/reports")

        if not reports_dir.exists():
            print(f"❌ Reports directory not found: {reports_dir}")
            return 1

        if args.suite_name:
            results_file = reports_dir / f"results_{args.suite_name}.csv"
            if not results_file.exists():
                print(f"❌ Results file not found: {results_file}")
                print(f"\n💡 Available CSV files:")
                for csv_file in sorted(reports_dir.glob("*.csv")):
                    print(f"   - {csv_file.name}")
                return 1
            print(f"ℹ️  Using results for suite: {args.suite_name}")
        else:
            csv_files = list(reports_dir.glob("results*.csv"))

            if not csv_files:
                print(f"❌ No results CSV files found in {reports_dir}")
                return 1
            elif len(csv_files) == 1:
                results_file = csv_files[0]
                print(f"ℹ️  Auto-detected results file: {results_file.name}")
            else:
                if (reports_dir / "results.csv").exists():
                    results_file = reports_dir / "results.csv"
                else:
                    results_file = max(csv_files, key=lambda p: p.stat().st_mtime)
                print(f"ℹ️  Multiple CSV files found, using: {results_file.name}")

        args.results = str(results_file)

    print(f"📄 Results CSV: {args.results}")
    print(f"📁 Baselines dir: {args.baselines_dir}")
    print()

    manager = BaselineManager(args.results, args.baselines_dir)

    current = manager.load_current_results()
    if not current:
        print("❌ No results to process")
        return 1

    print("\n--- Quantization Validation ---")
    quality_issues = []
    export_issues = []

    for key, result in current.items():
        quality = validate_quantization_quality(result)
        export_val = validate_qdq_export(result)

        if not quality.is_acceptable:
            quality_issues.append(
                f"{result.model}/{result.feature}: {quality.formatted_drop}"
            )

        if not export_val.is_valid:
            export_issues.append(
                f"{result.model}/{result.feature}: AIMET={export_val.aimet_acc:.4f}, "
                f"QDQ={export_val.qdq_acc:.4f} (diff: {export_val.diff_abs:+.4f})"
            )

    if quality_issues:
        print(f"⚠️  Quantization Issues ({len(quality_issues)}):")
        for issue in quality_issues:
            print(f"  - {issue}")
        print("\n⚠️  FP32→AIMET quantization check detected issues")
    else:
        print("✅ All tests have acceptable quantization accuracy")

    if export_issues:
        print(f"\n⚠️  Export Validation Issues ({len(export_issues)}):")
        for issue in export_issues:
            print(f"  - {issue}")
        print("\n⚠️  AIMET→QDQ export validation detected issues")
    else:
        print("✅ All QDQ exports validated successfully")

    if args.action in ["store", "run"]:
        print("\n--- Storing Baseline ---")
        manager.save_baseline(current)

    if args.action in ["compare", "run"]:
        print("\n--- Comparing with Baseline ---")
        baseline = manager.load_baseline()

        if baseline:
            regressions, improvements, unchanged = manager.compare(current, baseline)
            runtime_regressions, memory_regressions = manager.compare_metrics(
                current, baseline
            )

            markdown = ReportGenerator.generate_markdown(
                current,
                baseline,
                regressions,
                improvements,
                unchanged,
                runtime_regressions=runtime_regressions,
                memory_regressions=memory_regressions,
                baseline_source=args.baseline_source,
            )

            if args.github_summary:
                ReportGenerator.write_github_summary(markdown)
            else:
                print("\n" + markdown)

            if args.output_file:
                _write_summary_json(
                    args.output_file,
                    current,
                    regressions,
                    improvements,
                    unchanged,
                )

            print(f"\n{'=' * 60}")
            print(f"📊 Comparison Summary")
            print(f"{'=' * 60}")
            print(f"✅ Unchanged:    {len(unchanged)}")
            print(f"📈 Improvements: {len(improvements)}")
            print(f"⚠️ Regressions:  {len(regressions)}")

            if runtime_regressions or memory_regressions:
                print(f"\n⏱️  Performance Warnings:")
                if runtime_regressions:
                    print(
                        f"   Runtime: {len(runtime_regressions)} test(s) exceeded threshold"
                    )
                if memory_regressions:
                    print(
                        f"   Memory:  {len(memory_regressions)} test(s) exceeded threshold"
                    )

            print(f"{'=' * 60}")

            if regressions:
                print(f"\n⚠️  {len(regressions)} regression(s) detected")
            else:
                print(f"\n✅ All tests passed or within threshold!")
        else:
            print("ℹ️  First run - no baseline to compare")
            markdown = ReportGenerator.generate_markdown(
                current,
                {},
                [],
                [],
                [],
                baseline_source=args.baseline_source,
            )

            if args.github_summary:
                ReportGenerator.write_github_summary(markdown)
            else:
                print("\n" + markdown)

            if args.output_file:
                _write_summary_json(
                    args.output_file,
                    current,
                    regressions=[],
                    improvements=[],
                    unchanged=[],
                )

            print("\nℹ️  Baseline saved. Next run will compare against this baseline.")

    print("\n✅ Baseline operations completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())

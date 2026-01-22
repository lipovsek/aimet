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
    techniques: Optional[str] = None


@dataclass
class QualityCheck:
    """Quality validation for FP32 → AIMET quantization."""

    model: str
    feature: str
    fp32_acc: float
    aimet_acc: float
    drop_abs: float
    drop_pct: float

    @property
    def status_emoji(self) -> str:
        """Get emoji based on quantization quality."""
        abs_drop = abs(self.drop_abs)
        if abs_drop < 1.0:  # Less than 1 percentage point
            return "✅"
        else:
            return "⚠️"

    @property
    def is_acceptable(self) -> bool:
        """Check if quantization quality is acceptable."""
        return abs(self.drop_abs) < 1.0  # Less than 1 percentage point

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


def validate_quantization_quality(result: TestResult) -> QualityCheck:
    """
    Validate FP32 → AIMET quantization quality.

    Args:
        result: Test result with FP32 and AIMET accuracies (in percentage format 0-100)

    Returns:
        QualityCheck with drop metrics and status
    """
    # Values are already percentages, so difference is in percentage points
    drop_abs = result.aimet_accuracy - result.fp32_accuracy

    # Calculate percentage change (e.g., 2% drop from 80% = -2.5% change)
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
    Compute overall test status based on all validations.

    Args:
        quality: Quantization quality check
        export_val: QDQ export validation
        baseline_comp: Optional baseline comparison

    Returns:
        Status emoji: ✅ PASS / ⚠️ WARNING / ❌ FAIL
    """
    if not quality.is_acceptable:
        return "❌"

    if not export_val.is_valid:
        return "❌"

    if baseline_comp and baseline_comp.diff < -5.0:
        return "❌"

    if quality.status_emoji == "⚠️" or export_val.status_emoji == "⚠️":
        return "⚠️"

    if baseline_comp and baseline_comp.is_regression:
        return "⚠️"

    return "✅"


class BaselineManager:
    """Manage baseline storage and comparison."""

    def __init__(
        self,
        results_csv: str = "ONNXRegression/reports/results.csv",
        baselines_dir: str = "ONNXRegression/baselines",
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
                key = f"{row['Model']}_{row['Feature']}"

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

                qdq_acc = safe_float(
                    row.get("QDQ Accuracy") or row.get("ONNX Accuracy", 0)
                )

                results[key] = TestResult(
                    model=row["Model"],
                    feature=row["Feature"],
                    fp32_accuracy=safe_float(row.get("FP32_accuracy")),
                    aimet_accuracy=safe_float(row.get("AIMET Accuracy")),
                    qdq_accuracy=qdq_acc,
                    qnn_latency_ms=qnn_latency,
                    techniques=row.get("Techniques", ""),
                )

        print(f"✓ Loaded {len(results)} test results from CSV")
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
                "techniques": result.techniques,
            }

        with open(self.baseline_file, "w") as f:
            json.dump(baseline_data, f, indent=2)

        print(f"✓ Baseline saved to: {self.baseline_file}")

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

            # Values are already percentages, so diff is in percentage points
            diff = curr_acc - base_acc
            diff_pct = (diff / base_acc * 100) if base_acc > 0 else 0

            comp = Comparison(
                model=curr_result.model,
                feature=curr_result.feature,
                baseline=base_acc,
                current=curr_acc,
                diff=diff,
                diff_pct=diff_pct,
            )

            if comp.is_regression:
                regressions.append(comp)
            elif comp.is_improvement:
                improvements.append(comp)
            else:
                unchanged.append(comp)

        return regressions, improvements, unchanged


class ReportGenerator:
    """Generate markdown reports for baseline comparison."""

    @staticmethod
    def generate_markdown(
        current: Dict[str, TestResult],
        baseline: Dict[str, Dict],
        regressions: List[Comparison],
        improvements: List[Comparison],
        unchanged: List[Comparison],
    ) -> str:
        """Generate markdown report with quality and export validations."""
        lines = []

        lines.append("## 📊 Results Comparison\n")

        if not baseline:
            lines.append("### ℹ️  First Run - No Baseline\n")
            lines.append("Showing quantization accuracy checks:\n")
            lines.append(
                "| Model | Feature | Config | FP32 | AIMET | Accuracy (FP32/AIMET) | QDQ | Export Status |"
            )
            lines.append(
                "|-------|---------|--------|------|-------|----------------------|-----|---------------|"
            )

            for result in current.values():
                quality = validate_quantization_quality(result)
                export_val = validate_qdq_export(result)
                config = result.techniques or ""

                # Values are already percentages, display directly
                lines.append(
                    f"| {result.model} | {result.feature} | {config} | "
                    f"{result.fp32_accuracy:.3f}% | {result.aimet_accuracy:.3f}% | "
                    f"{quality.formatted_drop} | {result.qdq_accuracy:.3f}% | "
                    f"{export_val.status_emoji} |"
                )

            lines.append("")
            lines.append(
                "\n**Legend:**\n"
                "- Config: Quantization techniques and parameters applied\n"
                "- Accuracy (FP32/AIMET): Compares original vs quantized accuracy\n"
                "  - ✅ Within 1% of FP32 | ⚠️ Over 1% drop from FP32\n"
                "- Export Status: ✅ <0.5pp diff | ⚠️ 0.5-1pp diff | ❌ >1pp diff\n"
            )

        else:
            # Calculate quality status counts
            passed_count = 0
            warning_count = 0
            failed_count = 0

            for result in current.values():
                quality = validate_quantization_quality(result)
                export_val = validate_qdq_export(result)

                # Find if this result has a baseline comparison
                key = f"{result.model}_{result.feature}"
                baseline_comp = None
                for comp in regressions + improvements + unchanged:
                    if comp.model == result.model and comp.feature == result.feature:
                        baseline_comp = comp
                        break

                overall_status = compute_overall_status(
                    quality, export_val, baseline_comp
                )

                if overall_status == "✅":
                    passed_count += 1
                elif overall_status == "⚠️":
                    warning_count += 1
                elif overall_status == "❌":
                    failed_count += 1

            lines.append(
                f"### Summary\n\n"
                f"**Baseline Comparison** (vs previous run's AIMET accuracy):\n"
                f"- ✅ Stable: {len(unchanged)}\n"
                f"- 📈 Improvements: {len(improvements)}\n"
                f"- ⚠️ Regressions: {len(regressions)}\n\n"
                f"**Quantization Status** (AIMET quantization vs FP32 original):\n"
                f"- ✅ Passed: {passed_count} tests (<1% loss)\n"
                f"- ⚠️ Warnings: {warning_count} tests\n"
                f"- ❌ Failed: {failed_count} tests (>1% loss)\n"
            )

            if regressions:
                lines.append("\n### ⚠️ Regressions\n")
                lines.append(
                    "**Legend:**\n"
                    "- Config: Quantization techniques and parameters applied\n"
                    "- Accuracy (FP32/AIMET): Compares original vs quantized accuracy\n"
                    "  - ✅ Within 1% of FP32 | ⚠️ Over 1% drop from FP32\n"
                    "- vs Baseline: Difference from previous run's AIMET accuracy\n"
                    "- Status: Overall test result considering all validation checks\n\n"
                )
                lines.append(
                    "| Model | Feature | Config | Baseline (AIMET) | Current (AIMET) | vs Baseline | Accuracy (FP32/AIMET) | Status |"
                )
                lines.append(
                    "|-------|---------|--------|------------------|-----------------|-------------|----------------------|--------|"
                )
                for r in sorted(regressions, key=lambda x: x.diff):
                    key = f"{r.model}_{r.feature}"
                    curr_result = current.get(key)
                    if curr_result:
                        quality = validate_quantization_quality(curr_result)
                        export_val = validate_qdq_export(curr_result)
                        overall_status = compute_overall_status(quality, export_val, r)
                        config = curr_result.techniques or ""
                    else:
                        quality = None
                        overall_status = "⚠️"
                        config = ""

                    # Values are already percentages, display directly
                    lines.append(
                        f"| {r.emoji} {r.model} | {r.feature} | {config} | "
                        f"{r.baseline:.3f}% | {r.current:.3f}% | "
                        f"{r.diff:+.5f} ({r.diff_pct:+.1f}%) | "
                        f"{quality.formatted_drop if quality else 'N/A'} | "
                        f"{overall_status} |"
                    )
                lines.append("")

            if improvements:
                lines.append("### 📈 Improvements\n")
                lines.append(
                    "**Legend:**\n"
                    "- Config: Quantization techniques and parameters applied\n"
                    "- Accuracy (FP32/AIMET): Compares original vs quantized accuracy\n"
                    "  - ✅ Within 1% of FP32 | ⚠️ Over 1% drop from FP32\n"
                    "- vs Baseline: Difference from previous run's AIMET accuracy\n"
                    "- Status: Overall test result considering all validation checks\n\n"
                )
                lines.append(
                    "| Model | Feature | Config | Baseline (AIMET) | Current (AIMET) | vs Baseline | Accuracy (FP32/AIMET) | Status |"
                )
                lines.append(
                    "|-------|---------|--------|------------------|-----------------|-------------|----------------------|--------|"
                )
                for r in sorted(improvements, key=lambda x: x.diff, reverse=True):
                    key = f"{r.model}_{r.feature}"
                    curr_result = current.get(key)
                    if curr_result:
                        quality = validate_quantization_quality(curr_result)
                        export_val = validate_qdq_export(curr_result)
                        overall_status = compute_overall_status(quality, export_val, r)
                        config = curr_result.techniques or ""
                    else:
                        quality = None
                        overall_status = "✅"
                        config = ""

                    # Values are already percentages, display directly
                    lines.append(
                        f"| {r.emoji} {r.model} | {r.feature} | {config} | "
                        f"{r.baseline:.3f}% | {r.current:.3f}% | "
                        f"{r.diff:+.5f} ({r.diff_pct:+.1f}%) | "
                        f"{quality.formatted_drop if quality else 'N/A'} | "
                        f"{overall_status} |"
                    )
                lines.append("")

            if unchanged:
                lines.append("<details>")
                lines.append("<summary>✅ Stable Tests (click to expand)</summary>\n")
                lines.append(
                    "**Legend:**\n"
                    "- Config: Quantization techniques and parameters applied\n"
                    "- Accuracy (FP32/AIMET): Compares original vs quantized accuracy\n"
                    "  - ✅ Within 1% of FP32 | ⚠️ Over 1% drop from FP32\n"
                    "- vs Baseline: Difference from previous run's AIMET accuracy\n"
                    "- Status: Overall test result considering all validation checks\n\n"
                )
                lines.append(
                    "| Model | Feature | Config | Baseline (AIMET) | Current (AIMET) | vs Baseline | Accuracy (FP32/AIMET) | Status |"
                )
                lines.append(
                    "|-------|---------|--------|------------------|-----------------|-------------|----------------------|--------|"
                )
                for r in unchanged:
                    key = f"{r.model}_{r.feature}"
                    curr_result = current.get(key)
                    if curr_result:
                        quality = validate_quantization_quality(curr_result)
                        export_val = validate_qdq_export(curr_result)
                        overall_status = compute_overall_status(quality, export_val, r)
                        config = curr_result.techniques or ""
                    else:
                        quality = None
                        overall_status = "✅"
                        config = ""

                    # Values are already percentages, display directly
                    lines.append(
                        f"| {r.model} | {r.feature} | {config} | "
                        f"{r.baseline:.3f}% | {r.current:.3f}% | "
                        f"{r.diff:+.5f} | "
                        f"{quality.formatted_drop if quality else 'N/A'} | "
                        f"{overall_status} |"
                    )
                lines.append("</details>\n")

        return "\n".join(lines)

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

        print("✓ Report written to GitHub step summary")


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
        default="ONNXRegression/baselines",
        help="Directory for baseline files",
    )

    parser.add_argument(
        "--github-summary",
        dest="github_summary",
        action="store_true",
        help="Write report to GitHub step summary",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("AIMET Baseline Comparison")
    print("=" * 60)

    if not args.results:
        reports_dir = Path("ONNXRegression/reports")

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

            markdown = ReportGenerator.generate_markdown(
                current, baseline, regressions, improvements, unchanged
            )

            if args.github_summary:
                ReportGenerator.write_github_summary(markdown)
            else:
                print("\n" + markdown)

            print(f"\n{'=' * 60}")
            print(f"📊 Comparison Summary")
            print(f"{'=' * 60}")
            print(f"✅ Unchanged:    {len(unchanged)}")
            print(f"📈 Improvements: {len(improvements)}")
            print(f"⚠️  Regressions:  {len(regressions)}")
            print(f"{'=' * 60}")

            if regressions:
                print(f"\n⚠️  {len(regressions)} regression(s) detected")
            else:
                print(f"\n✅ All tests passed or within threshold!")
        else:
            print("ℹ️  First run - no baseline to compare")
            markdown = ReportGenerator.generate_markdown(current, {}, [], [], [])

            if args.github_summary:
                ReportGenerator.write_github_summary(markdown)
            else:
                print("\n" + markdown)

            print("\nℹ️  Baseline saved. Next run will compare against this baseline.")

    print("\n✅ Baseline operations completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())

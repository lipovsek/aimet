# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# pylint: disable=missing-module-docstring

"""
Baseline Comparison and Reporting

This script:
1. Stores current results as baseline for next run
2. Compares current results with previous baseline
3. Generates GitHub-style markdown report
4. Exits with error if regressions detected

Usage:
    # Store baseline
    python baseline_comparison.py store

    # Compare and report
    python baseline_comparison.py compare --github-summary

    # Both (typical workflow)
    python baseline_comparison.py run --github-summary
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
    onnx_accuracy: float
    qnn_latency_ms: Optional[float] = None


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
        """Check if this is a regression (>1% drop)."""
        return self.diff < -0.01

    @property
    def is_improvement(self) -> bool:
        """Check if this is an improvement (>1% gain)."""
        return self.diff > 0.01

    @property
    def emoji(self) -> str:
        """Get emoji based on severity."""
        if self.diff < -0.05:
            return "🔴"  # Major regression (>5%)
        elif self.diff < -0.01:
            return "⚠️"  # Minor regression (>1%)
        elif self.diff > 0.01:
            return "✅"  # Improvement
        else:
            return "➖"  # Stable


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
            print(f"No results found at {self.results_csv}")
            return {}

        results = {}
        with open(self.results_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row['Model']}_{row['Feature']}"

                # Parse values with fallbacks
                def safe_float(value, default=0.0):
                    try:
                        return float(value or default)
                    except (ValueError, TypeError):
                        return default

                # Parse QNN latency (remove " ms" suffix)
                qnn_latency_str = row.get("QNN Latency", "")
                if qnn_latency_str and qnn_latency_str != "None":
                    qnn_latency = safe_float(qnn_latency_str.replace(" ms", ""), None)
                else:
                    qnn_latency = None

                results[key] = TestResult(
                    model=row["Model"],
                    feature=row["Feature"],
                    fp32_accuracy=safe_float(row.get("FP32_accuracy")),
                    aimet_accuracy=safe_float(row.get("AIMET Accuracy")),
                    onnx_accuracy=safe_float(row.get("ONNX Accuracy")),
                    qnn_latency_ms=qnn_latency,
                )

        print(f"✓ Loaded {len(results)} test results")
        return results

    def save_baseline(self, results: Dict[str, TestResult]) -> None:
        """Save current results as baseline."""
        baseline = {}
        for key, result in results.items():
            baseline[key] = {
                "fp32_accuracy": result.fp32_accuracy,
                "aimet_accuracy": result.aimet_accuracy,
                "onnx_accuracy": result.onnx_accuracy,
                "qnn_latency_ms": result.qnn_latency_ms,
            }

        with open(self.baseline_file, "w") as f:
            json.dump(baseline, f, indent=2)

        print(f"✓ Baseline saved to {self.baseline_file} ({len(baseline)} entries)")

    def load_baseline(self) -> Dict[str, Dict]:
        """Load previous baseline."""
        if not self.baseline_file.exists():
            print("ℹ️  No baseline found (first run)")
            return {}

        with open(self.baseline_file, "r") as f:
            baseline = json.load(f)

        print(f"✓ Loaded baseline with {len(baseline)} entries")
        return baseline

    def compare(
        self, current: Dict[str, TestResult], baseline: Dict[str, Dict]
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
                print(f"  New test: {key} (no baseline)")
                continue

            base_data = baseline[key]
            aimet_diff = curr_result.aimet_accuracy - base_data["aimet_accuracy"]

            comp = Comparison(
                model=curr_result.model,
                feature=curr_result.feature,
                baseline=base_data["aimet_accuracy"],
                current=curr_result.aimet_accuracy,
                diff=aimet_diff,
                diff_pct=(aimet_diff / base_data["aimet_accuracy"] * 100)
                if base_data["aimet_accuracy"] > 0
                else 0,
            )

            if comp.is_regression:
                regressions.append(comp)
            elif comp.is_improvement:
                improvements.append(comp)
            else:
                unchanged.append(comp)

        return regressions, improvements, unchanged


class ReportGenerator:
    """Generate comparison reports."""

    @staticmethod
    def generate_markdown(
        current: Dict[str, TestResult],
        baseline: Dict[str, Dict],
        regressions: List[Comparison],
        improvements: List[Comparison],
        unchanged: List[Comparison],
    ) -> str:
        """Generate markdown report."""
        lines = []

        lines.append("## 📊 Results Comparison\n")

        if not baseline:
            # First run - no comparison
            lines.append("ℹ️  **First run** - no baseline to compare against\n")
            lines.append("### Current Results\n")
            lines.append("| Model | Feature | FP32 Acc | AIMET Acc | ONNX Acc |")
            lines.append("|-------|---------|----------|-----------|----------|")
            for key, data in sorted(current.items()):
                lines.append(
                    f"| {data.model} | {data.feature} | "
                    f"{data.fp32_accuracy:.3f} | "
                    f"{data.aimet_accuracy:.3f} | "
                    f"{data.onnx_accuracy:.3f} |"
                )
        else:
            # Comparison with baseline
            total = len(regressions) + len(improvements) + len(unchanged)
            lines.append(
                f"**Tests:** {total} | "
                f"✅ Passing: {len(unchanged)} | "
                f"📈 Improved: {len(improvements)} | "
                f"⚠️  Regressed: {len(regressions)}\n"
            )

            # Regressions (most important!)
            if regressions:
                lines.append("### ⚠️  Regressions Detected\n")
                lines.append("| Model | Feature | Baseline | Current | Change |")
                lines.append("|-------|---------|----------|---------|--------|")
                for r in sorted(regressions, key=lambda x: x.diff):
                    lines.append(
                        f"| {r.emoji} {r.model} | {r.feature} | "
                        f"{r.baseline:.3f} | {r.current:.3f} | "
                        f"**{r.diff:+.3f}** ({r.diff_pct:+.1f}%) |"
                    )
                lines.append("")

            # Improvements
            if improvements:
                lines.append("### 📈 Improvements\n")
                lines.append("| Model | Feature | Baseline | Current | Change |")
                lines.append("|-------|---------|----------|---------|--------|")
                for r in sorted(improvements, key=lambda x: x.diff, reverse=True):
                    lines.append(
                        f"| {r.emoji} {r.model} | {r.feature} | "
                        f"{r.baseline:.3f} | {r.current:.3f} | "
                        f"{r.diff:+.3f} ({r.diff_pct:+.1f}%) |"
                    )
                lines.append("")

            # Unchanged (collapsed)
            if unchanged:
                lines.append("<details>")
                lines.append("<summary>✅ Stable Tests (click to expand)</summary>\n")
                lines.append("| Model | Feature | Baseline | Current | Change |")
                lines.append("|-------|---------|----------|---------|--------|")
                for r in unchanged:
                    lines.append(
                        f"| {r.model} | {r.feature} | "
                        f"{r.baseline:.3f} | {r.current:.3f} | "
                        f"{r.diff:+.3f} |"
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

        print("✓ Report written to GitHub summary")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare test results with baseline")

    parser.add_argument(
        "action",
        choices=["store", "compare", "run"],
        help="Action: store baseline, compare with baseline, or both",
    )

    parser.add_argument(
        "--results",
        default=None,  # Auto-detect if not specified
        help="Path to results CSV (auto-detects if not specified)",
    )

    parser.add_argument(
        "--suite-name",
        dest="suite_name",  # Added: explicit dest for hyphenated arg
        default=None,
        help="Suite name (for finding results_<suite>.csv)",
    )

    parser.add_argument(
        "--baselines-dir",
        default="ONNXRegression/baselines",
        help="Directory for baseline files",
    )

    parser.add_argument(
        "--github-summary",
        action="store_true",
        help="Write report to GitHub step summary",
    )

    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        dest="fail_on_regression",  # Fixed: use underscores for dest
        default=True,
        help="Exit with error if regressions detected (default: True)",
    )

    args = parser.parse_args()

    # Auto-detect results file if not specified
    if not args.results:
        reports_dir = Path("ONNXRegression/reports")

        if args.suite_name:
            # Look for suite-specific results
            results_file = reports_dir / f"results_{args.suite_name}.csv"
            if not results_file.exists():
                print(f"❌ Results file not found: {results_file}")
                print(
                    f"   Looked for suite-specific file based on --suite-name={args.suite_name}"
                )
                return 1
        else:
            # Auto-detect: look for any results_*.csv or results.csv
            csv_files = list(reports_dir.glob("results*.csv"))

            if not csv_files:
                print(f"❌ No results CSV files found in {reports_dir}")
                return 1
            elif len(csv_files) == 1:
                results_file = csv_files[0]
                print(f"ℹ️  Auto-detected results file: {results_file.name}")
            else:
                # Multiple files - prefer results.csv, otherwise take most recent
                if (reports_dir / "results.csv").exists():
                    results_file = reports_dir / "results.csv"
                else:
                    results_file = max(csv_files, key=lambda p: p.stat().st_mtime)
                print(f"ℹ️  Multiple CSV files found, using: {results_file.name}")

        args.results = str(results_file)

    # Initialize manager
    manager = BaselineManager(args.results, args.baselines_dir)

    # Load current results
    current = manager.load_current_results()
    if not current:
        print("❌ No results to process")
        return 1

    # Action: Store baseline
    if args.action in ["store", "run"]:
        manager.save_baseline(current)

    # Action: Compare
    if args.action in ["compare", "run"]:
        baseline = manager.load_baseline()

        if baseline:
            regressions, improvements, unchanged = manager.compare(current, baseline)

            # Generate report
            markdown = ReportGenerator.generate_markdown(
                current, baseline, regressions, improvements, unchanged
            )

            # Output
            if args.github_summary:
                ReportGenerator.write_github_summary(markdown)
            else:
                print("\n" + markdown)

            # Summary
            print(f"\n{'=' * 60}")
            print(f"Regressions: {len(regressions)}")
            print(f"Improvements: {len(improvements)}")
            print(f"Unchanged: {len(unchanged)}")
            print(f"{'=' * 60}")

            # Exit with error if regressions found
            if regressions and args.fail_on_regression:
                print(f"\n❌ {len(regressions)} regression(s) detected!")
                return 1
            else:
                print(f"\n✅ All tests passed!")
                return 0
        else:
            # No baseline - first run
            markdown = ReportGenerator.generate_markdown(current, {}, [], [], [])

            if args.github_summary:
                ReportGenerator.write_github_summary(markdown)
            else:
                print("\n" + markdown)

            print("\nℹ️  First run - baseline will be available for next run")
            return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# pylint: disable=missing-module-docstring

"""
Suite Runner for AIMET ONNX Regression

This module enables batch execution of multiple model configurations as a suite.
It supports:
- Predefined suites (aimet_only, aimet_plus_ontarget)
- Custom suite files
- Configuration overrides applied to all tests
- Filtering by model or feature name
- Consolidated reporting

"""

import argparse
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Reuse the single-config pipeline
from ONNXRegression.runner import run_single_config


def _load_suite(path: Path) -> Dict:
    """
    Load and validate a suite configuration file.

    Suite files define:
    - include: List of config files to run
    - overrides: Parameters to apply to all configs
    - metadata: Suite name, description, etc.

    Args:
        path: Path to suite YAML file

    Returns:
        Suite configuration dictionary

    Raises:
        ValueError: If suite file is invalid
    """
    with open(path, "r", encoding="utf-8") as f:
        suite = yaml.safe_load(f)

    if not isinstance(suite, dict) or "include" not in suite:
        raise ValueError(
            f"Suite YAML must be a dictionary with an 'include' list. "
            f"Got: {type(suite)}"
        )

    # Set defaults
    suite.setdefault("suite_name", path.stem)
    suite.setdefault("description", "")
    suite.setdefault("overrides", {})

    return suite


def _apply_overrides_to_config_file(
    cfg_path: Path, overrides: Dict, tmpdir: Path
) -> Path:
    """
    Apply suite-level overrides to a configuration file.

    Creates a modified copy of the config with overrides applied.
    Setting an override value to None deletes that key from the config.

    Args:
        cfg_path: Original config file path
        overrides: Dictionary of key-value overrides
        tmpdir: Temporary directory for modified config

    Returns:
        Path to modified config file

    Example:
        Overrides: {"eval_samples": 100, "qnn_options": None}
        Result: Sets eval_samples to 100, removes qnn_options entirely
    """
    # Load original config
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file at {cfg_path}")

    # Apply overrides
    for key, value in overrides.items():
        if value is None:
            # None means delete the key
            if key in config:
                del config[key]
        else:
            # Otherwise set/update the value
            config[key] = value

    # Save modified config to temp directory
    out_path = tmpdir / cfg_path.name
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return out_path


def main():
    """
    Main entry point for suite execution.

    Parses command-line arguments, loads configurations,
    applies overrides, runs tests, and generates consolidated reports.
    """
    # ============ Argument Parsing ============
    parser = argparse.ArgumentParser(
        description="Run a suite of AIMET ONNX regression tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run predefined suite
  python suite_runner.py --suite aimet_only

  # Run with filtering
  python suite_runner.py --suite aimet_plus_ontarget --filter quantsim

  # Custom suite file
  python suite_runner.py --suite-file my_tests.yaml

    """,
    )

    parser.add_argument(
        "--suite",
        choices=["aimet_only", "aimet_plus_ontarget"],
        default="aimet_only",
        help="Predefined suite to run (default: aimet_only)",
    )

    parser.add_argument(
        "--suite-file",
        type=str,
        default=None,
        help="Path to custom suite YAML (overrides --suite)",
    )

    parser.add_argument(
        "--configs-dir",
        type=str,
        default="ONNXRegression/configs",
        help="Directory containing model config YAMLs",
    )

    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter configs by substring (e.g., 'quantsim', 'resnet50')",
    )

    parser.add_argument(
        "--out-prefix", type=str, default=None, help="Prefix for output CSV/HTML files"
    )

    parser.add_argument(
        "--key-order",
        type=str,
        default=None,
        help="Comma-separated column order for reports",
    )

    parser.add_argument(
        "--title", type=str, default=None, help="Custom title for HTML report"
    )

    parser.add_argument(
        "--subtitle",
        type=str,
        default=None,
        help="Subtitle for HTML report (e.g., device info, date)",
    )

    args = parser.parse_args()

    # ============ Setup Directories ============
    configs_dir = Path(args.configs_dir)
    suites_dir = Path("ONNXRegression/suites")
    reports_dir = Path("ONNXRegression/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ============ Load Suite Configuration ============
    if args.suite_file:
        suite_path = Path(args.suite_file)
    else:
        suite_path = suites_dir / f"{args.suite}.yaml"

    print(f"Loading suite: {suite_path}")
    suite = _load_suite(suite_path)

    suite_name = suite.get("suite_name", suite_path.stem)
    description = suite.get("description", "")
    overrides = suite.get("overrides", {})
    include_list = suite["include"]

    if description:
        print(f"Description: {description}")

    if overrides:
        print(f"Overrides: {overrides}")

    # ============ Resolve Config Files ============
    config_paths: List[Path] = []

    for config_name in include_list:
        # Handle absolute and relative paths
        config_path = Path(config_name)
        if not config_path.is_absolute():
            config_path = configs_dir / config_name

        # Check if file exists
        if not config_path.exists():
            print(f"[WARNING] Config not found: {config_path} (skipping)")
            continue

        # Apply filter if specified
        if args.filter and args.filter not in config_path.name:
            continue

        config_paths.append(config_path)

    if not config_paths:
        print("[ERROR] No configs matched criteria. Nothing to run.")
        sys.exit(2)

    print(f"\nFound {len(config_paths)} configs to run:")
    for path in config_paths:
        print(f"  - {path.name}")

    # ============ Execute Suite ============
    print(f"\n{'=' * 60}")
    print(f"Starting suite: {suite_name}")
    print(f"{'=' * 60}")

    # Use temporary directory for modified configs
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        all_results = []

        for idx, cfg_path in enumerate(config_paths, 1):
            print(f"\n[{idx}/{len(config_paths)}] Processing: {cfg_path.name}")

            # Apply overrides if any
            if overrides:
                cfg_to_run = _apply_overrides_to_config_file(
                    cfg_path, overrides, tmpdir
                )
            else:
                cfg_to_run = cfg_path

            # Run the configuration
            try:
                result = run_single_config(str(cfg_to_run))
                all_results.append(result)
            except Exception as e:
                print(f"[ERROR] Failed to run {cfg_path.name}: {e}")
                # Continue with other configs
                continue

        # ============ Generate Consolidated Reports ============
        print(f"\n{'=' * 60}")
        print("Generating consolidated reports...")
        print(f"{'=' * 60}")

        from ONNXRegression.report.report_writer import write_csv, write_html

        # Determine output filenames
        if args.out_prefix:
            out_prefix = args.out_prefix
        else:
            out_prefix = f"results_{suite_name}"

        csv_path = reports_dir / f"{out_prefix}.csv"
        html_path = reports_dir / f"{out_prefix}.html"

        # Parse column order if specified
        key_order = None
        if args.key_order:
            key_order = [s.strip() for s in args.key_order.split(",")]

        # Determine what to hide based on suite type
        hide_prefixes = None
        if suite_name == "aimet_only":
            # Hide QNN-related columns for AIMET-only suite
            hide_prefixes = [
                "qnn_",  # qnn_latency, qnn_accuracy
                "ai_hub_qnn",  # AI Hub QNN job URLs
            ]

        # Set report titles
        default_titles = {
            "aimet_only": "AIMET PTQ - Host Evaluation Only",
            "aimet_plus_ontarget": "AIMET PTQ - Host + On-Device (QNN)",
        }

        page_title = args.title or default_titles.get(suite_name, "AIMET PTQ Report")

        if args.subtitle:
            subtitle = args.subtitle
        else:
            subtitle = f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        # Write reports
        write_csv(
            all_results, str(csv_path), key_order=key_order, hide_prefixes=hide_prefixes
        )

        write_html(
            all_results,
            str(html_path),
            key_order=key_order,
            hide_prefixes=hide_prefixes,
            page_title=page_title,
            subtitle=subtitle,
        )

        # ============ Summary ============
        print(f"\n✅ Suite completed: {suite_name}")
        print(f"Configs run: {len(all_results)}/{len(config_paths)}")

        if len(all_results) < len(config_paths):
            failed_count = len(config_paths) - len(all_results)
            print(f"⚠️  {failed_count} configs failed")

        print(f"\nReports generated:")
        print(f"  CSV:  {csv_path}")
        print(f"  HTML: {html_path}")

        # Show summary statistics if available
        if all_results:
            accuracies = [
                r.get("AIMET Accuracy", 0)
                for r in all_results
                if r.get("AIMET Accuracy") is not None
            ]
            if accuracies:
                avg_acc = sum(accuracies) / len(accuracies)
                print(f"\nAverage AIMET Accuracy: {avg_acc:.2%}")


if __name__ == "__main__":
    main()

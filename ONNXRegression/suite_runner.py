# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# pylint: disable=missing-module-docstring

"""
Suite Runner for AIMET ONNX and Torch Regression

This module executes a suite of AIMET quantization tests using the hierarchical
configuration system. Supports both AIMET ONNX and AIMET Torch frameworks.

Key Features:
- Load suite definition (profile + models + test_filter)
- Discover available tests from model configs
- Filter tests by suite and/or command-line flags
- Filter unsupported features based on framework (e.g., lite_mp not in Torch)
- Merge configs for each test (defaults → profile → model → test)
- Execute tests and collect results
- Generate consolidated reports

Usage:
    python suite_runner.py --suite nightly
    python suite_runner.py --suite nightly --filter-model resnet
    python suite_runner.py --suite nightly --filter-test quantsim_int8
    python suite_runner.py --suite nightly --filter-model resnet --filter-test quantsim

Design:
- Suite files define: profile + models + test_filter
- Config loader merges: defaults → profile → model → test
- Each test runs with merged config via run_single_config()
- Reports generated with suite-specific naming and filtering
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import yaml

from ONNXRegression.config_loader import load_config, list_tests, validate_config
from ONNXRegression.runner import run_single_config
from ONNXRegression.report.report_writer import write_csv, write_html


TORCH_UNSUPPORTED_FEATURES = {"lite_mp", "mixed_precision"}


def load_suite_file(suite_path: Path) -> Dict[str, Any]:
    """
    Load and validate a suite configuration file.

    Suite files define:
    - profile: Which runtime profile to use (e.g., nightly, smoke)
    - models: List of model YAML files to test
    - test_filter: Which tests to run from each model (empty = all)
    - metadata: Suite name, description, notification settings

    Args:
        suite_path: Path to suite YAML file

    Returns:
        Suite configuration dictionary

    Raises:
        ValueError: If suite file is invalid or missing required fields
    """
    if not suite_path.exists():
        raise FileNotFoundError(f"Suite file not found: {suite_path}")

    with open(suite_path, "r", encoding="utf-8") as f:
        suite = yaml.safe_load(f)

    if not isinstance(suite, dict):
        raise ValueError(f"Suite file must be a dictionary, got: {type(suite)}")

    if "models" not in suite:
        raise ValueError(
            f"Suite missing 'models' list: {suite_path}\n"
            f"Each suite must specify which models to test."
        )

    if not isinstance(suite["models"], list):
        raise ValueError(f"'models' must be a list in suite: {suite_path}")

    suite.setdefault("suite_name", suite_path.stem)
    suite.setdefault("description", "")
    suite.setdefault("profile", None)
    suite.setdefault("test_filter", [])
    suite.setdefault("config_overrides", {})  # Suite-level config overrides

    # Support 'framework' at suite level as a shorthand for config_overrides
    if "framework" in suite and "framework" not in suite["config_overrides"]:
        suite["config_overrides"]["framework"] = suite["framework"]

    return suite


def main():
    """
    Main entry point for suite execution.

    Flow:
    1. Parse command-line arguments
    2. Load suite file (get profile, models, test_filter)
    3. For each model:
       a. Discover available tests
       b. Apply suite test_filter
       c. Apply command-line filters
       d. Filter unsupported features based on framework
       e. For each matching test:
          - Merge configs via config_loader
          - Execute test via run_single_config()
    4. Generate consolidated reports
    """
    parser = argparse.ArgumentParser(
        description="Run a suite of AIMET ONNX/Torch regression tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python suite_runner.py --suite nightly
  python suite_runner.py --suite nightly --filter-model resnet
  python suite_runner.py --suite nightly --filter-test quantsim_int8
  python suite_runner.py --suite nightly --filter-model resnet --filter-test quantsim

Suite files location: ONNXRegression/suites/
        """,
    )

    parser.add_argument(
        "--suite",
        required=True,
        help="Suite name (e.g., 'nightly'). Loads from suites/<suite>.yaml",
    )

    parser.add_argument(
        "--filter-model",
        dest="filter_model",
        help="Filter models by substring (e.g., 'resnet' matches resnet50)",
    )

    parser.add_argument(
        "--filter-test",
        dest="filter_test",
        help="Filter tests by substring (e.g., 'quantsim' matches quantsim_int8)",
    )

    parser.add_argument(
        "--out-prefix",
        dest="out_prefix",
        help="Output file prefix (default: results_<suite_name>)",
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Show test matrix without executing"
    )

    args = parser.parse_args()

    suites_dir = Path("ONNXRegression/suites")
    reports_dir = Path("ONNXRegression/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    suite_path = suites_dir / f"{args.suite}.yaml"

    if not suite_path.exists():
        print(f"❌ Suite file not found: {suite_path}")
        print(f"\n💡 Available suites in {suites_dir}:")
        for suite_file in sorted(suites_dir.glob("*.yaml")):
            print(f"  - {suite_file.stem}")
        sys.exit(1)

    print(f"Loading suite: {suite_path}")

    try:
        suite = load_suite_file(suite_path)
    except (ValueError, FileNotFoundError) as e:
        print(f"❌ Failed to load suite: {e}")
        sys.exit(1)

    suite_name = suite["suite_name"]
    description = suite.get("description", "")
    profile = suite.get("profile")
    models = suite["models"]
    test_filter = suite.get("test_filter", [])
    config_overrides = suite.get("config_overrides", {})

    print(f"\n{'=' * 60}")
    print(f"Suite: {suite_name}")
    if description:
        print(f"Description: {description}")
    print(f"Profile: {profile or '(none - using defaults)'}")
    print(f"Models: {len(models)}")
    print(f"Test Filter: {test_filter or '(all tests)'}")
    if config_overrides:
        print(f"Config Overrides: {config_overrides}")
    if args.filter_model:
        print(f"Model Filter (CLI): {args.filter_model}")
    if args.filter_test:
        print(f"Test Filter (CLI): {args.filter_test}")
    print(f"{'=' * 60}\n")

    test_configs = []

    for model_yaml in models:
        if args.filter_model and args.filter_model not in model_yaml:
            print(f"⏭️  Skipping {model_yaml} (filtered by --filter-model)")
            continue

        try:
            available_tests = list_tests(model_yaml)

            print(f"\n📋 Model: {model_yaml}")
            print(f"   Available tests: {available_tests}")

            if test_filter:
                tests_to_run = [t for t in available_tests if t in test_filter]
                print(f"   After suite filter: {tests_to_run}")
            else:
                tests_to_run = available_tests
                print(f"   Running all tests")

            if args.filter_test:
                tests_to_run = [t for t in tests_to_run if args.filter_test in t]
                print(f"   After CLI filter: {tests_to_run}")

            if not tests_to_run:
                print(f"   ⚠️  No tests match filters - skipping model")
                continue

            for test_name in tests_to_run:
                try:
                    merged_config = load_config(model_yaml, test_name, profile)

                    # Apply suite-level config overrides (e.g., framework: torch)
                    config_overrides = suite.get("config_overrides", {})
                    for key, value in config_overrides.items():
                        merged_config[key] = value

                    validate_config(merged_config)

                    framework = merged_config.get("framework", "onnx").lower()
                    feature = merged_config.get("feature", "").lower()

                    if framework == "torch" and feature in TORCH_UNSUPPORTED_FEATURES:
                        print(
                            f"   ⏭️  Skipping {test_name}: feature '{feature}' not available in AIMET Torch"
                        )
                        continue

                    test_configs.append(
                        {
                            "model_yaml": model_yaml,
                            "test_name": test_name,
                            "config": merged_config,
                        }
                    )

                except Exception as e:
                    print(f"   ⚠️  Failed to load {test_name}: {e}")
                    continue

        except FileNotFoundError as e:
            print(f"⚠️  Error loading model config: {e}")
            continue
        except Exception as e:
            print(f"⚠️  Unexpected error processing {model_yaml}: {e}")
            continue

    if not test_configs:
        print("\n❌ No tests to run after filtering!")
        print("\n💡 Tips:")
        print("  - Check that suite models exist in configs/models/")
        print("  - Check that test_filter matches actual test names")
        print("  - Try without filters to see all available tests")
        sys.exit(2)

    print(f"\n{'=' * 60}")
    print(f"Total tests to run: {len(test_configs)}")
    print(f"{'=' * 60}\n")

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN - Test Matrix")
        print("=" * 60)

        for idx, test_info in enumerate(test_configs, 1):
            config = test_info["config"]
            framework = config.get("framework", "onnx")
            print(f"\n{idx}. {config['model_name']}/{test_info['test_name']}")
            print(f"   Framework: {framework}")
            print(f"   Feature: {config['feature']}")
            print(
                f"   Samples: calib={config.get('calib_samples')}, eval={config.get('eval_samples')}"
            )
            print(f"   QNN: {'Enabled' if config.get('qnn_options') else 'Disabled'}")

        print(f"\n{'=' * 60}")
        print(f"Total: {len(test_configs)} tests")
        print(f"Estimated time: ~{len(test_configs) * 20} minutes (20 min/test)")
        print(f"{'=' * 60}")
        sys.exit(0)

    all_results = []

    for idx, test_info in enumerate(test_configs, 1):
        model_yaml = test_info["model_yaml"]
        test_name = test_info["test_name"]
        config = test_info["config"]

        print(f"\n{'=' * 60}")
        print(
            f"Progress: [{idx}/{len(test_configs)}] ({idx / len(test_configs) * 100:.1f}%)"
        )
        print(f"Test: {model_yaml} / {test_name}")
        print(f"Estimated time remaining: ~{(len(test_configs) - idx) * 20} minutes")
        print(f"{'=' * 60}\n")

        print(f"  Model: {config['model_name']}")
        print(f"  Framework: {config.get('framework', 'onnx')}")
        print(f"  Feature: {config['feature']}")
        print(
            f"  Samples: calib={config.get('calib_samples')}, eval={config.get('eval_samples')}"
        )
        print(f"  QNN: {'Enabled' if config.get('qnn_options') else 'Disabled'}")

        try:
            # Run the test using the merged config
            # Skip individual reports
            result = run_single_config(config, skip_reports=True)
            all_results.append(result)
            print(f"  ✅ Success")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback

            traceback.print_exc()
            continue

    if not all_results:
        print("\n❌ No successful test runs!")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("Generating consolidated reports...")
    print(f"{'=' * 60}")

    out_prefix = args.out_prefix or f"results_{suite_name}"
    csv_path = reports_dir / f"{out_prefix}.csv"
    html_path = reports_dir / f"{out_prefix}.html"

    hide_prefixes = None
    if profile in ["nightly", "smoke"]:
        hide_prefixes = ["qnn_", "ai_hub_qnn"]

    page_title = f"AIMET Regression - {suite_name}"
    subtitle = f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    if profile:
        subtitle += f" | Profile: {profile}"

    write_csv(all_results, str(csv_path), hide_prefixes=hide_prefixes)
    write_html(
        all_results,
        str(html_path),
        hide_prefixes=hide_prefixes,
        page_title=page_title,
        subtitle=subtitle,
    )

    print(f"\n✅ Suite completed: {suite_name}")
    print(f"Tests run: {len(all_results)}/{len(test_configs)}")

    if len(all_results) < len(test_configs):
        failed_count = len(test_configs) - len(all_results)
        print(f"⚠️  {failed_count} tests failed")

    print(f"\nReports generated:")
    print(f"  CSV:  {csv_path}")
    print(f"  HTML: {html_path}")

    if all_results:
        accuracies = [
            r.get("AIMET Accuracy", 0)
            for r in all_results
            if r.get("AIMET Accuracy") is not None
        ]
        if accuracies:
            avg_acc = sum(accuracies) / len(accuracies)
            min_acc = min(accuracies)
            max_acc = max(accuracies)
            print(f"\nAccuracy Summary:")
            print(f"  Average: {avg_acc:.2%}")
            print(f"  Min: {min_acc:.2%}")
            print(f"  Max: {max_acc:.2%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

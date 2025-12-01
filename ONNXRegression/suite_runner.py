# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# pylint: disable=missing-module-docstring

"""
Suite Runner for AIMET ONNX Regression

This module executes a suite of AIMET quantization tests using the new
hierarchical configuration system.

Key Features:
- Load suite definition (profile + models + test_filter)
- Discover available tests from model configs
- Filter tests by suite and/or command-line flags
- Merge configs for each test (defaults → profile → model → test)
- Execute tests and collect results
- Generate consolidated reports

Usage:
    # Run full suite
    python suite_runner.py --suite nightly

    # Filter by model
    python suite_runner.py --suite nightly --filter-model resnet

    # Filter by test
    python suite_runner.py --suite nightly --filter-test quantsim_int8

    # Combine filters
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

# ==================== Internal Imports ====================
from ONNXRegression.config_loader import load_config, list_tests, validate_config
from ONNXRegression.runner import run_single_config
from ONNXRegression.report.report_writer import write_csv, write_html


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

    # Validate required fields
    if "models" not in suite:
        raise ValueError(
            f"Suite missing 'models' list: {suite_path}\n"
            f"Each suite must specify which models to test."
        )

    if not isinstance(suite["models"], list):
        raise ValueError(f"'models' must be a list in suite: {suite_path}")

    # Set defaults for optional fields
    suite.setdefault("suite_name", suite_path.stem)
    suite.setdefault("description", "")
    suite.setdefault("profile", None)  # Optional profile
    suite.setdefault("test_filter", [])  # Empty = run all tests

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
       d. For each matching test:
          - Merge configs via config_loader
          - Execute test via run_single_config()
    4. Generate consolidated reports
    """
    # ============ Argument Parsing ============
    parser = argparse.ArgumentParser(
        description="Run a suite of AIMET ONNX regression tests (NEW CONFIG SYSTEM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run nightly suite (fast, no QNN)
  python suite_runner.py --suite nightly

  # Filter by model name
  python suite_runner.py --suite nightly --filter-model resnet

  # Filter by test name
  python suite_runner.py --suite nightly --filter-test quantsim_int8

  # Combine filters
  python suite_runner.py --suite nightly --filter-model resnet --filter-test quantsim

Suite files location: ONNXRegression/suites/
Available suites: nightly (add smoke, weekly, release later)

What happens:
1. Load suite definition (profile + models + test_filter)
2. Discover tests from each model config
3. Apply filters (suite + command-line)
4. Execute each test with merged config
5. Generate consolidated CSV/HTML reports
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

    # ============ Setup Paths ============
    suites_dir = Path("ONNXRegression/suites")
    reports_dir = Path("ONNXRegression/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ============ Load Suite Configuration ============
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

    print(f"\n{'=' * 60}")
    print(f"Suite: {suite_name}")
    if description:
        print(f"Description: {description}")
    print(f"Profile: {profile or '(none - using defaults)'}")
    print(f"Models: {len(models)}")
    print(f"Test Filter: {test_filter or '(all tests)'}")
    if args.filter_model:
        print(f"Model Filter (CLI): {args.filter_model}")
    if args.filter_test:
        print(f"Test Filter (CLI): {args.filter_test}")
    print(f"{'=' * 60}\n")

    # ============ Build Test Matrix ============
    # For each model, expand into individual test configs
    test_configs = []

    for model_yaml in models:
        # Apply command-line model filter
        if args.filter_model and args.filter_model not in model_yaml:
            print(f"⏭️  Skipping {model_yaml} (filtered by --filter-model)")
            continue

        try:
            # Get list of available tests in this model
            available_tests = list_tests(model_yaml)

            print(f"\n📋 Model: {model_yaml}")
            print(f"   Available tests: {available_tests}")

            # Apply suite test_filter (from suite YAML)
            if test_filter:
                # Suite specifies which tests to run
                tests_to_run = [t for t in available_tests if t in test_filter]
                print(f"   After suite filter: {tests_to_run}")
            else:
                # No suite filter - run all tests
                tests_to_run = available_tests
                print(f"   Running all tests")

            # Apply command-line test filter
            if args.filter_test:
                tests_to_run = [t for t in tests_to_run if args.filter_test in t]
                print(f"   After CLI filter: {tests_to_run}")

            if not tests_to_run:
                print(f"   ⚠️  No tests match filters - skipping model")
                continue

            # For each matching test, load merged config
            for test_name in tests_to_run:
                try:
                    # Merge configs: defaults → profile → model → test
                    merged_config = load_config(model_yaml, test_name, profile)

                    # Validate the merged config
                    validate_config(merged_config)

                    # Add to test matrix
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

    # ============ Dry Run (Preview) ============
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN - Test Matrix")
        print("=" * 60)

        for idx, test_info in enumerate(test_configs, 1):
            config = test_info["config"]
            print(f"\n{idx}. {config['model_name']}/{test_info['test_name']}")
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

    # ============ Execute Tests ============
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

    # ============ Generate Consolidated Reports ============
    if not all_results:
        print("\n❌ No successful test runs!")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("Generating consolidated reports...")
    print(f"{'=' * 60}")

    # Determine output filenames
    out_prefix = args.out_prefix or f"results_{suite_name}"
    csv_path = reports_dir / f"{out_prefix}.csv"
    html_path = reports_dir / f"{out_prefix}.html"

    # Determine what to hide based on profile
    # For host-only profiles (nightly), hide QNN columns
    hide_prefixes = None
    if profile in ["nightly", "smoke"]:
        # These profiles don't run QNN, so hide those columns
        hide_prefixes = ["qnn_", "ai_hub_qnn"]

    # Generate reports
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

    # ============ Summary ============
    print(f"\n✅ Suite completed: {suite_name}")
    print(f"Tests run: {len(all_results)}/{len(test_configs)}")

    if len(all_results) < len(test_configs):
        failed_count = len(test_configs) - len(all_results)
        print(f"⚠️  {failed_count} tests failed")

    print(f"\nReports generated:")
    print(f"  CSV:  {csv_path}")
    print(f"  HTML: {html_path}")

    # Show summary statistics
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

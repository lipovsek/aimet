# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring

"""
Workflow utilities for ONNX regression testing.

Provides utilities for:
- Baseline setup and verification
- Environment lockfile generation with metadata
- AI Hub authentication and configuration
- Model dependency installation
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import yaml


class BaselineSetup:
    """Handle baseline setup and verification."""

    def __init__(self, suite: str):
        self.suite = suite
        self.baselines_dir = Path("ONNXRegression/baselines")
        self.downloaded_dir = self.baselines_dir / "downloaded"

    def setup(self) -> bool:
        """
        Setup baseline for comparison.

        Returns:
            True if baseline exists, False if this is first run
        """
        print("=" * 60)
        print(f"Baseline Setup for Suite: {self.suite}")
        print("=" * 60)

        baseline_file = self.downloaded_dir / "latest.json"

        if baseline_file.exists():
            print("✓ Previous baseline found")

            # Create baselines directory and copy
            self.baselines_dir.mkdir(parents=True, exist_ok=True)

            target = self.baselines_dir / "latest.json"
            import shutil

            shutil.copy(baseline_file, target)

            file_size = target.stat().st_size
            print(f"  Copied to: {target}")
            print(f"  Size: {file_size} bytes")

            return True
        else:
            print("ℹ️  No previous baseline found")
            print("This is either:")
            print("  - First run on this branch")
            print("  - Baseline artifact expired (>30 days old)")
            print("")
            print("A new baseline will be created from this run's results.")

            # Create directory for new baseline
            self.baselines_dir.mkdir(parents=True, exist_ok=True)

            return False


class LockfileGenerator:
    """Generate environment lockfiles with metadata."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.reports_dir = Path("ONNXRegression/reports")

    def generate(self) -> tuple[Path, Path]:
        """
        Generate lockfile and metadata.

        Returns:
            Tuple of (lockfile_path, metadata_path)
        """
        print("=" * 60)
        print("Generating Environment Lockfile")
        print("=" * 60)

        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Generate lockfile
        lockfile_path = self.reports_dir / f"requirements-{self.run_id}.lock"
        print("Generating lockfile...")

        try:
            result = subprocess.run(
                ["uv", "pip", "freeze"], capture_output=True, text=True, check=True
            )
            lockfile_path.write_text(result.stdout)
        except subprocess.CalledProcessError:
            # Fallback to pip if UV not available
            result = subprocess.run(
                ["pip", "freeze"], capture_output=True, text=True, check=True
            )
            lockfile_path.write_text(result.stdout)

        # Generate metadata
        metadata_path = self.reports_dir / f"metadata-{self.run_id}.json"
        print("Generating metadata...")

        metadata = self._collect_metadata()
        metadata_path.write_text(json.dumps(metadata, indent=2))

        # Show summary
        package_count = len(
            [
                line
                for line in lockfile_path.read_text().split("\n")
                if "==" in line and not line.startswith("#")
            ]
        )

        print(f"✓ Lockfile generated: {lockfile_path.name}")
        print(f"✓ Metadata generated: {metadata_path.name}")
        print(f"  Python packages: {package_count}")
        print("")
        print("📦 To reproduce this environment:")
        print(f"   uv pip install -r {lockfile_path.name}")

        return lockfile_path, metadata_path

    def _collect_metadata(self) -> Dict[str, Any]:
        """Collect environment metadata."""
        metadata = {
            "run_id": self.run_id,
            "branch": os.environ.get("GITHUB_REF_NAME", "unknown"),
            "suite": os.environ.get("INPUT_SUITE", "unknown"),
        }

        # Python version
        try:
            result = subprocess.run(
                ["python", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                metadata["python_version"] = result.stdout.strip()
        except:
            pass

        # UV version
        try:
            result = subprocess.run(
                ["uv", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                metadata["uv_version"] = result.stdout.strip()
        except:
            pass

        # GPU info
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                metadata["gpu"] = result.stdout.strip().split("\n")[0]
        except:
            metadata["gpu"] = "N/A"

        # CUDA version
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "CUDA Version" in line:
                        parts = line.split("CUDA Version:")
                        if len(parts) > 1:
                            metadata["cuda_version"] = parts[1].strip().split()[0]
                            break
        except:
            pass

        if "cuda_version" not in metadata:
            metadata["cuda_version"] = "N/A"

        # Driver version
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                metadata["driver_version"] = result.stdout.strip().split("\n")[0]
        except:
            metadata["driver_version"] = "N/A"

        return metadata


class AIHubConfig:
    """Configure Qualcomm AI Hub authentication."""

    @staticmethod
    def configure(api_token: Optional[str] = None) -> bool:
        """
        Configure AI Hub authentication for DEV environment.
        Creates ~/.qai_hub/client.ini with DEV URLs.

        Args:
            api_token: AI Hub API token (reads from env if not provided)

        Returns:
            True if configured successfully, False otherwise
        """
        print("=" * 60)
        print("Configuring Qualcomm AI Hub")
        print("=" * 60)

        token = api_token or os.environ.get("QAI_HUB_API_TOKEN")

        if not token:
            print("⚠️  No AI Hub API token found")
            print("   Set QAI_HUB_API_TOKEN environment variable")
            return False

        try:
            # Create config directory
            config_dir = Path.home() / ".qai_hub"
            config_dir.mkdir(parents=True, exist_ok=True)

            # Write configuration file for DEV environment
            config_file = config_dir / "client.ini"
            config_content = f"""[api]
api_token = {token}
api_url = https://dev.aihub.qualcomm.com
web_url = https://dev.aihub.qualcomm.com
verbose = True
"""
            config_file.write_text(config_content)

            print("✓ AI Hub configuration created")
            print(f"  Config file: {config_file}")

            # Verify configuration
            result = subprocess.run(
                ["qai-hub", "list-devices"], capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                print("✓ AI Hub connection verified")
                # Show first few devices
                lines = result.stdout.split("\n")[:5]
                for line in lines:
                    if line.strip():
                        print(f"  {line}")
                return True
            else:
                print("⚠️  Could not verify AI Hub connection")
                if result.stderr:
                    print(f"  Error: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("⚠️  AI Hub verification timed out")
            return False
        except Exception as e:
            print(f"❌ AI Hub configuration error: {e}")
            return False


class ModelDependencyInstaller:
    """
    Install model-specific dependencies from suite configuration.

    NOTE: This class installs ALL model extras from a suite file upfront.
    Use this for:
    - CLI: `python -m ONNXRegression.workflow.utils install-deps --suite nightly-torch`
    - Workflow pre-install step (before running tests)

    For per-model installation inside test loops, use `install_model_extras()` instead.
    """

    # Common packages needed by certain model types
    COMMON_PACKAGES: List[str] = [
        "object-detection-metrics",  # For YOLO mAP evaluation
    ]

    def __init__(self, suite_path: str):
        self.suite_path = Path(suite_path)

    def _run_install(self, packages: List[str]) -> bool:
        """
        Run uv pip install for given packages.

        Args:
            packages: List of package specifiers

        Returns:
            True if successful, False otherwise
        """
        cmd = ["uv", "pip", "install", "-q"] + packages

        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def install(self) -> None:
        """
        Install model-specific dependencies from suite YAML file.

        Reads the suite file, extracts model names, and installs:
        1. Common packages (e.g., object-detection-metrics)
        2. Per-model extras via: qai-hub-models[model-name]
        """
        print("=" * 60)
        print("Installing Model Dependencies")
        print("=" * 60)

        if not self.suite_path.exists():
            print(f"❌ Suite file not found: {self.suite_path}")
            return

        with open(self.suite_path) as f:
            suite = yaml.safe_load(f)

        models = suite.get("models", [])
        if not models:
            print("ℹ️  No models found in suite")
            return

        # Install common packages
        if self.COMMON_PACKAGES:
            print(f"Installing common packages: {self.COMMON_PACKAGES}")
            for package in self.COMMON_PACKAGES:
                if self._run_install([package]):
                    print(f"  ✓ {package}")
                else:
                    print(f"  ⚠️  {package} (failed, may not be needed)")
            print("")

        # Install per-model extras
        print(f"Installing model extras ({len(models)} models):")
        for model_yaml in models:
            # Extract model name from path like 'models/yolov5.yaml'
            model_name = model_yaml.replace("models/", "").replace(".yaml", "")

            if self._run_install([f"qai-hub-models[{model_name}]"]):
                print(f"  ✓ {model_name}")
            else:
                print(f"  - {model_name} (no extras or already installed)")

        print("")
        print("✓ Model dependencies installed")


import subprocess


def install_model_extras(model_name: str, use_uv: bool = True) -> None:
    """
    Install qai-hub-models extras for a single model.

    Use this for per-model installation inside test loops.
    For bulk installation from suite file, use ModelDependencyInstaller.

    Args:
        model_name: Model name (e.g., "yolov8_det", "resnet50")
        use_uv: Use uv pip instead of pip
    """
    pip_cmd = ["uv", "pip", "install", "-q"] if use_uv else ["pip", "install", "-q"]
    try:
        subprocess.run(
            [*pip_cmd, f"qai-hub-models[{model_name}]"],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        pass  # No extras for this model


def main():
    """CLI entry point for workflow helpers."""
    import argparse

    parser = argparse.ArgumentParser(description="Workflow helper utilities")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Baseline setup
    baseline_parser = subparsers.add_parser(
        "setup-baseline", help="Setup baseline for comparison"
    )
    baseline_parser.add_argument(
        "--suite", default=os.environ.get("INPUT_SUITE", "nightly")
    )

    # Lockfile generation
    lockfile_parser = subparsers.add_parser(
        "generate-lockfile", help="Generate environment lockfile"
    )
    lockfile_parser.add_argument(
        "--run-id", default=os.environ.get("GITHUB_RUN_ID", "unknown")
    )

    # AI Hub config
    aihub_parser = subparsers.add_parser(
        "configure-aihub", help="Configure AI Hub authentication"
    )
    aihub_parser.add_argument("--token", help="AI Hub API token (or use env var)")

    # Model dependencies
    deps_parser = subparsers.add_parser(
        "install-deps", help="Install model-specific dependencies from suite"
    )
    deps_parser.add_argument(
        "--suite",
        required=True,
        help="Suite name (e.g., 'nightly-torch')",
    )

    args = parser.parse_args()

    if args.command == "setup-baseline":
        setup = BaselineSetup(args.suite)
        has_baseline = setup.setup()

        # Set GitHub environment variable
        if "GITHUB_ENV" in os.environ:
            with open(os.environ["GITHUB_ENV"], "a") as f:
                f.write(f"HAS_BASELINE={'true' if has_baseline else 'false'}\n")

        exit(0)

    elif args.command == "generate-lockfile":
        generator = LockfileGenerator(args.run_id)
        lockfile, metadata = generator.generate()
        exit(0)

    elif args.command == "configure-aihub":
        success = AIHubConfig.configure(args.token)
        exit(0 if success else 1)

    elif args.command == "install-deps":
        suite_path = f"ONNXRegression/suites/{args.suite}.yaml"
        installer = ModelDependencyInstaller(suite_path)
        installer.install()
        exit(0)

    else:
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main()

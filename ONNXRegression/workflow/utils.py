# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# pylint: disable=missing-module-docstring

"""
Workflow utilities for ONNX regression testing.

Provides utilities for:
- Baseline setup and verification
- Environment lockfile generation with metadata
- AI Hub authentication and configuration
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any


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

    else:
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main()

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# pylint: disable=missing-module-docstring

"""
Artifact management for baseline tracking with branch isolation.

Handles downloading baselines via GitHub Actions artifacts with smart
branch-specific isolation strategy.

Key Features:
- Branch isolation (feature branches use own baselines)
- Develop tracks develop quality over time
- Clear error messages and debugging output
- Robust API error handling
"""

import os
import requests
import zipfile
import tempfile
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass


@dataclass
class WorkflowConfig:
    """Configuration from GitHub Actions environment."""

    token: str
    repo: str
    workflow_file: str
    current_branch: str
    current_run_id: str
    suite: str

    @classmethod
    def from_env(cls):
        """Create config from environment variables."""
        return cls(
            token=os.environ["GITHUB_TOKEN"],
            repo=os.environ["GITHUB_REPOSITORY"],
            workflow_file=os.environ.get("WORKFLOW_FILE", "onnx-nightly.yaml"),
            current_branch=os.environ["GITHUB_REF_NAME"],
            current_run_id=os.environ["GITHUB_RUN_ID"],
            suite=os.environ.get("INPUT_SUITE", "nightly"),
        )


class BaselineStrategy:
    """Determine baseline strategy based on branch."""

    def __init__(self, current_branch: str):
        self.current_branch = current_branch

    def is_main_branch(self) -> bool:
        """Check if current branch is develop (main branch)."""
        return self.current_branch == "develop"

    def get_baseline_branch(self) -> str:
        """
        Determine which branch's baseline to use.

        Strategy:
        - All branches use their own baseline (track own progress)
        """
        return self.current_branch

    def get_description(self) -> str:
        """Get human-readable description of strategy."""
        if self.is_main_branch():
            return "Main branch: Tracking develop quality over time"
        else:
            return "Feature branch: Tracking development progress"


class ArtifactManager:
    """Manage GitHub Actions artifacts for baseline storage."""

    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.api_base = "https://github.qualcomm.com/api/v3"
        self.headers = {
            "Authorization": f"token {config.token}",
            "Accept": "application/vnd.github.v3+json",
        }
        self.strategy = BaselineStrategy(config.current_branch)

    def get_artifact_name(self, branch: str, suite: str) -> str:
        """
        Generate artifact name for branch and suite.

        Format:
        - develop: baseline-{suite}
        - other branches: baseline-{suite}-{safe-branch}

        Examples:
            baseline-nightly (develop)
            baseline-nightly-feature-test-adaround (feature branch)
            baseline-smoke-feature-fix-quantsim (feature branch)
        """
        # For develop branch, don't include branch name
        if branch == "develop":
            return f"baseline-{suite}"

        # For feature branches, include normalized branch name
        safe_branch = self._normalize_branch_name(branch)
        return f"baseline-{suite}-{safe_branch}"

    def download_baseline(self, output_dir: Path) -> Optional[Path]:
        """
        Download baseline for current branch.

        Args:
            output_dir: Where to save downloaded baseline

        Returns:
            Path to downloaded baseline, or None if not found (first run)
        """
        baseline_branch = self.strategy.get_baseline_branch()
        artifact_name = self.get_artifact_name(baseline_branch, self.config.suite)

        print("=" * 60)
        print("📥 BASELINE DOWNLOAD")
        print("=" * 60)
        print(f"Current branch: {self.config.current_branch}")
        print(f"Suite: {self.config.suite}")
        print(f"Strategy: {self.strategy.get_description()}")
        print(f"Looking for artifact: {artifact_name}")
        print()

        # Get successful runs from baseline branch
        runs = self._get_successful_runs(
            branch=baseline_branch, exclude_current=True, limit=10
        )

        if not runs:
            print(f"ℹ️  No previous successful runs found on branch {baseline_branch}")
            print()
            print("   This is normal for:")
            print("   - First run on this branch")
            print("   - Previous runs failed")
            print("   - Workflow was renamed")
            print()
            print("✓ Baseline download completed (no baseline available)")
            return None

        print(f"✓ Found {len(runs)} previous successful runs on {baseline_branch}")
        print(f"  (excluding current run {self.config.current_run_id})")
        print()

        # Search for artifact in recent runs
        print("Searching for baseline artifact in recent runs...")
        for run in runs:
            run_id = run["id"]
            run_number = run.get("run_number", "?")
            run_date = run.get("created_at", "unknown")[:10]

            print(f"  Checking run #{run_number} (ID: {run_id}, {run_date})...")

            artifact = self._find_artifact_in_run(run_id, artifact_name)
            if artifact:
                print(f"  ✓ Found baseline artifact!")
                print(f"    Artifact ID: {artifact['id']}")
                print(f"    Size: {artifact['size_in_bytes']} bytes")
                print()

                # Download it
                baseline_path = self._download_artifact(artifact, output_dir)

                if baseline_path:
                    print("✓ Baseline download completed successfully")
                    print(f"  File: {baseline_path}")
                    print(f"  Size: {baseline_path.stat().st_size} bytes")
                    print(f"  From run: #{run_number} (ID: {run_id})")
                    print()
                    return baseline_path

        print(f"ℹ️  No baseline artifact found in recent {baseline_branch} runs")
        print()
        print("   Possible reasons:")
        print("   - Artifact expired (>30 days old)")
        print("   - Baseline artifact name changed")
        print("   - Previous runs didn't upload baseline")
        print()
        print("✓ Baseline download completed (no baseline available)")
        return None

    def _get_successful_runs(
        self, branch: str, exclude_current: bool = True, limit: int = 10
    ) -> List[Dict]:
        """Get list of successful workflow runs for specific branch."""
        url = (
            f"{self.api_base}/repos/{self.config.repo}/actions/"
            f"workflows/{self.config.workflow_file}/runs"
        )

        params = {"status": "success", "branch": branch, "per_page": limit}

        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=30)
            resp.raise_for_status()

            runs = resp.json().get("workflow_runs", [])

            if exclude_current:
                runs = [r for r in runs if str(r["id"]) != self.config.current_run_id]

            return runs

        except requests.exceptions.RequestException as e:
            print(f"⚠️  Error querying GitHub API: {e}")
            return []

    def _find_artifact_in_run(self, run_id: int, artifact_name: str) -> Optional[Dict]:
        """Find specific artifact in a run."""
        url = (
            f"{self.api_base}/repos/{self.config.repo}/actions/runs/{run_id}/artifacts"
        )

        try:
            resp = requests.get(url, headers=self.headers, timeout=30)
            if not resp.ok:
                return None

            artifacts = resp.json().get("artifacts", [])

            for artifact in artifacts:
                if artifact["name"] == artifact_name:
                    return artifact

            return None

        except requests.exceptions.RequestException:
            return None

    def _download_artifact(self, artifact: Dict, output_dir: Path) -> Optional[Path]:
        """Download and extract artifact."""
        artifact_id = artifact["id"]
        url = f"{self.api_base}/repos/{self.config.repo}/actions/artifacts/{artifact_id}/zip"

        try:
            print(f"  Downloading artifact...")
            resp = requests.get(url, headers=self.headers, stream=True, timeout=120)
            resp.raise_for_status()

            # Download to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                for chunk in resp.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                tmp_path = tmp.name

            print(f"  Extracting...")

            # Extract
            output_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(tmp_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)

            os.unlink(tmp_path)

            # Verify extraction
            baseline_file = output_dir / "latest.json"
            if baseline_file.exists():
                return baseline_file
            else:
                print(f"  ⚠️  Artifact downloaded but latest.json not found inside")
                return None

        except Exception as e:
            print(f"  ⚠️  Error downloading artifact: {e}")
            return None

    def _normalize_branch_name(self, branch: str) -> str:
        """
        Convert branch name to artifact-safe name.

        Examples:
            develop → develop
            feature/test-adaround → feature-test-adaround
            feature/fix/bug → feature-fix-bug
        """
        return branch.replace("/", "-").replace("_", "-").lower()


def main():
    """CLI entry point for artifact management."""
    import argparse

    parser = argparse.ArgumentParser(description="Manage workflow artifacts")
    parser.add_argument("action", choices=["download-baseline"])
    parser.add_argument("--output-dir", default="ONNXRegression/baselines/downloaded")

    args = parser.parse_args()

    try:
        config = WorkflowConfig.from_env()
        manager = ArtifactManager(config)

        if args.action == "download-baseline":
            output_dir = Path(args.output_dir)
            baseline = manager.download_baseline(output_dir)
            exit(0)

    except KeyError as e:
        print(f"❌ Missing required environment variable: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()

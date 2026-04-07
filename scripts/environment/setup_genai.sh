#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# Sets up a plain docker/pod for GenAI development or CI.
# Mirrors the setup steps from .github/workflows/genai-scorecard.yaml
#
# Usage:
#   bash setup_genai.sh                                  # Interactive dev (defaults)
#   bash setup_genai.sh --with-aws                       # Include AWS CLI
#   bash setup_genai.sh --wheels-dir /path/to/wheels     # Use pre-built wheels
#   bash setup_genai.sh --repo-dir /path/to/aimet        # Override repo location
#
# Assumes:
#   - Running inside a CUDA-enabled container
#   - AIMET repo is available at $REPO_DIR (default: /scratch/aimet)

set -euo pipefail

# -----------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------
INSTALL_AWS=false
WHEELS_DIR=""
REPO_DIR="${REPO_DIR:-/scratch/aimet}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-aws)     INSTALL_AWS=true; shift ;;
    --wheels-dir)   WHEELS_DIR="$2"; shift 2 ;;
    --repo-dir)     REPO_DIR="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,/^$/s/^# //p' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# Default wheels directory if not specified
if [ -z "$WHEELS_DIR" ]; then
  WHEELS_DIR="${REPO_DIR}/downloads"
fi

VENV_DIR="${REPO_DIR}/.venv"

echo "=== GenAI Dev Setup ==="
echo "Repo:   $REPO_DIR"
echo "Venv:   $VENV_DIR"
echo "Wheels: $WHEELS_DIR"

# -----------------------------------------------------------------------
# System dependencies
# -----------------------------------------------------------------------
echo "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
  ca-certificates \
  curl \
  git \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libgomp1 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  lsb-release \
  python3 \
  python3-pip \
  python3-venv \
  unzip \
  zip \
  zstd
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------
# AWS CLI v2 (optional, for CI uploads to S3)
# -----------------------------------------------------------------------
if $INSTALL_AWS; then
  echo "Installing AWS CLI v2..."
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  unzip -q /tmp/awscliv2.zip -d /tmp
  sudo /tmp/aws/install
  rm -rf /tmp/awscliv2.zip /tmp/aws
fi

# -----------------------------------------------------------------------
# Git configuration (non-interactive mode)
# -----------------------------------------------------------------------
echo "Configuring git for non-interactive mode..."
git config --global core.askPass ""
git config --global credential.helper ""
git config --global --add safe.directory '*'

# -----------------------------------------------------------------------
# Python virtual environment
# -----------------------------------------------------------------------
echo "Setting up Python venv at $VENV_DIR..."
python3 -m venv "$VENV_DIR"
. "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools

# -----------------------------------------------------------------------
# Python dependencies
# -----------------------------------------------------------------------
echo "Installing Python dependencies..."
pip install qai_hub_models
pip install -r "$REPO_DIR/GenAITests/requirements.txt"

# -----------------------------------------------------------------------
# AIMET
# -----------------------------------------------------------------------
# If pre-built wheels exist, install them. Otherwise install from PyPI
# for dependencies only, and prepend local source to PYTHONPATH so
# local code always takes priority over the PyPI packages.
if ls "$WHEELS_DIR"/*.whl 1>/dev/null 2>&1; then
  echo "Installing AIMET from pre-built wheels..."
  pip install "$WHEELS_DIR"/*.whl
else
  echo "No pre-built wheels found. Installing from PyPI (for dependencies)..."
  pip install aimet-torch aimet-onnx
fi

# -----------------------------------------------------------------------
# Environment variables
# -----------------------------------------------------------------------
echo "Configuring environment..."
EXTRA_ENV="
# GenAI environment
export GIT_CLONE_PROTECTION_ACTIVE=\"false\"
export GIT_TERMINAL_PROMPT=\"0\"
export MPLBACKEND=\"Agg\"
export QT_QPA_PLATFORM=\"offscreen\"
"

if [ -n "${SAML2AWS_APP_ID:-}" ]; then
  EXTRA_ENV+="export SAML2AWS_APP_ID=\"$SAML2AWS_APP_ID\"
"
fi

echo "$EXTRA_ENV" >> "$VENV_DIR/bin/activate"

# Re-source to pick up the new vars
. "$VENV_DIR/bin/activate"

# Also add to .bashrc so new shells get it
echo ". $VENV_DIR/bin/activate" >> ~/.bashrc

# -----------------------------------------------------------------------
# HuggingFace authentication
# -----------------------------------------------------------------------
# Source HF_TOKEN from profile.d if not already set
if [ -z "${HF_TOKEN:-}" ] && [ -f /etc/profile.d/hf-token.sh ]; then
  . /etc/profile.d/hf-token.sh
fi

if [ -n "${HF_TOKEN:-}" ]; then
  echo "Logging in to HuggingFace..."
  hf auth login --token "$HF_TOKEN"
else
  echo "WARNING: HF_TOKEN not set. Set it locally or run: hf auth login"
fi

# -----------------------------------------------------------------------
# Verify
# -----------------------------------------------------------------------
echo ""
echo "=== Verifying installation ==="
python -c "
try:
    import aimet_torch; print(f'aimet_torch: {aimet_torch.__version__}')
except ImportError:
    print('aimet_torch: not installed')
try:
    import aimet_onnx; print(f'aimet_onnx: {aimet_onnx.__version__}')
except ImportError:
    print('aimet_onnx: not installed')
"
echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate the virtual environment, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To use local source over PyPI packages, run:"
echo "  export PYTHONPATH=$REPO_DIR/TrainingExtensions/torch/src/python:$REPO_DIR/TrainingExtensions/onnx/src/python:$REPO_DIR/TrainingExtensions/common/src/python:$REPO_DIR:\$PYTHONPATH"

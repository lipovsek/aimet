#!/bin/bash

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# Only run on macOS
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "Skipping macOS deps installation (not on macOS)"
    exit 0
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

. "${SCRIPT_DIR}/../all/util/common.sh"

set_strict_mode

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

echo "Updating Homebrew..."
brew update

echo "Installing dependencies..."
brew install \
    cmake \
    eigen \
    git \
    jq \
    pandoc \
    pkg-config \
    python@3.10 \
    uv \
    zlib

# Ensure Xcode Command Line Tools are installed (provides clang, make, etc.)
if ! xcode-select -p &> /dev/null; then
    echo "Installing Xcode Command Line Tools..."
    xcode-select --install
    echo "Please complete the Xcode Command Line Tools installation and re-run this script."
    exit 1
fi

echo "macOS dependencies installed successfully."

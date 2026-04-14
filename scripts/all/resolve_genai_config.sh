#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# Resolve the scorecard config file for a given variant.
#
# In ad-hoc mode, decodes a base64 config to a temp file.
# In regression mode, uses the checked-in regression config.
#
# Usage:
#   scripts/resolve_scorecard_config.sh <variant> [b64_config]
#
# Outputs (via GITHUB_OUTPUT if set, otherwise stdout):
#   path=<resolved config path>

set -euo pipefail

VARIANT="${1:?Usage: $0 <variant> [b64_config]}"
B64_CONFIG="${2:-}"

CONFIG_DIR="GenAILab/configs"
STAGING_PATH="GenAILab/scorecard_config.yaml"

if [ -n "$B64_CONFIG" ]; then
  if ! echo "$B64_CONFIG" | base64 -d > "$STAGING_PATH" 2>/dev/null; then
    echo "Error: Failed to decode base64 config. Please ensure the input is valid base64."
    exit 1
  fi
  if [ ! -s "$STAGING_PATH" ]; then
    echo "Error: Decoded config file is empty."
    exit 1
  fi
  echo "Saved ad-hoc config to $STAGING_PATH"
  cat "$STAGING_PATH"
  CONFIG_PATH="$STAGING_PATH"
else
  CONFIG_PATH="$CONFIG_DIR/${VARIANT}_regression.yaml"
  if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Regression config not found: $CONFIG_PATH"
    exit 1
  fi
  echo "Using regression config: $CONFIG_PATH"
fi

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "path=$CONFIG_PATH" >> "$GITHUB_OUTPUT"
else
  echo "path=$CONFIG_PATH"
fi

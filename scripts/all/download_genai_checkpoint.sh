#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# Download an exported checkpoint from S3 and place it in the exports directory.
#
# Usage:
#   ./scripts/download_checkpoint.sh <s3-url>
#
# Example:
#   ./scripts/download_checkpoint.sh https://my-bucket.s3.amazonaws.com/exports/2026/03/04/12345678/Llama-3.2-1B-Instruct_143022.zip

set -euo pipefail
AWS_PROFILE="genai-laboratory"

# Ensure AWS CLI is installed
if ! command -v aws &>/dev/null; then
  echo "AWS CLI not found, installing..."
  curl -fSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  unzip -q /tmp/awscliv2.zip -d /tmp
  sudo /tmp/aws/install
  rm -rf /tmp/awscliv2.zip /tmp/aws
  echo "AWS CLI installed"
fi

# Ensure saml2aws is installed for AWS credential management
if ! command -v saml2aws &>/dev/null; then
  echo "saml2aws not found, installing..."
  SAML2AWS_VERSION="2.36.17"
  curl -fSL "https://github.com/Versent/saml2aws/releases/download/v${SAML2AWS_VERSION}/saml2aws_${SAML2AWS_VERSION}_linux_amd64.tar.gz" -o /tmp/saml2aws.tar.gz
  tar -xzf /tmp/saml2aws.tar.gz -C /tmp saml2aws
  sudo mv /tmp/saml2aws /usr/local/bin/
  rm -f /tmp/saml2aws.tar.gz /tmp/saml2aws
  echo "saml2aws installed to /usr/local/bin/saml2aws"
fi

# Configure saml2aws if the profile is not present
if [ ! -f ~/.saml2aws ] || ! grep -q "^name\s*=\s*${AWS_PROFILE}\s*$" ~/.saml2aws; then
  echo "Profile '${AWS_PROFILE}' not found in ~/.saml2aws, configuring..."
  if [ -z "${SAML2AWS_APP_ID:-}" ]; then
    echo "SAML2AWS_APP_ID is not set."
    read -rp "Enter your Azure AD App ID: " SAML2AWS_APP_ID
    if [ -z "$SAML2AWS_APP_ID" ]; then
      echo "Error: App ID cannot be empty." >&2
      exit 1
    fi
  fi
  saml2aws configure \
    --idp-provider AzureAD \
    --url https://account.activedirectory.windowsazure.com \
    --app-id "$SAML2AWS_APP_ID" \
    --username "${USER}@qti.qualcomm.com" \
    --mfa Auto \
    --profile "$AWS_PROFILE" \
    --skip-prompt
  echo "saml2aws profile '${AWS_PROFILE}' configured."
fi

# Ensure we have valid AWS credentials
saml2aws login

if [ -z "${1:-}" ]; then
  echo "Usage: $0 <s3-url>" >&2
  echo "  s3-url: S3 URL to an exported checkpoint zip" >&2
  exit 1
fi

URL="$1"
EXPORTS_DIR="GenAILab/artifacts/exports"

# Derive zip filename from URL
ZIP_NAME="$(basename "$URL")"
DIR_NAME="${ZIP_NAME%.zip}"

if [ -z "$DIR_NAME" ] || [ "$DIR_NAME" = "$ZIP_NAME" ]; then
  echo "Error: URL does not point to a .zip file: $URL" >&2
  exit 1
fi

DEST="$EXPORTS_DIR/$DIR_NAME"
if [ -d "$DEST" ]; then
  echo "Checkpoint already exists at $DEST" >&2
  echo ""
  echo "model_id: $DEST"
  exit 0
fi

mkdir -p "$EXPORTS_DIR"
TMP_ZIP="$(mktemp)"
trap 'rm -f "$TMP_ZIP"' EXIT

# Convert https:// URLs to s3:// URIs
# https://bucket.s3.amazonaws.com/key -> s3://bucket/key
# https://bucket.s3.region.amazonaws.com/key -> s3://bucket/key
if [[ "$URL" == https://*.s3*.amazonaws.com/* ]]; then
  BUCKET="$(echo "$URL" | sed -E 's|https://([^.]+)\.s3[^/]*\.amazonaws\.com/.*|\1|')"
  KEY="$(echo "$URL" | sed -E 's|https://[^/]+/(.*)|/\1|')"
  S3_URI="s3://${BUCKET}${KEY}"
elif [[ "$URL" == s3://* ]]; then
  S3_URI="$URL"
else
  echo "Error: Unrecognized URL format: $URL" >&2
  echo "  Expected https://<bucket>.s3.amazonaws.com/<key> or s3://<bucket>/<key>" >&2
  exit 1
fi

# Use the saml profile that saml2aws writes credentials to
echo "Downloading $ZIP_NAME from $S3_URI..."
aws --profile "$AWS_PROFILE" s3 cp "$S3_URI" "$TMP_ZIP"

echo "Extracting to $EXPORTS_DIR/..."
unzip -q "$TMP_ZIP" -d "$EXPORTS_DIR"

# The zip filename may differ from the directory inside it (the Lambda strips
# dates from filenames), so read the actual top-level directory from the zip.
INNER_DIR="$(unzip -Z1 "$TMP_ZIP" | head -1 | cut -d/ -f1)"
EXTRACTED_DIR="$EXPORTS_DIR/${INNER_DIR:-$DIR_NAME}"

echo ""
echo "model_id: $EXTRACTED_DIR"

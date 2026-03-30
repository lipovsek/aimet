#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# Wrapper that makes kubectl exec behave like SSH for rsync.
# rsync calls: <transport> <host> <command...>
# We ignore <host> and run <command> via kubectl exec.

NAMESPACE="$KUBE_NAMESPACE"
POD="$KUBE_POD"

# First arg is the "host" (rsync convention), rest is the command to run
shift
kubectl exec -i -n "$NAMESPACE" "$POD" -- "$@"

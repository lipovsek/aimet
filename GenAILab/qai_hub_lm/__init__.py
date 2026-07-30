# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# Package-wide version for the shippable qai_hub_lm island (schema + driver +
# transforms) -- the unit synced to / pip-installable in AI Hub Models. Alpha
# (pre-0.1): makes NO cross-version compatibility guarantees yet. Intended to be
# bumped on a release cadence and, later, stamped into emitted recipe.yaml files
# and checked (exact-match) on load. That machinery is not wired up yet -- this
# is just the marker.
__version__ = "0.1.0a1"

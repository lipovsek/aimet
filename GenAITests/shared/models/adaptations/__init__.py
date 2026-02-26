# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Model adaptations for GenAI testing."""

from GenAITests.shared.models.adaptations.sha import (
    LlamaSHAAdaptation,
    Qwen3SHAAdaptation,
)
from GenAITests.shared.models.adaptations.sha_conv import (
    LlamaSHAConvAdaptation,
    Qwen3SHAConvAdaptation,
)
from GenAITests.shared.models.adaptations.fast_exportable import (
    Qwen2VLFastExportableAdaptation,
)

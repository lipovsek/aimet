# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


from packaging.version import parse
import torch

if parse(torch.__version__) < parse("2.4.0"):
    # Monkey-patch torch.onnx.export to soothe CI pipeline with torch < 2.4.0
    _export = torch.onnx.export

    def export(*args, **kwargs):
        kwargs.pop("dynamo", None)
        return _export(*args, **kwargs)

    torch.onnx.export = export

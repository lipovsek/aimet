# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: skip-file
# imports start
import numpy as np
import onnx
import onnxruntime as ort
from aimet_onnx import encodings_to_onnx_qdq
# imports end

# Load the exported artifacts
# An AIMET export produces these two files side by side.
model = onnx.load_model('./quantized_mobilenet_v2.onnx')
encodings = './quantized_mobilenet_v2.encodings'
# End of loading the exported artifacts

# Convert to QDQ
qdq_model = encodings_to_onnx_qdq(model, encodings)
onnx.save_model(qdq_model, './quantized_mobilenet_v2_qdq.onnx')
# End of convert to QDQ

# Run the QDQ model
session = ort.InferenceSession(qdq_model.SerializeToString())
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = session.run(None, {'input': dummy_input})
# End of running the QDQ model

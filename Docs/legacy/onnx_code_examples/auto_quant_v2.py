# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

""" Code example for AutoQuantV2 """

import math
import onnxruntime as ort
import numpy as np
from onnxsim import simplify

from aimet_onnx.auto_quant_v2 import AutoQuant
from aimet_onnx.adaround.adaround_weight import AdaroundParameters

# Step 1. Define constants
EVAL_DATASET_SIZE = 5000
CALIBRATION_DATASET_SIZE = 500
BATCH_SIZE = 32

# Step 2. Prepare model and dataloader
onnx_model = Model()
# Simplify the model
onnx_model, _ = simplify(onnx_model)

input_shape = (1, 3, 224, 224)
dummy_data = np.random.randn(*input_shape).astype(np.float32)
dummy_input = {'input': dummy_data}

# NOTE: Use your dataloader. It should iterate over unlabelled dataset.
#       Its data will be directly fed as input to the onnx model's inference session.
unlabelled_data_loader = DataLoader(data=data, batch_size=BATCH_SIZE,
                                    iterations=math.ceil(CALIBRATION_DATASET_SIZE / BATCH_SIZE))

# Step 3. Prepare eval callback
# NOTE: In the actual use cases, the users should implement this part to serve
#       their own goals, maintaining the function signature.
def eval_callback(session: ort.InferenceSession, num_of_samples: Optional[int] = None) -> float:
    data_loader = EvalDataLoader()
    if num_of_samples:
        iterations = math.ceil(num_of_samples / data_loader.batch_size)
    else:
        iterations = len(data_loader)
    batch_cntr = 1
    acc_top1 = 0
    acc_top5 = 0
    for input_data, target in data_loader:
        pred = session.run(None, {'input': input_data})

        batch_avg_top_1_5 = accuracy(pred, target, topk=(1, 5))

        acc_top1 += batch_avg_top_1_5[0].item()
        acc_top5 += batch_avg_top_1_5[1].item()

        batch_cntr += 1
        if batch_cntr > iterations:
            break
    acc_top1 /= iterations
    acc_top5 /= iterations
    return acc_top1

# Step 4. Create AutoQuant object
auto_quant = AutoQuant(onnx_model,
                       dummy_input,
                       unlabelled_data_loader,
                       eval_callback)

# Step 5. (Optional) Set AdaRound params
ADAROUND_DATASET_SIZE = 2000
adaround_data_loader = DataLoader(data=data, batch_size=BATCH_SIZE,
                                  iterations=math.ceil(ADAROUND_DATASET_SIZE / BATCH_SIZE))
adaround_params = AdaroundParameters(adaround_data_loader, num_batches=len(adaround_data_loader))
auto_quant.set_adaround_params(adaround_params)

# Step 6. Run AutoQuant
sim, initial_accuracy = auto_quant.run_inference()
model, optimized_accuracy, encoding_path = auto_quant.optimize(allowed_accuracy_drop=0.01)

print(f"- Quantized Accuracy (before optimization): {initial_accuracy:.4f}")
print(f"- Quantized Accuracy (after optimization):  {optimized_accuracy:.4f}")

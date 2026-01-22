# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=missing-docstring

# [setup]
import torch
from torchvision.models import mobilenet_v2
from aimet_torch.common.quantsim_config.utils import get_path_for_per_channel_config
from aimet_torch import QuantizationSimModel
from aimet_torch.v2.mixed_precision import MixedPrecisionConfigurator

input_shape = (1, 3, 224, 224)
dummy_input = torch.randn(input_shape).cuda()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = mobilenet_v2(pretrained=True).eval().to(device)

# create the sim object. Feel free to change the default settings as you wish
quant_sim = QuantizationSimModel(model,
                                 dummy_input=dummy_input,
                                 default_param_bw=8,
                                 default_output_bw=8,
                                 config_file=get_path_for_per_channel_config())

# create the MMP configurator object
mp_configurator = MixedPrecisionConfigurator(quant_sim)
# [set_precision_leaf]
mp_configurator.set_precision(quant_sim.model.features[1].conv[0][0], activation='int16', param={'weight': 'int16'})
# [set_precision_non_leaf]
mp_configurator.set_precision(quant_sim.model.features[3].conv[1], activation='int16', param={'weight': 'int16'})
# [set_precision_type]
mp_configurator.set_precision(torch.nn.AvgPool2d, activation='int16')
# [set_precision_model_input]
mp_configurator.set_model_input_precision(activation='int16')
# [set_precision_model_output]
mp_configurator.set_model_output_precision(activation='int16')
# [apply]
mp_configurator.apply()

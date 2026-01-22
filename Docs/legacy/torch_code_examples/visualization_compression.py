# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

""" Code examples for visualization APIs """

# Visualization imports
from decimal import Decimal
import torch
from torchvision import models
import aimet_common.defs
import aimet_torch.defs
import aimet_torch.utils
from aimet_common.utils import start_bokeh_server_session
from aimet_torch.compress import ModelCompressor
from aimet_torch.visualize_serialized_data import VisualizeCompression

# End of import statements


def model_compression_with_visualization(eval_func):
    """
    Code example for compressing a model with a visualization url provided.
    """
    process = None
    try:
        visualization_url, process = start_bokeh_server_session()

        input_shape = (1, 3, 224, 224)
        model = models.resnet18(pretrained=True).to(torch.device('cuda'))

        modules_to_ignore = [model.conv1]

        greedy_params = aimet_common.defs.GreedySelectionParameters(target_comp_ratio=Decimal(0.65),
                                                                    num_comp_ratio_candidates=10,
                                                                    saved_eval_scores_dict=
                                                                    '../data/resnet18_eval_scores.pkl')

        auto_params = aimet_torch.defs.SpatialSvdParameters.AutoModeParams(greedy_params,
                                                                           modules_to_ignore=modules_to_ignore)

        params = aimet_torch.defs.SpatialSvdParameters(aimet_torch.defs.SpatialSvdParameters.Mode.auto, auto_params,
                                                       multiplicity=8)

        # If no visualization URL is provided, during model compression execution no visualizations will be published.
        ModelCompressor.compress_model(model=model, eval_callback=eval_func, eval_iterations=5,
                                       input_shape=input_shape,
                                       compress_scheme=aimet_common.defs.CompressionScheme.spatial_svd,
                                       cost_metric=aimet_common.defs.CostMetric.mac, parameters=params,
                                       visualization_url=None)

        comp_ratios_file_path = './data/greedy_selection_comp_ratios_list.pkl'
        eval_scores_path = '../data/resnet18_eval_scores.pkl'

        # A user can visualize the eval scores dictionary and optimal compression ratios by executing the following code.
        compression_visualizations = VisualizeCompression(visualization_url)
        compression_visualizations.display_eval_scores(eval_scores_path)
        compression_visualizations.display_comp_ratio_plot(comp_ratios_file_path)
    finally:
        if process:
            process.terminate()
            process.join()

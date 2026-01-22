# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Class for visualizing after compression is completed"""

import pickle
import pandas as pd
from bokeh.models import ColumnDataSource, DataTable, TableColumn
from aimet_torch.common.compression_algo import CompressionAlgo
from aimet_torch.common.bokeh_plots import BokehServerSession
from aimet_torch.common import plotting_utils


class VisualizeCompression:
    """Updates bokeh server session document and publishes graphs/tables to the server with session id compression."""

    def __init__(self, visualization_url):
        self.bokeh_session = BokehServerSession(
            visualization_url, session_id="compression"
        )
        self.__document = self.bokeh_session.document

    def display_eval_scores(self, saved_eval_scores_dict_path):
        """
        Publishes the evaluation scores table to the server.

        :param saved_eval_scores_dict_path: file path to the evaluation scores for each layer
        :return: None
        """
        with open(saved_eval_scores_dict_path, "rb") as infile:
            eval_scores_dict = pickle.load(infile)

        eval_scores_data_frame = pd.DataFrame.from_dict(eval_scores_dict).T
        eval_scores_data_frame.columns = eval_scores_data_frame.columns.map(str)
        eval_scores_data_frame.insert(0, "layers", eval_scores_data_frame.index)

        source = ColumnDataSource(data=eval_scores_data_frame)
        columns = [
            TableColumn(field=Ci, title=Ci) for Ci in eval_scores_data_frame.columns
        ]  # bokeh columns
        eval_scores_data_table = DataTable(source=source, columns=columns, width=1500)

        self.__document.add_root(eval_scores_data_table)

    def display_comp_ratio_plot(self, comp_ratio_list_path):
        """
        Publishes the optimal compression ratios to the server.

        :param comp_ratio_list_path: Path to the pkl file with compression ratios for each layer
        :return: None
        """
        layer_comp_ratio_list = CompressionAlgo.unpickle_comp_ratios_list(
            comp_ratio_list_path=comp_ratio_list_path
        )

        # visualize comp ratios vs layers in a plot and add it to a server session document.
        comp_ratios = []
        layer_names = []
        for layer_name, comp_ratio in layer_comp_ratio_list:
            comp_ratios.append(comp_ratio)
            layer_names.append(layer_name)

        plot = plotting_utils.plot_optimal_compression_ratios(comp_ratios, layer_names)
        self.__document.add_root(plot)

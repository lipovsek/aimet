# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Creates stats for SVD"""

import logging


class SvdStatistics:
    """Class that models statistics returned as a result of doing SVD compression"""

    def __init__(
        self,
        base_accuracy,
        comp_accuracy,
        cost_metric,
        best_index,
        mem_comp_ratio,
        mac_comp_ratio,
        rank_stats_list,
    ):
        self.baseline_model_accuracy = base_accuracy
        self.compressed_model_accuracy = comp_accuracy
        self.cost_metric = cost_metric
        self.best_rank_index = best_index
        self.memory_compression_ratio = mem_comp_ratio
        self.mac_compression_ratio = mac_comp_ratio
        self.per_rank_index = rank_stats_list

    class PerRankIndex:
        """Inner-class that models SVD statistics for a given rank index"""

        def __init__(
            self, rank_index, model_accuracy, model_compression_ratio, layer_stats_list
        ):
            self.rank_index = rank_index
            self.model_accuracy = model_accuracy
            self.model_compression_ratio = model_compression_ratio
            self.per_selected_layer = layer_stats_list

    class PerSelectedLayer:
        """Inner-class that models SVD statistics for a given layer selected for compression"""

        def __init__(self, name, rank, compression_ratio):
            self.layer_name = name
            self.rank = rank
            self.compression_ratio = compression_ratio

    def pretty_print(self, logger):
        """Function that logs the SVD statistics in a pretty format."""
        if logger is None:
            logger = logging.getLogger()

        logger.info(
            "*************************************************************************************************"
        )
        logger.info("Compressed Model SVD Statistics")
        logger.info(
            "Baseline Model Accuracy: %r   Compressed Model Accuracy: %r ",
            self.baseline_model_accuracy,
            self.compressed_model_accuracy,
        )
        logger.info(
            "Cost Metric: %s   Best Rank Index: %r",
            self.cost_metric,
            self.best_rank_index,
        )
        logger.info(
            "Memory Compression Ratio: %r   MAC Compression Ratio: %r",
            self.memory_compression_ratio,
            self.mac_compression_ratio,
        )
        logger.info(
            "*************************************************************************************************"
        )
        logger.info("Per Rank Index Model SVD Statistics")
        logger.info(
            "-------------------------------------------------------------------------------------------------"
        )
        for rank_index_stats in self.per_rank_index:
            logger.info(
                "Rank Index: {:2}     Accuracy: {}   Compression Ratio: {}".format(
                    rank_index_stats.rank_index,
                    rank_index_stats.model_accuracy,
                    rank_index_stats.model_compression_ratio,
                )
            )
            for layer_stats in self.per_rank_index[
                rank_index_stats.rank_index
            ].per_selected_layer:
                logger.info(
                    "Layer Name: %s   Rank: %r   Compression Ratio: %r",
                    layer_stats.layer_name,
                    layer_stats.rank,
                    layer_stats.compression_ratio,
                )
            logger.info(
                "---------------------------------------------------------------------------------------------"
            )
        logger.info(
            "*************************************************************************************************"
        )
        logger.info("")

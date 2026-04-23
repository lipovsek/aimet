# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Quant Analyzer for AIMET v2"""

import os
import contextlib
from collections import namedtuple
from typing import Tuple, List, Type, Optional, Generator
import torch

from aimet_torch.common.quant_analyzer import export_stats_histogram_plot
from aimet_torch._base.quant_analyzer import QuantAnalyzerBase
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.nn.base import BaseQuantizationMixin
from aimet_torch.quantization.base import QuantizerBase
from aimet_torch.quantization.encoding_analyzer import _HistogramObserver, _Histogram
from aimet_torch.batch_norm_fold import fold_all_batch_norms


V1Encoding = namedtuple("V1Encoding", ["min", "max"])


class QuantAnalyzer(QuantAnalyzerBase):
    """
    QuantAnalyzer tool provides

     1) model sensitivity to weight and activation quantization
     2) per layer sensitivity analysis
     3) per layer encoding (min - max range)
     4) per PDF analysis and
     5) per layer MSE analysis
    """

    @staticmethod
    def _get_quantsim_cls() -> Type[QuantizationSimModel]:
        return QuantizationSimModel

    @staticmethod
    def _get_quant_wrapper_type() -> Tuple[Type]:
        return (BaseQuantizationMixin,)

    def _create_and_export_stats_histogram_plot(
        self,
        quantizer: QuantizerBase,
        results_dir: str,
        title: str,
    ):
        """
        For given quantizer, create and export histogram (PDF) of statistics in html format.

        :param quantizer: Quantizer.
        :param results_dir: Directory to save the results.
        :param title: Title of the plot.
        """
        if isinstance(quantizer.encoding_analyzer.observer, _HistogramObserver):
            os.makedirs(results_dir, exist_ok=True)

            v2_histograms = quantizer.encoding_analyzer.observer.get_stats()
            histograms = self._convert_to_v1_histograms(v2_histograms)
            encodings = self._get_quantizer_encodings(quantizer)

            for index, (histogram, encoding) in enumerate(zip(histograms, encodings)):
                export_stats_histogram_plot(
                    histogram, encoding, results_dir, title=f"{title}_{index}"
                )

    @staticmethod
    def _enable_disable_quantizers(quantizers: List[QuantizerBase], enabled: bool):
        """
        For given list of quantizers, set (enable/disable) quantizer's enabled.

        :param quantizers: List of quantizers.
        :param enabled: Enabled flag.
        """
        raise RuntimeError("Changing enabled attribute is not allowed in quantsim v2")

    @classmethod
    def _disable_param_quantizers(cls, sim: QuantizationSimModel):
        # pylint: disable=protected-access
        ctx = contextlib.ExitStack()
        for quant_wrapper in cls._get_quantized_modules(sim):
            ctx.enter_context(quant_wrapper._remove_param_quantizers())
        return ctx

    @classmethod
    def _disable_activation_quantizers(cls, sim: QuantizationSimModel):
        # pylint: disable=protected-access
        ctx = contextlib.ExitStack()
        for quant_wrapper in cls._get_quantized_modules(sim):
            ctx.enter_context(quant_wrapper._remove_activation_quantizers())
        return ctx

    @staticmethod
    def _disable_quant_wrapper(module: BaseQuantizationMixin):
        # pylint: disable=protected-access
        return module._remove_all_quantizers()

    @staticmethod
    def _convert_to_v1_histograms(histograms: List[_Histogram]) -> List:
        v1_histograms = []
        for hist in histograms:
            assert hist is not None, "Cannot find histogram data in quantizer"
            hist_sum = torch.sum(hist.histogram).item()
            v1_hist = []
            for bin_edge, hist_value in zip(hist.bin_edges, hist.histogram):
                v1_hist.append((bin_edge.item(), hist_value.item() / hist_sum))
            v1_histograms.append(v1_hist)

        return v1_histograms

    @staticmethod
    @contextlib.contextmanager
    def _recompute_param_histogram(quantizer: QuantizerBase, param: torch.nn.Parameter):
        with quantizer.compute_encodings():
            _ = quantizer(param)
        yield
        quantizer.encoding_analyzer.reset_stats()

    @staticmethod
    def _is_quantizer_enabled(quantizer: Optional[QuantizerBase]):
        return quantizer is not None

    @classmethod
    def _get_quantizer_encodings(cls, quantizer: QuantizerBase) -> Optional[List]:
        v1_encodings = []

        encoding = quantizer.get_encodings()
        if not encoding:
            return None

        flatten_min = encoding.min.flatten()
        flatten_max = encoding.max.flatten()

        for encoding_min, encoding_max in zip(flatten_min, flatten_max):
            v1_encodings.append(
                V1Encoding(min=encoding_min.item(), max=encoding_max.item())
            )

        return v1_encodings

    @staticmethod
    def _get_quantized_modules(
        sim: QuantizationSimModel,
    ) -> Generator[BaseQuantizationMixin, None, None]:
        for module in sim.model.modules():
            if isinstance(module, BaseQuantizationMixin):
                yield module

    @staticmethod
    def _fold_all_batch_norms(*args, **kwargs):
        return fold_all_batch_norms(*args, **kwargs)

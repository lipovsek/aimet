# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Torch quantsim utility functions."""

from unittest.mock import MagicMock, patch, call

import pytest

from GenAILab.bench.precision import (
    Granularity,
    PrecisionConfig,
    WeightPrecision,
    int4,
    int8,
)


class TestApplyBlockGranularityToDecoderStack:
    def test_lpbq_calls_grouped_blockwise(self):
        from GenAILab.qai_hub_lm.backends.torch import quantsim_utils

        precision = PrecisionConfig.from_dict(
            {
                "blocks": {
                    "default": {"qtype": 4, "granularity": "LPBQ", "block_size": 128}
                }
            }
        )
        mock_sim = MagicMock()

        with patch.object(
            quantsim_utils, "set_grouped_blockwise_quantization_for_weights"
        ) as mock_gbq:
            quantsim_utils._apply_block_granularity_to_decoder_stack(
                mock_sim, precision
            )
            mock_gbq.assert_called_once()
            kwargs = mock_gbq.call_args
            assert kwargs[1]["bitwidth"] == 4
            assert kwargs[1]["block_size"] == 128
            assert kwargs[1]["symmetric"] is True
            assert kwargs[1]["decompressed_bw"] == 8

    def test_bq_calls_blockwise(self):
        from GenAILab.qai_hub_lm.backends.torch import quantsim_utils

        precision = PrecisionConfig.from_dict(
            {"blocks": {"default": {"qtype": 4, "granularity": "BQ", "block_size": 64}}}
        )
        mock_sim = MagicMock()

        with patch.object(
            quantsim_utils, "set_blockwise_quantization_for_weights"
        ) as mock_bq:
            quantsim_utils._apply_block_granularity_to_decoder_stack(
                mock_sim, precision
            )
            mock_bq.assert_called_once()
            kwargs = mock_bq.call_args
            assert kwargs[1]["bitwidth"] == 4
            assert kwargs[1]["block_size"] == 64
            assert kwargs[1]["symmetric"] is True

    def test_pcq_does_not_call_block_functions(self):
        from GenAILab.qai_hub_lm.backends.torch import quantsim_utils

        precision = PrecisionConfig.from_dict({"blocks": {"default": {"qtype": 8}}})
        mock_sim = MagicMock()

        with (
            patch.object(
                quantsim_utils, "set_grouped_blockwise_quantization_for_weights"
            ) as mock_gbq,
            patch.object(
                quantsim_utils, "set_blockwise_quantization_for_weights"
            ) as mock_bq,
        ):
            quantsim_utils._apply_block_granularity_to_decoder_stack(
                mock_sim, precision
            )
            mock_gbq.assert_not_called()
            mock_bq.assert_not_called()


class TestSetLmHeadPrecision:
    def test_lpbq_calls_grouped_blockwise(self):
        from GenAILab.qai_hub_lm.backends.torch import quantsim_utils

        precision = WeightPrecision(
            qtype=int4, granularity=Granularity.LPBQ, block_size=128
        )
        mock_sim = MagicMock()

        with patch.object(
            quantsim_utils, "set_grouped_blockwise_quantization_for_weights"
        ) as mock_gbq:
            quantsim_utils._set_lm_head_precision(mock_sim, precision)
            mock_gbq.assert_called_once()
            assert mock_gbq.call_args[1]["bitwidth"] == 4
            assert mock_gbq.call_args[1]["block_size"] == 128

    def test_bq_calls_blockwise(self):
        from GenAILab.qai_hub_lm.backends.torch import quantsim_utils

        precision = WeightPrecision(
            qtype=int4, granularity=Granularity.BQ, block_size=64
        )
        mock_sim = MagicMock()

        with patch.object(
            quantsim_utils, "set_blockwise_quantization_for_weights"
        ) as mock_bq:
            quantsim_utils._set_lm_head_precision(mock_sim, precision)
            mock_bq.assert_called_once()

    def test_pcq_sets_bitwidth_directly(self):
        from GenAILab.qai_hub_lm.backends.torch import quantsim_utils

        precision = WeightPrecision(qtype=int8)
        mock_sim = MagicMock()

        quantsim_utils._set_lm_head_precision(mock_sim, precision)
        assert mock_sim.model.model.lm_head.param_quantizers["weight"].bitwidth == 8

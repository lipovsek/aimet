# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ONNX generator utility functions."""

from unittest.mock import MagicMock, patch

import pytest

from GenAILab.shared.models.base import SimCollection
from GenAILab.shared.models.generator import Generator, VLM_Generator


class TestBuildFpMode:
    def test_returns_context_manager_factory(self):
        from GenAILab.onnx.models.utils.generator_utils import _build_fp_mode

        mock_backbone = MagicMock()
        mock_backbone._remove_quantization_nodes.return_value.__enter__ = MagicMock()
        mock_backbone._remove_quantization_nodes.return_value.__exit__ = MagicMock(
            return_value=False
        )
        collection = SimCollection(backbone=mock_backbone)

        fp_mode = _build_fp_mode(collection)
        assert callable(fp_mode)

    def test_calls_remove_and_rebuild(self):
        from GenAILab.onnx.models.utils.generator_utils import _build_fp_mode

        mock_backbone = MagicMock()
        mock_backbone._remove_quantization_nodes.return_value.__enter__ = MagicMock()
        mock_backbone._remove_quantization_nodes.return_value.__exit__ = MagicMock(
            return_value=False
        )
        collection = SimCollection(backbone=mock_backbone)

        fp_mode = _build_fp_mode(collection)
        with fp_mode():
            mock_backbone._remove_quantization_nodes.assert_called()
            mock_backbone._rebuild_session.assert_called()

    def test_handles_visual_model(self):
        from GenAILab.onnx.models.utils.generator_utils import _build_fp_mode

        mock_backbone = MagicMock()
        mock_backbone._remove_quantization_nodes.return_value.__enter__ = MagicMock()
        mock_backbone._remove_quantization_nodes.return_value.__exit__ = MagicMock(
            return_value=False
        )
        mock_visual = MagicMock()
        mock_visual._remove_quantization_nodes.return_value.__enter__ = MagicMock()
        mock_visual._remove_quantization_nodes.return_value.__exit__ = MagicMock(
            return_value=False
        )

        collection = SimCollection(backbone=mock_backbone, visual=mock_visual)

        fp_mode = _build_fp_mode(collection)
        with fp_mode():
            mock_visual._remove_quantization_nodes.assert_called()
            mock_visual._rebuild_session.assert_called()


class TestGeneratorFactory:
    def test_wraps_backbone_in_torch_onnx_interface(self):
        from GenAILab.onnx.models.utils.generator_utils import generator_factory

        mock_backbone = MagicMock()
        mock_backbone._remove_quantization_nodes.return_value.__enter__ = MagicMock()
        mock_backbone._remove_quantization_nodes.return_value.__exit__ = MagicMock(
            return_value=False
        )
        mock_config = MagicMock()
        collection = SimCollection(backbone=mock_backbone, config=mock_config)

        tok = MagicMock()
        tok.eos_token_id = 0

        with patch(
            "GenAILab.onnx.models.utils.generator_utils.TorchONNXInterface"
        ) as mock_interface:
            mock_interface.return_value = MagicMock()
            mock_interface.return_value.config = mock_config
            mock_interface.return_value.dtype = MagicMock()

            gen = generator_factory(
                sim_collection=collection,
                generator_cls=Generator,
                tokenizer=tok,
                sequence_length=8,
                context_length=32,
            )
            mock_interface.assert_called_once_with(mock_backbone, mock_config)
            assert isinstance(gen, Generator)

    def test_vlm_wraps_both_models(self):
        from GenAILab.onnx.models.utils.generator_utils import generator_factory

        mock_backbone = MagicMock()
        mock_backbone._remove_quantization_nodes.return_value.__enter__ = MagicMock()
        mock_backbone._remove_quantization_nodes.return_value.__exit__ = MagicMock(
            return_value=False
        )
        mock_visual = MagicMock()
        mock_visual._remove_quantization_nodes.return_value.__enter__ = MagicMock()
        mock_visual._remove_quantization_nodes.return_value.__exit__ = MagicMock(
            return_value=False
        )
        mock_config = MagicMock()
        mock_config.text_config = MagicMock()
        embedding = MagicMock()

        collection = SimCollection(
            backbone=mock_backbone,
            visual=mock_visual,
            embedding=embedding,
            config=mock_config,
        )

        tok = MagicMock()
        tok.eos_token_id = 0

        with patch(
            "GenAILab.onnx.models.utils.generator_utils.TorchONNXInterface"
        ) as mock_interface:
            mock_bb_wrapped = MagicMock()
            mock_vis_wrapped = MagicMock()
            mock_interface.side_effect = [mock_bb_wrapped, mock_vis_wrapped]

            gen = generator_factory(
                sim_collection=collection,
                generator_cls=VLM_Generator,
                tokenizer=tok,
                sequence_length=8,
                context_length=32,
            )
            assert mock_interface.call_count == 2
            assert isinstance(gen, VLM_Generator)

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Torch generator utility functions."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from aimet_torch.quantsim import QuantizationSimModel
from GenAILab.shared.models.base import SimCollection
from GenAILab.shared.models.generator import Generator, VLM_Generator


class TestPlaceCollection:
    def test_places_backbone(self):
        from GenAILab.torch.models.utils.generator_utils import place_collection

        mock_sim = MagicMock(spec=QuantizationSimModel)
        mock_sim.model = torch.nn.Linear(4, 4)
        collection = SimCollection(backbone=mock_sim)

        with patch(
            "GenAILab.torch.models.utils.generator_utils.place_model"
        ) as mock_place:
            mock_place.return_value.__enter__ = MagicMock()
            mock_place.return_value.__exit__ = MagicMock(return_value=False)
            with place_collection(collection, torch.device("cpu")):
                pass
            mock_place.assert_called()

    def test_skips_none_fields(self):
        from GenAILab.torch.models.utils.generator_utils import place_collection

        collection = SimCollection(backbone=MagicMock())
        collection.backbone = None
        collection.visual = None
        collection.embedding = None

        # Should not raise even with all None fields
        with patch(
            "GenAILab.torch.models.utils.generator_utils.place_model"
        ) as mock_place:
            mock_place.return_value.__enter__ = MagicMock()
            mock_place.return_value.__exit__ = MagicMock(return_value=False)
            with place_collection(collection, torch.device("cpu")):
                pass


class TestBuildFpMode:
    def test_returns_context_manager_factory(self):
        from GenAILab.torch.models.utils.generator_utils import _build_fp_mode

        mock_sim = MagicMock()
        mock_sim.model = torch.nn.Linear(4, 4)
        collection = SimCollection(backbone=mock_sim)

        with patch(
            "GenAILab.torch.models.utils.generator_utils.remove_all_quantizers"
        ) as mock_remove:
            mock_remove.return_value.__enter__ = MagicMock()
            mock_remove.return_value.__exit__ = MagicMock(return_value=False)
            fp_mode = _build_fp_mode(collection)
            assert callable(fp_mode)
            with fp_mode():
                mock_remove.assert_called()

    def test_handles_visual_model(self):
        from GenAILab.torch.models.utils.generator_utils import _build_fp_mode

        mock_backbone = MagicMock()
        mock_backbone.model = torch.nn.Linear(4, 4)
        mock_visual = MagicMock()
        mock_visual.model = torch.nn.Linear(4, 4)
        embedding = torch.nn.Embedding(10, 4)

        collection = SimCollection(
            backbone=mock_backbone, visual=mock_visual, embedding=embedding
        )

        with patch(
            "GenAILab.torch.models.utils.generator_utils.remove_all_quantizers"
        ) as mock_remove:
            mock_remove.return_value.__enter__ = MagicMock()
            mock_remove.return_value.__exit__ = MagicMock(return_value=False)
            fp_mode = _build_fp_mode(collection)
            with fp_mode():
                # Should be called for backbone and visual
                assert mock_remove.call_count >= 2


class TestGeneratorFactory:
    def test_creates_generator_for_llm(self):
        from GenAILab.torch.models.utils.generator_utils import generator_factory

        mock_sim = MagicMock()
        mock_sim.model = torch.nn.Linear(4, 4)
        collection = SimCollection(backbone=mock_sim)

        tok = MagicMock()
        tok.eos_token_id = 0

        with patch(
            "GenAILab.torch.models.utils.generator_utils.remove_all_quantizers"
        ) as mock_remove:
            mock_remove.return_value.__enter__ = MagicMock()
            mock_remove.return_value.__exit__ = MagicMock(return_value=False)
            gen = generator_factory(
                sim_collection=collection,
                generator_cls=Generator,
                tokenizer=tok,
                sequence_length=8,
                context_length=32,
            )
            assert isinstance(gen, Generator)

    def test_creates_vlm_generator_for_vlm(self):
        from GenAILab.torch.models.utils.generator_utils import generator_factory

        mock_backbone = MagicMock()
        mock_backbone.model = torch.nn.Linear(4, 4)
        mock_visual = MagicMock()
        mock_visual.model = torch.nn.Linear(4, 4)
        embedding = torch.nn.Embedding(10, 4)
        mock_config = MagicMock()

        collection = SimCollection(
            backbone=mock_backbone,
            visual=mock_visual,
            embedding=embedding,
            config=mock_config,
        )

        tok = MagicMock()
        tok.eos_token_id = 0

        with patch(
            "GenAILab.torch.models.utils.generator_utils.remove_all_quantizers"
        ) as mock_remove:
            mock_remove.return_value.__enter__ = MagicMock()
            mock_remove.return_value.__exit__ = MagicMock(return_value=False)
            gen = generator_factory(
                sim_collection=collection,
                generator_cls=VLM_Generator,
                tokenizer=tok,
                sequence_length=8,
                context_length=32,
            )
            assert isinstance(gen, VLM_Generator)

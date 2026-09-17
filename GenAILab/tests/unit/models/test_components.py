# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the component registry and the generalized SimCollection."""

from unittest.mock import MagicMock

import pytest

from GenAILab.qai_hub_lm.models.base import VLM, SimCollection
from GenAILab.qai_hub_lm.models.components import (
    AUDIO,
    COMPONENTS,
    VISUAL,
    ComponentKind,
    input_encoders,
    model_components,
    prefill_only_keys,
    spec,
)
from GenAILab.qai_hub_lm.schema.components import (
    ALL_COMPONENTS,
    BACKBONE,
    MODALITY_COMPONENTS,
)


class TestRegistry:
    def test_schema_and_model_layers_agree_on_names_and_order(self):
        # The schema half is the source of truth for names/order; the model half
        # adds hook wiring. Drift between them would desync recipe/precision
        # keys from the sims they configure.
        assert tuple(COMPONENTS) == tuple(MODALITY_COMPONENTS)
        assert ALL_COMPONENTS == (BACKBONE, *COMPONENTS)

    def test_backbone_is_not_a_modality_component(self):
        assert BACKBONE not in COMPONENTS

    def test_spec_names_match_their_keys(self):
        for name, component in COMPONENTS.items():
            assert component.name == name

    def test_visual_and_audio_are_input_encoders(self):
        assert VISUAL.kind is ComponentKind.input_encoder
        assert AUDIO.kind is ComponentKind.input_encoder
        assert set(input_encoders()) == {VISUAL, AUDIO}

    def test_spec_lookup_rejects_unknown_with_helpful_message(self):
        with pytest.raises(KeyError, match="Unknown component"):
            spec("olfactory")

    def test_prefill_only_keys_covers_every_component(self):
        keys = prefill_only_keys()
        assert "pixel_values" in keys
        assert "input_features" in keys
        assert "input_features_mask" in keys

    def test_audio_token_id_attr_matches_hf_config_field(self):
        # Qwen3-ASR and Gemma4 both expose config.audio_token_id.
        assert AUDIO.token_id_attr == "audio_token_id"
        assert VISUAL.token_id_attr == "image_token_id"


class TestModelComponents:
    def test_vlm_declares_visual_by_default(self):
        # Every pre-audio model class inherited vision, so the default keeps
        # them working unchanged.
        assert model_components(VLM) == (VISUAL,)

    def test_declaration_is_returned_in_canonical_order(self):
        class Both(VLM):
            COMPONENTS = (AUDIO.name, VISUAL.name)

            @classmethod
            def instantiate_position_processor(cls):
                return None

        assert model_components(Both) == (VISUAL, AUDIO)

    def test_unknown_declared_component_raises(self):
        class Bogus(VLM):
            COMPONENTS = ("telepathy",)

            @classmethod
            def instantiate_position_processor(cls):
                return None

        with pytest.raises(ValueError, match="unknown component"):
            model_components(Bogus)

    def test_plain_class_declares_nothing(self):
        assert model_components(object) == ()


class TestSupportsComponent:
    """Capability gating: a declared component may still be absent per checkpoint."""

    def test_defaults_to_the_class_declaration(self):
        config = MagicMock()
        assert VLM.supports_component("visual", config)
        assert not VLM.supports_component("audio", config)

    def test_gemma4_gates_audio_on_audio_config(self):
        # Gemma4 declares both encoders, but publishes variants with no audio
        # tower — building an audio sim for those would fail on a None module.
        from GenAILab.qai_hub_lm.models.gemma4 import Gemma4_VLM

        assert Gemma4_VLM.COMPONENTS == ("visual", "audio")

        with_audio = MagicMock()
        with_audio.audio_config = MagicMock()
        with_audio.vision_config = MagicMock()
        assert Gemma4_VLM.supports_component("audio", with_audio)
        assert Gemma4_VLM.supports_component("visual", with_audio)

        without_audio = MagicMock()
        without_audio.audio_config = None
        without_audio.vision_config = MagicMock()
        assert not Gemma4_VLM.supports_component("audio", without_audio)
        assert Gemma4_VLM.supports_component("visual", without_audio)

    def test_qwen3_asr_supports_only_audio(self):
        from GenAILab.qai_hub_lm.models.qwen3_asr import Qwen3ASR_LM

        config = MagicMock()
        assert Qwen3ASR_LM.supports_component("audio", config)
        assert not Qwen3ASR_LM.supports_component("visual", config)


def _audio_config(channels):
    class _Cfg:
        subsampling_conv_channels = channels

    config = MagicMock()
    config.audio_config = _Cfg()
    return config


class TestGemma4AudioShapeMath:
    def test_soft_token_count_follows_the_mask_halving(self):
        from GenAILab.qai_hub_lm.models.gemma4 import Gemma4_VLM

        # Two stride-2 subsampling layers, each slicing the mask [:, ::2].
        config = _audio_config([128, 32])
        assert Gemma4_VLM.audio_soft_tokens(config, 800) == 200
        assert Gemma4_VLM.audio_soft_tokens(config, 400) == 100
        # Odd lengths round up at each halving.
        assert Gemma4_VLM.audio_soft_tokens(config, 3) == 1
        assert Gemma4_VLM.audio_soft_tokens(config, 5) == 2

    def test_halving_depth_follows_the_channel_count(self):
        from GenAILab.qai_hub_lm.models.gemma4 import Gemma4_VLM

        assert Gemma4_VLM.audio_soft_tokens(_audio_config([128]), 800) == 400
        assert Gemma4_VLM.audio_soft_tokens(_audio_config([128, 32, 8]), 800) == 100

    def test_missing_channels_raises_rather_than_guessing(self):
        from GenAILab.qai_hub_lm.models.gemma4 import Gemma4_VLM

        config = MagicMock()
        config.audio_config = object()  # no subsampling_conv_channels
        with pytest.raises(ValueError, match="subsampling_conv_channels"):
            Gemma4_VLM.get_num_mel_bins(config)
        with pytest.raises(ValueError, match="subsampling_conv_channels"):
            Gemma4_VLM.audio_soft_tokens(config, 800)

    def test_mel_bins_come_from_subsampling_conv_channels(self):
        from GenAILab.qai_hub_lm.models.gemma4 import Gemma4_VLM

        # Gemma4AudioConfig has no feature-width field, but the encoder sizes
        # input_proj_linear from subsampling_conv_channels[0], so that value IS
        # the mel-bin count.
        assert Gemma4_VLM.get_num_mel_bins(_audio_config([128, 32])) == 128
        assert Gemma4_VLM.get_num_mel_bins(_audio_config([32, 4])) == 32


class TestPerComponentExportPath:
    def test_default_follows_the_model_wide_setting(self):
        assert VLM.use_dynamo_export_for("visual") is VLM.use_dynamo_export()
        assert VLM.use_dynamo_export_for() is VLM.use_dynamo_export()

    def test_gemma4_uses_dynamo_only_for_audio(self):
        # Measured: the audio tower cannot be torchscript-traced (masking_utils
        # vmap) but exports via torch.export; vision/backbone still trace.
        from GenAILab.qai_hub_lm.models.gemma4 import Gemma4_VLM

        assert Gemma4_VLM.use_dynamo_export_for("audio") is True
        assert Gemma4_VLM.use_dynamo_export_for("visual") is False
        assert Gemma4_VLM.use_dynamo_export() is False

    def test_qwen3_asr_keeps_torchscript(self):
        # Verified by smoke test: torch.jit.trace works with an all-ones mask.
        from GenAILab.qai_hub_lm.models.qwen3_asr import Qwen3ASR_LM

        assert Qwen3ASR_LM.use_dynamo_export_for("audio") is False


class TestSimCollectionComponents:
    def test_components_are_keyword_only(self):
        backbone, visual, embedding = (MagicMock() for _ in range(3))
        with pytest.raises(TypeError):
            SimCollection(backbone, visual, embedding)

    def test_visual_kwarg_sets_visual_component(self):
        visual = MagicMock()
        sc = SimCollection(backbone=MagicMock(), visual=visual)
        assert sc.component("visual") is visual
        assert sc.is_vlm()

    def test_audio_kwarg_accepted(self):
        audio = MagicMock()
        sc = SimCollection(backbone=MagicMock(), audio=audio)
        assert sc.audio is audio
        assert sc.has("audio")
        # An audio-only model is NOT a VLM.
        assert not sc.is_vlm()
        assert sc.visual is None

    def test_unknown_component_kwarg_rejected(self):
        with pytest.raises(TypeError):
            SimCollection(backbone=MagicMock(), **{"taste": MagicMock()})

    def test_component_accessor_rejects_unknown_name(self):
        sc = SimCollection(backbone=MagicMock())
        with pytest.raises(KeyError, match="Unknown component"):
            sc.component("taste")

    def test_component_attributes_are_writable(self):
        # bench code and tests do `collection.visual = None` to disable it.
        sc = SimCollection(backbone=MagicMock(), visual=MagicMock())
        sc.visual = None
        assert sc.visual is None
        assert not sc.is_vlm()
        assert not sc.has("visual")

    def test_absent_component_reads_as_none_not_attribute_error(self):
        sc = SimCollection(backbone=MagicMock())
        assert sc.visual is None
        assert sc.audio is None

    def test_present_components_is_in_canonical_order(self):
        sc = SimCollection(
            backbone=MagicMock(),
            audio=MagicMock(),
            visual=MagicMock(),
        )
        assert sc.present_components() == ("visual", "audio")

    def test_iter_members_yields_backbone_components_and_embedding(self):
        backbone, visual, audio, embedding = (MagicMock() for _ in range(4))
        sc = SimCollection(
            backbone=backbone,
            embedding=embedding,
            visual=visual,
            audio=audio,
        )
        members = list(sc.iter_members())
        assert backbone in members
        assert visual in members
        assert audio in members
        assert embedding in members
        # config / extras / position_id_processor are not placeable sims.
        assert sc.config not in members or sc.config is None

    def test_iter_members_yields_none_for_absent_components(self):
        # Callers filter on isinstance, so a None placeholder is fine — but the
        # count must stay stable so placement logic cannot silently skip one.
        sc = SimCollection(backbone=MagicMock())
        assert len(list(sc.iter_members())) == 2 + len(COMPONENTS)


class TestDeclaredComponentsAreImplemented:
    """Replaces what ``@abstractmethod`` guaranteed before the hooks became
    concrete (so an audio-only model need not supply vision hooks)."""

    def test_every_registered_model_implements_its_component_hooks(self):
        import inspect

        from GenAILab.bench.yaml_config_parser import YAMLConfigParser
        from GenAILab.qai_hub_lm.models.base import VLM

        missing = []
        for name, model_cls in YAMLConfigParser.model_lookup.items():
            if not issubclass(model_cls, VLM):
                continue
            for component in model_components(model_cls):
                for hook in (
                    component.build_wrapper,
                    component.sample_inputs,
                    component.input_names,
                ):
                    own = inspect.getattr_static(model_cls, hook, None)
                    base = inspect.getattr_static(VLM, hook, None)
                    if own is base:
                        missing.append(f"{name} ({model_cls.__name__}): {hook}")
        assert not missing, "Declared components with unimplemented hooks: " + str(
            missing
        )


class TestAudioFeaturePadding:
    """``input_features`` has no agreed layout -- Qwen3-ASR ``[B, mel, time]``,
    Gemma4 ``[B, time, mel]`` -- and padding the wrong axis fails only in
    onnxruntime, so each model owns ``pad_audio_item``."""

    @staticmethod
    def _item(shape, frames):
        import torch

        return {
            "input_features": torch.zeros(shape),
            "input_features_mask": torch.ones((1, frames), dtype=torch.long),
        }

    def test_base_refuses_to_guess_the_layout(self):
        from GenAILab.qai_hub_lm.models.generator import VLM_Generator

        gen = MagicMock()
        with pytest.raises(NotImplementedError, match="pad_audio_item"):
            VLM_Generator.pad_audio_item(gen, self._item((1, 300, 128), 300), 800)

    def test_gemma4_pads_the_time_axis_not_the_mel_axis(self):
        from GenAILab.qai_hub_lm.models.gemma4 import Gemma4_VLM_Generator

        gen = MagicMock()
        # (B, time, mel) = (1, 300, 128) -> time padded to 800, mel untouched.
        item = Gemma4_VLM_Generator.pad_audio_item(
            gen, self._item((1, 300, 128), 300), 800
        )
        assert tuple(item["input_features"].shape) == (1, 800, 128)
        assert tuple(item["input_features_mask"].shape) == (1, 800)

    def test_gemma4_mask_stays_a_real_validity_mask(self):
        from GenAILab.qai_hub_lm.models.gemma4 import Gemma4_VLM_Generator

        gen = MagicMock()
        item = Gemma4_VLM_Generator.pad_audio_item(
            gen, self._item((1, 300, 128), 300), 800
        )
        mask = item["input_features_mask"]
        assert mask[:, :300].all()
        assert not mask[:, 300:].any()

    def test_gemma4_over_length_truncates_time_only(self):
        from GenAILab.qai_hub_lm.models.gemma4 import Gemma4_VLM_Generator

        gen = MagicMock()
        with pytest.warns(UserWarning, match="still exceeding audio_frames"):
            item = Gemma4_VLM_Generator.pad_audio_item(
                gen, self._item((1, 900, 128), 900), 800
            )
        assert tuple(item["input_features"].shape) == (1, 800, 128)
        assert tuple(item["input_features_mask"].shape) == (1, 800)

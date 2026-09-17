# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Schema-side tests for the audio component: recipe chain + precision block.

The cache-identity tests here matter as much as the round-trips: adding audio
must not perturb the hashes of configs that have no audio, or every cached
recipe artifact silently invalidates.
"""

import pytest

from GenAILab.bench.precision import PrecisionConfig
from GenAILab.qai_hub_lm.schema.precision import AudioPrecisionSchema, PrecisionSchema
from GenAILab.qai_hub_lm.schema.recipe import Recipe


def _names(chain):
    return [s.name for s in chain]


class TestAudioRecipeChain:
    def test_audio_chain_parses_and_gets_default_calibration(self):
        r = Recipe.model_validate({"backbone": [{"name": "SeqMSE"}], "audio": []})
        # Rule 1 appends Calibration to any chain with no terminal step.
        assert _names(r.audio) == ["Calibration"]

    def test_audio_absent_by_default(self):
        r = Recipe.model_validate([{"name": "Calibration"}])
        assert r.audio is None
        assert r.visual is None

    def test_single_step_dict_under_audio_is_wrapped(self):
        r = Recipe.model_validate(
            {"backbone": [{"name": "Calibration"}], "audio": {"name": "SeqMSE"}}
        )
        assert _names(r.audio) == ["SeqMSE", "Calibration"]

    def test_named_chains_includes_audio_in_canonical_order(self):
        r = Recipe.model_validate(
            {
                "backbone": [{"name": "Calibration"}],
                "audio": [{"name": "Calibration"}],
                "visual": [{"name": "Calibration"}],
            }
        )
        assert [c for c, _ in r._named_chains()] == ["backbone", "visual", "audio"]

    def test_round_trip_through_to_components(self):
        r = Recipe.model_validate(
            {
                "backbone": [{"name": "Calibration"}],
                "audio": [{"name": "Calibration"}],
            }
        )
        assert Recipe.model_validate(r.to_components()) == r

    def test_presim_prefix_must_match_across_backbone_and_audio(self):
        # Rule 3: pre-sim steps act on the whole float model, so an audio chain
        # cannot omit what backbone declares.
        with pytest.raises(ValueError, match="Pre-sim prefix"):
            Recipe.model_validate(
                {
                    "backbone": [
                        {"name": "SpinQuant", "enable_r1": True},
                        {"name": "Calibration"},
                    ],
                    "audio": [{"name": "Calibration"}],
                }
            )

    def test_matching_presim_prefix_across_audio_is_accepted(self):
        r = Recipe.model_validate(
            {
                "backbone": [
                    {"name": "SpinQuant", "enable_r1": True},
                    {"name": "Calibration"},
                ],
                "audio": [
                    {"name": "SpinQuant", "enable_r1": True},
                    {"name": "Calibration"},
                ],
            }
        )
        assert _names(r.audio) == ["SpinQuant", "Calibration"]

    def test_phased_steps_works_for_audio(self):
        r = Recipe.model_validate(
            {
                "backbone": [
                    {"name": "SpinQuant", "enable_r1": True},
                    {"name": "Calibration"},
                ],
                "audio": [
                    {"name": "SpinQuant", "enable_r1": True},
                    {"name": "Calibration"},
                ],
            }
        )
        pre, on_sim = r.phased_steps("audio")
        assert _names(pre) == ["SpinQuant"]
        assert _names(on_sim) == ["Calibration"]


class TestAudioPrecisionSchema:
    def test_audio_absent_by_default(self):
        assert PrecisionSchema().audio is None

    def test_audio_block_defaults_to_int8_weight_int16_acts(self):
        p = PrecisionSchema.model_validate({"audio": {}})
        assert p.audio.weight.qtype == "int8"
        assert p.audio.activations == "int16"

    def test_float_audio_weight_rejected_and_names_the_component(self):
        with pytest.raises(ValueError, match="audio.weight"):
            AudioPrecisionSchema.model_validate({"weight": {"qtype": "float16"}})

    def test_float_visual_weight_error_still_names_visual(self):
        # The shared base must not blur which component failed.
        with pytest.raises(ValueError, match="visual.weight"):
            PrecisionSchema.model_validate({"visual": {"weight": {"qtype": "float16"}}})

    def test_component_accessor(self):
        p = PrecisionSchema.model_validate({"audio": {}})
        assert p.component("audio") is p.audio
        assert p.component("visual") is None
        with pytest.raises(KeyError, match="Unknown component"):
            p.component("nose")

    def test_extra_keys_still_forbidden(self):
        with pytest.raises(ValueError):
            PrecisionSchema.model_validate({"audio": {"weightt": {"qtype": "int8"}}})


class TestPrecisionConfigCacheIdentityUnchanged:
    """Adding audio must not move the cache keys of audio-free configs."""

    def test_text_only_to_dict_has_no_component_keys(self):
        d = PrecisionConfig().to_dict()
        assert "audio" not in d
        assert "visual" not in d

    def test_text_only_weight_identity_has_no_component_keys(self):
        d = PrecisionConfig().weight_identity()
        assert "audio_weight" not in d
        assert "visual_weight" not in d

    def test_visual_only_identity_is_unchanged_by_audio_support(self):
        pc = PrecisionConfig()
        pc.ensure_component_defaults("visual")
        assert set(pc.weight_identity()) == {
            "blocks",
            "lm_head",
            "embedding",
            "visual_weight",
        }
        assert "audio" not in pc.to_dict()

    def test_audio_defaults_are_recorded_once_set(self):
        pc = PrecisionConfig()
        pc.ensure_component_defaults("audio")
        assert pc.audio_weight.qtype.bits == 8
        assert pc.audio_activations.bits == 16
        assert "audio_weight" in pc.weight_identity()
        assert pc.to_dict()["audio"]["weight"]["qtype"] == "int8"

    def test_visual_key_precedes_audio_key_for_stable_hashing(self):
        pc = PrecisionConfig()
        pc.ensure_component_defaults("visual")
        pc.ensure_component_defaults("audio")
        keys = list(pc.to_dict())
        assert keys.index("visual") < keys.index("audio")

    def test_unknown_component_rejected(self):
        with pytest.raises(KeyError, match="Unknown component"):
            PrecisionConfig().ensure_component_defaults("smell")

    def test_from_schema_populates_audio(self):
        schema = PrecisionSchema.model_validate({"audio": {"activations": "int8"}})
        pc = PrecisionConfig.from_schema(schema)
        assert pc.audio_weight.qtype.bits == 8
        assert pc.audio_activations.bits == 8
        assert pc.visual_weight is None

    def test_legacy_from_dict_populates_audio(self):
        pc = PrecisionConfig.from_dict({"audio": {"weight": {"qtype": "int8"}}})
        assert pc.audio_weight.qtype.bits == 8

    def test_legacy_from_dict_rejects_float_audio_weight(self):
        with pytest.raises(ValueError, match="audio.weight"):
            PrecisionConfig.from_dict({"audio": {"weight": {"qtype": "float16"}}})

    def test_component_accessors(self):
        pc = PrecisionConfig()
        pc.ensure_component_defaults("audio")
        assert pc.component_weight("audio") is pc.audio_weight
        assert pc.component_activations("audio") is pc.audio_activations
        assert pc.component_weight("visual") is None

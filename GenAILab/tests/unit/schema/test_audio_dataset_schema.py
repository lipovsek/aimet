# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Schema tests for the audio/ASR dataset spec (schema island: no torch)."""

import pytest
from pydantic import TypeAdapter, ValidationError

from GenAILab.qai_hub_lm.schema.dataset import (
    DatasetSpec,
    LibriSpeechSpec,
    dataset_name_of,
    dataset_names,
    spec_for_dataset,
)

_ADAPTER = TypeAdapter(DatasetSpec)


class TestLibriSpeechSpecVocabulary:
    def test_name_is_in_the_dataset_vocabulary(self):
        assert "LibriSpeech" in dataset_names()

    def test_spec_lookup_round_trips(self):
        assert spec_for_dataset("LibriSpeech") is LibriSpeechSpec
        assert dataset_name_of(LibriSpeechSpec) == "LibriSpeech"


class TestLibriSpeechSpecRoundTrip:
    def test_discriminated_union_selects_the_librispeech_spec(self):
        spec = _ADAPTER.validate_python(
            {"name": "LibriSpeech", "split": "test.clean", "num_samples": 8}
        )
        assert isinstance(spec, LibriSpeechSpec)
        assert spec.split == "test.clean"
        assert spec.num_samples == 8

    def test_all_fields_round_trip(self):
        payload = {
            "name": "LibriSpeech",
            "split": "validation.clean",
            "num_samples": 32,
            "language": "English",
            "include_reference": True,
        }
        spec = _ADAPTER.validate_python(payload)
        assert spec.model_dump(exclude_unset=True) == payload

    def test_n_window_is_not_a_yaml_knob(self):
        with pytest.raises(ValidationError):
            _ADAPTER.validate_python({"name": "LibriSpeech", "n_window": 50})

    def test_omitted_fields_are_not_recorded(self):
        # Fields carry no defaults of their own: an omitted field must stay
        # omitted so the dataset's own default applies.
        spec = _ADAPTER.validate_python({"name": "LibriSpeech"})
        assert spec.model_dump(exclude_unset=True) == {"name": "LibriSpeech"}

    def test_unknown_field_is_rejected(self):
        with pytest.raises(ValidationError):
            _ADAPTER.validate_python({"name": "LibriSpeech", "sample_rate": 16000})

    def test_bad_field_type_is_rejected(self):
        with pytest.raises(ValidationError):
            _ADAPTER.validate_python({"name": "LibriSpeech", "num_samples": "many"})

    def test_usable_as_an_interleaved_source(self):
        spec = _ADAPTER.validate_python(
            {
                "name": "Interleaved",
                "source_datasets": [
                    {"name": "LibriSpeech", "split": "validation.clean"},
                    {"name": "Wikitext", "split": "train"},
                ],
            }
        )
        assert isinstance(spec.source_datasets[0], LibriSpeechSpec)

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Export utilities for QuantizationSimModel"""

from contextlib import contextmanager
import json
import os
from typing import List
import tempfile

import torch
from aimet_common import quantsim
from aimet_torch.experimental.v2.quantsim.export_utils import EncodingType
from aimet_torch.common.defs import QuantizationDataType
from aimet_torch.v2.quantsim import QuantizationSimModel as QuantizationSimModelV2
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantization.affine.quantizer import GroupedBlockQuantizeDequantize
from ....models.test_models import SingleResidualWithAvgPool


@contextmanager
def swap_encoding_version(version="1.0.0"):
    old_version = quantsim.encoding_version
    quantsim.encoding_version = version

    yield

    quantsim.encoding_version = old_version


def test_export_1_0_0_per_tensor():
    model = SingleResidualWithAvgPool().eval()
    dummy_inp = torch.randn(1, 3, 32, 32)
    for qsim in (
        QuantizationSimModelV2(
            model, dummy_input=dummy_inp, default_param_bw=4, default_output_bw=16
        ),
    ):
        qtzrs_before = [
            q.__dict__.copy()
            for q in qsim.model.modules()
            if isinstance(q, QuantizerBase)
        ]
        qsim.compute_encodings(lambda m, _: m(dummy_inp), None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with swap_encoding_version():
                qsim.export(tmp_dir, "qsim_export", dummy_inp)

            with open(os.path.join(tmp_dir, "qsim_export.encodings"), "r") as f:
                encodings = json.load(f)
        qtzrs_after = [
            q.__dict__.copy()
            for q in qsim.model.modules()
            if isinstance(q, QuantizerBase)
        ]

        """
        Quantizer attributes should not change after export
        """
        assert len(qtzrs_before) == len(qtzrs_after)
        for before, after in zip(qtzrs_before, qtzrs_after):
            assert before.keys() == after.keys()
            assert all(before[key] is after[key] for key in before)

        assert encodings["version"] == "1.0.0"
        assert isinstance(encodings["activation_encodings"], List)
        assert isinstance(encodings["param_encodings"], List)

        for encoding in encodings["activation_encodings"]:
            assert len(encoding.keys()) == 7
            assert "name" in encoding
            assert encoding.get("dtype") == QuantizationDataType.int.name.upper()
            assert encoding.get("enc_type") == EncodingType.PER_TENSOR.name
            assert "scale" in encoding
            assert isinstance(encoding["scale"], List) and len(encoding["scale"]) == 1
            assert "offset" in encoding
            assert isinstance(encoding["scale"], List) and len(encoding["scale"]) == 1
            assert encoding.get("bw") in [4, 16]
            if encoding["bw"] == 4:
                assert encoding.get("is_sym")
            else:
                assert encoding.get("is_sym") is False


def test_export_1_0_0_fp16():
    model = SingleResidualWithAvgPool().eval()
    dummy_inp = torch.randn(1, 3, 32, 32)
    for qsim in (
        QuantizationSimModelV2(
            model,
            dummy_input=dummy_inp,
            default_param_bw=16,
            default_output_bw=16,
            default_data_type=QuantizationDataType.float,
        ),
    ):
        qtzrs_before = [
            q.__dict__.copy()
            for q in qsim.model.modules()
            if isinstance(q, QuantizerBase)
        ]
        qsim.compute_encodings(lambda m, _: m(dummy_inp), None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with swap_encoding_version():
                qsim.export(tmp_dir, "qsim_export", dummy_inp)

            with open(os.path.join(tmp_dir, "qsim_export.encodings"), "r") as f:
                encodings = json.load(f)
        qtzrs_after = [
            q.__dict__.copy()
            for q in qsim.model.modules()
            if isinstance(q, QuantizerBase)
        ]

        """
        Quantizer attributes should not change after export
        """
        assert len(qtzrs_before) == len(qtzrs_after)
        for before, after in zip(qtzrs_before, qtzrs_after):
            assert before.keys() == after.keys()
            assert all(before[key] is after[key] for key in before)

        assert encodings["version"] == "1.0.0"
        assert isinstance(encodings["activation_encodings"], List)
        assert isinstance(encodings["param_encodings"], List)

        for encoding in (
            encodings["activation_encodings"] + encodings["param_encodings"]
        ):
            assert len(encoding.keys()) == 4
            assert "name" in encoding
            assert encoding.get("dtype") == QuantizationDataType.float.name.upper()
            assert encoding.get("enc_type") == EncodingType.PER_TENSOR.name
            assert encoding.get("bw") == 16


def test_export_1_0_0_per_channel():
    quantsim_config = {
        "defaults": {
            "ops": {"is_output_quantized": "True"},
            "params": {"is_quantized": "True", "is_symmetric": "True"},
            "per_channel_quantization": "True",
        },
        "params": {"bias": {"is_quantized": "False"}},
        "op_type": {},
        "supergroups": [],
        "model_input": {},
        "model_output": {},
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(os.path.join(tmp_dir, "config_file.json"), "w") as f:
            json.dump(quantsim_config, f)

        model = SingleResidualWithAvgPool().eval()
        dummy_inp = torch.randn(1, 3, 32, 32)
        for qsim in (
            QuantizationSimModelV2(
                model,
                dummy_input=dummy_inp,
                default_param_bw=4,
                config_file=os.path.join(tmp_dir, "config_file.json"),
            ),
        ):
            qtzrs_before = [
                q.__dict__.copy()
                for q in qsim.model.modules()
                if isinstance(q, QuantizerBase)
            ]
            qsim.compute_encodings(lambda m, _: m(dummy_inp), None)

            with swap_encoding_version():
                qsim.export(tmp_dir, "qsim_export", dummy_inp)

            with open(os.path.join(tmp_dir, "qsim_export.encodings"), "r") as f:
                encodings = json.load(f)
            qtzrs_after = [
                q.__dict__.copy()
                for q in qsim.model.modules()
                if isinstance(q, QuantizerBase)
            ]

            """
            Quantizer attributes should not change after export
            """
            assert len(qtzrs_before) == len(qtzrs_after)
            for before, after in zip(qtzrs_before, qtzrs_after):
                assert before.keys() == after.keys()
                assert all(before[key] is after[key] for key in before)

        assert encodings["version"] == "1.0.0"
        assert isinstance(encodings["activation_encodings"], List)
        assert isinstance(encodings["param_encodings"], List)

        param_name_to_shape = {}
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                param_name_to_shape[name + ".weight"] = module.weight.shape

        for encoding in encodings["param_encodings"]:
            assert encoding["name"] in param_name_to_shape
            assert len(encoding.keys()) == 7
            assert "name" in encoding
            assert encoding.get("dtype") == QuantizationDataType.int.name.upper()
            assert encoding.get("enc_type") == EncodingType.PER_CHANNEL.name
            assert encoding.get("bw") == 4
            assert encoding.get("is_sym")
            assert len(encoding["scale"]) == param_name_to_shape[encoding["name"]][0]
            assert len(encoding["offset"]) == param_name_to_shape[encoding["name"]][0]


def test_export_1_0_0_bq_lpbq():
    model = SingleResidualWithAvgPool().eval()
    dummy_inp = torch.randn(1, 3, 32, 32)
    qsim = QuantizationSimModelV2(model, dummy_input=dummy_inp, default_param_bw=4)
    qsim.model.conv2.param_quantizers["weight"] = GroupedBlockQuantizeDequantize(
        shape=(16, 4, 1, 1),
        bitwidth=4,
        symmetric=True,
        block_size=(-1, -1, -1, -1),
        decompressed_bw=8,
        block_grouping=None,
    )
    qtzrs_before = [
        q.__dict__.copy() for q in qsim.model.modules() if isinstance(q, QuantizerBase)
    ]
    qsim.compute_encodings(lambda m, _: m(dummy_inp), None)

    with tempfile.TemporaryDirectory() as tmp_dir:
        with swap_encoding_version():
            qsim.export(tmp_dir, "qsim_export", dummy_inp)

        with open(os.path.join(tmp_dir, "qsim_export.encodings"), "r") as f:
            encodings = json.load(f)
    qtzrs_after = [
        q.__dict__.copy() for q in qsim.model.modules() if isinstance(q, QuantizerBase)
    ]

    """
    Quantizer attributes should not change after export
    """
    assert len(qtzrs_before) == len(qtzrs_after)
    for before, after in zip(qtzrs_before, qtzrs_after):
        assert before.keys() == after.keys()
        assert all(before[key] is after[key] for key in before)

    assert encodings["version"] == "1.0.0"
    assert isinstance(encodings["activation_encodings"], List)
    assert isinstance(encodings["param_encodings"], List)
    conv_weight_encoding = [
        encoding
        for encoding in encodings["param_encodings"]
        if encoding["name"] == "conv2.weight"
    ][0]
    assert len(conv_weight_encoding.keys()) == 8
    assert conv_weight_encoding["dtype"] == QuantizationDataType.int.name.upper()
    assert conv_weight_encoding["enc_type"] == EncodingType.PER_BLOCK.name
    assert conv_weight_encoding["is_sym"]
    assert conv_weight_encoding["bw"] == 4
    assert len(conv_weight_encoding["scale"]) == 64
    assert len(conv_weight_encoding["offset"]) == 64
    assert conv_weight_encoding["block_size"] == 8

    qsim.model.conv2.param_quantizers["weight"] = GroupedBlockQuantizeDequantize(
        shape=(16, 4, 1, 1),
        bitwidth=4,
        symmetric=True,
        block_size=(-1, -1, -1, -1),
        decompressed_bw=8,
        block_grouping=(1, 4, 1, 1),
    )
    qtzrs_before = [
        q.__dict__.copy() for q in qsim.model.modules() if isinstance(q, QuantizerBase)
    ]
    qsim.compute_encodings(lambda m, _: m(dummy_inp), None)

    with tempfile.TemporaryDirectory() as tmp_dir:
        with swap_encoding_version():
            qsim.export(tmp_dir, "qsim_export", dummy_inp)

        with open(os.path.join(tmp_dir, "qsim_export.encodings"), "r") as f:
            encodings = json.load(f)
    qtzrs_after = [
        q.__dict__.copy() for q in qsim.model.modules() if isinstance(q, QuantizerBase)
    ]

    """
    Quantizer attributes should not change after export
    """
    assert len(qtzrs_before) == len(qtzrs_after)
    for before, after in zip(qtzrs_before, qtzrs_after):
        assert before.keys() == after.keys()
        assert all(before[key] is after[key] for key in before)

    assert encodings["version"] == "1.0.0"
    assert isinstance(encodings["activation_encodings"], List)
    assert isinstance(encodings["param_encodings"], List)
    conv_weight_encoding = [
        encoding
        for encoding in encodings["param_encodings"]
        if encoding["name"] == "conv2.weight"
    ][0]
    assert len(conv_weight_encoding.keys()) == 10
    assert conv_weight_encoding["dtype"] == QuantizationDataType.int.name.upper()
    assert conv_weight_encoding["enc_type"] == EncodingType.LPBQ.name
    assert conv_weight_encoding["is_sym"]
    assert conv_weight_encoding["bw"] == 8
    assert len(conv_weight_encoding["scale"]) == 16
    assert len(conv_weight_encoding["offset"]) == 16
    assert conv_weight_encoding["block_size"] == 8
    assert len(conv_weight_encoding["per_block_int_scale"]) == 64
    assert conv_weight_encoding["compressed_bw"] == 4

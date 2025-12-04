# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
import yaml
import pytest

try:
    from aimet_onnx.common.quantsim_config.v2.parser import Qtype, OpConfig
except ImportError:
    from aimet_torch.common.quantsim_config.v2.parser import Qtype, OpConfig


def test_qtype_str_parsing():
    qtype_str = "int8[channel_axis=1, block_axis=2, block_size=4, min=-6.0, max=6.0]"
    qtype = Qtype.from_str(qtype_str)

    assert qtype.dtype == "int8"
    assert qtype.channel_axis == 1
    assert qtype.block_axis == 2
    assert qtype.block_size == 4
    assert qtype.min == -6.0
    assert qtype.max == 6.0

    reconstructed_str = qtype.to_str()
    assert reconstructed_str == qtype_str

    for qtype_str in (
        "int4",
        "int8",
        "int16",
        "uint4",
        "uint8",
        "uint16",
        "int32",
        "float",
    ):
        qtype = Qtype.from_str(qtype_str)

        assert qtype.dtype == qtype_str
        assert (
            qtype.channel_axis
            == qtype.block_axis
            == qtype.block_size
            == qtype.min
            == qtype.max
            == None
        )

    reconstructed_str = qtype.to_str()
    assert reconstructed_str == qtype_str

    for qtype_str in ("int7", "float16", "uint8[channel_axis=-1, unknown_param=5]"):
        with pytest.raises(ValueError):
            Qtype.from_str(qtype_str)


def test_op_parsing():
    yaml_config = yaml.safe_load("""
Conv:
  supported_qtypes:
    # Only W8A8, W8A16, and W16A16 are supported (No W16A8)
    - X: uint8
      W: int8[channel_axis=0, block_axis=1]
      B: int32
      Y: uint8
    - X: uint16
      W: int8[channel_axis=0, block_axis=1] | int16[channel_axis=0, block_axis=1]
      B: int32
      Y: uint16""")

    op_config = OpConfig.from_dict("Conv", yaml_config["Conv"])
    assert op_config.op_type == "Conv"
    assert len(op_config.kernels) == 3

    assert op_config.kernels[0].inputs["X"] == Qtype(dtype="uint8")
    assert op_config.kernels[0].inputs["W"] == Qtype(
        dtype="int8", channel_axis=0, block_axis=1
    )
    assert op_config.kernels[0].inputs["B"] == Qtype(dtype="int32")

    assert op_config.kernels[0].outputs["Y"] == Qtype(dtype="uint8")
    assert op_config.kernels[1].inputs["X"] == Qtype(dtype="uint16")
    assert op_config.kernels[1].inputs["W"] == Qtype(
        dtype="int8", channel_axis=0, block_axis=1
    )
    assert op_config.kernels[1].inputs["B"] == Qtype(dtype="int32")
    assert op_config.kernels[1].outputs["Y"] == Qtype(dtype="uint16")

    assert op_config.kernels[2].inputs["X"] == Qtype(dtype="uint16")
    assert op_config.kernels[2].inputs["W"] == Qtype(
        dtype="int16", channel_axis=0, block_axis=1
    )
    assert op_config.kernels[2].inputs["B"] == Qtype(dtype="int32")
    assert op_config.kernels[2].outputs["Y"] == Qtype(dtype="uint16")

    yaml_config = yaml.safe_load("""
Softmax:
  supported_qtypes:
    # Softmax output encoding is always 0 to 1
    - input: uint8
      output: uint8[min=0.0, max=1.0]
    - input: uint16
      output: uint16[min=0.0, max=1.0]""")

    op_config = OpConfig.from_dict("Softmax", yaml_config["Softmax"])
    assert op_config.op_type == "Softmax"
    assert len(op_config.kernels) == 2

    assert op_config.kernels[0].inputs["input"] == Qtype(dtype="uint8")
    assert op_config.kernels[0].outputs["output"] == Qtype(
        dtype="uint8", min=0.0, max=1.0
    )
    assert op_config.kernels[1].inputs["input"] == Qtype(dtype="uint16")
    assert op_config.kernels[1].outputs["output"] == Qtype(
        dtype="uint16", min=0.0, max=1.0
    )

    yaml_config = yaml.safe_load("""
Relu:
  supported_qtypes:
    # ReLU output encoding is always non-negative
    - X: uint8
      Y: uint8[min=0.0]
    - X: uint16
      Y: uint16[min=0.0]""")

    op_config = OpConfig.from_dict("Relu", yaml_config["Relu"])
    assert op_config.op_type == "Relu"
    assert len(op_config.kernels) == 2
    assert op_config.kernels[0].inputs["X"] == Qtype(dtype="uint8")
    assert op_config.kernels[0].outputs["Y"] == Qtype(dtype="uint8", min=0.0)
    assert op_config.kernels[1].inputs["X"] == Qtype(dtype="uint16")
    assert op_config.kernels[1].outputs["Y"] == Qtype(dtype="uint16", min=0.0)

    yaml_config = yaml.safe_load("""
Resize:
  supported_qtypes:
    # HTP Resize requires input and output to share the same encoding
    - X: uint8[min=a, max=b]
      scales: float
      roi: float
      Y: uint8[min=a, max=b]
    - X: uint16[min=a, max=b]
      scales: float
      roi: float
      Y: uint16[min=a, max=b]""")

    op_config = OpConfig.from_dict("Resize", yaml_config["Resize"])
    assert op_config.op_type == "Resize"
    assert len(op_config.kernels) == 2
    assert op_config.kernels[0].inputs["X"] == Qtype(dtype="uint8", min="a", max="b")
    assert op_config.kernels[0].inputs["scales"] == Qtype(dtype="float")
    assert op_config.kernels[0].inputs["sizes"] is None
    assert op_config.kernels[0].outputs["Y"] == Qtype(dtype="uint8", min="a", max="b")
    assert op_config.kernels[1].inputs["X"] == Qtype(dtype="uint16", min="a", max="b")
    assert op_config.kernels[1].inputs["scales"] == Qtype(dtype="float")
    assert op_config.kernels[1].inputs["sizes"] is None
    assert op_config.kernels[1].outputs["Y"] == Qtype(dtype="uint16", min="a", max="b")

    yaml_config = yaml.safe_load("""
LayerNormalization:
  supported_qtypes:
    # 1. Per-channel quantization is not supported
    # 2. If weight is uint16, HTP requires it to be non-negative
    - X: uint8
      Scale: int8 | uint8
      B: int32
      Y: uint8
      "*": float
    - X: uint16
      Scale: uint8 | int16 | uint16[min=0.0]
      B: int32
      Y: uint16
      "*": float""")

    op_config = OpConfig.from_dict(
        "LayerNormalization", yaml_config["LayerNormalization"]
    )
    assert op_config.op_type == "LayerNormalization"
    assert len(op_config.kernels) == 5
    assert op_config.kernels[0].inputs["X"] == Qtype(dtype="uint8")
    assert op_config.kernels[0].inputs["Scale"] == Qtype(dtype="int8")
    assert op_config.kernels[0].inputs["B"] == Qtype(dtype="int32")
    assert op_config.kernels[0].outputs["Y"] == Qtype(dtype="uint8")

    assert op_config.kernels[1].inputs["X"] == Qtype(dtype="uint8")
    assert op_config.kernels[1].inputs["Scale"] == Qtype(dtype="uint8")
    assert op_config.kernels[1].inputs["B"] == Qtype(dtype="int32")
    assert op_config.kernels[1].outputs["Y"] == Qtype(dtype="uint8")

    assert op_config.kernels[2].inputs["X"] == Qtype(dtype="uint16")
    assert op_config.kernels[2].inputs["Scale"] == Qtype(dtype="uint8")
    assert op_config.kernels[2].inputs["B"] == Qtype(dtype="int32")
    assert op_config.kernels[2].outputs["Y"] == Qtype(dtype="uint16")

    assert op_config.kernels[3].inputs["X"] == Qtype(dtype="uint16")
    assert op_config.kernels[3].inputs["Scale"] == Qtype(dtype="int16")
    assert op_config.kernels[3].inputs["B"] == Qtype(dtype="int32")
    assert op_config.kernels[3].outputs["Y"] == Qtype(dtype="uint16")

    assert op_config.kernels[4].inputs["X"] == Qtype(dtype="uint16")
    assert op_config.kernels[4].inputs["Scale"] == Qtype(dtype="uint16", min=0.0)
    assert op_config.kernels[4].inputs["B"] == Qtype(dtype="int32")
    assert op_config.kernels[4].outputs["Y"] == Qtype(dtype="uint16")

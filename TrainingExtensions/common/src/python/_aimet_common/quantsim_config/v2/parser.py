# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Iterator
from dataclasses import dataclass
import re
import functools
import itertools
import onnx
from ...onnx._utils import _is_float


@dataclass(frozen=True)
class Variable:
    name: str


@dataclass
class Qtype:
    dtype: str
    channel_axis: int | None = None
    block_axis: int | None = None
    block_size: int | None = None
    min: float | Variable | None = None
    max: float | Variable | None = None

    _allowed_dtypes = (
        "int4",
        "int8",
        "int16",
        "uint4",
        "uint8",
        "uint16",
        "int32",
        "float",
    )

    def __post_init__(self):
        if self.dtype not in self._allowed_dtypes:
            raise RuntimeError(
                f"Unsupported dtype '{self}'. Expected one of {self._allowed_dtypes}."
            )

        if self.channel_axis is not None:
            try:
                self.channel_axis = int(self.channel_axis)
            except ValueError as e:
                raise ValueError(
                    f"channel_axis must be an integer. Got '{self}'."
                ) from e

        if self.block_axis is not None:
            try:
                self.block_axis = int(self.block_axis)
            except ValueError as e:
                raise ValueError(f"block_axis must be an integer. Got '{self}'.") from e

        if self.block_size is not None:
            try:
                self.block_size = int(self.block_size)
            except ValueError as e:
                raise ValueError(f"block_size must be an integer. Got '{self}'.") from e

        if self.block_axis is not None and self.channel_axis is None:
            raise ValueError(
                f"block_axis is specified but channel_axis is missing: {self}"
            )

        if self.block_size is not None and self.block_axis is None:
            raise ValueError(
                f"block_size is specified but block_axis is missing: {self}"
            )

        if self.min is not None:
            try:
                self.min = float(self.min)
            except ValueError:
                try:
                    self.min = Variable(name=str(self.min))
                except ValueError as e:
                    raise ValueError(
                        f"min must be a float or Variable. Got '{self}'."
                    ) from e

        if self.max is not None:
            try:
                self.max = float(self.max)
            except ValueError:
                try:
                    self.max = Variable(name=str(self.max))
                except ValueError as e:
                    raise ValueError(
                        f"max must be a float or Variable. Got '{self}'."
                    ) from e

    def __repr__(self) -> str:
        return self.to_str()

    def to_str(self) -> str:
        optional_kvs = {
            "channel_axis": self.channel_axis,
            "block_axis": self.block_axis,
            "block_size": self.block_size,
            "min": self.min,
            "max": self.max,
        }
        optional_kvs = ", ".join(
            [
                f"{key}={value}"
                for key, value in optional_kvs.items()
                if value is not None
            ]
        )

        if optional_kvs:
            return f"{self.dtype}[{optional_kvs}]"

        return self.dtype

    @classmethod
    def from_str(cls, qtype_str: str) -> "Qtype":
        """ """
        if not qtype_str:
            raise NotImplementedError("Qtype configuration is missing.")

        dtype_str, kwargs = cls._parse_qtype_str(qtype_str)

        unknown_keys = kwargs.keys() - {
            "channel_axis",
            "block_axis",
            "block_size",
            "min",
            "max",
        }
        if unknown_keys:
            raise ValueError(
                f"Qtype configuration string '{qtype_str}' contains unknown keys: {unknown_keys}."
            )

        return cls(dtype=dtype_str, **kwargs)

    @classmethod
    def _parse_qtype_str(cls, qtype_str: str) -> tuple[str, dict]:
        pattern = cls._regex_pattern()

        match = pattern.match(qtype_str.replace(" ", ""))
        if not match:
            raise ValueError(f"Qtype configuration string '{qtype_str}' is invalid.")

        dtype_str, kwargs = match.groups()

        if kwargs:
            kwargs = dict(kv.split("=") for kv in kwargs.split(","))
        else:
            kwargs = {}

        return dtype_str, kwargs

    @classmethod
    @functools.lru_cache()
    def _regex_pattern(cls):
        dtype_str = "|".join(cls._allowed_dtypes)
        optional_kvs = r"(?:channel_axis|block_axis|block_size|min|max)=(?:[^\]]+)"
        optional_kvs = f"{optional_kvs}(?:,{optional_kvs})*"
        pattern = r"^({dtype_str})(?:\[({optional_kvs})\])?$".format(
            dtype_str=dtype_str, optional_kvs=optional_kvs
        )
        return re.compile(pattern)


@dataclass(frozen=True)
class Kernel:
    inputs: dict[str, Qtype | None]
    outputs: dict[str, Qtype | None]

    @classmethod
    def from_dict(cls, op_type: str, config: dict) -> Iterator["Kernel"]:
        schema = onnx.defs.get_schema(op_type)
        fp_tensor_names = set(
            t.name
            for t in itertools.chain(schema.inputs, schema.outputs)
            if _is_float(schema, t)
        )

        if "*" not in config and config.keys() != fp_tensor_names:
            raise RuntimeError(
                f"Kernel configuration keys do not match schema for op {schema.name}."
                f" Expected keys: {fp_tensor_names}, "
                f"but got keys: {set(config.keys())}."
            )

        for t in itertools.chain(schema.inputs, schema.outputs):
            if not _is_float(schema, t) and config.get(t.name):
                raise RuntimeError(
                    ("Input" if t in schema.inputs else "Output")
                    + f" '{t.name}' of op '{schema.name}' is not float type, "
                    + "but a quantization type is specified in the configuration."
                )

        default = config.get("*")
        inputs = {
            inp.name: config.get(inp.name, default)
            if inp.name in fp_tensor_names
            else None
            for inp in schema.inputs
        }
        outputs = {
            out.name: config.get(out.name, default)
            if out.name in fp_tensor_names
            else None
            for out in schema.outputs
        }

        combinations = itertools.product(
            *(
                qtype_str.split("|") if qtype_str is not None else [None]
                for qtype_str in itertools.chain(inputs.values(), outputs.values())
            )
        )
        for qtypes in combinations:
            yield cls(
                inputs={
                    inp.name: Qtype.from_str(qtype_str)
                    if qtype_str is not None
                    else None
                    for inp, qtype_str in zip(schema.inputs, qtypes)
                },
                outputs={
                    out.name: Qtype.from_str(qtype_str)
                    if qtype_str is not None
                    else None
                    for out, qtype_str in zip(
                        schema.outputs, qtypes[len(schema.inputs) :]
                    )
                },
            )


@dataclass(frozen=True)
class OpConfig:
    op_type: str
    kernels: list[Kernel]

    @classmethod
    def from_dict(cls, op_type: str, config: dict) -> "OpConfig":
        kernels = []

        for kernel_config in config.get("supported_qtypes", []):
            kernels += Kernel.from_dict(op_type, kernel_config)

        return cls(
            op_type=op_type,
            kernels=kernels,
        )

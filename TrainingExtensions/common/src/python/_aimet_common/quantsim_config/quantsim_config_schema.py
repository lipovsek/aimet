# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Schema used to validate json configuration file. Docs: https://json-schema.org/learn/"""

QUANTSIM_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "defaults": {
            "type": "object",
            "properties": {
                "ops": {
                    "type": "object",
                    "properties": {
                        "is_input_quantized": {
                            "type": "string",
                            "pattern": "^True$|^False$",
                        },
                        "is_output_quantized": {
                            "type": "string",
                            "pattern": "^True$|^False$",
                        },
                        "is_symmetric": {"type": "string", "pattern": "^True$|^False$"},
                    },
                    "additionalProperties": False,
                },
                "params": {
                    "type": "object",
                    "properties": {
                        "is_quantized": {"type": "string", "pattern": "^True$|^False$"},
                        "is_symmetric": {"type": "string", "pattern": "^True$|^False$"},
                    },
                    "additionalProperties": False,
                },
                "strict_symmetric": {"type": "string", "pattern": "^True$|^False$"},
                "unsigned_symmetric": {"type": "string", "pattern": "^True$|^False$"},
                "per_channel_quantization": {
                    "type": "string",
                    "pattern": "^True$|^False$",
                },
                "supported_kernels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "activation": {
                                "type": "object",
                                "properties": {
                                    "bitwidth": {
                                        "type": "integer",
                                        "enum": [4, 8, 16, 32],
                                    },
                                    "dtype": {
                                        "type": "string",
                                        "pattern": "^int$|^float$",
                                    },
                                },
                                "required": ["bitwidth", "dtype"],
                                "additionalProperties": False,
                            },
                            "param": {
                                "type": "object",
                                "properties": {
                                    "bitwidth": {
                                        "type": "integer",
                                        "enum": [4, 8, 16, 32],
                                    },
                                    "dtype": {
                                        "type": "string",
                                        "pattern": "^int$|^float$",
                                    },
                                },
                                "required": ["bitwidth", "dtype"],
                                "additionalProperties": False,
                            },
                        },
                        "required": ["activation", "param"],
                        "additionalProperties": False,
                    },
                    "minItems": 1,
                    "additionalItems": False,
                },
                "hw_version": {"type": "string"},
            },
            "required": ["ops", "params"],
            "additionalProperties": False,
        },
        "params": {
            "type": "object",
            "patternProperties": {
                ".*": {
                    "type": "object",
                    "properties": {
                        "is_quantized": {"type": "string", "pattern": "^True$|^False$"},
                        "is_symmetric": {"type": "string", "pattern": "^True$|^False$"},
                    },
                    "additionalProperties": False,
                }
            },
        },
        "op_type": {
            "type": "object",
            "patternProperties": {
                ".*": {
                    "type": "object",
                    "properties": {
                        "is_input_quantized": {
                            "type": "string",
                            "pattern": "^True$|^False$",
                        },
                        "is_output_quantized": {
                            "type": "string",
                            "pattern": "^True$|^False$",
                        },
                        "is_symmetric": {"type": "string", "pattern": "^True$|^False$"},
                        "params": {
                            "type": "object",
                            "patternProperties": {
                                ".*": {
                                    "type": "object",
                                    "properties": {
                                        "is_quantized": {
                                            "type": "string",
                                            "pattern": "^True$|^False$",
                                        },
                                        "is_symmetric": {
                                            "type": "string",
                                            "pattern": "^True$|^False$",
                                        },
                                    },
                                    "additionalProperties": False,
                                }
                            },
                        },
                        "supported_kernels": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "activation": {
                                        "type": "object",
                                        "properties": {
                                            "bitwidth": {
                                                "type": "integer",
                                                "enum": [4, 8, 16, 32],
                                            },
                                            "dtype": {
                                                "type": "string",
                                                "pattern": "^int$|^float$",
                                            },
                                        },
                                        "required": ["bitwidth", "dtype"],
                                        "additionalProperties": False,
                                    },
                                    "param": {
                                        "type": "object",
                                        "properties": {
                                            "bitwidth": {
                                                "type": "integer",
                                                "enum": [4, 8, 16, 32],
                                            },
                                            "dtype": {
                                                "type": "string",
                                                "pattern": "^int$|^float$",
                                            },
                                        },
                                        "required": ["bitwidth", "dtype"],
                                        "additionalProperties": False,
                                    },
                                },
                                "required": ["activation"],
                                "additionalProperties": False,
                            },
                            "minItems": 1,
                            "additionalItems": False,
                        },
                        "per_channel_quantization": {
                            "type": "string",
                            "pattern": "^True$|^False$",
                        },
                        "encoding_constraints": {
                            "type": "object",
                            "properties": {
                                "min": {"type": "number"},
                                "max": {"type": "number"},
                            },
                            "required": ["min"],
                        },
                    },
                    "additionalProperties": False,
                }
            },
        },
        "supergroups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                    }
                },
                "required": ["op_list"],
                "additionalProperties": False,
            },
            "additionalItems": False,
        },
        "supergroup_pass_list": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
        },
        "model_input": {
            "type": "object",
            "properties": {
                "is_input_quantized": {"type": "string", "pattern": "^True$|^False$"}
            },
            "additionalProperties": False,
        },
        "model_output": {
            "type": "object",
            "properties": {
                "is_output_quantized": {"type": "string", "pattern": "^True$|^False$"}
            },
            "additionalProperties": False,
        },
    },
    "required": [
        "defaults",
        "params",
        "op_type",
        "supergroups",
        "model_input",
        "model_output",
    ],
    "additionalProperties": False,
}

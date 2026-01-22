# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Constants for arch checker."""


# pylint: disable=too-few-public-methods
class ArchCheckerReportConstants:
    """Constants for arch checker report."""

    OP_STRUCT_OP_TYPE = "OpStructure"
    DF_GRAPH_NODENAME = "Graph/Layer_name"
    DF_ISSUE = "Issue"
    DF_RECOMM = "Recommendation"

    UNDEFINED_ISSUE = "Undefined issue from check: {}"
    UNDEFINED_RECOMM = "Undefined recommendation from check: {}"

    OUTPUT_CSV_HEADER = [DF_GRAPH_NODENAME, DF_ISSUE, DF_RECOMM]

    ERR_MSG_DICT = {
        "_check_conv_channel_32_base": {
            DF_ISSUE: "The channel size of input/output tensor of this convolution is not a multiple of 32",
            DF_RECOMM: "Try adjusting the channels to multiple of 32 to get better performance.",
        },
        "_check_conv_channel_larger_than_32": {
            DF_ISSUE: "The channel size of input/output tensor of this convolution is smaller than 32",
            DF_RECOMM: "Try adjusting the channels to multiple of 32 to get better performance.",
        },
        "_activation_checks": {
            "PRelu": {
                DF_ISSUE: "PRelu activation function degenerates performance.",
                DF_RECOMM: "Try use Relu instead.",
            },
            "SiLU": {
                DF_ISSUE: "SiLU (Swish) activation function degenerates performance.",
                DF_RECOMM: "Try use Hard SiLU (hardswish) instaed.",
            },
        },
        "_check_batch_norm_fold": {
            DF_ISSUE: "The batch norm layer cannot be folded to immediate conv/linear layer. Quantizing standalone BN can degenerate performance.",
            DF_RECOMM: "Try remove the standalone BN or move the BN adjacent to Conv.",
        },
        "_check_intermediate_padding": {
            DF_ISSUE: "This convolution includes intermediate padding that degenerates performance.",
            DF_RECOMM: "Try move all padding to the first convolution in the sequence: [Conv -> Activation -> (Optionally) BN -> Conv].",
        },
        "_check_foldable_bn_with_split": {
            DF_ISSUE: "This structure: (conv1, conv2, ...) -> split_node(concat) -> BN degenerates performance",
            DF_RECOMM: "Try transform the structure so that BN can be folded to conv.",
        },
    }

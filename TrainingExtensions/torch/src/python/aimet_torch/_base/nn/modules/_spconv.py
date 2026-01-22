# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=abstract-method, arguments-differ, unused-argument
"""Custom module definitions for spconv"""

import torch

__all__ = [
    "CustomSparseConv3d",
    "CustomSparseConv3d_WithIndicesFeatures",
    "CustomSparseConv3DLayer",
    "SparseTensorWrapper",
    "CustomScatterDense",
    "ScatterDense",
]

try:
    import spconv.pytorch as spconv
except ImportError as e:
    CustomSparseConv3d = None
    CustomSparseConv3d_WithIndicesFeatures = None
    CustomSparseConv3DLayer = None
    SparseTensorWrapper = None
    CustomScatterDense = None
    ScatterDense = None
else:

    class CustomSparseConv3d(torch.autograd.Function):
        """
        Custom Sparse Conv3d autograd function
        """

        @staticmethod
        def symbolic(g, dense_inputs, weight, bias, all_sp_conv_attrs):
            """
            Symbolic method (static) for Custom sparse Conv3d
            :param g: ONNX graph object
            :param dense_inputs: Dense inputs
            :param weight: weight value
            :param bias: bias value
            :param all_sp_conv_attrs: spconv attributes
            :return: Added op to the graph object
            """
            attrs = {}
            for k, v in all_sp_conv_attrs.items():
                if v:
                    if isinstance(v, str):
                        attrs[k + "_s"] = v
                    else:
                        attrs[k + "_i"] = v
            if bias:
                return g.op(
                    "spconv::SparseConvolution", dense_inputs, weight, bias, **attrs
                )
            return g.op("spconv::SparseConvolution", dense_inputs, weight, **attrs)

        @staticmethod
        def forward(ctx, dense_inputs, weight, bias, all_sp_conv_attrs):
            """
            forward method (static) for Custom sparse Conv3d
            :param ctx: context object
            :param dense_inputs: Dense inputs
            :param weight: weight value
            :param bias: bias value
            :param all_sp_conv_attrs: spconv attributes
            :return: Dense tensor
            """
            device = weight.device
            dense_inputs = dense_inputs.to(device)
            sp_conv_attrs = {}
            ignore = [
                "ndim",
                "output_bound",
                "input_spatial_shape",
                "activation",
                "subm",
                "batch_size",
                "spatial_shape",
                "input_shape",
                "inverse",
                "transposed",
                "rulebook",
                "output_shape",
                "output_spatial_shape",
                "output_padding",
            ]
            for k, v in all_sp_conv_attrs.items():
                if k in ignore:
                    continue
                sp_conv_attrs[k] = v
            sp_conv_attrs["bias"] = sp_conv_attrs.get("bias", False)
            conv3d = torch.nn.Conv3d(**sp_conv_attrs)

            with torch.no_grad():
                conv3d.weight.copy_(weight.detach().permute(0, 4, 1, 2, 3))
                if sp_conv_attrs["bias"]:
                    conv3d.bias.copy_(bias.detach())
            conv3d = conv3d.to(device)

            out = conv3d(dense_inputs)
            return out

    class CustomSparseConv3d_WithIndicesFeatures(torch.autograd.Function):
        """
        Custom Sparse Conv3d (with indices and features as inputs) autograd function
        """

        @staticmethod
        def symbolic(g, indices, features, weight, bias, all_sp_conv_attrs):
            """
            Symbolic method (static) for Custom sparse Conv3d (with indices and features as inputs)
            :param g: ONNX graph object
            :param indices: Indices input
            :param features: Features input
            :param weight: weight value
            :param bias: bias value
            :param all_sp_conv_attrs: spconv attributes
            :return: Added op to the graph object
            """
            remove = ["spatial_shape", "batch_size"]
            attrs = {}
            for k, v in all_sp_conv_attrs.items():
                if k not in remove and v:
                    if isinstance(v, str):
                        attrs[k + "_s"] = v
                    else:
                        attrs[k + "_i"] = v
            if bias:
                return g.op(
                    "spconv::SparseConvolution",
                    indices,
                    features,
                    weight,
                    bias,
                    **attrs,
                )
            return g.op("spconv::SparseConvolution", indices, features, weight, **attrs)

        @staticmethod
        def forward(ctx, indices, features, weight, bias, all_sp_conv_attrs):
            """
            forward method (static) for Custom sparse Conv3d (with indices and features as inputs)
            :param ctx: context object
            :param indices: Indices input
            :param features: Features input
            :param weight: weight value
            :param bias: bias value
            :param all_sp_conv_attrs: spconv attributes
            :return: Dense tensor
            """
            device = weight.device
            indices = indices.to(device)
            features = features.to(device)
            sp_conv_attrs = {}
            ignore = [
                "ndim",
                "output_bound",
                "input_spatial_shape",
                "activation",
                "subm",
                "batch_size",
                "spatial_shape",
                "input_shape",
                "inverse",
                "transposed",
                "rulebook",
                "output_shape",
                "output_spatial_shape",
                "output_padding",
            ]
            for k, v in all_sp_conv_attrs.items():
                if k in ignore:
                    continue
                sp_conv_attrs[k] = v
            sp_conv_attrs["bias"] = sp_conv_attrs.get("bias", False)
            conv3d = torch.nn.Conv3d(**sp_conv_attrs)

            with torch.no_grad():
                conv3d.weight.copy_(weight.detach().permute(0, 4, 1, 2, 3))
                if sp_conv_attrs["bias"]:
                    conv3d.bias.copy_(bias.detach())
            conv3d = conv3d.to(device)

            dense_inputs = features.reshape(
                all_sp_conv_attrs["batch_size"],
                features.shape[1],
                *all_sp_conv_attrs["spatial_shape"],
            )
            dense_inputs = dense_inputs.to(device)

            out = conv3d(dense_inputs)
            return out

    # pylint: disable=too-many-arguments, super-with-arguments
    class CustomSparseConv3DLayer(torch.nn.Module):
        """
        SparseConv3D op implementation
        """

        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
        ):
            super(CustomSparseConv3DLayer, self).__init__()
            activation = "None"  # "ReLU"
            self.sp_conv_3d = spconv.SparseConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=1,
                algo=spconv.ConvAlgo.Native,
            )  # doesn't support groups as of now
            self.bias_available = bias
            if not bias:
                with torch.no_grad():
                    self.sp_conv_3d.bias = torch.nn.Parameter(torch.zeros(out_channels))
            self.conv_attrs_dict = {
                "in_channels": self.sp_conv_3d.in_channels,
                "out_channels": self.sp_conv_3d.out_channels,
                "kernel_size": self.sp_conv_3d.kernel_size,
                "stride": self.sp_conv_3d.stride,
                "padding": self.sp_conv_3d.padding,
                "dilation": self.sp_conv_3d.dilation,
                "subm": int(self.sp_conv_3d.subm),
                "ndim": self.sp_conv_3d.ndim,
                "output_bound": 20000,
                "activation": activation,
                "groups": groups,
            }

        def forward_with_indices_features(self, indices, features):
            """
            forward with indices and features as inputs
            :param indices: Indices input
            :param features: Features input
            :return: Dense tensor output
            """
            spatial_shape = [
                indices[:, 1].max().item() + 1,
                indices[:, 2].max().item() + 1,
                indices[:, 3].max().item() + 1,
            ]
            batch_size = indices[:, 0].max().item() + 1
            if torch.jit.is_tracing():
                self.conv_attrs_dict["spatial_shape"] = spatial_shape
                self.conv_attrs_dict["batch_size"] = batch_size
                self.conv_attrs_dict["input_spatial_shape"] = spatial_shape
                self.conv_attrs_dict["output_bound"] = features.shape[0]
                self.conv_attrs_dict["input_shape"] = features.shape
                self.conv_attrs_dict["rulebook"] = "subm" + str(
                    self.conv_attrs_dict["subm"]
                )
                self.conv_attrs_dict["transposed"] = 0
                self.conv_attrs_dict["inverse"] = 0

                self.conv_attrs_dict = dict(
                    sorted(self.conv_attrs_dict.items(), key=lambda x: (x[0], x[1]))
                )
                return CustomSparseConv3d_WithIndicesFeatures.apply(
                    indices,
                    features,
                    self.sp_conv_3d.weight,
                    self.sp_conv_3d.bias,
                    self.conv_attrs_dict,
                )

            sp_tensor = spconv.SparseConvTensor(
                features=features,
                indices=indices,
                spatial_shape=spatial_shape,
                batch_size=batch_size,
            )
            saved_bias_zero = self.sp_conv_3d.bias
            if not self.bias_available:
                self.sp_conv_3d.bias = None
            sp_conv_outs = self.sp_conv_3d(sp_tensor)
            dense_outs = sp_conv_outs.dense()
            if not self.bias_available:
                self.sp_conv_3d.bias = saved_bias_zero
            return dense_outs

        def forward_with_dense_input(self, dense_inp):
            """
            Forward-pass routine for SparseConv3D op
            """
            if isinstance(dense_inp, (tuple, list)) and len(dense_inp) == 2:
                return self.forward_with_indices_features(*tuple(dense_inp))

            if isinstance(dense_inp, spconv.SparseConvTensor):
                dense_inp = dense_inp.dense(channels_first=True)

            if torch.jit.is_tracing():
                self.conv_attrs_dict["input_spatial_shape"] = dense_inp.shape[2:]
                self.conv_attrs_dict["spatial_shape"] = dense_inp.shape[2:]
                self.conv_attrs_dict["batch_size"] = dense_inp.shape[0]
                self.conv_attrs_dict["output_bound"] = (
                    dense_inp.shape[0]
                    * dense_inp.shape[2]
                    * dense_inp.shape[3]
                    * dense_inp.shape[4]
                )
                self.conv_attrs_dict["input_shape"] = [
                    self.conv_attrs_dict["output_bound"],
                    dense_inp.shape[1],
                ]
                self.conv_attrs_dict["rulebook"] = "subm" + str(
                    self.conv_attrs_dict["subm"]
                )
                self.conv_attrs_dict["transposed"] = 0
                self.conv_attrs_dict["inverse"] = 0

                self.conv_attrs_dict = dict(
                    sorted(self.conv_attrs_dict.items(), key=lambda x: (x[0], x[1]))
                )
                return CustomSparseConv3d.apply(
                    dense_inp,
                    self.sp_conv_3d.weight,
                    self.sp_conv_3d.bias,
                    self.conv_attrs_dict,
                )

            # Dense to Sparse Conversion
            dense_inp = dense_inp.permute(0, 2, 3, 4, 1)  # N D H W C
            indices = (
                torch.stack(
                    torch.meshgrid(
                        torch.arange(dense_inp.shape[0]),
                        torch.arange(dense_inp.shape[1]),
                        torch.arange(dense_inp.shape[2]),
                        torch.arange(dense_inp.shape[3]),
                        indexing="ij",
                    ),
                    dim=-1,
                )
                .reshape(-1, 4)
                .int()
            )
            features = dense_inp.reshape(-1, dense_inp.shape[4])
            spatial_shape = dense_inp.shape[1:-1]
            batch_size = dense_inp.shape[0]
            sp_tensor = spconv.SparseConvTensor(
                features=features,
                indices=indices,
                spatial_shape=spatial_shape,
                batch_size=batch_size,
            )

            saved_bias_zero = self.sp_conv_3d.bias
            if not self.bias_available:
                self.sp_conv_3d.bias = None
            sp_conv_outs = self.sp_conv_3d(sp_tensor)
            dense_outs = sp_conv_outs.dense()
            if not self.bias_available:
                self.sp_conv_3d.bias = saved_bias_zero
            return dense_outs

        def forward(self, *args):
            """
            Forward pass for Custom SparseConv3d layer
            :param args: Either one dense input of format NCDHW or two inputs (indices, features) both in dense form
            :return: Dense tensor
            """
            if len(args) == 2:
                return self.forward_with_indices_features(*args)
            return self.forward_with_dense_input(*args)

    # pylint: disable=useless-super-delegation
    class SparseTensorWrapper(torch.nn.Module):
        """
        Custom SparsetensorWrapper class for SparseConvTensor
        """

        def __init__(self):
            super(SparseTensorWrapper, self).__init__()

        def forward_with_indices_and_features(self, coords, voxels):
            """
            forward pass with indices and features as inputs
            :param coords: Indices input
            :param voxels: Features input
            :return: Sparse tensor
            """
            # dense_inp is expected to be in N C D H W format
            if torch.jit.is_tracing():
                return coords, voxels

            spatial_shape = [
                coords[:, 1].max() + 1,
                coords[:, 2].max() + 1,
                coords[:, 3].max() + 1,
            ]
            return spconv.SparseConvTensor(
                features=voxels,
                indices=coords,
                spatial_shape=spatial_shape,
                batch_size=coords[:, 0].max() + 1,
            )

        def forward_with_dense_input(self, dense_inp):
            """
            forward pass with single dense input (NCDHW format)
            :param dense_inp: Dense input
            :return: Sparse tensor
            """
            if isinstance(dense_inp, tuple) and len(dense_inp) == 2:
                return self.forward_with_indices_and_features(*dense_inp)

            # dense_inp is expected to be in N C D H W format
            if torch.jit.is_tracing():
                return dense_inp

            dense_inp = dense_inp.permute(0, 2, 3, 4, 1)
            # Considering all indices as dense
            indices = (
                torch.stack(
                    torch.meshgrid(
                        torch.arange(dense_inp.shape[0]),
                        torch.arange(dense_inp.shape[1]),
                        torch.arange(dense_inp.shape[2]),
                        torch.arange(dense_inp.shape[3]),
                        indexing="ij",
                    ),
                    dim=-1,
                )
                .reshape(-1, 4)
                .int()
            )
            features = dense_inp.reshape(-1, dense_inp.shape[4])
            spatial_shape = dense_inp.shape[1:-1]
            return spconv.SparseConvTensor(
                features=features,
                indices=indices,
                spatial_shape=spatial_shape,
                batch_size=dense_inp.shape[0],
            )

        def forward(self, *args):
            """
            Forward pass for SparseConvTensor's custom implementation
            :param args: Either one dense input of format NCDHW or two inputs (indices, features) both in dense form
            :return: Sparse tensor
            """
            if len(args) == 2:
                return self.forward_with_indices_and_features(*args)
            return self.forward_with_dense_input(*args)

    class CustomScatterDense(torch.autograd.Function):
        """
        Custom Scatter Dense autograd function
        """

        @staticmethod
        def symbolic(g, dense_inputs, attrs):
            """
            Symbolic method (static) for ScatterDense
            :param g:ONNX graph object
            :param dense_inputs: Dense inputs
            :param attrs: ScatterDense attributes
            :return: Added op to the graph object
            """
            save_attrs = {}
            for k, v in attrs.items():
                if isinstance(v, str):
                    save_attrs[k + "_s"] = v
                else:
                    save_attrs[k + "_i"] = v
            return g.op("spconv::ScatterDense", dense_inputs, **save_attrs)

        @staticmethod
        def forward(ctx, dense_inputs, attrs):
            """
            forward method (static) for ScatterDense
            :param ctx: context object
            :param dense_inputs: Dense inputs
            :param attrs: ScatterDense attributes
            :return: Dense tensor
            """
            return dense_inputs

    class ScatterDense(torch.nn.Module):
        """
        ScatterDense custom implementation
        """

        def __init__(self):
            super(ScatterDense, self).__init__()

        def forward(self, inputs):
            """
            Forward pass for ScatterDense
            :param inputs: Sparse Inputs
            :return: Dense tensor
            """
            if torch.jit.is_tracing():
                attrs = {
                    "format": "xyz",
                    "input_spatial_shape": inputs.shape[2:],
                    "output_shape": inputs.shape,
                }
                return CustomScatterDense.apply(inputs, attrs)

            return (
                inputs.dense()
                if isinstance(inputs, spconv.SparseConvTensor)
                else inputs
            )

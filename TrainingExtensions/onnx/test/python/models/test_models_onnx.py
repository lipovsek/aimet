# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Dummy onnx models for testing"""

import onnx
import numpy as np


def create_initializer_tensor(
    name: str,
    tensor_array: np.ndarray,
    data_type: onnx.TensorProto = onnx.TensorProto.FLOAT,
) -> onnx.TensorProto:
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist(),
    )

    return initializer_tensor


class ModelWithMultipleInputs:
    @staticmethod
    def get_model():
        model_input_name = "X1"
        X1 = onnx.helper.make_tensor_value_info(
            model_input_name, onnx.TensorProto.FLOAT, [1, 3, 32, 32]
        )

        model_input_name = "X2"
        X2 = onnx.helper.make_tensor_value_info(
            model_input_name, onnx.TensorProto.FLOAT, [1, 3, 32, 32]
        )

        conv1_output_node_name = "Conv1_Y"

        # Dummy weights for conv.
        conv1_in_channels = 3
        conv1_out_channels = 3
        conv1_kernel_shape = (3, 3)
        conv1_pads = (1, 1, 1, 1)
        conv1_W = np.ones(
            shape=(conv1_out_channels, conv1_in_channels, *conv1_kernel_shape)
        ).astype(np.float32)
        conv1_B = np.ones(shape=(conv1_out_channels)).astype(np.float32)
        # Create the initializer tensor for the weights.
        conv1_W_initializer_tensor_name = "Conv1_W"
        conv1_W_initializer_tensor = create_initializer_tensor(
            name=conv1_W_initializer_tensor_name,
            tensor_array=conv1_W,
            data_type=onnx.TensorProto.FLOAT,
        )
        conv1_B_initializer_tensor_name = "Conv1_B"
        conv1_B_initializer_tensor = create_initializer_tensor(
            name=conv1_B_initializer_tensor_name,
            tensor_array=conv1_B,
            data_type=onnx.TensorProto.FLOAT,
        )

        conv1_node = onnx.helper.make_node(
            name="Conv1",
            op_type="Conv",
            inputs=[
                "X1",
                conv1_W_initializer_tensor_name,
                conv1_B_initializer_tensor_name,
            ],
            outputs=[conv1_output_node_name],
            kernel_shape=conv1_kernel_shape,
            pads=conv1_pads,
        )

        add_0_node_name = "ADD_0"

        add_0_node = onnx.helper.make_node(
            name=add_0_node_name,  # Name is optional.
            op_type="Add",
            # Must follow the order of input and output definitions.
            # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
            inputs=[conv1_output_node_name, "X2"],
            outputs=[add_0_node_name],
        )

        conv2_output_node_name = "Conv2_Y"

        # Dummy weights for conv.
        conv2_in_channels = 3
        conv2_out_channels = 3
        conv2_kernel_shape = (3, 3)
        conv2_pads = (1, 1, 1, 1)
        conv2_W = np.ones(
            shape=(conv2_out_channels, conv2_in_channels, *conv2_kernel_shape)
        ).astype(np.float32)
        conv2_B = np.ones(shape=(conv2_out_channels)).astype(np.float32)
        # Create the initializer tensor for the weights.
        conv2_W_initializer_tensor_name = "Conv2_W"
        conv2_W_initializer_tensor = create_initializer_tensor(
            name=conv2_W_initializer_tensor_name,
            tensor_array=conv2_W,
            data_type=onnx.TensorProto.FLOAT,
        )
        conv2_B_initializer_tensor_name = "Conv2_B"
        conv2_B_initializer_tensor = create_initializer_tensor(
            name=conv2_B_initializer_tensor_name,
            tensor_array=conv2_B,
            data_type=onnx.TensorProto.FLOAT,
        )

        conv2_node = onnx.helper.make_node(
            name="Conv2",  # Name is optional.
            op_type="Conv",
            # Must follow the order of input and output definitions.
            # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
            inputs=[
                add_0_node_name,
                conv2_W_initializer_tensor_name,
                conv2_B_initializer_tensor_name,
            ],
            outputs=[conv2_output_node_name],
            # The following arguments are attributes.
            kernel_shape=conv2_kernel_shape,
            # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
            pads=conv2_pads,
        )

        add_1_node_name = "ADD_1"

        add_1_node = onnx.helper.make_node(
            name=add_1_node_name,  # Name is optional.
            op_type="Add",
            # Must follow the order of input and output definitions.
            # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
            inputs=[conv2_output_node_name, "X1"],
            outputs=["Y"],
        )

        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(
            model_output_name, onnx.TensorProto.FLOAT, [1, 3, 32, 32]
        )

        graph_def = onnx.helper.make_graph(
            nodes=[conv1_node, add_0_node, conv2_node, add_1_node],
            name="ConvReluNet",
            inputs=[X1, X2],  # Graph input
            outputs=[Y],  # Graph output
            initializer=[
                conv1_W_initializer_tensor,
                conv1_B_initializer_tensor,
                conv2_W_initializer_tensor,
                conv2_B_initializer_tensor,
            ],
        )
        # Create the model (ModelProto)
        return onnx.helper.make_model(
            graph_def,
            producer_name="onnx-example",
            opset_imports=[onnx.helper.make_operatorsetid("", 20)],
            ir_version=10,
        )


def model_with_multiple_inputs():
    return ModelWithMultipleInputs.get_model()


class ModelWithMultipleOutputs:
    @staticmethod
    def get_model():
        model_input_name = "X1"
        X1 = onnx.helper.make_tensor_value_info(
            model_input_name, onnx.TensorProto.FLOAT, [1, 3, 32, 32]
        )

        model_input_name = "X2"
        X2 = onnx.helper.make_tensor_value_info(
            model_input_name, onnx.TensorProto.FLOAT, [1, 3, 32, 32]
        )

        conv1_output_node_name = "Conv1_Y"

        # Dummy weights for conv.
        conv1_in_channels = 3
        conv1_out_channels = 3
        conv1_kernel_shape = (3, 3)
        conv1_pads = (1, 1, 1, 1)
        conv1_W = np.ones(
            shape=(conv1_out_channels, conv1_in_channels, *conv1_kernel_shape)
        ).astype(np.float32)
        conv1_B = np.ones(shape=(conv1_out_channels)).astype(np.float32)
        # Create the initializer tensor for the weights.
        conv1_W_initializer_tensor_name = "Conv1_W"
        conv1_W_initializer_tensor = create_initializer_tensor(
            name=conv1_W_initializer_tensor_name,
            tensor_array=conv1_W,
            data_type=onnx.TensorProto.FLOAT,
        )
        conv1_B_initializer_tensor_name = "Conv1_B"
        conv1_B_initializer_tensor = create_initializer_tensor(
            name=conv1_B_initializer_tensor_name,
            tensor_array=conv1_B,
            data_type=onnx.TensorProto.FLOAT,
        )

        conv1_node = onnx.helper.make_node(
            name="Conv1",
            op_type="Conv",
            inputs=[
                "X1",
                conv1_W_initializer_tensor_name,
                conv1_B_initializer_tensor_name,
            ],
            outputs=[conv1_output_node_name],
            kernel_shape=conv1_kernel_shape,
            pads=conv1_pads,
        )

        add_0_node_name = "ADD_0"

        add_0_node = onnx.helper.make_node(
            name=add_0_node_name,  # Name is optional.
            op_type="Add",
            # Must follow the order of input and output definitions.
            # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
            inputs=[conv1_output_node_name, "X2"],
            outputs=[add_0_node_name],
        )

        conv2_output_node_name = "Conv2_Y"

        # Dummy weights for conv.
        conv2_in_channels = 3
        conv2_out_channels = 3
        conv2_kernel_shape = (3, 3)
        conv2_pads = (1, 1, 1, 1)
        conv2_W = np.ones(
            shape=(conv2_out_channels, conv2_in_channels, *conv2_kernel_shape)
        ).astype(np.float32)
        conv2_B = np.ones(shape=(conv2_out_channels)).astype(np.float32)
        # Create the initializer tensor for the weights.
        conv2_W_initializer_tensor_name = "Conv2_W"
        conv2_W_initializer_tensor = create_initializer_tensor(
            name=conv2_W_initializer_tensor_name,
            tensor_array=conv2_W,
            data_type=onnx.TensorProto.FLOAT,
        )
        conv2_B_initializer_tensor_name = "Conv2_B"
        conv2_B_initializer_tensor = create_initializer_tensor(
            name=conv2_B_initializer_tensor_name,
            tensor_array=conv2_B,
            data_type=onnx.TensorProto.FLOAT,
        )

        conv2_node = onnx.helper.make_node(
            name="Conv2",  # Name is optional.
            op_type="Conv",
            # Must follow the order of input and output definitions.
            # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
            inputs=[
                add_0_node_name,
                conv2_W_initializer_tensor_name,
                conv2_B_initializer_tensor_name,
            ],
            outputs=[conv2_output_node_name],
            # The following arguments are attributes.
            kernel_shape=conv2_kernel_shape,
            # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
            pads=conv2_pads,
        )

        add_1_node_name = "ADD_1"

        add_1_node = onnx.helper.make_node(
            name=add_1_node_name,  # Name is optional.
            op_type="Add",
            # Must follow the order of input and output definitions.
            # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
            inputs=[conv2_output_node_name, "X1"],
            outputs=["Y"],
        )

        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(
            model_output_name, onnx.TensorProto.FLOAT, [1, 3, 32, 32]
        )

        conv1_output = onnx.helper.make_tensor_value_info(
            conv1_output_node_name, onnx.TensorProto.FLOAT, [1, 3, 32, 32]
        )

        graph_def = onnx.helper.make_graph(
            nodes=[conv1_node, add_0_node, conv2_node, add_1_node],
            name="ConvReluNet",
            inputs=[X1, X2],  # Graph input
            outputs=[Y, conv1_output],  # Graph output
            initializer=[
                conv1_W_initializer_tensor,
                conv1_B_initializer_tensor,
                conv2_W_initializer_tensor,
                conv2_B_initializer_tensor,
            ],
        )
        # Create the model (ModelProto)
        return onnx.helper.make_model(
            graph_def,
            producer_name="onnx-example",
            opset_imports=[onnx.helper.make_operatorsetid("", 20)],
            ir_version=10,
        )


def model_with_multiple_outputs():
    return ModelWithMultipleOutputs.get_model()

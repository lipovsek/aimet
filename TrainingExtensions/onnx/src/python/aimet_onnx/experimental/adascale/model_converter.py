# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from aimet_common.utils import AimetLogger

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AdaScale)
import onnx
import os
from aimet_onnx.utils import ModelProto
from onnx.utils import extract_model
from onnx2torch import convert
from aimet_onnx.experimental.adascale.onnx2torch_ext import *  # pylint: disable=wildcard-import, unused-wildcard-import
from aimet_onnx.experimental.adascale.onnx2torch_ext import (
    copy_pt_weights_to_onnx as copy_wts,
)


class ModelConverter:
    """
    Given a onnx ModelProto:
    1. Create a pytorch model for a onnx subgraph
    2. Copy weights from pytorch to onnx
    """

    def __init__(self, model: ModelProto, checkpointdir: str):
        self.model = model
        self.checkpoint = checkpointdir
        self.fp_model_path = self._get_onnx_fp_model()
        self.ctr = 0

    def _get_onnx_fp_model(self):
        os.makedirs(self.checkpoint, exist_ok=True)
        onnx_model_path = os.path.join(self.checkpoint, "fp_model.onnx")
        onnx.save_model(
            self.model,
            onnx_model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="fp_model.data",
        )
        return onnx_model_path

    def _get_onnx_subgraph(self, block_input_output_names):
        """
        Given a onnx block end points get onnx subgraph
        """
        block_model_path = os.path.join(self.checkpoint, f"block_{self.ctr}_fp32.onnx")
        block_input_names, block_output_names = block_input_output_names
        try:
            extract_model(
                self.fp_model_path,
                block_model_path,
                block_input_names,
                block_output_names,
            )
            block_fp32_model = onnx.load(block_model_path)
            self.ctr += 1
            return block_fp32_model, block_model_path
        except:
            raise RuntimeError(  # pylint: disable=raise-missing-from
                f"Unable to extract onnx subgraph for given block input/output {block_input_output_names}"
            )

    def get_pt_block(self, block_input_output_names):
        """
        Given a onnx block end points get a pytorch block
        """
        onnx_block, block_model_path = self._get_onnx_subgraph(block_input_output_names)
        return convert(onnx_block), block_model_path

    def copy_pt_weights_to_onnx(self, pt_block, onnx_model):
        """
        Given a pt_block with adascale params computed, copy the params to onnx model
        """
        copy_wts(pt_block, onnx_model)

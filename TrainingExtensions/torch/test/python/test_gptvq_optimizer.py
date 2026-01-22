# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Test GPTVQ optimizer"""

import pytest
import torch

from aimet_torch.gptvq.gptvq_optimizer import GPTVQOptimizer
from aimet_torch.gptvq.utils import manipulate_inverse_hessian_diagonal


class TestGPTVQOptimizer:
    @pytest.mark.parametrize("vector_dim", [1, 2, 4])
    @pytest.mark.parametrize("num_blocks_per_column", [24, 12, 6])
    @pytest.mark.parametrize("min_value", [0, 16])
    def test_update_weight_block(self, vector_dim, num_blocks_per_column, min_value):
        start_index = 0
        num_of_centroids = 64

        weight_block = torch.zeros(768, 128)
        original_sliced_weight = weight_block[
            :, start_index : start_index + vector_dim
        ].clone()
        codebook = torch.arange(
            start=min_value,
            end=min_value + num_blocks_per_column * num_of_centroids * vector_dim,
            dtype=torch.float32,
        )
        codebook = codebook.reshape(num_blocks_per_column, num_of_centroids, vector_dim)

        updated_weight_block, _ = GPTVQOptimizer._update_weight_block(
            weight_block[:, start_index : start_index + vector_dim].clone(),
            codebook,
            vector_dim=vector_dim,
            num_blocks_per_column=num_blocks_per_column,
        )

        assert updated_weight_block.shape == original_sliced_weight.shape
        assert not torch.allclose(updated_weight_block, original_sliced_weight)

        updated_weight_block = updated_weight_block.reshape(
            num_blocks_per_column, -1, vector_dim
        )
        for group_index in range(num_blocks_per_column):
            current_group_weight = updated_weight_block[group_index]
            corresponding_codebook = codebook[group_index]

            # Since weights is zero tensor, the rounded weight should be first vector in codebook, which is the nearest vector
            nearest_vector = corresponding_codebook[0]
            assert all([row.equal(nearest_vector) for row in current_group_weight])

    @pytest.mark.parametrize("ndim", [None, 2, 3])
    def test_manipulate_hessian_inverse_diagonal(self, ndim):
        vector_dim = 2
        num_elements = 32

        tensor = torch.zeros(24, num_elements, vector_dim)
        if ndim == 2:
            hessian_inverse_diagonal = torch.zeros(1, vector_dim)
        elif ndim == 3:
            hessian_inverse_diagonal = torch.zeros(1, num_elements, vector_dim)
        else:
            hessian_inverse_diagonal = None

        manipulated_tensor = manipulate_inverse_hessian_diagonal(
            tensor, hessian_inverse_diagonal
        )
        if ndim == 2:
            assert torch.equal(hessian_inverse_diagonal, manipulated_tensor)
        elif ndim == 3:
            assert manipulated_tensor.shape == (1, num_elements, 1, vector_dim)
        else:
            assert torch.equal(
                manipulated_tensor, torch.ones(tensor.shape[-1], device=tensor.device)
            )

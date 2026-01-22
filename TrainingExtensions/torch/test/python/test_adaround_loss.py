# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Test AdaroundLoss"""

import numpy as np
import unittest.mock

import torch
import torch.nn.functional as functional
from aimet_torch._base.adaround.adaround_loss import (
    AdaroundLoss,
    AdaroundHyperParameters,
)


class TestAdaroundLoss(unittest.TestCase):
    """Test AdaroundLoss"""

    def test_compute_recon_loss(self):
        """test compute reconstruction loss using dummy input and target tensors"""
        np.random.seed(0)
        inp = np.random.rand(32, 3, 12, 12)
        target = np.random.rand(32, 3, 12, 12)
        inp_tensor = torch.from_numpy(inp)
        target_tensor = torch.from_numpy(target)

        recons_loss = AdaroundLoss.compute_recon_loss(inp_tensor, target_tensor)
        print(recons_loss.item())
        self.assertAlmostEqual(
            recons_loss.item(),
            functional.mse_loss(inp_tensor, target_tensor, reduction="none")
            .sum(1)
            .mean()
            .item(),
            places=5,
        )

        # Linear layer
        inp = np.random.rand(32, 10)
        target = np.random.rand(32, 10)
        inp_tensor = torch.from_numpy(inp)
        target_tensor = torch.from_numpy(target)

        recons_loss = AdaroundLoss.compute_recon_loss(inp_tensor, target_tensor)
        print(recons_loss.item())
        self.assertAlmostEqual(
            recons_loss.item(),
            functional.mse_loss(inp_tensor, target_tensor, reduction="none")
            .sum(1)
            .mean()
            .item(),
            places=5,
        )

    def test_compute_round_loss(self):
        """test compute rounding loss"""
        np.random.seed(0)
        alpha = np.random.rand(1, 3, 12, 12)
        alpha_tensor = torch.from_numpy(alpha)

        # Since warm start is 0.2 (20%), cut iter < 2000 (20% of 10000 iterations) will have rounding loss = 0
        opt_params = AdaroundHyperParameters(
            num_iterations=10000, reg_param=0.01, beta_range=(20, 2), warm_start=0.2
        )
        cur_iter = 10
        round_loss_1 = AdaroundLoss.compute_round_loss(
            alpha_tensor, opt_params, cur_iter
        )
        self.assertEqual(round_loss_1, 0)

        cur_iter = 8000
        round_loss_2 = AdaroundLoss.compute_round_loss(
            alpha_tensor, opt_params, cur_iter
        )
        self.assertAlmostEqual(round_loss_2.item(), 4.266156963161077, places=5)

    def test_compute_beta(self):
        """test compute beta"""
        num_iterations = 10000
        cur_iter = 8000
        beta_range = (20, 2)
        warm_start = 0.2
        self.assertEqual(
            AdaroundLoss._compute_beta(
                num_iterations, cur_iter, beta_range, warm_start
            ),
            4.636038969321072,
        )

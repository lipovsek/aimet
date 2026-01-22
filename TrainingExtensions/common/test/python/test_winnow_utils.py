# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""This file contains unit tests for testing the utilities in winnow_utils.py."""

import unittest

try:
    from aimet_onnx.common.winnow.winnow_utils import (
        get_indices_among_ones_of_overlapping_ones,
        update_winnowed_channels,
    )
except ImportError:
    from aimet_torch.common.winnow.winnow_utils import (
        get_indices_among_ones_of_overlapping_ones,
        update_winnowed_channels,
    )


class TestWinnowUtils(unittest.TestCase):
    """Test winnow_utils module"""

    def test_get_indices_among_ones_of_overlapping_ones(self):
        """Test the get_indices_among_ones_of_overlapping_ones() utility"""

        more_ones_mask = [1, 1, 0, 1, 0, 0, 1]
        less_ones_mask = [1, 0, 0, 1, 0, 0, 0]
        self.assertEqual(
            [0, 2],
            get_indices_among_ones_of_overlapping_ones(more_ones_mask, less_ones_mask),
        )

        more_ones_mask = [0, 0, 0]
        less_ones_mask = [0, 0, 0]
        self.assertEqual(
            [],
            get_indices_among_ones_of_overlapping_ones(more_ones_mask, less_ones_mask),
        )

        more_ones_mask = [1, 1, 1]
        less_ones_mask = [1, 1, 1]
        self.assertEqual(
            [0, 1, 2],
            get_indices_among_ones_of_overlapping_ones(more_ones_mask, less_ones_mask),
        )

        more_ones_mask = [0, 0, 1]
        less_ones_mask = [0, 0, 1]
        self.assertEqual(
            [0],
            get_indices_among_ones_of_overlapping_ones(more_ones_mask, less_ones_mask),
        )

    def test_update_winnowed_channels(self):
        """Test the update_winnowed_channels utility"""
        original_mask = [1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1]
        new_mask = [1, 1, 0, 0, 1, 0, 1]
        update_winnowed_channels(original_mask, new_mask)
        self.assertEqual([1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], original_mask)

        # Test that assertion is raised if length of new mask is different than number of ones in original mask
        with self.assertRaises(AssertionError):
            original_mask = [1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1]
            new_mask = [1, 1, 0, 0, 1, 0]
            update_winnowed_channels(original_mask, new_mask)

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Module to keep track of information related to channels in a layer
that have zero planes."""

from collections import OrderedDict


class PolySlice:
    """Specifies one or more slices in one or more dimensions of a multi-dimensional array;
    can be used to for example specify slices to be removed from a tensor."""

    def __init__(self, dim=None, index=None):
        """dim is an int; indices may be a single int or list of int's"""
        self._slices_by_dim = {}  # a set per dimension
        if dim is not None:
            assert index not in (None, [])
            self.set(dim, index)

    def __repr__(self):
        """Printable representation of the object."""
        slices_by_dim = self.get_all()
        repr_str = ""
        for dim, indices in slices_by_dim.items():
            slices = ", ".join(str(idx) for idx in indices)
            repr_str += "dim." + str(dim) + ": " + slices + "  "
        return repr_str

    def __eq__(self, poly_slice):
        """Compares the argument PolySlice with the self PolySlice and
        returns True if they are equal. Otherwise, returns False."""
        return poly_slice._slices_by_dim == self._slices_by_dim  # pylint: disable=protected-access

    def set(self, dim, index):
        """Set the specified slices for the specified dimension"""
        self._slices_by_dim[dim] = set()
        self.add(dim, index)

    def add(self, dim, index):
        """Add one or more slices in the specified dimension"""
        to_add = index if isinstance(index, list) else [index]
        if dim in self._slices_by_dim:
            for idx in to_add:
                self._slices_by_dim[dim].add(idx)
        else:
            self._slices_by_dim[dim] = set(index)

    @property
    def num_dims(self):
        """Returns the number of dimensions of the PolySlice object."""
        return len(self._slices_by_dim)

    def get_dims(self):
        """Returns the dimensions which have zero channels"""
        return sorted(self._slices_by_dim)

    def get_slices(self, dim):
        """Returns the indices which have zero channels."""
        return sorted(self._slices_by_dim[dim])

    def get_all(self):
        """Returns all the dimensions and the corresponding channels
        with zero planes."""
        result = OrderedDict()
        for dim in sorted(self._slices_by_dim):
            result[dim] = sorted(list(self._slices_by_dim[dim]))
        return result

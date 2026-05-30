from __future__ import annotations

import unittest

import numpy as np

from hardware.control_basis import make_control_basis
from hardware.transition_tuner import FixedGrid


class ControlBasisTests(unittest.TestCase):
    def build_grid(self) -> FixedGrid:
        return FixedGrid(
            n_sites=8,
            edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)],
            base_coupling=1.0,
            input_index=0,
            target_indices=[6, 7],
        )

    def test_required_site_and_edge_bases_exist(self):
        basis = make_control_basis(self.build_grid())
        for key in ["uniform", "linear_gradient", "input_window", "target_window", "center_bowl", "boundary_suppression"]:
            self.assertIn(key, basis.site_basis)
            self.assertEqual(basis.site_basis[key].shape, (8,))
        for key in ["uniform_edges", "input_to_target_gradient", "target_corridor", "boundary_edges", "center_edges"]:
            self.assertIn(key, basis.edge_basis)

    def test_edge_basis_contains_only_existing_edges(self):
        basis = make_control_basis(self.build_grid())
        allowed = {tuple(sorted(edge)) for edge in self.build_grid().edges}
        for edge_map in basis.edge_basis.values():
            self.assertTrue(set(edge_map).issubset(allowed))

    def test_boundary_suppression_is_larger_at_edges_than_middle(self):
        basis = make_control_basis(self.build_grid())
        boundary = basis.site_basis["boundary_suppression"]
        self.assertGreater(boundary[0], boundary[len(boundary) // 2])

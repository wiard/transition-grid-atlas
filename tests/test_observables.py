from __future__ import annotations

import unittest

import numpy as np

from engine.lindblad import psi_to_rho
from engine.observables import (
    edge_contact,
    frame_correlation,
    ipr,
    probability_from_psi,
    probability_from_rho,
    x_mean,
    x_variance,
)


class ObservableTests(unittest.TestCase):
    def test_probability_from_psi_sums_to_one(self):
        psi = np.array([1.0 + 0.0j, 1.0j], dtype=np.complex128)
        p = probability_from_psi(psi)
        self.assertAlmostEqual(float(np.sum(p)), 1.0, places=12)

    def test_probability_from_rho_sums_to_one(self):
        psi = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
        p = probability_from_rho(psi_to_rho(psi / np.linalg.norm(psi)))
        self.assertAlmostEqual(float(np.sum(p)), 1.0, places=12)

    def test_x_mean_and_variance_match_simple_distribution(self):
        p = np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float64)
        self.assertAlmostEqual(x_mean(p), 1.5, places=12)
        self.assertAlmostEqual(x_variance(p), 0.25, places=12)

    def test_ipr_distinguishes_localized_and_spread_distributions(self):
        delta = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        spread = np.full(4, 0.25, dtype=np.float64)
        self.assertGreater(ipr(delta), ipr(spread))

    def test_edge_contact_detects_boundary_mass(self):
        p = np.array([0.6, 0.2, 0.2, 0.0], dtype=np.float64)
        self.assertTrue(edge_contact(p, edge_width=1, threshold=0.5))
        self.assertFalse(edge_contact(np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float64), edge_width=1, threshold=0.2))

    def test_frame_correlation_is_bounded(self):
        a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        b = np.array([0.0, 0.8, 0.2], dtype=np.float64)
        corr = frame_correlation(a, b)
        self.assertGreaterEqual(corr, -1.0)
        self.assertLessEqual(corr, 1.0)

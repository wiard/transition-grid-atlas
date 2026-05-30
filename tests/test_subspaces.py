from __future__ import annotations

import unittest

import numpy as np

from hardware.subspaces import (
    complement_projector,
    diagonal_phase_noise,
    information_subspace,
    noise_overlap_with_subspace,
    projector_from_basis,
    suppression_score,
)


class SubspaceTests(unittest.TestCase):
    def test_projector_is_hermitian_and_idempotent(self):
        U = information_subspace(4, 0, [3])
        P = projector_from_basis(U)
        self.assertTrue(np.allclose(P, np.conjugate(P.T)))
        self.assertTrue(np.allclose(P @ P, P))

    def test_complement_projector_sums_to_identity(self):
        U = information_subspace(5, 0, [3, 4])
        P = projector_from_basis(U)
        Q = complement_projector(P)
        self.assertTrue(np.allclose(P + Q, np.eye(5, dtype=np.complex128)))

    def test_noise_inside_information_subspace_has_higher_overlap_than_outside_noise(self):
        U = information_subspace(4, 0, [3])
        inside_noise = [diagonal_phase_noise(np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64))]
        outside_noise = [diagonal_phase_noise(np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64))]
        self.assertGreater(
            noise_overlap_with_subspace(inside_noise, U),
            noise_overlap_with_subspace(outside_noise, U),
        )

    def test_suppression_score_is_bounded(self):
        U = information_subspace(4, 0, [3])
        noise_ops = [diagonal_phase_noise(np.array([0.5, 0.2, 0.1, 0.5], dtype=np.float64))]
        score = suppression_score(noise_ops, U)
        self.assertGreaterEqual(score, -1.0e-12)
        self.assertLessEqual(score, 1.0 + 1.0e-12)

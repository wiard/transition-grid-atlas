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
    transport_subspace_from_hamiltonian,
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

    def test_fixed_geometric_subspace_overlap_is_invariant_for_fixed_noise(self):
        U = information_subspace(5, 0, [3, 4])
        noise_ops = [
            diagonal_phase_noise(np.array([0.0, 0.3, 0.6, 0.2, 0.1], dtype=np.float64)),
            diagonal_phase_noise(np.array([0.0, 0.1, 0.4, 0.3, 0.2], dtype=np.float64)),
        ]
        overlap0 = noise_overlap_with_subspace(noise_ops, U)
        overlap1 = noise_overlap_with_subspace(noise_ops, U)
        self.assertAlmostEqual(overlap0, overlap1)

    def test_transport_subspace_from_hamiltonian_changes_when_h_changes(self):
        H0 = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.complex128,
        )
        H1 = np.array(H0, copy=True)
        H1[1, 1] = 0.7
        H1[2, 2] = -0.4

        U0 = transport_subspace_from_hamiltonian(H0, 0, [3])
        U1 = transport_subspace_from_hamiltonian(H1, 0, [3])
        P0 = projector_from_basis(U0)
        P1 = projector_from_basis(U1)
        self.assertFalse(np.allclose(P0, P1))

from __future__ import annotations

import unittest

import numpy as np

from hardware.transition_tuner import (
    FixedGrid,
    TransitionTunerConfig,
    apply_transition_controls,
    build_base_hamiltonian,
    noise_operators_from_profiles,
    random_transition_search,
    transport_efficiency,
)


class TransitionTunerTests(unittest.TestCase):
    def build_grid(self) -> FixedGrid:
        return FixedGrid(
            n_sites=6,
            edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
            base_coupling=1.0,
            input_index=0,
            target_indices=[4, 5],
        )

    def build_config(self) -> TransitionTunerConfig:
        return TransitionTunerConfig(
            max_relative_coupling_delta=0.15,
            max_onsite_shift=0.10,
            n_candidates=80,
            seed=7,
            time_min=0.0,
            time_max=12.0,
            n_time_samples=60,
            transport_weight=1.0,
            suppression_weight=0.5,
            control_penalty_weight=0.01,
        )

    def build_noise_ops(self) -> list[np.ndarray]:
        return noise_operators_from_profiles(
            [
                [0.0, 0.2, 0.8, 1.0, 0.5, 0.1],
                [0.0, 0.1, 0.4, 0.8, 0.4, 0.1],
            ]
        )

    def test_build_base_hamiltonian_is_hermitian(self):
        H0 = build_base_hamiltonian(self.build_grid())
        self.assertTrue(np.allclose(H0, np.conjugate(H0.T)))

    def test_apply_transition_controls_does_not_create_new_nonzero_off_diagonal_edges(self):
        grid = self.build_grid()
        H0 = build_base_hamiltonian(grid)
        tuned = apply_transition_controls(
            H0,
            grid,
            {(0, 1): 1.1, (2, 3): 0.9},
            np.zeros(grid.n_sites, dtype=np.float64),
        )
        base_support = (np.abs(H0) > 1.0e-12).astype(int)
        tuned_support = (np.abs(tuned) > 1.0e-12).astype(int)
        self.assertTrue(np.array_equal(base_support - np.eye(grid.n_sites, dtype=int), tuned_support - np.eye(grid.n_sites, dtype=int)))

    def test_transport_efficiency_is_bounded(self):
        grid = self.build_grid()
        H0 = build_base_hamiltonian(grid)
        times = np.linspace(0.0, 12.0, 60, dtype=np.float64)
        efficiency = transport_efficiency(H0, grid.input_index, list(grid.target_indices), times)
        self.assertGreaterEqual(efficiency, 0.0)
        self.assertLessEqual(efficiency, 1.0 + 1.0e-12)

    def test_random_transition_search_is_deterministic_with_fixed_seed(self):
        grid = self.build_grid()
        noise_ops = self.build_noise_ops()
        config = self.build_config()
        result_a = random_transition_search(grid, noise_ops, config)
        result_b = random_transition_search(grid, noise_ops, config)
        self.assertEqual(result_a, result_b)

    def test_random_transition_search_reports_baseline_and_best_metrics(self):
        result = random_transition_search(self.build_grid(), self.build_noise_ops(), self.build_config())
        self.assertGreaterEqual(result.best_objective + 1.0e-12, result.baseline_objective)
        self.assertGreaterEqual(result.baseline_transport_efficiency, 0.0)
        self.assertGreaterEqual(result.best_transport_efficiency, 0.0)
        self.assertGreaterEqual(result.baseline_suppression_score, 0.0)
        self.assertGreaterEqual(result.best_suppression_score, 0.0)

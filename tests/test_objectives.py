from __future__ import annotations

import unittest

import numpy as np

from hardware.objectives import (
    control_cost,
    evaluate_motor_metrics,
    motor_objective,
    noise_metric_decomposition,
    projector_from_transport_subspace,
)
from hardware.subspaces import diagonal_phase_noise
from hardware.transition_tuner import FixedGrid, build_base_hamiltonian


class ObjectiveTests(unittest.TestCase):
    def build_grid(self) -> FixedGrid:
        return FixedGrid(
            n_sites=6,
            edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
            base_coupling=1.0,
            input_index=0,
            target_indices=[4, 5],
        )

    def test_o_info_equals_internal_plus_leak(self):
        grid = self.build_grid()
        H = build_base_hamiltonian(grid)
        P = projector_from_transport_subspace(H, grid.input_index, list(grid.target_indices))
        noise = [diagonal_phase_noise(np.array([0.0, 0.2, 0.4, 0.6, 0.3, 0.1], dtype=np.float64))]
        metrics = noise_metric_decomposition(noise, P)
        self.assertAlmostEqual(
            metrics["noise_action_on_info"],
            metrics["noise_internal"] + metrics["noise_leakage"],
            places=10,
        )

    def test_metrics_are_finite_and_bounded(self):
        grid = self.build_grid()
        H = build_base_hamiltonian(grid)
        times = np.linspace(0.0, 12.0, 40, dtype=np.float64)
        metrics = evaluate_motor_metrics(
            H,
            [diagonal_phase_noise(np.array([0.0, 0.2, 0.4, 0.6, 0.3, 0.1], dtype=np.float64))],
            grid,
            {"k1": 0.05},
            times,
            {"transport": 1.0, "noise_action": 0.5, "leakage": 0.25, "control_cost": 0.01},
        )
        self.assertTrue(np.isfinite(metrics.objective))
        self.assertGreaterEqual(metrics.transport_efficiency, 0.0)
        self.assertLessEqual(metrics.transport_efficiency, 1.0 + 1.0e-12)
        self.assertGreaterEqual(metrics.noise_internal, 0.0)
        self.assertGreaterEqual(metrics.noise_leakage, 0.0)
        self.assertGreaterEqual(metrics.noise_action_on_info, 0.0)
        self.assertGreaterEqual(metrics.suppression_score, -1.0e-12)
        self.assertLessEqual(metrics.suppression_score, 1.0 + 1.0e-12)

    def test_objective_changes_with_weights_and_cost(self):
        objective_a = motor_objective(
            transport_efficiency=0.8,
            noise_action_on_info=0.2,
            noise_leakage=0.1,
            control_cost=0.05,
            weights={"transport": 1.0, "noise_action": 0.5, "leakage": 0.25, "control_cost": 0.01},
        )
        objective_b = motor_objective(
            transport_efficiency=0.8,
            noise_action_on_info=0.2,
            noise_leakage=0.1,
            control_cost=0.05,
            weights={"transport": 0.5, "noise_action": 1.0, "leakage": 0.5, "control_cost": 0.01},
        )
        self.assertNotEqual(objective_a, objective_b)
        self.assertGreater(control_cost({"a": 0.2, "b": 0.2}), control_cost({"a": 0.1, "b": 0.1}))

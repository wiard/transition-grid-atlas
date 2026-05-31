from __future__ import annotations

import unittest

import numpy as np

from hardware.control_basis import make_control_basis
from hardware.control_knobs import ControlKnob, KnobRegistry
from hardware.objectives import ObjectiveNormalizationScale, evaluate_motor_metrics
from hardware.transition_motor import (
    build_control_hamiltonian,
    finite_difference_gradient,
    projected_gradient_ascent,
    random_restart_transition_motor_search,
    transition_motor_config_from_dict,
)
from hardware.transition_tuner import FixedGrid, build_base_hamiltonian, noise_operators_from_profiles


class TransitionMotorTests(unittest.TestCase):
    def build_grid(self) -> FixedGrid:
        return FixedGrid(
            n_sites=8,
            edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)],
            base_coupling=1.0,
            input_index=0,
            target_indices=[6, 7],
        )

    def build_registry(self) -> KnobRegistry:
        return KnobRegistry(
            knobs=(
                ControlKnob("grad", "onsite_phase", "g", "linear_gradient", -0.1, 0.1, 0.0, "arb", "grad", "grad"),
                ControlKnob("corridor", "coherent_coupling", "c", "target_corridor", -0.15, 0.15, 0.0, "arb", "corridor", "corridor"),
            )
        )

    def build_metrics_objective(self):
        grid = self.build_grid()
        basis = make_control_basis(grid)
        registry = self.build_registry()
        H0 = build_base_hamiltonian(grid)
        noise_ops = noise_operators_from_profiles([[0.0, 0.1, 0.3, 0.5, 0.3, 0.1, 0.0, 0.0]])
        times = np.linspace(0.0, 12.0, 40, dtype=np.float64)
        weights = {"transport": 1.0, "noise_action": 0.5, "leakage": 0.25, "control_cost": 0.01}

        def metrics_fn(theta):
            H = build_control_hamiltonian(H0, grid, theta, basis, registry)
            return evaluate_motor_metrics(H, noise_ops, grid, theta, times, weights, n_modes=2)

        return H0, grid, basis, registry, noise_ops, metrics_fn

    def test_linear_gradient_sets_linear_diagonal_pattern(self):
        H0, grid, basis, registry, _, _ = self.build_metrics_objective()
        H = build_control_hamiltonian(H0, grid, {"grad": 0.05, "corridor": 0.0}, basis, registry)
        diag_delta = np.real(np.diag(H - H0))
        diffs = np.diff(diag_delta)
        self.assertTrue(np.allclose(diffs, diffs[0]))

    def test_build_control_hamiltonian_preserves_hermiticity_and_no_new_edges(self):
        H0, grid, basis, registry, _, _ = self.build_metrics_objective()
        H = build_control_hamiltonian(H0, grid, {"grad": 0.05, "corridor": 0.1}, basis, registry)
        self.assertTrue(np.allclose(H, np.conjugate(H.T)))
        offdiag_mask = ~np.eye(grid.n_sites, dtype=bool)
        base_support = np.abs(H0[offdiag_mask]) > 1.0e-12
        support = np.abs(H[offdiag_mask]) > 1.0e-12
        self.assertTrue(np.array_equal(base_support, support))

    def test_finite_difference_gradient_returns_all_knobs(self):
        _, _, _, registry, _, metrics_fn = self.build_metrics_objective()
        gradient = finite_difference_gradient(lambda theta: metrics_fn(theta).objective, registry.defaults(), registry, eps=1.0e-4)
        self.assertEqual(set(gradient), set(registry.names()))

    def test_projected_search_is_deterministic_and_respects_ranges(self):
        _, _, _, registry, _, metrics_fn = self.build_metrics_objective()
        theta_a, value_a = projected_gradient_ascent(
            lambda theta: metrics_fn(theta).objective,
            registry.defaults(),
            registry,
            n_steps=10,
            step_size=0.05,
            eps=1.0e-4,
        )
        theta_b, value_b = projected_gradient_ascent(
            lambda theta: metrics_fn(theta).objective,
            registry.defaults(),
            registry,
            n_steps=10,
            step_size=0.05,
            eps=1.0e-4,
        )
        self.assertEqual(theta_a, theta_b)
        self.assertAlmostEqual(value_a, value_b)
        registry.validate_theta(theta_a)

    def test_random_restart_search_improves_or_matches_baseline(self):
        H0, grid, basis, registry, noise_ops, metrics_fn = self.build_metrics_objective()

        class Config:
            objective_weights = {"transport": 1.0, "noise_action": 0.5, "leakage": 0.25, "control_cost": 0.01}
            time_min = 0.0
            time_max = 12.0
            n_time_samples = 40
            n_transport_modes = 2
            finite_diff_eps = 1.0e-4
            optimizer_steps = 10
            optimizer_step_size = 0.05
            random_restarts = 3
            seed = 42

        result = random_restart_transition_motor_search(H0, grid, noise_ops, basis, registry, Config())
        self.assertGreaterEqual(result.best_metrics.objective + 1.0e-12, result.baseline_metrics.objective)
        registry.validate_theta(result.best_theta)

    def test_transition_motor_config_resolves_objective_mode_registry(self):
        config, _ = transition_motor_config_from_dict(
            {
                "transition_motor": {
                    "grid": {
                        "n_sites": 4,
                        "edges": [[0, 1], [1, 2], [2, 3]],
                        "base_coupling": 1.0,
                        "input_index": 0,
                        "target_indices": [2, 3],
                    },
                    "knobs": [
                        {
                            "name": "grad",
                            "family": "onsite_phase",
                            "symbol": "g",
                            "basis_name": "linear_gradient",
                            "min_value": -0.1,
                            "max_value": 0.1,
                            "default": 0.0,
                            "units": "arb",
                            "description": "grad",
                            "hardware_meaning": "grad",
                        }
                    ],
                    "noise": {"profiles": [[0.0, 0.2, 0.1, 0.0]]},
                    "objective": {
                        "selected_mode": "noise_mode",
                        "normalization_scales": {
                            "transport": 0.05,
                            "noise_action": 0.0025,
                            "leakage": 0.0010,
                            "control_cost": 0.04,
                        },
                        "objective_mode_registry": {
                            "default_mode": "detector_mode",
                            "modes": [
                                {
                                    "name": "detector_mode",
                                    "mode": "normalized",
                                    "transport": 3.0,
                                    "noise_action": 0.0,
                                    "leakage": 0.0,
                                    "control_cost": 0.005,
                                    "description": "detector",
                                },
                                {
                                    "name": "noise_mode",
                                    "mode": "calibrated",
                                    "transport": 0.25,
                                    "noise_action": 4.0,
                                    "leakage": 1.0,
                                    "control_cost": 0.01,
                                    "description": "noise",
                                },
                            ],
                        },
                    },
                    "optimizer": {
                        "time_min": 0.0,
                        "time_max": 8.0,
                        "n_time_samples": 20,
                        "n_transport_modes": 2,
                        "finite_diff_eps": 1.0e-4,
                        "optimizer_steps": 5,
                        "optimizer_step_size": 0.05,
                        "random_restarts": 1,
                        "seed": 42,
                    },
                }
            }
        )
        self.assertEqual(config.objective_mode, "calibrated")
        self.assertEqual(config.selected_objective_mode, "noise_mode")
        self.assertEqual(config.objective_weights["noise_action"], 4.0)
        self.assertEqual(config.objective_mode_registry.names(), ["detector_mode", "noise_mode"])

    def test_random_restart_search_supports_normalized_objective_mode(self):
        H0, grid, basis, registry, noise_ops, metrics_fn = self.build_metrics_objective()

        class Config:
            objective_weights = {"transport": 0.25, "noise_action": 4.0, "leakage": 1.0, "control_cost": 0.01}
            objective_mode = "normalized"
            normalization_scales = ObjectiveNormalizationScale(
                transport=0.05,
                noise_action=0.0025,
                leakage=0.0010,
                control_cost=0.04,
            )
            time_min = 0.0
            time_max = 12.0
            n_time_samples = 40
            n_transport_modes = 2
            finite_diff_eps = 1.0e-4
            optimizer_steps = 10
            optimizer_step_size = 0.05
            random_restarts = 2
            seed = 42

        result = random_restart_transition_motor_search(H0, grid, noise_ops, basis, registry, Config())
        self.assertAlmostEqual(result.baseline_metrics.objective, 0.0, places=10)
        self.assertGreaterEqual(result.best_metrics.objective + 1.0e-12, result.baseline_metrics.objective)
        registry.validate_theta(result.best_theta)

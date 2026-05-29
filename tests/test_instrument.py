from __future__ import annotations

import unittest

from engine import run_transport_simulation
from engine.observables import classify_transport_regime
from explorer.parameter_sweep import detect_phase_boundary_zones, run_parameter_sweep
from validation.statistics import two_window_scaling_metrics

import numpy as np


def build_small_config():
    return {
        "simulation": {
            "grid_size": 15,
            "steps": 12,
            "t_max_steps": 12,
            "dt": 0.20,
            "hopping": 1.0,
            "initial_site": "center",
            "feedback_smoothing": 1.0,
            "fit_start_step": 2,
            "fit_end_step": None,
            "validation_tolerance": 1.0e-12,
        },
        "parameters": {
            "W": 0.8,
            "bias": 0.05,
            "eta": 0.35,
            "gamma": 0.0,
            "seed": 7,
        },
        "montecarlo": {
            "runs": 5,
            "base_seed": 100,
            "vary": {"W_std": 0.1, "bias_std": 0.01, "eta_std": 0.05, "gamma_std": 0.0},
        },
        "sweep": {
            "W_values": [0.0, 1.0],
            "bias_values": [0.0],
            "eta_values": [0.0, 0.4],
            "gamma_values": [0.0],
            "phase_map_axes": ["W", "eta"],
            "phase_metric": "alpha",
            "phase_valid_only": True,
            "boundary_detection": {"high_alpha_threshold": 0.8, "low_alpha_threshold": 0.2},
        },
        "hunter": {
            "target_alpha": 0.5,
            "rounds": 2,
            "samples_per_round": 4,
            "shrink_factor": 0.5,
            "search_center": {"W": 1.0, "bias": 0.05, "eta": 0.3, "gamma": 0.0},
            "search_span": {"W": 0.5, "bias": 0.05, "eta": 0.3, "gamma": 0.0},
        },
    }


class TransitionGridAtlasTests(unittest.TestCase):
    def test_two_window_scaling_detects_stable_fit(self):
        x = np.linspace(1.0, 6.0, 6, dtype=np.float64)
        y = 0.50 * x
        metrics = two_window_scaling_metrics(x=x, y=y)
        self.assertTrue(metrics["stable"])
        self.assertLessEqual(metrics["relative_shift"], 0.10)

    def test_two_window_scaling_detects_transient_shift(self):
        x = np.linspace(1.0, 6.0, 6, dtype=np.float64)
        y = np.array([0.20, 0.40, 0.60, 1.60, 2.00, 2.40], dtype=np.float64)
        metrics = two_window_scaling_metrics(x=x, y=y)
        self.assertFalse(metrics["stable"])
        self.assertGreater(metrics["relative_shift"], 0.10)

    def test_diffusive_candidate_requires_tight_alpha_and_high_r2(self):
        valid = classify_transport_regime(
            alpha=0.50,
            r2=0.97,
            unitarity_error_value=1.0e-15,
            hermitian_error_value=0.0,
            scaling_window_stable=True,
        )
        rejected = classify_transport_regime(
            alpha=0.56,
            r2=0.97,
            unitarity_error_value=1.0e-15,
            hermitian_error_value=0.0,
            scaling_window_stable=True,
        )
        transient = classify_transport_regime(
            alpha=0.50,
            r2=0.97,
            unitarity_error_value=1.0e-15,
            hermitian_error_value=0.0,
            scaling_window_stable=False,
        )
        self.assertEqual(valid, "VALID_DIFFUSIVE_CANDIDATE")
        self.assertEqual(rejected, "WEAK_FIT")
        self.assertEqual(transient, "WEAK_FIT")

    def test_unitary_run_has_expected_fields(self):
        result = run_transport_simulation(build_small_config())
        self.assertIn("alpha", result)
        self.assertIn("r2", result)
        self.assertIn("unitarity_error", result)
        self.assertIn("hermitian_error", result)
        self.assertIn("early_alpha", result)
        self.assertIn("late_alpha", result)
        self.assertIn("alpha_window_shift", result)
        self.assertLess(result["hermitian_error"], 1.0e-12)

    def test_dissipative_run_is_marked_invalid_nonunitary(self):
        config = build_small_config()
        result = run_transport_simulation(config, parameters={"gamma": 0.20})
        self.assertEqual(result["status"], "INVALID_NONUNITARY")

    def test_sweep_produces_expected_number_of_points(self):
        config = build_small_config()
        results = run_parameter_sweep(config)
        self.assertEqual(len(results), 4)
        zones = detect_phase_boundary_zones(
            config,
            [
                {
                    "alpha": 0.9,
                    "status": "VALID_BALLISTIC",
                    "parameters": {"W": 0.0, "bias": 0.0, "eta": 0.4, "gamma": 0.0},
                },
                {
                    "alpha": 0.1,
                    "status": "VALID_LOCALIZED",
                    "parameters": {"W": 1.0, "bias": 0.0, "eta": 0.4, "gamma": 0.0},
                },
            ],
        )
        self.assertEqual(len(zones), 1)
        self.assertEqual(zones[0]["status"], "PHASE_BOUNDARY_ZONE")


if __name__ == "__main__":
    unittest.main()

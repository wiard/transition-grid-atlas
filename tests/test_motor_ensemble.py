from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from hardware.control_basis import make_control_basis
from hardware.control_knobs import knob_registry_from_dicts
from hardware.motor_ensemble import (
    MotorEnsembleSampleResult,
    apply_fabrication_disorder,
    detector_distribution_at_time,
    detector_final_probability,
    detector_success_probability,
    run_single_motor_ensemble_sample,
    run_transition_motor_ensemble_study,
    sample_correlated_profile,
    sample_fabrication_disorder_profiles,
    sample_phase_noise_profiles,
    summarize_motor_ensemble,
    transition_motor_ensemble_from_dict,
    write_motor_ensemble_csv,
    write_motor_ensemble_summary_json,
)
from hardware.transition_tuner import FixedGrid
from run import run_transition_motor_ensemble_mode


class MotorEnsembleTests(unittest.TestCase):
    def build_config(self, csv_path: str, summary_path: str) -> dict[str, object]:
        return {
            "transition_motor_ensemble": {
                "grid": {
                    "n_sites": 6,
                    "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
                    "base_coupling": 1.0,
                    "input_index": 0,
                    "target_indices": [4, 5],
                },
                "knobs": [
                    {
                        "name": "path_coupling_boost",
                        "family": "coherent_coupling",
                        "symbol": "a_path",
                        "basis_name": "target_corridor",
                        "min_value": -0.1,
                        "max_value": 0.1,
                        "default": 0.0,
                        "units": "arb",
                        "description": "path",
                        "hardware_meaning": "path",
                    },
                    {
                        "name": "onsite_phase_gradient",
                        "family": "onsite_phase",
                        "symbol": "b_grad",
                        "basis_name": "linear_gradient",
                        "min_value": -0.1,
                        "max_value": 0.1,
                        "default": 0.0,
                        "units": "arb",
                        "description": "grad",
                        "hardware_meaning": "grad",
                    },
                ],
                "fabrication_disorder": {
                    "n_samples": 4,
                    "onsite_sigma": 0.05,
                    "correlation_length_sites": 2.0,
                    "seed": 11,
                },
                "phase_noise": {
                    "n_profiles_per_sample": 2,
                    "profile_sigma": 0.7,
                    "correlation_length_sites": 2.0,
                    "seed": 21,
                },
                "objective_weights": {
                    "transport": 1.0,
                    "noise_action": 0.5,
                    "leakage": 0.25,
                    "control_cost": 0.01,
                },
                "optimizer": {
                    "time_min": 0.0,
                    "time_max": 12.0,
                    "n_time_samples": 40,
                    "n_transport_modes": 2,
                    "finite_diff_eps": 1.0e-4,
                    "optimizer_steps": 12,
                    "optimizer_step_size": 0.05,
                    "random_restarts": 2,
                    "seed": 7,
                },
                "success_criteria": {
                    "min_objective_gain": -1.0e-9,
                    "min_detector_gain": -0.05,
                    "min_noise_action_reduction": -1.0e-9,
                },
                "outputs": {
                    "csv_path": csv_path,
                    "summary_path": summary_path,
                },
            }
        }

    def test_correlated_profile_shape_reproducibility_and_zero_sigma(self):
        rng_a = np.random.default_rng(123)
        rng_b = np.random.default_rng(123)
        profile_a = sample_correlated_profile(8, 0.4, 2.0, rng_a)
        profile_b = sample_correlated_profile(8, 0.4, 2.0, rng_b)
        self.assertEqual(profile_a.shape, (8,))
        self.assertTrue(np.allclose(profile_a, profile_b))
        self.assertTrue(np.allclose(sample_correlated_profile(8, 0.0, 2.0, np.random.default_rng(1)), 0.0))
        with self.assertRaises(ValueError):
            sample_correlated_profile(8, 0.4, 0.0, np.random.default_rng(1))

    def test_fabrication_and_phase_noise_profiles_are_reproducible(self):
        fab_a = sample_fabrication_disorder_profiles(3, 8, 0.1, 2.0, 55)
        fab_b = sample_fabrication_disorder_profiles(3, 8, 0.1, 2.0, 55)
        noise_a = sample_phase_noise_profiles(2, 8, 1.0, 2.0, 99)
        noise_b = sample_phase_noise_profiles(2, 8, 1.0, 2.0, 99)
        self.assertTrue(all(np.allclose(a, b) for a, b in zip(fab_a, fab_b)))
        self.assertTrue(all(np.allclose(a, b) for a, b in zip(noise_a, noise_b)))

    def test_apply_fabrication_disorder_changes_only_diagonal(self):
        H = np.zeros((4, 4), dtype=np.complex128)
        H[0, 1] = H[1, 0] = 1.0
        H[1, 2] = H[2, 1] = 0.5
        disorder = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float64)
        out = apply_fabrication_disorder(H, disorder)
        self.assertTrue(np.allclose(np.diag(out), disorder))
        self.assertTrue(np.allclose(np.triu(out, 1), np.triu(H, 1)))
        self.assertTrue(np.allclose(np.tril(out, -1), np.tril(H, -1)))
        self.assertTrue(np.allclose(out, np.conjugate(out.T)))

    def test_detector_distribution_and_probabilities_are_bounded(self):
        H = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        distribution = detector_distribution_at_time(H, 0, np.pi / 4.0)
        self.assertAlmostEqual(float(np.sum(distribution)), 1.0, places=10)
        success = detector_success_probability(H, 0, [1], np.linspace(0.0, np.pi, 20))
        final = detector_final_probability(H, 0, [1], np.pi / 2.0)
        self.assertGreaterEqual(success, 0.0)
        self.assertLessEqual(success, 1.0 + 1.0e-12)
        self.assertGreaterEqual(final, 0.0)
        self.assertLessEqual(final, 1.0 + 1.0e-12)

    def test_run_single_sample_returns_bounded_metrics(self):
        config = transition_motor_ensemble_from_dict(
            self.build_config("outputs/tmp_motor_ensemble.csv", "outputs/tmp_motor_ensemble.json")
        )
        grid = config.motor_config.grid
        registry = config.motor_config.knob_registry
        basis = make_control_basis(grid)
        result = run_single_motor_ensemble_sample(
            grid,
            registry,
            basis,
            onsite_disorder=np.linspace(-0.05, 0.05, grid.n_sites, dtype=np.float64),
            noise_profiles=sample_phase_noise_profiles(2, grid.n_sites, 0.7, 2.0, 99),
            motor_config=config.motor_config,
            success_criteria=config.success_criteria,
        )
        self.assertIsInstance(result, MotorEnsembleSampleResult)
        self.assertGreaterEqual(result.baseline_detector_success, 0.0)
        self.assertLessEqual(result.baseline_detector_success, 1.0 + 1.0e-12)
        self.assertGreaterEqual(result.best_detector_success, 0.0)
        self.assertLessEqual(result.best_detector_success, 1.0 + 1.0e-12)

    def test_summary_and_writers_work(self):
        results = [
            MotorEnsembleSampleResult(0, 0.4, 0.5, 0.3, 0.35, 0.5, 0.6, 0.2, 0.1, 0.1, 0.09, 0.0, 0.01, 0.4, 0.45, 0.1, 0.1, 0.1, 0.01, 0.05, 2, 3, True),
            MotorEnsembleSampleResult(1, 0.4, 0.38, 0.3, 0.29, 0.5, 0.48, 0.2, 0.21, 0.1, 0.11, 0.0, 0.02, 0.4, 0.39, -0.02, -0.02, -0.01, -0.01, -0.01, 1, 2, False),
        ]
        summary = summarize_motor_ensemble(results)
        self.assertAlmostEqual(summary.success_rate, 0.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "ensemble.csv"
            json_path = Path(tmpdir) / "summary.json"
            self.assertTrue(write_motor_ensemble_csv(csv_path, results).exists())
            self.assertTrue(write_motor_ensemble_summary_json(json_path, summary).exists())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["n_samples"], 2)
        with self.assertRaises(ValueError):
            summarize_motor_ensemble([])

    def test_module_and_cli_smoke_work_with_tmp_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = str(Path(tmpdir) / "motor_ensemble.csv")
            summary_path = str(Path(tmpdir) / "motor_ensemble.json")
            config = self.build_config(csv_path, summary_path)
            results, summary, csv_output, summary_output = run_transition_motor_ensemble_study(config)
            self.assertEqual(len(results), 4)
            self.assertEqual(summary.n_samples, 4)
            self.assertTrue(csv_output.exists())
            self.assertTrue(summary_output.exists())

            stream = io.StringIO()
            with redirect_stdout(stream):
                exit_code = run_transition_motor_ensemble_mode(config)
            self.assertEqual(exit_code, 0)
            output = stream.getvalue()
            self.assertIn("Transition Motor Ensemble Study", output)
            self.assertIn("csv_path =", output)

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from hardware.objective_mode_comparison import (
    ObjectiveModeComparisonSample,
    bootstrap_delta_ci,
    plot_objective_mode_comparison,
    run_objective_mode_comparison,
    run_scale_sensitivity,
    summarize_objective_mode_comparison,
    write_comparison_csv,
    write_comparison_summary_json,
)


class ObjectiveModeComparisonTests(unittest.TestCase):
    def _sample(
        self,
        *,
        sample_id: int,
        raw_detector_success: float,
        normalized_detector_success: float,
        raw_noise_action: float,
        normalized_noise_action: float,
        raw_leakage: float,
        normalized_leakage: float,
        raw_control_cost: float,
        normalized_control_cost: float,
        raw_objective: float,
        normalized_objective: float,
    ) -> ObjectiveModeComparisonSample:
        detector_delta = normalized_detector_success - raw_detector_success
        noise_action_delta = raw_noise_action - normalized_noise_action
        leakage_delta = raw_leakage - normalized_leakage
        control_cost_delta = raw_control_cost - normalized_control_cost
        objective_delta = normalized_objective - raw_objective
        return ObjectiveModeComparisonSample(
            sample_id=sample_id,
            raw_detector_success=raw_detector_success,
            normalized_detector_success=normalized_detector_success,
            raw_noise_action=raw_noise_action,
            normalized_noise_action=normalized_noise_action,
            raw_leakage=raw_leakage,
            normalized_leakage=normalized_leakage,
            raw_control_cost=raw_control_cost,
            normalized_control_cost=normalized_control_cost,
            raw_objective=raw_objective,
            normalized_objective=normalized_objective,
            detector_delta=detector_delta,
            noise_action_delta=noise_action_delta,
            leakage_delta=leakage_delta,
            control_cost_delta=control_cost_delta,
            objective_delta=objective_delta,
            normalized_wins_detector=detector_delta > 0.0,
            normalized_wins_noise_action=noise_action_delta > 0.0,
            normalized_wins_leakage=leakage_delta > 0.0,
            normalized_wins_objective=objective_delta > 0.0,
            normalized_wins_control_cost=control_cost_delta > 0.0,
        )

    def _base_transition_motor_config(self, *, normalized: bool) -> dict[str, object]:
        block: dict[str, object] = {
            "transition_motor": {
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
                "noise": {
                    "profiles": [
                        [0.0, 0.1, 0.3, 0.3, 0.1, 0.0],
                        [0.0, 0.0, 0.1, 0.2, 0.1, 0.0],
                    ]
                },
                "optimizer": {
                    "time_min": 0.0,
                    "time_max": 8.0,
                    "n_time_samples": 20,
                    "n_transport_modes": 2,
                    "finite_diff_eps": 1.0e-4,
                    "optimizer_steps": 6,
                    "optimizer_step_size": 0.05,
                    "random_restarts": 1,
                    "seed": 7,
                },
                "outputs": {
                    "sensitivity_csv_path": "outputs/test_objective_mode_comparison_sensitivity.csv",
                },
            }
        }
        if normalized:
            block["transition_motor"]["objective_mode"] = {
                "name": "normalized_noise",
                "mode": "normalized",
                "weights": {
                    "transport": 1.0,
                    "noise_action": 1.0,
                    "leakage": 0.5,
                    "control_cost": 0.05,
                },
                "normalization": {
                    "transport_scale": 0.05,
                    "noise_action_scale": 0.002,
                    "leakage_scale": 0.001,
                    "control_cost_scale": 0.04,
                },
                "description": "normalized",
            }
        else:
            block["transition_motor"]["objective_weights"] = {
                "transport": 1.0,
                "noise_action": 0.5,
                "leakage": 0.25,
                "control_cost": 0.01,
            }
        return block

    def _write_config_bundle(self, root: Path) -> dict[str, object]:
        raw_path = root / "raw.yaml"
        normalized_path = root / "normalized.yaml"
        raw_path.write_text(yaml.safe_dump(self._base_transition_motor_config(normalized=False), sort_keys=False), encoding="utf-8")
        normalized_path.write_text(yaml.safe_dump(self._base_transition_motor_config(normalized=True), sort_keys=False), encoding="utf-8")
        return {
            "objective_mode_comparison": {
                "base_raw_config": str(raw_path),
                "base_normalized_config": str(normalized_path),
                "ensemble": {
                    "n_samples": 2,
                    "onsite_sigma": 0.05,
                    "fabrication_correlation_length_sites": 2.0,
                    "fabrication_seed": 11,
                    "n_noise_profiles_per_sample": 2,
                    "phase_noise_sigma": 0.6,
                    "phase_noise_correlation_length_sites": 2.0,
                    "phase_noise_seed": 21,
                },
                "bootstrap": {
                    "n_bootstrap": 200,
                    "ci": 0.95,
                    "seed": 123,
                },
                "calibration_sensitivity": {
                    "enabled": True,
                    "scale_multipliers": [0.5, 1.0],
                },
                "outputs": {
                    "csv_path": str(root / "comparison.csv"),
                    "summary_path": str(root / "comparison.json"),
                    "plot_path": str(root / "comparison.png"),
                },
            }
        }

    def test_summary_and_all_core_win_rate(self):
        samples = [
            self._sample(
                sample_id=0,
                raw_detector_success=0.50,
                normalized_detector_success=0.60,
                raw_noise_action=0.20,
                normalized_noise_action=0.10,
                raw_leakage=0.12,
                normalized_leakage=0.11,
                raw_control_cost=0.03,
                normalized_control_cost=0.04,
                raw_objective=0.5,
                normalized_objective=0.7,
            ),
            self._sample(
                sample_id=1,
                raw_detector_success=0.50,
                normalized_detector_success=0.45,
                raw_noise_action=0.20,
                normalized_noise_action=0.19,
                raw_leakage=0.12,
                normalized_leakage=0.13,
                raw_control_cost=0.05,
                normalized_control_cost=0.04,
                raw_objective=0.5,
                normalized_objective=0.45,
            ),
        ]
        summary = summarize_objective_mode_comparison(samples)
        self.assertAlmostEqual(summary.detector_win_rate, 0.5)
        self.assertAlmostEqual(summary.noise_action_win_rate, 1.0)
        self.assertAlmostEqual(summary.leakage_win_rate, 0.5)
        self.assertAlmostEqual(summary.objective_win_rate, 0.5)
        self.assertAlmostEqual(summary.control_cost_win_rate, 0.5)
        self.assertAlmostEqual(summary.all_core_win_rate, 0.5)
        with self.assertRaises(ValueError):
            summarize_objective_mode_comparison([])

    def test_bootstrap_delta_ci_is_deterministic(self):
        values = [0.1, 0.2, 0.15, 0.25]
        first = bootstrap_delta_ci(values, n_bootstrap=200, seed=123)
        second = bootstrap_delta_ci(values, n_bootstrap=200, seed=123)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], sum(values) / len(values))
        self.assertGreaterEqual(first[1], sum(values) / len(values))
        with self.assertRaises(ValueError):
            bootstrap_delta_ci([], n_bootstrap=10)

    def test_delta_interpretation_signs(self):
        sample = self._sample(
            sample_id=0,
            raw_detector_success=0.50,
            normalized_detector_success=0.55,
            raw_noise_action=0.20,
            normalized_noise_action=0.10,
            raw_leakage=0.12,
            normalized_leakage=0.10,
            raw_control_cost=0.05,
            normalized_control_cost=0.03,
            raw_objective=0.50,
            normalized_objective=0.60,
        )
        self.assertGreater(sample.detector_delta, 0.0)
        self.assertGreater(sample.noise_action_delta, 0.0)
        self.assertGreater(sample.leakage_delta, 0.0)
        self.assertGreater(sample.control_cost_delta, 0.0)

    def test_writers_and_plot(self):
        samples = [
            self._sample(
                sample_id=0,
                raw_detector_success=0.50,
                normalized_detector_success=0.60,
                raw_noise_action=0.20,
                normalized_noise_action=0.10,
                raw_leakage=0.12,
                normalized_leakage=0.11,
                raw_control_cost=0.03,
                normalized_control_cost=0.04,
                raw_objective=0.5,
                normalized_objective=0.7,
            )
        ]
        summary = summarize_objective_mode_comparison(samples)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = write_comparison_csv(root / "comparison.csv", samples)
            json_path = write_comparison_summary_json(
                root / "summary.json",
                summary,
                bootstrap_ci={"detector_delta": (0.01, 0.02)},
                scale_sensitivity=[{"scale_multiplier": 1.0, "mean_objective_delta": 0.1}],
            )
            plot_path = plot_objective_mode_comparison(samples, root / "comparison.png")
            self.assertTrue(csv_path.exists())
            self.assertTrue(plot_path.exists())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["n_samples"], 1)

    def test_tiny_comparison_run_and_scale_sensitivity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._write_config_bundle(Path(tmpdir))
            samples, summary = run_objective_mode_comparison(config)
            self.assertEqual(len(samples), 2)
            self.assertEqual(summary.n_samples, 2)
            self.assertTrue(all(sample.sample_id in {0, 1} for sample in samples))
            first = samples[0]
            self.assertAlmostEqual(
                first.detector_delta,
                first.normalized_detector_success - first.raw_detector_success,
            )
            self.assertAlmostEqual(
                first.noise_action_delta,
                first.raw_noise_action - first.normalized_noise_action,
            )
            self.assertAlmostEqual(
                first.control_cost_delta,
                first.raw_control_cost - first.normalized_control_cost,
            )
            sensitivity = run_scale_sensitivity(config)
            self.assertEqual(len(sensitivity), 2)
            self.assertEqual([item["scale_multiplier"] for item in sensitivity], [0.5, 1.0])


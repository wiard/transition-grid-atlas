from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from hardware.time_resolution_audit import (
    TimeResolutionAuditRun,
    adjusted_n_time_samples,
    plot_time_resolution_audit,
    run_time_resolution_audit,
    summarize_time_resolution_audit,
    with_time_resolution_multiplier,
    write_time_resolution_csv,
    write_time_resolution_summary_json,
)


class TimeResolutionAuditTests(unittest.TestCase):
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
                    "optimizer_steps": 2,
                    "optimizer_step_size": 0.05,
                    "random_restarts": 1,
                    "seed": 7,
                },
                "outputs": {
                    "sensitivity_csv_path": "outputs/test_time_resolution_sensitivity.csv",
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

    def _write_comparison_bundle(self, root: Path) -> dict[str, object]:
        raw_path = root / "raw.yaml"
        normalized_path = root / "normalized.yaml"
        comparison_path = root / "comparison.yaml"
        raw_path.write_text(yaml.safe_dump(self._base_transition_motor_config(normalized=False), sort_keys=False), encoding="utf-8")
        normalized_path.write_text(yaml.safe_dump(self._base_transition_motor_config(normalized=True), sort_keys=False), encoding="utf-8")
        comparison_config = {
            "objective_mode_comparison": {
                "base_raw_config": str(raw_path),
                "base_normalized_config": str(normalized_path),
                "ensemble": {
                    "n_samples": 1,
                    "onsite_sigma": 0.05,
                    "fabrication_correlation_length_sites": 2.0,
                    "fabrication_seed": 11,
                    "n_noise_profiles_per_sample": 2,
                    "phase_noise_sigma": 0.6,
                    "phase_noise_correlation_length_sites": 2.0,
                    "phase_noise_seed": 21,
                },
                "bootstrap": {
                    "n_bootstrap": 50,
                    "ci": 0.95,
                    "seed": 123,
                },
                "calibration_sensitivity": {
                    "enabled": False,
                    "scale_multipliers": [1.0],
                },
                "outputs": {
                    "csv_path": str(root / "comparison.csv"),
                    "summary_path": str(root / "comparison.json"),
                    "plot_path": str(root / "comparison.png"),
                },
            }
        }
        comparison_path.write_text(yaml.safe_dump(comparison_config, sort_keys=False), encoding="utf-8")
        return {
            "time_resolution_audit": {
                "base_comparison_config": str(comparison_path),
                "time_step_multipliers": [0.5, 1.0, 2.0],
                "sensitivity_thresholds": {
                    "common_balanced_relative_change": 0.10,
                    "detector_delta_absolute_change": 0.01,
                    "noise_action_delta_absolute_change": 0.001,
                    "leakage_delta_absolute_change": 0.0005,
                },
                "outputs": {
                    "csv_path": str(root / "time_resolution.csv"),
                    "summary_path": str(root / "time_resolution.json"),
                    "plot_path": str(root / "time_resolution.png"),
                },
            },
            "__config_path__": str(root / "time_resolution.yaml"),
        }

    def test_adjusted_n_time_samples_multiplier_one(self):
        self.assertEqual(
            adjusted_n_time_samples(time_min=0.0, time_max=20.0, n_time_samples=100, time_step_multiplier=1.0),
            100,
        )

    def test_adjusted_n_time_samples_finer_and_coarser(self):
        self.assertGreater(
            adjusted_n_time_samples(time_min=0.0, time_max=20.0, n_time_samples=100, time_step_multiplier=0.5),
            100,
        )
        self.assertLess(
            adjusted_n_time_samples(time_min=0.0, time_max=20.0, n_time_samples=100, time_step_multiplier=2.0),
            100,
        )

    def test_adjusted_n_time_samples_invalid_multiplier(self):
        with self.assertRaises(ValueError):
            adjusted_n_time_samples(time_min=0.0, time_max=20.0, n_time_samples=100, time_step_multiplier=0.0)

    def test_adjusted_n_time_samples_invalid_time_bounds(self):
        with self.assertRaises(ValueError):
            adjusted_n_time_samples(time_min=5.0, time_max=5.0, n_time_samples=100, time_step_multiplier=1.0)

    def test_adjusted_n_time_samples_invalid_sample_count(self):
        with self.assertRaises(ValueError):
            adjusted_n_time_samples(time_min=0.0, time_max=20.0, n_time_samples=1, time_step_multiplier=1.0)

    def test_with_time_resolution_multiplier_does_not_mutate_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = self._write_comparison_bundle(root)
            original = json.loads(json.dumps(config))
            adjusted = with_time_resolution_multiplier(config, 0.5)

            self.assertEqual(config, original)
            self.assertNotIn("__raw_config_data__", config)
            self.assertGreater(
                adjusted["__raw_config_data__"]["transition_motor"]["optimizer"]["n_time_samples"],
                20,
            )
            self.assertGreater(
                adjusted["__normalized_config_data__"]["transition_motor"]["optimizer"]["n_time_samples"],
                20,
            )

    def test_with_time_resolution_multiplier_preserves_seeds_and_grid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = self._write_comparison_bundle(root)
            adjusted = with_time_resolution_multiplier(config, 2.0)
            raw_payload = adjusted["__raw_config_data__"]["transition_motor"]
            normalized_payload = adjusted["__normalized_config_data__"]["transition_motor"]

            self.assertEqual(raw_payload["optimizer"]["seed"], 7)
            self.assertEqual(normalized_payload["optimizer"]["seed"], 7)
            self.assertEqual(raw_payload["grid"]["edges"], [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
            self.assertEqual(normalized_payload["grid"]["edges"], [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])

    def test_with_time_resolution_multiplier_invalid_multiplier(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = self._write_comparison_bundle(root)
            with self.assertRaises(ValueError):
                with_time_resolution_multiplier(config, -1.0)

    def test_summary_detects_stable_metrics_below_thresholds(self):
        runs = [
            TimeResolutionAuditRun(0.5, 40, 0.2, -0.030, 0.8, 0.0020, 0.7, 0.00040, 0.8, 1.00, 0.2, -0.020, 0.8, 0.90),
            TimeResolutionAuditRun(1.0, 20, 0.2, -0.031, 0.8, 0.0021, 0.7, 0.00042, 0.8, 1.02, 0.2, -0.021, 0.8, 0.91),
            TimeResolutionAuditRun(2.0, 10, 0.2, -0.029, 0.8, 0.0020, 0.7, 0.00041, 0.8, 1.01, 0.2, -0.020, 0.8, 0.92),
        ]
        summary = summarize_time_resolution_audit(
            runs,
            sensitivity_thresholds={
                "common_balanced_relative_change": 0.10,
                "detector_delta_absolute_change": 0.01,
                "noise_action_delta_absolute_change": 0.001,
                "leakage_delta_absolute_change": 0.0005,
            },
        )
        self.assertFalse(summary.time_resolution_sensitive)
        self.assertTrue(summary.sign_stable_common_balanced)
        self.assertIn("Do not promote", summary.registry_recommendation)

    def test_summary_detects_sensitive_metrics_and_recommendation(self):
        runs = [
            TimeResolutionAuditRun(0.5, 40, 0.2, 0.015, 0.8, 0.0001, 0.7, 0.00010, 0.8, -0.50, 0.2, -0.010, 0.8, 0.20),
            TimeResolutionAuditRun(1.0, 20, 0.2, -0.020, 0.8, 0.0020, 0.7, 0.00070, 0.8, 1.00, 0.2, -0.015, 0.8, 0.90),
            TimeResolutionAuditRun(2.0, 10, 0.2, -0.040, 0.8, 0.0035, 0.7, 0.00140, 0.8, 1.90, 0.2, -0.030, 0.8, 1.60),
        ]
        summary = summarize_time_resolution_audit(
            runs,
            sensitivity_thresholds={
                "common_balanced_relative_change": 0.10,
                "detector_delta_absolute_change": 0.01,
                "noise_action_delta_absolute_change": 0.001,
                "leakage_delta_absolute_change": 0.0005,
            },
        )
        self.assertTrue(summary.time_resolution_sensitive)
        self.assertFalse(summary.sign_stable_detector_delta)
        self.assertIn("Add time_step_resolution", summary.registry_recommendation)

    def test_summary_requires_baseline_multiplier(self):
        runs = [
            TimeResolutionAuditRun(0.5, 40, 0.2, -0.030, 0.8, 0.0020, 0.7, 0.00040, 0.8, 1.00, 0.2, -0.020, 0.8, 0.90),
            TimeResolutionAuditRun(2.0, 10, 0.2, -0.029, 0.8, 0.0020, 0.7, 0.00041, 0.8, 1.01, 0.2, -0.020, 0.8, 0.92),
        ]
        with self.assertRaises(ValueError):
            summarize_time_resolution_audit(
                runs,
                sensitivity_thresholds={
                    "common_balanced_relative_change": 0.10,
                    "detector_delta_absolute_change": 0.01,
                    "noise_action_delta_absolute_change": 0.001,
                    "leakage_delta_absolute_change": 0.0005,
                },
            )

    def test_csv_json_and_plot_writers(self):
        runs = [
            TimeResolutionAuditRun(1.0, 20, 0.2, -0.031, 0.8, 0.0021, 0.7, 0.00042, 0.8, 1.02, 0.2, -0.021, 0.8, 0.91),
            TimeResolutionAuditRun(2.0, 10, 0.2, -0.029, 0.8, 0.0020, 0.7, 0.00041, 0.8, 1.01, 0.2, -0.020, 0.8, 0.92),
        ]
        summary = summarize_time_resolution_audit(
            runs,
            sensitivity_thresholds={
                "common_balanced_relative_change": 0.10,
                "detector_delta_absolute_change": 0.01,
                "noise_action_delta_absolute_change": 0.001,
                "leakage_delta_absolute_change": 0.0005,
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = write_time_resolution_csv(root / "audit.csv", runs)
            summary_path = write_time_resolution_summary_json(root / "audit.json", summary)
            plot_path = plot_time_resolution_audit(runs, root / "audit.png")

            self.assertTrue(csv_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(plot_path.exists())

            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn("n_time_samples", rows[0])

            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertIn("time_resolution_sensitive", payload)
            self.assertIn("registry_recommendation", payload)

    def test_tiny_time_resolution_audit_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = self._write_comparison_bundle(root)
            runs, summary = run_time_resolution_audit(config)

            self.assertEqual([run.time_step_multiplier for run in runs], [0.5, 1.0, 2.0])
            self.assertEqual(summary.baseline_multiplier, 1.0)
            self.assertEqual(len(runs), 3)
            self.assertTrue(all(run.n_time_samples >= 2 for run in runs))


if __name__ == "__main__":
    unittest.main()

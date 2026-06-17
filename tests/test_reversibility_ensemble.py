from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from hardware.reversibility_ensemble import (
    ReversibilityEnsembleSampleResult,
    plot_reversibility_ensemble,
    run_reversibility_ensemble,
    run_single_reversibility_ensemble_sample,
    safe_corrcoef,
    summarize_reversibility_ensemble,
    write_reversibility_ensemble_csv,
    write_reversibility_ensemble_summary_json,
)


class ReversibilityEnsembleTests(unittest.TestCase):
    def _base_pareto_config(self) -> dict[str, object]:
        return {
            "transition_motor_pareto": {
                "grid": {
                    "n_sites": 4,
                    "edges": [[0, 1], [1, 2], [2, 3]],
                    "base_coupling": 1.0,
                    "input_index": 0,
                    "target_indices": [2, 3],
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
                    }
                ],
                "fabrication_disorder": {
                    "n_samples": 2,
                    "onsite_sigma": 0.03,
                    "correlation_length_sites": 1.5,
                    "seed": 1,
                },
                "phase_noise": {
                    "n_profiles_per_sample": 1,
                    "profile_sigma": 0.3,
                    "correlation_length_sites": 1.5,
                    "seed": 2,
                },
                "optimizer": {
                    "time_min": 0.0,
                    "time_max": 6.0,
                    "n_time_samples": 18,
                    "n_transport_modes": 2,
                    "finite_diff_eps": 1.0e-4,
                    "optimizer_steps": 2,
                    "optimizer_step_size": 0.04,
                    "random_restarts": 1,
                    "seed": 5,
                },
                "success_criteria": {
                    "min_objective_gain": -1.0e-9,
                    "min_detector_gain": -0.05,
                    "min_noise_action_reduction": -1.0e-9,
                },
                "weight_sets": [
                    {
                        "name": "detector_max",
                        "transport": 1.0,
                        "noise_action": 0.25,
                        "leakage": 0.1,
                        "control_cost": 0.01,
                    },
                    {
                        "name": "balanced",
                        "transport": 1.0,
                        "noise_action": 0.5,
                        "leakage": 0.25,
                        "control_cost": 0.01,
                    },
                ],
                "pareto": {
                    "maximize": [
                        "mean_detector_success_gain",
                        "mean_noise_action_reduction",
                        "mean_noise_leakage_reduction",
                        "success_rate",
                    ],
                    "minimize": [
                        "mean_best_control_cost",
                        "mean_saturated_knobs",
                    ],
                },
                "outputs": {
                    "csv_path": "outputs/test_pareto.csv",
                    "summary_path": "outputs/test_pareto.json",
                    "plot_path": "results/renders/test_pareto.png",
                },
            }
        }

    def _audit_config(self, base_path: Path) -> dict[str, object]:
        return {
            "transition_motor_pareto_audit": {
                "base_config": str(base_path),
                "stress_weight_sets": [
                    {
                        "name": "normalized_noise_strong",
                        "mode": "normalized",
                        "transport": 1.0,
                        "noise_action": 3.0,
                        "leakage": 1.0,
                        "control_cost": 0.05,
                        "normalization": {
                            "transport_scale": 0.05,
                            "noise_action_scale": 0.002,
                            "leakage_scale": 0.001,
                            "control_cost_scale": 0.04,
                        },
                    }
                ],
                "fabrication_disorder": {
                    "n_samples": 2,
                    "onsite_sigma": 0.03,
                    "correlation_length_sites": 1.5,
                    "seed": 1,
                },
                "phase_noise": {
                    "n_profiles_per_sample": 1,
                    "profile_sigma": 0.3,
                    "correlation_length_sites": 1.5,
                    "seed": 2,
                },
                "optimizer": {
                    "time_min": 0.0,
                    "time_max": 6.0,
                    "n_time_samples": 18,
                    "n_transport_modes": 2,
                    "finite_diff_eps": 1.0e-4,
                    "optimizer_steps": 2,
                    "optimizer_step_size": 0.04,
                    "random_restarts": 1,
                    "seed": 5,
                },
                "reversibility": {
                    "enabled": True,
                    "dephasing_strength": 0.05,
                    "time_step_multiplier": 1.0,
                },
                "bootstrap": {"n_bootstrap": 10, "ci": 0.95, "seed": 7},
                "regime_distance": {
                    "metrics": [
                        "mean_detector_success_gain",
                        "mean_noise_action_reduction",
                        "mean_noise_leakage_reduction",
                        "success_rate",
                        "mean_best_control_cost",
                        "mean_saturated_knobs",
                    ]
                },
                "outputs": {
                    "csv_path": "outputs/test_audit.csv",
                    "summary_path": "outputs/test_audit.json",
                    "plot_path": "results/renders/test_audit.png",
                },
            }
        }

    def _ensemble_config(self, audit_path: Path, root: Path) -> dict[str, object]:
        return {
            "reversibility_ensemble": {
                "operating_modes": [
                    {"name": "balanced", "config": str(audit_path), "preset_name": "balanced"},
                    {
                        "name": "normalized_noise_strong",
                        "config": str(audit_path),
                        "preset_name": "normalized_noise_strong",
                    },
                ],
                "grid": {
                    "n_sites": 4,
                    "edges": [[0, 1], [1, 2], [2, 3]],
                    "base_coupling": 1.0,
                    "input_index": 0,
                    "target_indices": [2, 3],
                },
                "fabrication_disorder": {
                    "n_samples": 2,
                    "onsite_sigma": 0.03,
                    "correlation_length_sites": 1.5,
                    "seed": 1,
                },
                "phase_noise": {
                    "n_profiles_per_sample": 1,
                    "profile_sigma": 0.3,
                    "correlation_length_sites": 1.5,
                    "seed": 2,
                },
                "dephasing_strengths": [0.0, 0.1],
                "time": {
                    "time_min": 0.0,
                    "time_max": 6.0,
                    "n_time_samples": 18,
                    "time_step_multiplier": 1.0,
                    "forward_time_selection": "peak_target",
                },
                "selection": {"detector_tolerance_fraction": 0.05},
                "outputs": {
                    "csv_path": str(root / "ensemble.csv"),
                    "summary_path": str(root / "ensemble.json"),
                    "plot_path": str(root / "ensemble.png"),
                },
            },
            "__config_path__": str(root / "ensemble.yaml"),
        }

    def test_sample_result_dataclass(self):
        result = ReversibilityEnsembleSampleResult(
            sample_id=0,
            mode_name="balanced",
            dephasing_strength=0.05,
            detector_success=0.8,
            coherent_reversibility_score=1.0,
            coherent_loss_delta=0.0,
            open_reversibility_score=0.98,
            open_loss_delta=0.02,
            return_probability=0.98,
            phase_noise_profile_index=None,
            time_step_multiplier=1.0,
        )
        self.assertEqual(result.mode_name, "balanced")

    def test_safe_corrcoef_bounded(self):
        corr = safe_corrcoef(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
        self.assertGreaterEqual(corr, -1.0)
        self.assertLessEqual(corr, 1.0)

    def test_safe_corrcoef_constant_vector_zero(self):
        self.assertEqual(safe_corrcoef(np.array([1.0, 1.0]), np.array([2.0, 3.0])), 0.0)

    def test_summarize_selects_best_reversibility_and_open_loss(self):
        rows = [
            ReversibilityEnsembleSampleResult(0, "a", 0.05, 0.80, 1.0, 0.0, 0.99, 0.01, 0.99, None, 1.0),
            ReversibilityEnsembleSampleResult(1, "a", 0.05, 0.79, 1.0, 0.0, 0.98, 0.02, 0.98, None, 1.0),
            ReversibilityEnsembleSampleResult(0, "b", 0.05, 0.81, 1.0, 0.0, 0.95, 0.05, 0.95, None, 1.0),
            ReversibilityEnsembleSampleResult(1, "b", 0.05, 0.80, 1.0, 0.0, 0.94, 0.06, 0.94, None, 1.0),
        ]
        summaries, summary = summarize_reversibility_ensemble(rows, detector_tolerance_fraction=0.05)
        self.assertEqual(summary.best_reversibility_by_dephasing[0.05], "a")
        self.assertEqual(summary.lowest_open_loss_by_dephasing[0.05], "a")
        self.assertEqual(summary.preferred_mode_by_dephasing[0.05], "a")
        self.assertEqual(len(summaries), 2)

    def test_preferred_mode_uses_detector_tolerance_then_reversibility(self):
        rows = [
            ReversibilityEnsembleSampleResult(0, "a", 0.05, 0.80, 1.0, 0.0, 0.97, 0.03, 0.97, None, 1.0),
            ReversibilityEnsembleSampleResult(1, "a", 0.05, 0.80, 1.0, 0.0, 0.97, 0.03, 0.97, None, 1.0),
            ReversibilityEnsembleSampleResult(0, "b", 0.05, 0.83, 1.0, 0.0, 0.95, 0.05, 0.95, None, 1.0),
            ReversibilityEnsembleSampleResult(1, "b", 0.05, 0.83, 1.0, 0.0, 0.95, 0.05, 0.95, None, 1.0),
        ]
        _, summary = summarize_reversibility_ensemble(rows, detector_tolerance_fraction=0.05)
        self.assertEqual(summary.preferred_mode_by_dephasing[0.05], "a")

    def test_writers_and_plot(self):
        rows = [
            ReversibilityEnsembleSampleResult(0, "a", 0.0, 0.8, 1.0, 0.0, 1.0, 0.0, 1.0, None, 1.0),
            ReversibilityEnsembleSampleResult(0, "a", 0.1, 0.8, 1.0, 0.0, 0.95, 0.05, 0.95, None, 1.0),
        ]
        summaries, summary = summarize_reversibility_ensemble(rows, detector_tolerance_fraction=0.05)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = write_reversibility_ensemble_csv(tmp / "ensemble.csv", rows, summaries)
            json_path = write_reversibility_ensemble_summary_json(tmp / "ensemble.json", summary, summaries)
            plot_path = plot_reversibility_ensemble(summaries, summary, tmp / "ensemble.png")
            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())
            self.assertTrue(plot_path.exists())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertIn("mode_summaries", payload)
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertTrue(any(row["row_type"] == "sample" for row in csv_rows))
            self.assertTrue(any(row["row_type"] == "summary" for row in csv_rows))

    def test_run_single_sample_dephasing_effect(self):
        H = np.array(
            [[0.0, 1.0], [1.0, 0.0]],
            dtype=np.complex128,
        )
        clean = run_single_reversibility_ensemble_sample(
            sample_id=0,
            mode_name="test",
            H=H,
            input_index=0,
            target_indices=[1],
            dephasing_strength=0.0,
            time_min=0.0,
            time_max=np.pi / 4.0,
            n_time_samples=9,
            time_step_multiplier=1.0,
            forward_time_selection="final",
        )
        noisy = run_single_reversibility_ensemble_sample(
            sample_id=0,
            mode_name="test",
            H=H,
            input_index=0,
            target_indices=[1],
            dephasing_strength=0.2,
            time_min=0.0,
            time_max=np.pi / 4.0,
            n_time_samples=9,
            time_step_multiplier=1.0,
            forward_time_selection="final",
        )
        self.assertGreater(clean.open_reversibility_score, 0.999999)
        self.assertGreater(noisy.open_loss_delta, clean.open_loss_delta)

    def test_tiny_ensemble_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base_path = tmp / "base.yaml"
            audit_path = tmp / "audit.yaml"
            ensemble_path = tmp / "ensemble.yaml"
            base_path.write_text(yaml.safe_dump(self._base_pareto_config(), sort_keys=False), encoding="utf-8")
            audit_path.write_text(yaml.safe_dump(self._audit_config(base_path), sort_keys=False), encoding="utf-8")
            ensemble_config = self._ensemble_config(audit_path, tmp)
            ensemble_path.write_text(yaml.safe_dump(ensemble_config, sort_keys=False), encoding="utf-8")
            loaded = yaml.safe_load(ensemble_path.read_text(encoding="utf-8"))
            loaded["__config_path__"] = str(ensemble_path)
            results, summary = run_reversibility_ensemble(loaded)
            self.assertEqual(summary.n_samples, 2)
            self.assertEqual(summary.operating_modes, ["balanced", "normalized_noise_strong"])
            self.assertEqual(summary.dephasing_strengths, [0.0, 0.1])
            self.assertEqual(len(results), 8)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hardware.motor_pareto import (
    ObjectiveWeightSet,
    ParetoRunResult,
    ParetoSweepSummary,
    balanced_score,
    choose_best_balanced,
    dominates,
    load_weight_sets,
    mark_pareto_front,
    plot_pareto_results,
    run_pareto_weight_sweep,
    validate_weight_set,
    write_pareto_csv,
    write_pareto_summary_json,
)


class MotorParetoTests(unittest.TestCase):
    def build_result(
        self,
        name: str,
        *,
        detector: float,
        noise: float,
        leakage: float,
        success: float,
        cost: float,
        saturated: float,
        pareto: bool = False,
    ) -> ParetoRunResult:
        weights = ObjectiveWeightSet(name, 1.0, 0.5, 0.25, 0.01)
        return ParetoRunResult(
            name=name,
            weights=weights,
            n_samples=4,
            success_rate=success,
            detector_win_rate=success,
            noise_action_win_rate=success,
            leakage_win_rate=success,
            objective_win_rate=success,
            mean_detector_success_gain=detector,
            median_detector_success_gain=detector,
            mean_noise_action_reduction=noise,
            median_noise_action_reduction=noise,
            mean_noise_leakage_reduction=leakage,
            median_noise_leakage_reduction=leakage,
            mean_objective_gain=detector + noise + leakage,
            median_objective_gain=detector + noise + leakage,
            mean_best_control_cost=cost,
            mean_saturated_knobs=saturated,
            mean_near_bound_knobs=saturated,
            is_pareto_optimal=pareto,
        )

    def build_config(self, csv_path: str, summary_path: str, plot_path: str) -> dict[str, object]:
        return {
            "transition_motor_pareto": {
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
                    "n_samples": 3,
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
                "optimizer": {
                    "time_min": 0.0,
                    "time_max": 12.0,
                    "n_time_samples": 40,
                    "n_transport_modes": 2,
                    "finite_diff_eps": 1.0e-4,
                    "optimizer_steps": 8,
                    "optimizer_step_size": 0.05,
                    "random_restarts": 1,
                    "seed": 7,
                },
                "success_criteria": {
                    "min_objective_gain": -1.0e-9,
                    "min_detector_gain": -0.05,
                    "min_noise_action_reduction": -1.0e-9,
                },
                "weight_sets": [
                    {
                        "name": "detector_max",
                        "transport": 1.5,
                        "noise_action": 0.25,
                        "leakage": 0.1,
                        "control_cost": 0.01,
                    },
                    {
                        "name": "normalized_noise",
                        "mode": "normalized",
                        "transport": 1.0,
                        "noise_action": 2.0,
                        "leakage": 1.0,
                        "control_cost": 0.05,
                        "normalization": {
                            "transport_scale": 0.05,
                            "noise_action_scale": 0.002,
                            "leakage_scale": 0.001,
                            "control_cost_scale": 0.04,
                        },
                    },
                ],
                "pareto": {
                    "maximize": [
                        "mean_detector_success_gain",
                        "mean_noise_action_reduction",
                        "mean_noise_leakage_reduction",
                        "success_rate",
                    ],
                    "minimize": ["mean_best_control_cost", "mean_saturated_knobs"],
                },
                "outputs": {
                    "csv_path": csv_path,
                    "summary_path": summary_path,
                    "plot_path": plot_path,
                },
            }
        }

    def test_validate_and_load_weight_sets(self):
        validate_weight_set(ObjectiveWeightSet("ok", 1.0, 0.5, 0.1, 0.01))
        with self.assertRaises(ValueError):
            validate_weight_set(ObjectiveWeightSet("", 1.0, 0.5, 0.1, 0.01))
        with self.assertRaises(ValueError):
            validate_weight_set(ObjectiveWeightSet("bad", -1.0, 0.5, 0.1, 0.01))
        with self.assertRaises(ValueError):
            validate_weight_set(ObjectiveWeightSet("bad", 0.0, 0.0, 0.0, 0.01))
        loaded = load_weight_sets(
            {
                "weight_sets": [
                    {"name": "one", "transport": 1.0, "noise_action": 0.5, "leakage": 0.1, "control_cost": 0.01},
                    {
                        "name": "two",
                        "mode": "normalized",
                        "transport": 1.0,
                        "noise_action": 1.0,
                        "leakage": 0.5,
                        "control_cost": 0.05,
                        "normalization": {
                            "transport_scale": 0.05,
                            "noise_action_scale": 0.002,
                            "leakage_scale": 0.001,
                            "control_cost_scale": 0.04,
                        },
                    },
                ]
            }
        )
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[1].mode, "normalized")
        self.assertIsNotNone(loaded[1].normalization)

    def test_dominance_and_pareto_front(self):
        a = self.build_result("a", detector=0.2, noise=0.2, leakage=0.1, success=0.9, cost=0.1, saturated=1.0)
        b = self.build_result("b", detector=0.1, noise=0.1, leakage=0.05, success=0.8, cost=0.2, saturated=2.0)
        self.assertTrue(
            dominates(
                a,
                b,
                maximize=["mean_detector_success_gain", "mean_noise_action_reduction", "mean_noise_leakage_reduction", "success_rate"],
                minimize=["mean_best_control_cost", "mean_saturated_knobs"],
            )
        )
        self.assertFalse(dominates(a, a, maximize=["mean_detector_success_gain"], minimize=["mean_best_control_cost"]))
        marked = mark_pareto_front(
            [a, b],
            maximize=["mean_detector_success_gain", "mean_noise_action_reduction", "mean_noise_leakage_reduction", "success_rate"],
            minimize=["mean_best_control_cost", "mean_saturated_knobs"],
        )
        by_name = {item.name: item for item in marked}
        self.assertTrue(by_name["a"].is_pareto_optimal)
        self.assertFalse(by_name["b"].is_pareto_optimal)

    def test_balanced_score_and_writers(self):
        a = self.build_result("a", detector=0.2, noise=0.1, leakage=0.1, success=0.9, cost=0.2, saturated=2.0)
        b = self.build_result("b", detector=0.15, noise=0.15, leakage=0.12, success=0.9, cost=0.05, saturated=1.0)
        self.assertEqual(choose_best_balanced([a, b]), "b")
        normalized = {
            "mean_detector_success_gain": {"a": 1.0, "b": 0.0},
            "mean_noise_action_reduction": {"a": 0.0, "b": 1.0},
            "mean_noise_leakage_reduction": {"a": 0.0, "b": 1.0},
            "success_rate": {"a": 1.0, "b": 1.0},
            "mean_best_control_cost": {"a": 0.0, "b": 1.0},
            "mean_saturated_knobs": {"a": 0.0, "b": 1.0},
        }
        self.assertTrue(np.isfinite(balanced_score(a, normalized)))
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "pareto.csv"
            summary_path = Path(tmpdir) / "pareto.json"
            plot_path = Path(tmpdir) / "pareto.png"
            self.assertTrue(write_pareto_csv(csv_path, [a, b]).exists())
            summary = ParetoSweepSummary(
                n_weight_sets=2,
                pareto_optimal_names=["a", "b"],
                best_detector_name="a",
                best_noise_action_name="b",
                best_leakage_name="b",
                best_balanced_name="b",
            )
            self.assertTrue(write_pareto_summary_json(summary_path, [a, b], summary).exists())
            self.assertTrue(plot_pareto_results([a, b], plot_path).exists())
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertIn("summary", payload)

    def test_tiny_pareto_sweep_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self.build_config(
                str(Path(tmpdir) / "pareto.csv"),
                str(Path(tmpdir) / "pareto.json"),
                str(Path(tmpdir) / "pareto.png"),
            )
            results, summary = run_pareto_weight_sweep(config)
            self.assertEqual(len(results), 2)
            self.assertEqual(summary.n_weight_sets, 2)

from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from hardware.control_knobs import knob_registry_from_dicts
from hardware.motor_pareto import ObjectiveWeightSet, ParetoRunResult
from hardware.motor_pareto_audit import (
    KnobProfile,
    ParetoAuditResult,
    ParetoAuditSummary,
    ObjectiveComponentScale,
    compute_component_scale,
    compute_knob_profiles,
    compute_objective_components_for_result,
    compute_preset_confidence_intervals,
    is_compressed_frontier,
    load_audit_config,
    merge_base_and_stress_weight_sets,
    minmax_normalize_values,
    normalized_balanced_scores,
    pairwise_regime_distances,
    plot_pareto_audit,
    regime_metric_vectors,
    run_pareto_audit,
    summarize_regime_distances,
    write_pareto_audit_csv,
    write_pareto_audit_summary_json,
)


class MotorParetoAuditTests(unittest.TestCase):
    def test_component_scale_picks_dominant_component(self) -> None:
        result = self.build_result(
            "detector_max",
            transport=3.0,
            detector=0.05,
            noise=0.001,
            leakage=0.0005,
            cost=0.02,
        )
        scale = compute_component_scale(result)
        self.assertEqual(scale.dominant_component, "detector")

    def test_component_ratios_are_finite_with_zero_components(self) -> None:
        result = self.build_result(
            "detector_only",
            transport=1.0,
            detector=0.04,
            noise=0.0,
            leakage=0.0,
            cost=0.01,
        )
        scale = compute_component_scale(result)
        self.assertTrue(np.isfinite(scale.detector_to_noise_ratio))
        self.assertTrue(np.isfinite(scale.detector_to_leakage_ratio))

    def test_minmax_normalize_higher_is_better(self) -> None:
        normalized = minmax_normalize_values({"a": 1.0, "b": 3.0}, higher_is_better=True)
        self.assertLess(normalized["a"], normalized["b"])

    def test_minmax_normalize_lower_is_better(self) -> None:
        normalized = minmax_normalize_values({"a": 1.0, "b": 3.0}, higher_is_better=False)
        self.assertGreater(normalized["a"], normalized["b"])

    def test_merge_base_and_stress_weight_sets(self) -> None:
        base = self.build_base_pareto_config()
        audit = self.build_audit_config_dict()
        merged = merge_base_and_stress_weight_sets(base, audit)
        self.assertEqual(
            [item["name"] for item in merged["transition_motor_pareto"]["weight_sets"]],
            ["balanced", "noise_suppression", "ultra_detector", "ultra_noise"],
        )
        self.assertEqual(
            [item["name"] for item in base["transition_motor_pareto"]["weight_sets"]],
            ["balanced", "noise_suppression"],
        )

    def test_duplicate_preset_names_fail(self) -> None:
        base = self.build_base_pareto_config()
        audit = self.build_audit_config_dict()
        audit["transition_motor_pareto_audit"]["stress_weight_sets"][0]["name"] = "balanced"
        with self.assertRaises(ValueError):
            merge_base_and_stress_weight_sets(base, audit)

    def test_bootstrap_ci_is_deterministic(self) -> None:
        rows = {
            "balanced": [
                {"detector_success_gain": 0.01, "noise_action_reduction": 0.001, "noise_leakage_reduction": 0.0, "objective_gain": 0.02},
                {"detector_success_gain": 0.03, "noise_action_reduction": 0.002, "noise_leakage_reduction": 0.001, "objective_gain": 0.04},
            ]
        }
        first = compute_preset_confidence_intervals(rows, metrics=["detector_success_gain"], n_bootstrap=200, ci=0.95, seed=7)
        second = compute_preset_confidence_intervals(rows, metrics=["detector_success_gain"], n_bootstrap=200, ci=0.95, seed=7)
        self.assertEqual(first, second)

    def test_knob_profile_summary_computes_mean_std_and_bounds(self) -> None:
        registry = self.build_registry()
        rows = {
            "balanced": [
                {"best_theta": {"path_coupling_boost": 0.15, "onsite_phase_gradient": 0.05}},
                {"best_theta": {"path_coupling_boost": 0.15, "onsite_phase_gradient": -0.05}},
            ]
        }
        profiles = compute_knob_profiles(rows, ["path_coupling_boost", "onsite_phase_gradient"], registry)
        profile = profiles[0]
        self.assertAlmostEqual(profile.mean_theta["path_coupling_boost"], 0.15)
        self.assertAlmostEqual(profile.std_theta["onsite_phase_gradient"], 0.05)
        self.assertAlmostEqual(profile.fraction_at_upper_bound["path_coupling_boost"], 1.0)

    def test_regime_distances_find_closest_and_furthest(self) -> None:
        results = [
            self.build_result("a", detector=0.05, noise=0.001, leakage=0.001, success=0.8, cost=0.02, saturated=1.0),
            self.build_result("b", detector=0.051, noise=0.0011, leakage=0.0011, success=0.8, cost=0.021, saturated=1.1),
            self.build_result("c", detector=0.02, noise=0.01, leakage=0.008, success=0.9, cost=0.01, saturated=0.0),
        ]
        distances = pairwise_regime_distances(regime_metric_vectors(results))
        summary = summarize_regime_distances(distances)
        self.assertEqual(summary["closest_presets"], ("a", "b"))
        self.assertEqual(summary["most_separated_presets"], ("b", "c"))

    def test_compressed_frontier_threshold(self) -> None:
        self.assertTrue(is_compressed_frontier(0.10, threshold=0.15))
        self.assertFalse(is_compressed_frontier(0.20, threshold=0.15))

    def test_csv_json_and_plot_writers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            results = [self.build_result("a"), self.build_result("b", detector=0.03, noise=0.004, leakage=0.003)]
            audit_result = self.build_audit_result(results)
            csv_path = write_pareto_audit_csv(tmp / "audit.csv", audit_result)
            json_path = write_pareto_audit_summary_json(tmp / "audit.json", audit_result)
            plot_path = plot_pareto_audit(audit_result, tmp / "audit.png")
            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())
            self.assertTrue(plot_path.exists())

    def test_tiny_audit_run_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base_path = tmp / "base.yaml"
            audit_path = tmp / "audit.yaml"
            base_path.write_text(yaml.safe_dump(self.build_base_pareto_config()), encoding="utf-8")
            audit_path.write_text(yaml.safe_dump(self.build_audit_config_dict(base_path)), encoding="utf-8")
            loaded = load_audit_config(audit_path)
            self.assertIn("transition_motor_pareto_audit", loaded)
            result = run_pareto_audit(audit_path)
            self.assertEqual(result.summary.n_presets, 4)
            self.assertEqual(result.summary.base_preset_names, ["balanced", "noise_suppression"])
            self.assertEqual(result.summary.stress_preset_names, ["ultra_detector", "ultra_noise"])

    def build_result(
        self,
        name: str,
        *,
        transport: float = 1.0,
        noise_action_weight: float = 0.5,
        leakage_weight: float = 0.25,
        control_cost_weight: float = 0.01,
        detector: float = 0.04,
        noise: float = 0.002,
        leakage: float = 0.001,
        success: float = 0.8,
        cost: float = 0.02,
        saturated: float = 1.0,
    ) -> ParetoRunResult:
        return ParetoRunResult(
            name=name,
            weights=ObjectiveWeightSet(name, transport, noise_action_weight, leakage_weight, control_cost_weight),
            n_samples=3,
            success_rate=success,
            detector_win_rate=1.0,
            noise_action_win_rate=0.8,
            leakage_win_rate=0.6,
            objective_win_rate=1.0,
            mean_detector_success_gain=detector,
            median_detector_success_gain=detector,
            mean_noise_action_reduction=noise,
            median_noise_action_reduction=noise,
            mean_noise_leakage_reduction=leakage,
            median_noise_leakage_reduction=leakage,
            mean_objective_gain=detector + noise + leakage - cost * control_cost_weight,
            median_objective_gain=detector + noise + leakage - cost * control_cost_weight,
            mean_best_control_cost=cost,
            mean_saturated_knobs=saturated,
            mean_near_bound_knobs=saturated,
            is_pareto_optimal=True,
        )

    def build_registry(self):
        return knob_registry_from_dicts(
            [
                {
                    "name": "path_coupling_boost",
                    "family": "coherent_coupling",
                    "symbol": "a_path",
                    "basis_name": "target_corridor",
                    "min_value": -0.15,
                    "max_value": 0.15,
                    "default": 0.0,
                    "units": "relative_edge_scale",
                    "description": "Path knob",
                    "hardware_meaning": "Edge boost",
                },
                {
                    "name": "onsite_phase_gradient",
                    "family": "onsite_phase",
                    "symbol": "b_grad",
                    "basis_name": "linear_gradient",
                    "min_value": -0.10,
                    "max_value": 0.10,
                    "default": 0.0,
                    "units": "normalized_beta_shift",
                    "description": "Gradient knob",
                    "hardware_meaning": "Phase gradient",
                },
            ]
        )

    def build_audit_result(self, results: list[ParetoRunResult]) -> ParetoAuditResult:
        component_scales = [compute_component_scale(result) for result in results]
        normalized_scores = normalized_balanced_scores(results)
        knob_profiles = [
            KnobProfile(
                preset_name=result.name,
                mean_theta={"path_coupling_boost": 0.1, "onsite_phase_gradient": 0.05},
                std_theta={"path_coupling_boost": 0.02, "onsite_phase_gradient": 0.01},
                fraction_at_upper_bound={"path_coupling_boost": 0.5, "onsite_phase_gradient": 0.0},
                fraction_at_lower_bound={"path_coupling_boost": 0.0, "onsite_phase_gradient": 0.0},
            )
            for result in results
        ]
        distances = pairwise_regime_distances(regime_metric_vectors(results))
        distance_summary = summarize_regime_distances(distances)
        summary = ParetoAuditSummary(
            n_presets=len(results),
            base_preset_names=[results[0].name],
            stress_preset_names=[result.name for result in results[1:]],
            dominant_component_overall="detector",
            component_scaling_issue=True,
            mean_pairwise_regime_distance=float(distance_summary["mean_pairwise_regime_distance"]),
            closest_presets=tuple(distance_summary["closest_presets"]),
            most_separated_presets=tuple(distance_summary["most_separated_presets"]),
            compressed_frontier=False,
        )
        confidence_intervals = [
            {
                "preset_name": result.name,
                "metric": "detector_success_gain",
                "mean": result.mean_detector_success_gain,
                "median": result.median_detector_success_gain,
                "ci_low": result.mean_detector_success_gain - 0.01,
                "ci_high": result.mean_detector_success_gain + 0.01,
            }
            for result in results
        ]
        from hardware.motor_pareto_audit import PresetCI

        return ParetoAuditResult(
            pareto_results=results,
            component_scales=component_scales,
            normalized_scores=normalized_scores,
            confidence_intervals=[PresetCI(**item) for item in confidence_intervals],
            knob_profiles=knob_profiles,
            regime_distances=distances,
            summary=summary,
            best_normalized_score_name=max(normalized_scores.items(), key=lambda item: item[1])[0],
            n_samples_per_preset=3,
            bootstrap_n=200,
            bootstrap_ci=0.95,
        )

    def build_base_pareto_config(self) -> dict[str, object]:
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
                        "min_value": -0.15,
                        "max_value": 0.15,
                        "default": 0.0,
                        "units": "relative_edge_scale",
                        "description": "Path knob",
                        "hardware_meaning": "Edge boost",
                    },
                    {
                        "name": "onsite_phase_gradient",
                        "family": "onsite_phase",
                        "symbol": "b_grad",
                        "basis_name": "linear_gradient",
                        "min_value": -0.10,
                        "max_value": 0.10,
                        "default": 0.0,
                        "units": "normalized_beta_shift",
                        "description": "Gradient knob",
                        "hardware_meaning": "Phase gradient",
                    },
                ],
                "fabrication_disorder": {
                    "n_samples": 2,
                    "onsite_sigma": 0.03,
                    "correlation_length_sites": 1.5,
                    "seed": 1,
                },
                "phase_noise": {
                    "n_profiles_per_sample": 2,
                    "profile_sigma": 0.6,
                    "correlation_length_sites": 1.5,
                    "seed": 2,
                },
                "optimizer": {
                    "time_min": 0.0,
                    "time_max": 10.0,
                    "n_time_samples": 40,
                    "n_transport_modes": 2,
                    "finite_diff_eps": 1.0e-4,
                    "optimizer_steps": 6,
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
                    {"name": "balanced", "transport": 1.0, "noise_action": 0.5, "leakage": 0.25, "control_cost": 0.01},
                    {"name": "noise_suppression", "transport": 0.5, "noise_action": 1.5, "leakage": 0.5, "control_cost": 0.01},
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
                    "csv_path": "outputs/base.csv",
                    "summary_path": "outputs/base.json",
                    "plot_path": "results/renders/base.png",
                },
            }
        }

    def build_audit_config_dict(self, base_path: Path | None = None) -> dict[str, object]:
        return {
            "transition_motor_pareto_audit": {
                "base_config": str(base_path or "configs/hardware/transition_motor_pareto_demo.yaml"),
                "stress_weight_sets": [
                    {"name": "ultra_detector", "transport": 3.0, "noise_action": 0.0, "leakage": 0.0, "control_cost": 0.005},
                    {"name": "ultra_noise", "transport": 0.25, "noise_action": 4.0, "leakage": 1.0, "control_cost": 0.01},
                ],
                "bootstrap": {
                    "n_bootstrap": 200,
                    "ci": 0.95,
                    "seed": 11,
                },
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
                    "csv_path": "outputs/transition_motor_pareto_audit.csv",
                    "summary_path": "outputs/transition_motor_pareto_audit_summary.json",
                    "plot_path": "results/renders/transition_motor_pareto_audit.png",
                },
            }
        }


if __name__ == "__main__":
    unittest.main()

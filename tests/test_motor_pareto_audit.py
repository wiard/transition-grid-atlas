from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from hardware.motor_ensemble import DetailedMotorEnsembleSample, MotorEnsembleSampleResult
from hardware.motor_pareto import ObjectiveWeightSet
from hardware.motor_pareto_audit import (
    KnobProfileSummary,
    ParetoAuditRunResult,
    ParetoStressAuditSummary,
    compute_objective_scale_summary,
    normalized_objective_gain,
    pairwise_theta_distance_summary,
    run_transition_motor_pareto_audit,
    summarize_knob_profiles,
    transition_motor_pareto_audit_from_dict,
    write_pareto_audit_csv,
    write_pareto_audit_summary_json,
    plot_pareto_audit,
)
from hardware.objectives import MotorMetrics


class MotorParetoAuditTests(unittest.TestCase):
    def test_scale_summary_and_normalized_gain_are_finite(self) -> None:
        details = {
            "a": [self.build_detail(transport_gain=0.04, noise_gain=0.002, leakage_gain=0.001, best_cost=0.03)],
            "b": [self.build_detail(transport_gain=0.05, noise_gain=0.001, leakage_gain=0.0005, best_cost=0.02)],
        }
        scales = compute_objective_scale_summary(details)
        self.assertGreater(scales.transport_scale, 0.0)
        self.assertGreater(scales.raw_scale_ratio, 0.0)

        value = normalized_objective_gain(
            details["a"][0],
            ObjectiveWeightSet("balanced", 1.0, 0.5, 0.25, 0.01),
            scales,
        )
        self.assertTrue(np.isfinite(value))

    def test_summarize_knob_profiles_tracks_saturation(self) -> None:
        parsed = transition_motor_pareto_audit_from_dict(self.build_config())
        registry = parsed.motor_config.knob_registry
        details = [
            self.build_detail(best_theta={"path_coupling_boost": 0.15, "onsite_phase_gradient": 0.1}),
            self.build_detail(best_theta={"path_coupling_boost": 0.15, "onsite_phase_gradient": 0.0}),
        ]
        profiles = summarize_knob_profiles("demo", details, registry)
        by_name = {profile.knob: profile for profile in profiles}
        self.assertAlmostEqual(by_name["path_coupling_boost"].saturation_rate, 1.0)
        self.assertGreaterEqual(by_name["onsite_phase_gradient"].near_bound_rate, 0.5)

    def test_pairwise_distance_and_writers(self) -> None:
        result_a = self.build_audit_result("a", theta_shift=0.02)
        result_b = self.build_audit_result("b", theta_shift=0.08)
        mean_distance, min_distance = pairwise_theta_distance_summary([result_a, result_b])
        self.assertGreater(mean_distance, 0.0)
        self.assertGreater(min_distance, 0.0)

        summary = ParetoStressAuditSummary(
            n_weight_sets=2,
            n_samples_per_weight_set=2,
            pareto_optimal_names=["a"],
            best_detector_name="a",
            best_noise_action_name="b",
            best_leakage_name="b",
            best_cost_name="a",
            best_normalized_name="a",
            normalization_recommended=True,
            regime_assessment="constraint_shaped_family",
            mean_pairwise_theta_distance=mean_distance,
            min_pairwise_theta_distance=min_distance,
            objective_scale_summary=compute_objective_scale_summary(
                {"a": [self.build_detail()], "b": [self.build_detail(transport_gain=0.05)]}
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = write_pareto_audit_csv(tmp / "audit.csv", [result_a, result_b])
            json_path = write_pareto_audit_summary_json(tmp / "audit.json", [result_a, result_b], summary)
            plot_path = plot_pareto_audit([result_a, result_b], tmp / "plot.png")
            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())
            self.assertTrue(plot_path.exists())

    def test_tiny_pareto_audit_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = self.build_config(
                csv_path=tmp / "audit.csv",
                summary_path=tmp / "audit.json",
                plot_path=tmp / "audit.png",
            )
            results, summary, csv_path, summary_path, plot_path = run_transition_motor_pareto_audit(config)
            self.assertEqual(len(results), 2)
            self.assertEqual(summary.n_weight_sets, 2)
            self.assertTrue(csv_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(plot_path.exists())
            self.assertIn(summary.best_normalized_name, {result.name for result in results})

    def build_detail(
        self,
        *,
        transport_gain: float = 0.03,
        noise_gain: float = 0.001,
        leakage_gain: float = 0.0005,
        best_cost: float = 0.02,
        best_theta: dict[str, float] | None = None,
    ) -> DetailedMotorEnsembleSample:
        theta = {
            "path_coupling_boost": 0.0,
            "onsite_phase_gradient": 0.0,
        }
        if best_theta:
            theta.update(best_theta)
        sample = MotorEnsembleSampleResult(
            sample_id=0,
            baseline_detector_success=0.70,
            best_detector_success=0.74,
            baseline_detector_final=0.50,
            best_detector_final=0.54,
            baseline_transport_efficiency=0.70,
            best_transport_efficiency=0.70 + transport_gain,
            baseline_noise_action_on_info=0.10,
            best_noise_action_on_info=0.10 - noise_gain,
            baseline_noise_leakage=0.05,
            best_noise_leakage=0.05 - leakage_gain,
            baseline_control_cost=0.0,
            best_control_cost=best_cost,
            baseline_objective=0.60,
            best_objective=0.60 + transport_gain + noise_gain + leakage_gain - best_cost * 0.01,
            detector_success_gain=0.04,
            transport_gain=transport_gain,
            noise_action_reduction=noise_gain,
            noise_leakage_reduction=leakage_gain,
            objective_gain=transport_gain + noise_gain + leakage_gain - best_cost * 0.01,
            saturated_knobs=1,
            near_bound_knobs=1,
            success=True,
        )
        baseline_metrics = MotorMetrics(0.70, 0.02, 0.05, 0.10, 0.90, 0.0, 0.60)
        best_metrics = MotorMetrics(0.70 + transport_gain, 0.02, 0.05 - leakage_gain, 0.10 - noise_gain, 0.90, best_cost, sample.best_objective)
        return DetailedMotorEnsembleSample(
            sample=sample,
            baseline_theta={"path_coupling_boost": 0.0, "onsite_phase_gradient": 0.0},
            best_theta=theta,
            baseline_metrics=baseline_metrics,
            best_metrics=best_metrics,
        )

    def build_audit_result(self, name: str, *, theta_shift: float) -> ParetoAuditRunResult:
        knob_profiles = [
            KnobProfileSummary(name, "path_coupling_boost", theta_shift, theta_shift, 0.5, 0.5),
            KnobProfileSummary(name, "onsite_phase_gradient", -theta_shift, theta_shift, 0.0, 0.0),
        ]
        detector_mean = 0.05 + theta_shift
        noise_mean = 0.001 + theta_shift * 0.01
        leakage_mean = 0.0005 + theta_shift * 0.005
        objective_mean = 0.04 + theta_shift
        normalized_mean = 0.8 + theta_shift
        return ParetoAuditRunResult(
            name=name,
            weights=ObjectiveWeightSet(name, 1.0, 0.5, 0.25, 0.01),
            n_samples=2,
            success_rate=0.75,
            mean_detector_success_gain=detector_mean,
            mean_noise_action_reduction=noise_mean,
            mean_noise_leakage_reduction=leakage_mean,
            mean_objective_gain=objective_mean,
            mean_normalized_objective_gain=normalized_mean,
            mean_best_control_cost=0.03 + theta_shift,
            mean_saturated_knobs=2.0,
            mean_near_bound_knobs=2.0,
            mean_transport_term_gain=0.05,
            mean_noise_action_term_gain=0.002,
            mean_noise_leakage_term_gain=0.001,
            mean_control_cost_penalty=0.0003,
            detector_gain_ci_low=detector_mean - 0.01,
            detector_gain_ci_high=detector_mean + 0.01,
            noise_action_ci_low=noise_mean - 0.0003,
            noise_action_ci_high=noise_mean + 0.0003,
            leakage_ci_low=leakage_mean - 0.0002,
            leakage_ci_high=leakage_mean + 0.0002,
            objective_ci_low=objective_mean - 0.01,
            objective_ci_high=objective_mean + 0.01,
            normalized_objective_ci_low=normalized_mean - 0.1,
            normalized_objective_ci_high=normalized_mean + 0.1,
            mean_theta_l2=theta_shift,
            knob_profiles=knob_profiles,
            is_pareto_optimal=True,
        )

    def build_config(
        self,
        *,
        csv_path: Path | None = None,
        summary_path: Path | None = None,
        plot_path: Path | None = None,
    ) -> dict[str, object]:
        return {
            "transition_motor_pareto_audit": {
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
                    {
                        "name": "balanced",
                        "transport": 1.0,
                        "noise_action": 0.5,
                        "leakage": 0.25,
                        "control_cost": 0.01,
                    },
                    {
                        "name": "noise_extreme",
                        "transport": 0.3,
                        "noise_action": 2.0,
                        "leakage": 0.5,
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
                "audit": {
                    "n_bootstrap": 200,
                    "seed": 11,
                },
                "outputs": {
                    "csv_path": str(csv_path or Path("outputs/test_pareto_audit.csv")),
                    "summary_path": str(summary_path or Path("outputs/test_pareto_audit.json")),
                    "plot_path": str(plot_path or Path("results/renders/test_pareto_audit.png")),
                },
            }
        }


if __name__ == "__main__":
    unittest.main()

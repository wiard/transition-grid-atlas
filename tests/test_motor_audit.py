from __future__ import annotations

from dataclasses import dataclass
import unittest

from hardware.control_knobs import ControlKnob, KnobRegistry
from hardware.motor_audit import (
    ablate_knobs,
    count_saturated_knobs,
    knob_bound_statuses,
    knob_efficiency_from_ablation,
    run_control_limit_sweep,
    run_seed_stability_audit,
    scale_registry_limits,
    summarize_seed_stability,
)
from hardware.objectives import MotorMetrics
from hardware.transition_motor import TransitionMotorResult


class MotorAuditTests(unittest.TestCase):
    def build_registry(self) -> KnobRegistry:
        return KnobRegistry(
            knobs=(
                ControlKnob("a", "onsite_phase", "a", "uniform", -0.1, 0.1, 0.0, "arb", "a", "a"),
                ControlKnob("b", "coherent_coupling", "b", "uniform_edges", -0.2, 0.2, 0.0, "arb", "b", "b"),
            )
        )

    def test_bound_statuses_detect_bounds_and_relative_position(self):
        registry = self.build_registry()
        statuses = knob_bound_statuses({"a": 0.1, "b": -0.2}, registry)
        by_name = {item.name: item for item in statuses}
        self.assertAlmostEqual(by_name["a"].relative_position, 1.0)
        self.assertTrue(by_name["a"].at_upper_bound)
        self.assertTrue(by_name["b"].at_lower_bound)
        self.assertTrue(by_name["a"].near_bound)
        self.assertEqual(count_saturated_knobs(statuses), 2)

    def test_ablation_returns_one_result_per_knob(self):
        def evaluate(theta):
            objective = 1.0 + 2.0 * theta["a"] + 3.0 * theta["b"]
            return MotorMetrics(
                transport_efficiency=0.8 + 0.1 * theta["a"],
                noise_internal=0.0,
                noise_leakage=0.2 - 0.05 * theta["b"],
                noise_action_on_info=0.3 - 0.1 * theta["a"],
                suppression_score=0.7 + 0.1 * theta["a"],
                control_cost=theta["a"] ** 2 + theta["b"] ** 2,
                objective=objective,
            )

        best_theta = {"a": 0.1, "b": 0.2}
        default_theta = {"a": 0.0, "b": 0.0}
        results = ablate_knobs(best_theta, default_theta, evaluate)
        self.assertEqual({item.knob for item in results}, {"a", "b"})
        self.assertTrue(all(item.objective_loss_from_ablation == item.objective_loss_from_ablation for item in results))

    def test_gain_per_cost_handles_zero_theta(self):
        def evaluate(theta):
            return MotorMetrics(0.8, 0.0, 0.1, 0.2, 0.8, theta["a"] ** 2 + theta["b"] ** 2, 1.0 + theta["a"] + theta["b"])

        best_theta = {"a": 0.0, "b": 0.2}
        default_theta = {"a": 0.0, "b": 0.0}
        efficiencies = knob_efficiency_from_ablation(best_theta, ablate_knobs(best_theta, default_theta, evaluate))
        self.assertTrue(all(item.gain_per_cost == item.gain_per_cost for item in efficiencies))

    def test_scale_registry_limits_preserves_defaults(self):
        registry = self.build_registry()
        scaled_half = scale_registry_limits(registry, 0.5)
        scaled_full = scale_registry_limits(registry, 1.0)
        self.assertEqual(scaled_half.defaults(), registry.defaults())
        self.assertEqual(scaled_full.defaults(), registry.defaults())
        self.assertAlmostEqual(scaled_full.knobs[0].min_value, registry.knobs[0].min_value)
        self.assertGreater(scaled_full.knobs[1].max_value, scaled_half.knobs[1].max_value)

    def test_limit_sweep_and_seed_summary_are_well_formed(self):
        registry = self.build_registry()

        @dataclass(frozen=True)
        class DummyConfig:
            knob_registry: KnobRegistry
            seed: int

        config = DummyConfig(knob_registry=registry, seed=7)

        def make_metrics(objective, transport, noise_action, leakage):
            return MotorMetrics(transport, 0.0, leakage, noise_action, 1.0 - noise_action, 0.01, objective)

        def fake_run_motor_fn(config):
            best_theta = {knob.name: knob.max_value for knob in config.knob_registry.knobs}
            baseline = make_metrics(0.5, 0.7, 0.2, 0.1)
            best = make_metrics(0.5 + 0.1 * config.knob_registry.knobs[0].max_value, 0.75, 0.18, 0.09)
            return TransitionMotorResult(
                baseline_theta=config.knob_registry.defaults(),
                best_theta=best_theta,
                baseline_metrics=baseline,
                best_metrics=best,
                objective_improvement=best.objective - baseline.objective,
                top_sensitivities=[],
            )

        sweep = run_control_limit_sweep(config, [0.5, 1.0, 1.5], fake_run_motor_fn)
        self.assertEqual([item.scale for item in sweep], [0.5, 1.0, 1.5])

        stability = run_seed_stability_audit(config, [1, 2], fake_run_motor_fn)
        summary = summarize_seed_stability(stability)
        self.assertIn("mean_best_objective", summary)
        self.assertIn("std_best_objective", summary)
        self.assertIn("min_best_objective", summary)
        self.assertIn("max_best_objective", summary)

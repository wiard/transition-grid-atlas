from __future__ import annotations

import unittest

from hardware.objective_modes import (
    ObjectiveMode,
    ObjectiveNormalizationScale,
    default_objective_modes,
    estimate_normalization_scales_from_rows,
    evaluate_objective_mode,
    objective_mode_from_config,
    raw_motor_objective_from_mode,
)


class ObjectiveModesTests(unittest.TestCase):
    def test_valid_raw_mode_accepts_none_normalization(self):
        mode = ObjectiveMode(
            name="raw_balanced",
            mode="raw",
            transport_weight=1.0,
            noise_action_weight=0.5,
            leakage_weight=0.25,
            control_cost_weight=0.01,
            normalization=None,
            description="raw",
        )
        self.assertIsNone(mode.normalization)

    def test_normalized_mode_without_scales_fails(self):
        with self.assertRaises(ValueError):
            ObjectiveMode(
                name="normalized_noise",
                mode="normalized",
                transport_weight=1.0,
                noise_action_weight=1.0,
                leakage_weight=0.5,
                control_cost_weight=0.05,
                normalization=None,
                description="bad",
            )

    def test_negative_weights_and_nonpositive_scales_fail(self):
        with self.assertRaises(ValueError):
            ObjectiveMode(
                name="bad",
                mode="raw",
                transport_weight=-1.0,
                noise_action_weight=0.5,
                leakage_weight=0.25,
                control_cost_weight=0.01,
                normalization=None,
                description="bad",
            )
        with self.assertRaises(ValueError):
            ObjectiveMode(
                name="bad",
                mode="normalized",
                transport_weight=1.0,
                noise_action_weight=0.5,
                leakage_weight=0.25,
                control_cost_weight=0.01,
                normalization=ObjectiveNormalizationScale(0.0, 1.0, 1.0, 1.0),
                description="bad",
            )

    def test_raw_objective_reproduces_legacy_formula(self):
        mode = ObjectiveMode(
            name="raw_balanced",
            mode="raw",
            transport_weight=1.0,
            noise_action_weight=0.5,
            leakage_weight=0.25,
            control_cost_weight=0.01,
            normalization=None,
            description="raw",
        )
        objective = raw_motor_objective_from_mode(
            transport_efficiency=0.8,
            noise_action_on_info=0.2,
            noise_leakage=0.1,
            control_cost=0.05,
            mode=mode,
        )
        self.assertAlmostEqual(objective, 0.8 - 0.5 * 0.2 - 0.25 * 0.1 - 0.01 * 0.05)

    def test_normalized_objective_uses_scales_and_is_monotonic(self):
        mode = ObjectiveMode(
            name="normalized_noise",
            mode="normalized",
            transport_weight=1.0,
            noise_action_weight=1.0,
            leakage_weight=0.5,
            control_cost_weight=0.05,
            normalization=ObjectiveNormalizationScale(0.05, 0.002, 0.001, 0.04),
            description="normalized",
        )
        base = evaluate_objective_mode(
            transport_efficiency=0.75,
            noise_action_on_info=0.15,
            noise_leakage=0.08,
            control_cost=0.02,
            mode=mode,
        )
        better_transport = evaluate_objective_mode(
            transport_efficiency=0.80,
            noise_action_on_info=0.15,
            noise_leakage=0.08,
            control_cost=0.02,
            mode=mode,
        )
        worse_noise = evaluate_objective_mode(
            transport_efficiency=0.75,
            noise_action_on_info=0.20,
            noise_leakage=0.08,
            control_cost=0.02,
            mode=mode,
        )
        self.assertGreater(better_transport, base)
        self.assertLess(worse_noise, base)

    def test_estimate_normalization_scales_from_rows(self):
        rows = [
            {
                "detector_success_gain": 0.05,
                "noise_action_reduction": 0.002,
                "noise_leakage_reduction": 0.001,
                "best_control_cost": 0.04,
            },
            {
                "detector_success_gain": 0.03,
                "noise_action_reduction": 0.003,
                "noise_leakage_reduction": 0.002,
                "best_control_cost": 0.02,
            },
        ]
        scales = estimate_normalization_scales_from_rows(rows)
        self.assertGreater(scales.transport_scale, 0.0)
        self.assertGreater(scales.noise_action_scale, 0.0)
        self.assertGreater(scales.leakage_scale, 0.0)
        self.assertGreater(scales.control_cost_scale, 0.0)

    def test_default_modes_and_config_parser(self):
        modes = default_objective_modes(ObjectiveNormalizationScale(0.05, 0.002, 0.001, 0.04))
        names = [mode.name for mode in modes]
        self.assertIn("raw_detector_max", names)
        self.assertIn("normalized_noise", names)
        self.assertIsNone(objective_mode_from_config({}))
        parsed = objective_mode_from_config(
            {
                "objective_mode": {
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
            }
        )
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.mode, "normalized")
        self.assertEqual(parsed.name, "normalized_noise")

    def test_objective_mode_from_config_requires_scales_for_normalized(self):
        with self.assertRaises(ValueError):
            objective_mode_from_config(
                {
                    "objective_mode": {
                        "name": "normalized_noise",
                        "mode": "normalized",
                        "weights": {
                            "transport": 1.0,
                            "noise_action": 1.0,
                            "leakage": 0.5,
                            "control_cost": 0.05,
                        },
                        "description": "invalid",
                    }
                }
            )

    def test_estimate_normalization_scales_missing_column_fails(self):
        with self.assertRaises(ValueError):
            estimate_normalization_scales_from_rows(
                [
                    {
                        "detector_success_gain": 0.05,
                        "noise_action_reduction": 0.002,
                        "best_control_cost": 0.04,
                    }
                ]
            )

from __future__ import annotations

import unittest

from hardware.objective_modes import ObjectiveMode, ObjectiveModeRegistry


class ObjectiveModeRegistryTests(unittest.TestCase):
    def test_valid_registry_returns_names_and_mode(self):
        registry = ObjectiveModeRegistry(
            modes=(
                ObjectiveMode(
                    name="raw_mode",
                    mode="raw",
                    transport=1.0,
                    noise_action=0.5,
                    leakage=0.25,
                    control_cost=0.01,
                    description="legacy raw",
                ),
                ObjectiveMode(
                    name="calibrated_mode",
                    mode="calibrated",
                    transport=0.25,
                    noise_action=4.0,
                    leakage=1.0,
                    control_cost=0.01,
                    description="audit calibrated",
                ),
            ),
            default_mode="calibrated_mode",
        )
        self.assertEqual(registry.names(), ["raw_mode", "calibrated_mode"])
        self.assertEqual(registry.get("calibrated_mode").mode, "calibrated")

    def test_duplicate_mode_names_fail(self):
        with self.assertRaises(ValueError):
            ObjectiveModeRegistry(
                modes=(
                    ObjectiveMode(
                        name="dup",
                        mode="raw",
                        transport=1.0,
                        noise_action=0.5,
                        leakage=0.25,
                        control_cost=0.01,
                        description="a",
                    ),
                    ObjectiveMode(
                        name="dup",
                        mode="normalized",
                        transport=1.0,
                        noise_action=1.0,
                        leakage=1.0,
                        control_cost=1.0,
                        description="b",
                    ),
                ),
                default_mode="dup",
            )

    def test_non_positive_performance_family_fails(self):
        with self.assertRaises(ValueError):
            ObjectiveMode(
                name="invalid",
                mode="normalized",
                transport=0.0,
                noise_action=0.0,
                leakage=0.0,
                control_cost=0.1,
                description="invalid",
            )


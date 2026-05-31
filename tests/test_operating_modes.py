from __future__ import annotations

import unittest

from hardware.operating_modes import OperatingMode, OperatingModeRegistry


class OperatingModeRegistryTests(unittest.TestCase):
    def test_valid_registry_returns_names_and_mode(self):
        registry = OperatingModeRegistry(
            modes=(
                OperatingMode(
                    name="legacy_raw",
                    objective_mode="raw",
                    transport=1.0,
                    noise_action=0.5,
                    leakage=0.25,
                    control_cost=0.01,
                    description="legacy",
                ),
                OperatingMode(
                    name="noise_mode",
                    objective_mode="normalized",
                    transport=0.25,
                    noise_action=4.0,
                    leakage=1.0,
                    control_cost=0.01,
                    description="noise",
                ),
            ),
            default_mode="noise_mode",
        )
        self.assertEqual(registry.names(), ["legacy_raw", "noise_mode"])
        self.assertEqual(registry.get("noise_mode").objective_mode, "normalized")

    def test_duplicate_mode_names_fail(self):
        with self.assertRaises(ValueError):
            OperatingModeRegistry(
                modes=(
                    OperatingMode(
                        name="dup",
                        objective_mode="raw",
                        transport=1.0,
                        noise_action=0.5,
                        leakage=0.25,
                        control_cost=0.01,
                        description="a",
                    ),
                    OperatingMode(
                        name="dup",
                        objective_mode="normalized",
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
            OperatingMode(
                name="invalid",
                objective_mode="normalized",
                transport=0.0,
                noise_action=0.0,
                leakage=0.0,
                control_cost=0.1,
                description="invalid",
            )


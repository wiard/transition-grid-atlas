from __future__ import annotations

import unittest

from hardware.control_knobs import ControlKnob, KnobRegistry


class ControlKnobTests(unittest.TestCase):
    def build_registry(self) -> KnobRegistry:
        return KnobRegistry(
            knobs=(
                ControlKnob(
                    name="k1",
                    family="onsite_phase",
                    symbol="k1",
                    basis_name="uniform",
                    min_value=-0.1,
                    max_value=0.1,
                    default=0.0,
                    units="arb",
                    description="test",
                    hardware_meaning="test",
                ),
            )
        )

    def test_valid_registry_accepts_defaults(self):
        registry = self.build_registry()
        self.assertEqual(registry.defaults()["k1"], 0.0)

    def test_duplicate_knob_names_fail(self):
        with self.assertRaises(ValueError):
            KnobRegistry(
                knobs=(
                    ControlKnob("k", "onsite_phase", "k", "uniform", -0.1, 0.1, 0.0, "arb", "a", "b"),
                    ControlKnob("k", "onsite_phase", "k2", "uniform", -0.1, 0.1, 0.0, "arb", "a", "b"),
                )
            )

    def test_default_outside_range_fails(self):
        with self.assertRaises(ValueError):
            ControlKnob("k", "onsite_phase", "k", "uniform", -0.1, 0.1, 0.2, "arb", "a", "b")

    def test_unknown_theta_key_fails(self):
        registry = self.build_registry()
        with self.assertRaises(ValueError):
            registry.validate_theta({"unknown": 0.1})

    def test_clip_theta_works(self):
        registry = self.build_registry()
        clipped = registry.clip_theta({"k1": 1.5})
        self.assertAlmostEqual(clipped["k1"], 0.1)

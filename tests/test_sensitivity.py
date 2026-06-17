from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hardware.control_knobs import ControlKnob, KnobRegistry
from hardware.sensitivity import compute_sensitivity_matrix, top_sensitivities, write_sensitivity_csv


class SensitivityTests(unittest.TestCase):
    def build_registry(self) -> KnobRegistry:
        return KnobRegistry(
            knobs=(
                ControlKnob("x", "onsite_phase", "x", "uniform", -1.0, 1.0, 0.0, "arb", "x", "x"),
                ControlKnob("y", "onsite_phase", "y", "uniform", -1.0, 1.0, 0.0, "arb", "y", "y"),
            )
        )

    def test_sensitivity_entries_cover_all_metric_knob_pairs(self):
        registry = self.build_registry()

        class Metrics:
            def __init__(self, theta):
                self.transport_efficiency = theta["x"] + 2.0 * theta["y"]
                self.noise_action_on_info = theta["x"] ** 2 + theta["y"]

        entries = compute_sensitivity_matrix(
            lambda theta: Metrics(theta),
            {"x": 0.1, "y": -0.2},
            registry,
            eps=1.0e-4,
            metric_names=["transport_efficiency", "noise_action_on_info"],
        )
        self.assertEqual(len(entries), 4)
        self.assertTrue(all(abs(entry.central_difference) < 1.0e6 for entry in entries))

    def test_top_sensitivities_sort_by_absolute_value(self):
        entries = [
            type("Entry", (), {"metric": "a", "knob": "k1", "central_difference": 0.2}),
            type("Entry", (), {"metric": "b", "knob": "k2", "central_difference": -1.3}),
            type("Entry", (), {"metric": "c", "knob": "k3", "central_difference": 0.7}),
        ]
        top = top_sensitivities(entries, limit=2)
        self.assertEqual(top[0]["knob"], "k2")
        self.assertEqual(top[1]["knob"], "k3")

    def test_write_sensitivity_csv_writes_file(self):
        registry = self.build_registry()

        class Metrics:
            def __init__(self, theta):
                self.transport_efficiency = theta["x"]

        entries = compute_sensitivity_matrix(
            lambda theta: Metrics(theta),
            {"x": 0.0, "y": 0.0},
            registry,
            eps=1.0e-4,
            metric_names=["transport_efficiency"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = write_sensitivity_csv(Path(tmpdir) / "sensitivity.csv", entries)
            self.assertTrue(output.exists())

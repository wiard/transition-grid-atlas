from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hardware.ensemble_statistics import (
    bootstrap_mean_ci,
    compute_bootstrap_summary,
    compute_bound_pressure_summary,
    compute_tradeoff_summary,
    compute_win_rates,
    plot_ensemble_statistics,
    read_ensemble_csv,
    safe_corrcoef,
    write_statistics_json,
)


class EnsembleStatisticsTests(unittest.TestCase):
    def build_rows(self) -> list[dict[str, float]]:
        return [
            {
                "detector_success_gain": 0.10,
                "transport_gain": 0.10,
                "noise_action_reduction": 0.02,
                "noise_leakage_reduction": 0.03,
                "objective_gain": 0.08,
                "saturated_knobs": 2.0,
                "near_bound_knobs": 3.0,
            },
            {
                "detector_success_gain": -0.01,
                "transport_gain": -0.01,
                "noise_action_reduction": 0.00,
                "noise_leakage_reduction": -0.01,
                "objective_gain": 0.01,
                "saturated_knobs": 1.0,
                "near_bound_knobs": 2.0,
            },
            {
                "detector_success_gain": 0.04,
                "transport_gain": 0.04,
                "noise_action_reduction": 0.01,
                "noise_leakage_reduction": 0.02,
                "objective_gain": 0.03,
                "saturated_knobs": 3.0,
                "near_bound_knobs": 3.0,
            },
        ]

    def test_read_csv_and_required_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ensemble.csv"
            path.write_text(
                "detector_success_gain,transport_gain,noise_action_reduction,noise_leakage_reduction,objective_gain,saturated_knobs,near_bound_knobs\n"
                "0.1,0.1,0.02,0.03,0.08,2,3\n",
                encoding="utf-8",
            )
            rows = read_ensemble_csv(path)
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["detector_success_gain"], 0.1)

    def test_compute_win_rates_and_tolerance(self):
        win_rates = compute_win_rates(self.build_rows(), tolerance=0.0)
        self.assertEqual(win_rates.n_samples, 3)
        self.assertAlmostEqual(win_rates.detector_win_rate, 2.0 / 3.0)
        self.assertAlmostEqual(win_rates.noise_action_win_rate, 2.0 / 3.0)
        tighter = compute_win_rates(self.build_rows(), tolerance=0.015)
        self.assertAlmostEqual(tighter.noise_action_win_rate, 1.0 / 3.0)
        with self.assertRaises(ValueError):
            compute_win_rates([])

    def test_bootstrap_ci_is_deterministic_and_valid(self):
        values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        ci_a = bootstrap_mean_ci(values, n_bootstrap=500, seed=7)
        ci_b = bootstrap_mean_ci(values, n_bootstrap=500, seed=7)
        self.assertEqual(ci_a, ci_b)
        self.assertLessEqual(ci_a[0], float(np.mean(values)))
        self.assertGreaterEqual(ci_a[1], float(np.mean(values)))
        with self.assertRaises(ValueError):
            bootstrap_mean_ci(np.array([], dtype=np.float64))

    def test_bootstrap_summary_bound_pressure_and_tradeoffs(self):
        bootstrap = compute_bootstrap_summary(
            self.build_rows(),
            ["detector_success_gain", "noise_action_reduction", "noise_leakage_reduction", "objective_gain"],
            n_bootstrap=200,
            seed=11,
        )
        self.assertEqual(len(bootstrap), 4)

        bound = compute_bound_pressure_summary(self.build_rows())
        self.assertAlmostEqual(bound.mean_saturated_knobs, 2.0)
        self.assertAlmostEqual(bound.median_saturated_knobs, 2.0)

        tradeoffs = compute_tradeoff_summary(self.build_rows())
        self.assertGreaterEqual(tradeoffs.detector_vs_noise_corr, -1.0)
        self.assertLessEqual(tradeoffs.detector_vs_noise_corr, 1.0)
        self.assertEqual(safe_corrcoef(np.array([1.0, 1.0]), np.array([2.0, 3.0])), 0.0)

    def test_plot_and_json_writers(self):
        rows = self.build_rows()
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = Path(tmpdir) / "stats.png"
            json_path = Path(tmpdir) / "stats.json"
            self.assertTrue(plot_ensemble_statistics(rows, plot_path).exists())
            output = write_statistics_json(
                json_path,
                win_rates=compute_win_rates(rows),
                bootstrap=compute_bootstrap_summary(rows, ["detector_success_gain"], n_bootstrap=100, seed=3),
                bound_pressure=compute_bound_pressure_summary(rows),
                tradeoffs=compute_tradeoff_summary(rows),
            )
            self.assertTrue(output.exists())
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("win_rates", payload)

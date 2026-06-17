from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from hardware.transition_tuner import FixedGrid, TransitionTunerConfig
from hardware.wafer_ensemble import (
    EnsembleSampleResult,
    apply_fabrication_disorder,
    run_single_wafer_sample,
    run_wafer_ensemble_study,
    sample_correlated_profile,
    sample_fabrication_disorder_profiles,
    sample_phase_noise_profiles,
    summarize_ensemble,
    write_ensemble_csv,
    write_ensemble_summary_json,
)
from run import run_wafer_ensemble_mode


class WaferEnsembleTests(unittest.TestCase):
    def build_grid(self) -> FixedGrid:
        return FixedGrid(
            n_sites=8,
            edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)],
            base_coupling=1.0,
            input_index=0,
            target_indices=[6, 7],
        )

    def build_tuner(self) -> TransitionTunerConfig:
        return TransitionTunerConfig(
            max_relative_coupling_delta=0.10,
            max_onsite_shift=0.08,
            n_candidates=30,
            seed=7,
            time_min=0.0,
            time_max=12.0,
            n_time_samples=40,
            transport_weight=1.0,
            suppression_weight=0.5,
            control_penalty_weight=0.01,
        )

    def build_config(self, csv_path: str, summary_path: str) -> dict[str, object]:
        return {
            "wafer_ensemble": {
                "grid": {
                    "n_sites": 8,
                    "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]],
                    "base_coupling": 1.0,
                    "input_index": 0,
                    "target_indices": [6, 7],
                },
                "fabrication_disorder": {
                    "n_samples": 4,
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
                "tuner": {
                    "max_relative_coupling_delta": 0.10,
                    "max_onsite_shift": 0.08,
                    "n_candidates": 30,
                    "seed": 7,
                    "time_min": 0.0,
                    "time_max": 12.0,
                    "n_time_samples": 40,
                    "transport_weight": 1.0,
                    "suppression_weight": 0.5,
                    "control_penalty_weight": 0.01,
                },
                "outputs": {
                    "csv_path": csv_path,
                    "summary_path": summary_path,
                },
            }
        }

    def test_correlated_profile_shape_and_reproducibility(self):
        rng_a = np.random.default_rng(123)
        rng_b = np.random.default_rng(123)
        profile_a = sample_correlated_profile(12, 0.5, 2.0, rng_a)
        profile_b = sample_correlated_profile(12, 0.5, 2.0, rng_b)
        self.assertEqual(profile_a.shape, (12,))
        self.assertTrue(np.allclose(profile_a, profile_b))

    def test_fabrication_profiles_are_reproducible(self):
        first = sample_fabrication_disorder_profiles(3, 10, 0.1, 2.0, 55)
        second = sample_fabrication_disorder_profiles(3, 10, 0.1, 2.0, 55)
        self.assertEqual(len(first), 3)
        self.assertTrue(all(np.allclose(a, b) for a, b in zip(first, second)))

    def test_phase_noise_profiles_are_reproducible(self):
        first = sample_phase_noise_profiles(2, 10, 1.0, 2.0, 99)
        second = sample_phase_noise_profiles(2, 10, 1.0, 2.0, 99)
        self.assertEqual(len(first), 2)
        self.assertTrue(all(np.allclose(a, b) for a, b in zip(first, second)))

    def test_apply_fabrication_disorder_changes_only_diagonal(self):
        H = np.zeros((4, 4), dtype=np.complex128)
        H[0, 1] = H[1, 0] = 1.0
        H[1, 2] = H[2, 1] = 1.0
        disorder = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float64)
        out = apply_fabrication_disorder(H, disorder)
        self.assertTrue(np.allclose(np.triu(out, 1), np.triu(H, 1)))
        self.assertTrue(np.allclose(np.tril(out, -1), np.tril(H, -1)))
        self.assertTrue(np.allclose(np.diag(out), disorder))

    def test_run_single_wafer_sample_returns_bounded_metrics(self):
        result = run_single_wafer_sample(
            self.build_grid(),
            onsite_disorder=np.linspace(-0.05, 0.05, 8, dtype=np.float64),
            noise_profiles=sample_phase_noise_profiles(2, 8, 0.7, 2.0, 99),
            tuner_config=self.build_tuner(),
        )
        self.assertIsInstance(result, EnsembleSampleResult)
        self.assertGreaterEqual(result.baseline_transport_efficiency, 0.0)
        self.assertLessEqual(result.baseline_transport_efficiency, 1.0 + 1.0e-12)
        self.assertGreaterEqual(result.best_transport_efficiency, 0.0)
        self.assertLessEqual(result.best_transport_efficiency, 1.0 + 1.0e-12)

    def test_summarize_ensemble_computes_success_rate(self):
        summary = summarize_ensemble(
            [
                EnsembleSampleResult(0, 0.5, 0.6, 0.3, 0.2, 0.7, 0.8, 1.0, 1.1, 0.1, 0.1, 0.1, True),
                EnsembleSampleResult(1, 0.5, 0.4, 0.3, 0.31, 0.7, 0.69, 1.0, 0.98, -0.1, -0.01, -0.02, False),
            ]
        )
        self.assertAlmostEqual(summary.success_rate, 0.5)

    def test_write_ensemble_csv_writes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ensemble.csv"
            output = write_ensemble_csv(
                path,
                [EnsembleSampleResult(0, 0.5, 0.6, 0.3, 0.2, 0.7, 0.8, 1.0, 1.1, 0.1, 0.1, 0.1, True)],
            )
            self.assertTrue(output.exists())

    def test_write_ensemble_summary_json_writes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            summary = summarize_ensemble(
                [EnsembleSampleResult(0, 0.5, 0.6, 0.3, 0.2, 0.7, 0.8, 1.0, 1.1, 0.1, 0.1, 0.1, True)]
            )
            output = write_ensemble_summary_json(path, summary)
            self.assertTrue(output.exists())
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["n_samples"], 1)

    def test_module_and_cli_smoke_work_with_tmp_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = str(Path(tmpdir) / "ensemble.csv")
            summary_path = str(Path(tmpdir) / "summary.json")
            config = self.build_config(csv_path, summary_path)
            results, summary, csv_output, summary_output = run_wafer_ensemble_study(config)
            self.assertEqual(len(results), 4)
            self.assertEqual(summary.n_samples, 4)
            self.assertTrue(csv_output.exists())
            self.assertTrue(summary_output.exists())

            stream = io.StringIO()
            with redirect_stdout(stream):
                exit_code = run_wafer_ensemble_mode(config)
            self.assertEqual(exit_code, 0)
            output = stream.getvalue()
            self.assertIn("Synthetic wafer ensemble study", output)
            self.assertIn("csv_path =", output)

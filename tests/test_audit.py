from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from validation.audit import (
    build_audit_report,
    deduplicate_transient_rows,
    duplicate_parameter_keys,
    is_modern_strict_stable_diffusive,
    is_transient_diffusive_like,
)


def audit_thresholds():
    return {
        "diffusive_alpha_target": 0.5,
        "diffusive_alpha_tolerance": 0.035,
        "min_r2": 0.95,
        "max_unitarity_error": 1.0e-12,
        "max_hermitian_error": 1.0e-12,
        "required_gamma": 0.0,
        "max_alpha_window_shift": 0.05,
        "top_n": 20,
    }


class AuditTests(unittest.TestCase):
    def test_modern_strict_stable_requires_all_conditions(self):
        row = {
            "alpha": "0.51",
            "r2": "0.97",
            "unitarity_error": "1.0e-15",
            "hermitian_error": "0.0",
            "gamma": "0.0",
            "scaling_window_stable": "true",
            "alpha_window_shift": "0.03",
        }
        self.assertTrue(is_modern_strict_stable_diffusive(row, audit_thresholds()))

        row["alpha_window_shift"] = "0.06"
        self.assertFalse(is_modern_strict_stable_diffusive(row, audit_thresholds()))

    def test_legacy_diffusive_row_is_not_modern_strict_without_new_columns(self):
        legacy_row = {
            "status": "VALID_DIFFUSIVE_CANDIDATE",
            "alpha": "0.50",
            "r2": "0.98",
            "unitarity_error": "1.0e-15",
            "hermitian_error": "0.0",
            "gamma": "0.0",
            "scaling_window_stable": "",
            "alpha_window_shift": "",
        }
        self.assertFalse(is_modern_strict_stable_diffusive(legacy_row, audit_thresholds()))

    def test_transient_diffusive_like_is_counted_separately(self):
        row = {
            "alpha": "0.50",
            "r2": "0.98",
            "unitarity_error": "1.0e-15",
            "hermitian_error": "0.0",
            "gamma": "0.0",
            "scaling_window_stable": "false",
            "alpha_window_shift": "0.12",
        }
        self.assertFalse(is_modern_strict_stable_diffusive(row, audit_thresholds()))
        self.assertTrue(is_transient_diffusive_like(row, audit_thresholds()))

    def test_duplicate_parameter_keys_detected(self):
        rows = [
            {"W": "2.5", "eta": "0.4", "gamma": "0.0", "seed": "42", "mode": "sweep"},
            {"W": "2.5", "eta": "0.4", "gamma": "0.0", "seed": "42", "mode": "sweep"},
            {"W": "2.5", "eta": "0.4", "gamma": "0.0", "seed": "42", "mode": "single"},
        ]
        duplicates = duplicate_parameter_keys(rows)
        self.assertEqual(duplicates[0][0], ("2.5", "0.4", "0.0", "42", "sweep"))
        self.assertEqual(duplicates[0][1], 2)

    def test_missing_master_results_handled_gracefully(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            (project_root / "atlas" / "results").mkdir(parents=True, exist_ok=True)
            report = build_audit_report(config={"audit": audit_thresholds()}, project_root=project_root)
            self.assertIn("Total master ledger rows: 0", report)
            self.assertIn("no master ledger rows found", report)

    def test_deduplicate_transient_rows_keeps_best_per_parameter_key(self):
        rows = [
            {
                "W": "2.5",
                "eta": "0.4",
                "gamma": "0.0",
                "seed": "42",
                "mode": "sweep",
                "alpha": "0.49",
                "r2": "0.96",
                "unitarity_error": "2.0e-15",
                "hermitian_error": "0.0",
                "alpha_window_shift": "0.20",
                "scaling_window_stable": "false",
            },
            {
                "W": "2.5",
                "eta": "0.4",
                "gamma": "0.0",
                "seed": "42",
                "mode": "sweep",
                "alpha": "0.501",
                "r2": "0.97",
                "unitarity_error": "1.0e-15",
                "hermitian_error": "0.0",
                "alpha_window_shift": "0.10",
                "scaling_window_stable": "false",
            },
        ]
        deduped = deduplicate_transient_rows(rows, audit_thresholds())
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["alpha"], "0.501")

    def test_master_rows_are_not_mislabeled_as_latest_sweep_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            results_dir = project_root / "atlas" / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            master_csv = (
                "status,W,eta,gamma,seed,mode,alpha,r2,unitarity_error,hermitian_error,"
                "alpha_window_shift,scaling_window_stable\n"
                "WEAK_FIT,3.0,0.0,0.0,42,sweep,0.503,0.96,1e-15,0.0,0.72,false\n"
            )
            latest_sweep_csv = (
                "status,W,eta,gamma,seed,mode,alpha,r2,unitarity_error,hermitian_error,"
                "alpha_window_shift,scaling_window_stable\n"
                "WEAK_FIT,2.6,0.2,0.0,42,sweep,0.52,0.97,1e-15,0.0,0.15,false\n"
            )
            (results_dir / "master_results.csv").write_text(master_csv, encoding="utf-8")
            (results_dir / "sweep_20260529T999999Z.csv").write_text(latest_sweep_csv, encoding="utf-8")

            report = build_audit_report(config={"audit": audit_thresholds()}, project_root=project_root)
            self.assertIn("8A. Top transient diffusive-like rows in master ledger:", report)
            self.assertIn("W=3.0", report)
            self.assertIn("source=master_results.csv", report)
            self.assertNotIn("W=3.0, eta=0.0, gamma=0.0, seed=42, alpha=0.503, r2=0.96, unitarity_error=1e-15, alpha_window_shift=0.72, scaling_window_stable=false, source=sweep_20260529T999999Z.csv", report)

    def test_latest_sweep_rows_are_labeled_as_latest_sweep_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            results_dir = project_root / "atlas" / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            master_csv = (
                "status,W,eta,gamma,seed,mode,alpha,r2,unitarity_error,hermitian_error,"
                "alpha_window_shift,scaling_window_stable\n"
            )
            latest_sweep_csv = (
                "status,W,eta,gamma,seed,mode,alpha,r2,unitarity_error,hermitian_error,"
                "alpha_window_shift,scaling_window_stable\n"
                "WEAK_FIT,2.55,0.3,0.0,42,sweep,0.501,0.98,1e-15,0.0,0.11,false\n"
            )
            (results_dir / "master_results.csv").write_text(master_csv, encoding="utf-8")
            (results_dir / "sweep_20260529T111111Z.csv").write_text(latest_sweep_csv, encoding="utf-8")

            report = build_audit_report(config={"audit": audit_thresholds()}, project_root=project_root)
            self.assertIn("8B. Top transient diffusive-like rows in latest sweep:", report)
            self.assertIn("source=sweep_20260529T111111Z.csv", report)

    def test_transient_rows_are_deduplicated_in_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            results_dir = project_root / "atlas" / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            repeated_row = (
                "WEAK_FIT,2.5,0.4,0.0,42,sweep,0.501,0.98,1e-15,0.0,0.12,false\n"
            )
            master_csv = (
                "status,W,eta,gamma,seed,mode,alpha,r2,unitarity_error,hermitian_error,"
                "alpha_window_shift,scaling_window_stable\n"
                + repeated_row
                + repeated_row
            )
            (results_dir / "master_results.csv").write_text(master_csv, encoding="utf-8")

            report = build_audit_report(config={"audit": audit_thresholds()}, project_root=project_root)
            expected_line = (
                "status=WEAK_FIT, W=2.5, eta=0.4, gamma=0.0, seed=42, alpha=0.501, "
                "r2=0.98, unitarity_error=1e-15, alpha_window_shift=0.12, "
                "scaling_window_stable=false, source=master_results.csv"
            )
            self.assertEqual(report.count(expected_line), 1)


if __name__ == "__main__":
    unittest.main()

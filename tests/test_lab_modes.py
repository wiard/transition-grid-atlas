from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from engine.lab_modes import build_lab_config, normalize_theory_mode, run_lab_simulation
from tests.lab_fixtures import build_lab_test_config


class LabModeTests(unittest.TestCase):
    def test_normalize_theory_mode_supports_aliases(self):
        self.assertEqual(normalize_theory_mode("standard_qm"), "qm_free")
        self.assertEqual(normalize_theory_mode("qm_free"), "qm_free")
        self.assertEqual(normalize_theory_mode("lindblad"), "lindblad")

    def test_build_lab_config_defaults_rtt_to_experimental(self):
        config = build_lab_test_config(theory_mode="rtt")
        lab_cfg = build_lab_config(config)
        self.assertTrue(lab_cfg.rtt_experimental)
        self.assertEqual(lab_cfg.audit_feedback, "logged_only")

    def test_each_mode_has_explicit_w_and_gamma_contract(self):
        qm_cfg = build_lab_config(build_lab_test_config(theory_mode="qm_free"))
        self.assertEqual(qm_cfg.theory_mode, "qm_free")
        anderson_cfg = build_lab_config(build_lab_test_config(theory_mode="anderson"))
        self.assertGreater(anderson_cfg.W, 0.0)
        lindblad_cfg = build_lab_config(build_lab_test_config(theory_mode="lindblad"))
        self.assertGreaterEqual(lindblad_cfg.gamma, 0.0)

    def test_lab_run_produces_probability_frames(self):
        config = build_lab_test_config(theory_mode="qm_free")
        lab_cfg = build_lab_config(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_lab_simulation(
                config=config,
                run_id="qm_free_test",
                trajectories_dir=Path(tmpdir),
                lab_config=lab_cfg,
            )
        self.assertEqual(result["probability_frames"].ndim, 2)
        self.assertTrue(np.allclose(np.sum(result["probability_frames"], axis=1), 1.0, atol=1.0e-10))

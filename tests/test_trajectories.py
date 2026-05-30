from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from engine.trajectories import config_hash, load_trajectory_npz, save_trajectory_npz


class TrajectoryTests(unittest.TestCase):
    def test_config_hash_is_deterministic(self):
        config = {"mode": "qm_free", "gamma": 0.0, "W": 0.0}
        self.assertEqual(config_hash(config), config_hash(dict(reversed(list(config.items())))))

    def test_save_and_load_trajectory_npz_round_trip(self):
        probabilities = np.zeros((3, 8), dtype=np.float64)
        probabilities[:, 3] = 1.0
        times = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        x_mean = np.array([3.0, 3.0, 3.0], dtype=np.float64)
        x_var = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        metadata = {"run_id": "tiny", "theory_mode": "qm_free"}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tiny.npz"
            digest = save_trajectory_npz(
                path,
                probabilities=probabilities,
                times=times,
                x_mean=x_mean,
                x_var=x_var,
                metadata=metadata,
                extra_arrays={"ipr": np.array([1.0, 1.0, 1.0], dtype=np.float64)},
            )
            loaded = load_trajectory_npz(path)

        self.assertEqual(loaded["config_hash"], digest)
        self.assertEqual(loaded["metadata"]["run_id"], "tiny")
        self.assertTrue(np.allclose(loaded["probabilities"], probabilities))
        self.assertIn("ipr", loaded)

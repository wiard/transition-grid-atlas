from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from matplotlib import animation
import numpy as np

from interface.visualiser import animate_probability_trajectory


class VisualiserTests(unittest.TestCase):
    def test_visualiser_rejects_bad_shape(self):
        bad = np.ones(10, dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "bad.gif"
            with self.assertRaises(ValueError):
                animate_probability_trajectory(bad, out)

    def test_visualiser_writes_gif_for_small_trajectory(self):
        if not animation.writers.is_available("pillow"):
            raise unittest.SkipTest("pillow writer is not available in this environment")

        probabilities = np.zeros((3, 8), dtype=np.float64)
        probabilities[:, 3] = 1.0

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "tiny.gif"
            animate_probability_trajectory(probabilities, out)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)

from __future__ import annotations

import unittest

import numpy as np

from hardware.detector_mapping import (
    detector_probabilities_from_trajectory,
    transport_efficiency_to_targets,
)


class DetectorMappingTests(unittest.TestCase):
    def test_detector_probabilities_returns_last_frame(self):
        trajectory = np.array(
            [
                [0.2, 0.8, 0.0],
                [0.1, 0.3, 0.6],
            ],
            dtype=np.float64,
        )
        final_probabilities = detector_probabilities_from_trajectory(trajectory)
        self.assertTrue(np.allclose(final_probabilities, trajectory[-1]))

    def test_transport_efficiency_sums_targets(self):
        final_probabilities = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        efficiency = transport_efficiency_to_targets(final_probabilities, [1, 3])
        self.assertAlmostEqual(efficiency, 0.6)

from __future__ import annotations

import unittest

import numpy as np

from engine.wavepacket import gaussian_wavepacket


class WavepacketTests(unittest.TestCase):
    def test_gaussian_wavepacket_is_normalized(self):
        psi = gaussian_wavepacket(L=128, x0=64, sigma=8, k0=0.5)
        self.assertTrue(np.iscomplexobj(psi))
        self.assertAlmostEqual(float(np.sum(np.abs(psi) ** 2)), 1.0, places=12)

    def test_gaussian_wavepacket_rejects_bad_sigma(self):
        with self.assertRaises(ValueError):
            gaussian_wavepacket(L=128, x0=64, sigma=0, k0=0.0)

    def test_gaussian_wavepacket_rejects_bad_length(self):
        with self.assertRaises(ValueError):
            gaussian_wavepacket(L=0, x0=0, sigma=1.0, k0=0.0)

    def test_k0_changes_phase_not_initial_probability_mass(self):
        psi_a = gaussian_wavepacket(L=64, x0=32, sigma=6, k0=0.0)
        psi_b = gaussian_wavepacket(L=64, x0=32, sigma=6, k0=0.9)
        self.assertFalse(np.allclose(psi_a, psi_b))
        self.assertTrue(np.allclose(np.abs(psi_a) ** 2, np.abs(psi_b) ** 2))

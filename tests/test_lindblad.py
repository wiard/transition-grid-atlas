from __future__ import annotations

import unittest

import numpy as np

from engine.hamiltonian import build_tight_binding_hamiltonian
from engine.lindblad import coherence_norm, lindblad_dephasing_rhs, psi_to_rho, rk4_lindblad_step, trace_error
from engine.wavepacket import gaussian_wavepacket


class LindbladTests(unittest.TestCase):
    def test_psi_to_rho_has_trace_one(self):
        psi = gaussian_wavepacket(L=16, x0=8, sigma=2.0, k0=0.4)
        rho = psi_to_rho(psi)
        self.assertAlmostEqual(float(np.trace(rho).real), 1.0, places=12)

    def test_negative_gamma_fails(self):
        psi = gaussian_wavepacket(L=8, x0=4, sigma=1.0, k0=0.0)
        rho = psi_to_rho(psi)
        H = build_tight_binding_hamiltonian(size=8, hopping=1.0, disorder_strength=0.0, bias=0.0, seed=1)
        with self.assertRaises(ValueError):
            lindblad_dephasing_rhs(rho, H, -0.1)

    def test_gamma_zero_rhs_has_zero_trace_derivative(self):
        psi = gaussian_wavepacket(L=8, x0=4, sigma=1.0, k0=0.0)
        rho = psi_to_rho(psi)
        H = build_tight_binding_hamiltonian(size=8, hopping=1.0, disorder_strength=0.0, bias=0.0, seed=1)
        rhs = lindblad_dephasing_rhs(rho, H, 0.0)
        self.assertAlmostEqual(float(np.trace(rhs).real), 0.0, places=10)

    def test_rk4_step_preserves_trace_and_hermiticity(self):
        psi = gaussian_wavepacket(L=12, x0=6, sigma=1.5, k0=0.2)
        rho = psi_to_rho(psi)
        H = build_tight_binding_hamiltonian(size=12, hopping=1.0, disorder_strength=0.0, bias=0.0, seed=1)
        next_rho = rk4_lindblad_step(rho, H, 0.05, 0.02)
        self.assertLessEqual(trace_error(next_rho), 1.0e-9)
        self.assertTrue(np.allclose(next_rho, np.conjugate(next_rho.T), atol=1.0e-10))

    def test_dephasing_reduces_coherence_norm(self):
        psi = gaussian_wavepacket(L=12, x0=6, sigma=1.5, k0=0.4)
        rho = psi_to_rho(psi)
        H = build_tight_binding_hamiltonian(size=12, hopping=1.0, disorder_strength=0.0, bias=0.0, seed=1)
        next_rho = rk4_lindblad_step(rho, H, 0.8, 0.1)
        self.assertLess(coherence_norm(next_rho), coherence_norm(rho))

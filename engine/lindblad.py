"""Density-matrix Lindblad helpers for the Kinetic Transition Atlas."""

from __future__ import annotations

import numpy as np


def psi_to_rho(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=np.complex128)
    return np.outer(psi, np.conjugate(psi)).astype(np.complex128, copy=False)


def lindblad_dephasing_rhs(
    rho: np.ndarray,
    H: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Site-basis Lindblad dephasing RHS preserving trace analytically."""

    if gamma < 0:
        raise ValueError("gamma must be non-negative.")

    rho = np.asarray(rho, dtype=np.complex128)
    H = np.asarray(H, dtype=np.complex128)
    commutator = H @ rho - rho @ H
    coherent = -1j * commutator
    diagonal_part = np.diag(np.diag(rho))
    dephasing = np.float64(gamma) * (diagonal_part - rho)
    return (coherent + dephasing).astype(np.complex128, copy=False)


def rk4_lindblad_step(
    rho: np.ndarray,
    H: np.ndarray,
    gamma: float,
    dt: float,
) -> np.ndarray:
    if dt <= 0:
        raise ValueError("dt must be positive.")

    dt64 = np.float64(dt)
    k1 = lindblad_dephasing_rhs(rho, H, gamma)
    k2 = lindblad_dephasing_rhs(rho + 0.5 * dt64 * k1, H, gamma)
    k3 = lindblad_dephasing_rhs(rho + 0.5 * dt64 * k2, H, gamma)
    k4 = lindblad_dephasing_rhs(rho + dt64 * k3, H, gamma)

    out = rho + (dt64 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    out = 0.5 * (out + np.conjugate(out.T))

    trace = np.trace(out)
    if abs(trace) > 0.0:
        out = out / trace

    return out.astype(np.complex128, copy=False)


def trace_error(rho: np.ndarray) -> float:
    return float(abs(np.trace(np.asarray(rho, dtype=np.complex128)) - 1.0))


def coherence_norm(rho: np.ndarray) -> float:
    offdiag = np.asarray(rho, dtype=np.complex128) - np.diag(np.diag(rho))
    return float(np.linalg.norm(offdiag))

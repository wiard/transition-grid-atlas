"""Wave-packet initializers for the Kinetic Transition Atlas."""

from __future__ import annotations

import numpy as np


def gaussian_wavepacket(
    L: int,
    x0: float,
    sigma: float,
    k0: float,
    *,
    dtype=np.complex128,
) -> np.ndarray:
    """Build a normalized Gaussian wave packet on a 1D lattice."""

    if L <= 0:
        raise ValueError("L must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    x = np.arange(L, dtype=np.float64)
    envelope = np.exp(-((x - np.float64(x0)) ** 2) / (2.0 * np.float64(sigma) ** 2))
    phase = np.exp(1j * np.float64(k0) * x)
    psi = np.asarray(envelope * phase, dtype=dtype)
    norm = np.linalg.norm(psi)

    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Cannot normalize degenerate wave packet.")

    return (psi / norm).astype(np.complex128, copy=False)


def make_gaussian_packet(L: int, x0: float, sigma: float, k0: float) -> np.ndarray:
    """Backward-compatible alias for older lab callers."""

    return gaussian_wavepacket(L=L, x0=x0, sigma=sigma, k0=k0)

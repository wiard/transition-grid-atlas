"""Feedback operators for self-consistent transport experiments.

The feedback model here is intentionally simple and deterministic: the
instantaneous probability density is converted into a smoothed, centered
on-site potential. Because the potential is purely real and diagonal, it is
Hermitian and can be safely added to the base Hamiltonian.
"""

from __future__ import annotations

import numpy as np


def compute_density_feedback_profile(
    state: np.ndarray,
    eta: float,
    smoothing: float = 1.0,
) -> np.ndarray:
    """Return a real-valued feedback profile derived from the state density."""

    density = np.abs(state) ** 2
    density = density.astype(np.float64, copy=False)
    centered = density - np.mean(density, dtype=np.float64)

    if smoothing > 0.0 and centered.size >= 3:
        kernel = np.array([0.25, 0.50, 0.25], dtype=np.float64)
        smoothed = centered.copy()
        passes = max(1, int(round(smoothing)))
        for _ in range(passes):
            smoothed = np.convolve(smoothed, kernel, mode="same").astype(np.float64)
        centered = smoothed

    return np.float64(eta) * centered


def feedback_potential(state: np.ndarray, eta: float, smoothing: float = 1.0) -> np.ndarray:
    """Create a Hermitian diagonal matrix from the feedback profile."""

    profile = compute_density_feedback_profile(state=state, eta=eta, smoothing=smoothing)
    return np.diag(profile.astype(np.complex128))

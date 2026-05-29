"""Hamiltonian builders for Transition Grid Atlas.

The default model is a one-dimensional tight-binding transport chain with:

- nearest-neighbour hopping
- static on-site disorder
- a deterministic linear bias field

This file intentionally focuses only on model generation, not on validation,
plotting, or exploration logic.
"""

from __future__ import annotations

import numpy as np


def build_tight_binding_hamiltonian(
    size: int,
    hopping: float,
    disorder_strength: float,
    bias: float,
    seed: int,
) -> np.ndarray:
    """Construct a Hermitian tight-binding Hamiltonian.

    The on-site potential is:

    ``V_i = disorder_i + bias * x_i``

    where ``x_i`` is a centered coordinate on the grid and ``disorder_i`` is
    drawn deterministically from a seeded uniform distribution.
    """

    if size < 2:
        raise ValueError("grid size must be at least 2")

    rng = np.random.default_rng(seed)
    centered_positions = np.linspace(-1.0, 1.0, size, dtype=np.float64)
    disorder = rng.uniform(
        low=-0.5 * disorder_strength,
        high=0.5 * disorder_strength,
        size=size,
    ).astype(np.float64)
    onsite = disorder + np.float64(bias) * centered_positions

    hamiltonian = np.diag(onsite.astype(np.complex128))
    for index in range(size - 1):
        hamiltonian[index, index + 1] = np.complex128(-hopping)
        hamiltonian[index + 1, index] = np.complex128(-hopping)
    return hamiltonian.astype(np.complex128, copy=False)


def build_initial_state(size: int, initial_site: int | str = "center") -> np.ndarray:
    """Create a normalized delta-localized initial state vector."""

    if initial_site == "center":
        site = size // 2
    else:
        site = int(initial_site)
    if site < 0 or site >= size:
        raise ValueError(f"initial site {site} is outside grid of size {size}")

    state = np.zeros(size, dtype=np.complex128)
    state[site] = np.complex128(1.0)
    return state

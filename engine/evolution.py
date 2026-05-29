"""Time evolution for Transition Grid Atlas.

This module computes discrete-time propagation. For ``gamma = 0`` it produces
strictly unitary state updates from ``U = exp(-i H dt)``. For ``gamma > 0`` it
applies an additional deterministic attenuation operator after each step. That
attenuation is intentionally non-unitary and should be caught by the validation
layer as ``INVALID_NONUNITARY``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import expm

from engine.feedback import feedback_potential


def simulate_dynamics(
    base_hamiltonian: np.ndarray,
    initial_state: np.ndarray,
    dt: float,
    steps: int,
    eta: float,
    gamma: float,
    feedback_smoothing: float = 1.0,
) -> dict[str, Any]:
    """Simulate a transport trajectory and accumulate the total propagator."""

    size = base_hamiltonian.shape[0]
    centered_positions = np.abs(np.arange(size, dtype=np.float64) - np.float64((size - 1) / 2.0))

    normalized_state = initial_state.astype(np.complex128, copy=True)
    raw_state = normalized_state.copy()
    trajectory = [normalized_state.copy()]
    total_operator = np.eye(size, dtype=np.complex128)
    step_hamiltonians: list[np.ndarray] = []

    for _ in range(steps):
        step_hamiltonian = base_hamiltonian + feedback_potential(
            state=normalized_state,
            eta=eta,
            smoothing=feedback_smoothing,
        )
        unitary_step = expm((-1j) * step_hamiltonian * np.float64(dt)).astype(np.complex128)
        effective_step = unitary_step

        if gamma > 0.0:
            damping_profile = np.exp((-np.float64(gamma)) * np.float64(dt) * centered_positions).astype(np.float64)
            damping_operator = np.diag(damping_profile.astype(np.complex128))
            effective_step = damping_operator @ unitary_step

        total_operator = effective_step @ total_operator
        raw_state = effective_step @ raw_state

        state_norm = np.linalg.norm(raw_state)
        if state_norm <= 0.0:
            normalized_state = np.zeros_like(raw_state)
        else:
            normalized_state = (raw_state / state_norm).astype(np.complex128)

        trajectory.append(normalized_state.copy())
        step_hamiltonians.append(step_hamiltonian.astype(np.complex128))

    return {
        "trajectory": trajectory,
        "total_operator": total_operator.astype(np.complex128),
        "step_hamiltonians": step_hamiltonians,
    }

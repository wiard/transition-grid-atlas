"""Hermiticity validation helpers."""

from __future__ import annotations

import numpy as np


def hermitian_error(matrix: np.ndarray) -> float:
    """Return ||H - H^†||_inf for a candidate Hamiltonian."""

    deviation = matrix - matrix.conjugate().T
    return float(np.max(np.abs(deviation)))

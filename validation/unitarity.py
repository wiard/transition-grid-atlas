"""Unitary validation helpers."""

from __future__ import annotations

import numpy as np


def unitarity_error(operator: np.ndarray) -> float:
    """Return ||U U^† - I||_inf as a strict numerical sanity check."""

    identity = np.eye(operator.shape[0], dtype=np.complex128)
    deviation = operator @ operator.conjugate().T - identity
    return float(np.max(np.abs(deviation)))

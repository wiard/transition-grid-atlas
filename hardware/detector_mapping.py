"""Detector-array interpretations for KTA trajectory outputs."""

from __future__ import annotations

import numpy as np


def detector_probabilities_from_trajectory(probabilities: np.ndarray) -> np.ndarray:
    """Return the final output-channel distribution from a KTA trajectory."""

    array = np.asarray(probabilities, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("probabilities must have shape (n_times, n_sites)")
    return array[-1].astype(np.float64, copy=False)


def transport_efficiency_to_targets(
    final_probabilities: np.ndarray,
    target_indices: list[int],
) -> float:
    """Sum probability arriving at selected detector/output channels."""

    array = np.asarray(final_probabilities, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("final_probabilities must be one-dimensional")
    if not target_indices:
        return 0.0
    for index in target_indices:
        if index < 0 or index >= array.size:
            raise IndexError(f"target index out of range: {index}")
    return float(np.sum(array[target_indices], dtype=np.float64))

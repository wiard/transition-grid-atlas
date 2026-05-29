"""Monte Carlo summary helpers for Transition Grid Atlas."""

from __future__ import annotations

import numpy as np
from scipy import stats


def confidence_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Return a Student-t confidence interval around the sample mean."""

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return (float("nan"), float("nan"))
    if values.size == 1:
        value = float(values[0])
        return (value, value)

    mean = float(np.mean(values, dtype=np.float64))
    sem = stats.sem(values, ddof=1)
    interval = stats.t.interval(confidence, values.size - 1, loc=mean, scale=sem)
    return (float(interval[0]), float(interval[1]))


def summarise_samples(values: np.ndarray) -> dict[str, float]:
    """Return mean, standard deviation, and 95% confidence interval bounds."""

    values = np.asarray(values, dtype=np.float64)
    ci_low, ci_high = confidence_interval(values)
    return {
        "mean": float(np.mean(values, dtype=np.float64)),
        "std": float(np.std(values, dtype=np.float64, ddof=0)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }

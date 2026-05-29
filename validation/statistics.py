"""Statistical fit quality helpers."""

from __future__ import annotations

import numpy as np
from scipy import stats


def linear_regression_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Return slope, intercept, R², and RMSE for a linear fit."""

    if x.size < 2 or y.size < 2:
        return {"slope": 0.0, "intercept": 0.0, "r2": 0.0, "rmse": float("inf")}

    regression = stats.linregress(x.astype(np.float64), y.astype(np.float64))
    predicted = regression.intercept + regression.slope * x
    residual = y - predicted
    rmse = np.sqrt(np.mean(residual**2, dtype=np.float64))
    return {
        "slope": float(regression.slope),
        "intercept": float(regression.intercept),
        "r2": float(regression.rvalue**2),
        "rmse": float(rmse),
    }


def two_window_scaling_metrics(
    x: np.ndarray,
    y: np.ndarray,
    relative_shift_tolerance: float = 0.10,
) -> dict[str, float | bool]:
    """Compare early- and late-time scaling slopes for transient filtering.

    The input vectors are assumed to already represent the log-log transport fit
    window. The series is split into two halves and a linear fit is computed on
    each half. If the relative slope shift exceeds 10%, the run should be
    treated as a transient-dominated weak fit rather than a stable transport
    regime.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 4 or y.size < 4:
        return {
            "early_alpha": 0.0,
            "late_alpha": 0.0,
            "early_r2": 0.0,
            "late_r2": 0.0,
            "relative_shift": float("inf"),
            "stable": False,
        }

    midpoint = x.size // 2
    early_metrics = linear_regression_metrics(x[:midpoint], y[:midpoint])
    late_metrics = linear_regression_metrics(x[midpoint:], y[midpoint:])

    early_alpha = float(early_metrics["slope"])
    late_alpha = float(late_metrics["slope"])
    denominator = max(abs(early_alpha), abs(late_alpha), np.float64(1.0e-12))
    relative_shift = abs(early_alpha - late_alpha) / denominator

    return {
        "early_alpha": early_alpha,
        "late_alpha": late_alpha,
        "early_r2": float(early_metrics["r2"]),
        "late_r2": float(late_metrics["r2"]),
        "relative_shift": float(relative_shift),
        "stable": bool(relative_shift <= np.float64(relative_shift_tolerance)),
    }

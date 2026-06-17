"""Observable extraction and run classification for Transition Grid Atlas."""

from __future__ import annotations

from typing import Any

import numpy as np

from validation.statistics import linear_regression_metrics, two_window_scaling_metrics


def _position_grid(size: int) -> np.ndarray:
    return np.arange(size, dtype=np.float64)


def probability_from_psi(psi: np.ndarray) -> np.ndarray:
    """Return a normalized probability distribution from a wavefunction."""

    p = np.abs(np.asarray(psi, dtype=np.complex128)) ** 2
    total = float(np.sum(p, dtype=np.float64))
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("Invalid probability mass.")
    return (p / total).astype(np.float64, copy=False)


def probability_from_rho(rho: np.ndarray) -> np.ndarray:
    """Return a normalized site-basis probability distribution from rho."""

    p = np.real(np.diag(np.asarray(rho, dtype=np.complex128)))
    p = np.maximum(p, 0.0).astype(np.float64, copy=False)
    total = float(np.sum(p, dtype=np.float64))
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("Invalid density-matrix probability mass.")
    return (p / total).astype(np.float64, copy=False)


def x_mean(p: np.ndarray) -> float:
    x = np.arange(len(p), dtype=np.float64)
    return float(np.sum(x * np.asarray(p, dtype=np.float64), dtype=np.float64))


def x_variance(p: np.ndarray) -> float:
    x = np.arange(len(p), dtype=np.float64)
    mu = x_mean(p)
    return float(np.sum(((x - mu) ** 2) * np.asarray(p, dtype=np.float64), dtype=np.float64))


def ipr(p: np.ndarray) -> float:
    """Inverse participation ratio. Higher means more localized."""

    probabilities = np.asarray(p, dtype=np.float64)
    return float(np.sum(probabilities**2, dtype=np.float64))


def edge_contact(p: np.ndarray, edge_width: int = 4, threshold: float = 1.0e-3) -> bool:
    if edge_width <= 0:
        raise ValueError("edge_width must be positive.")
    probabilities = np.asarray(p, dtype=np.float64)
    mass = float(np.sum(probabilities[:edge_width], dtype=np.float64) + np.sum(probabilities[-edge_width:], dtype=np.float64))
    return bool(mass > threshold)


def frame_correlation(p_prev: np.ndarray, p_next: np.ndarray) -> float:
    a = np.asarray(p_prev, dtype=np.float64) - np.mean(p_prev, dtype=np.float64)
    b = np.asarray(p_next, dtype=np.float64) - np.mean(p_next, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_mean_square_displacement(trajectory: list[np.ndarray]) -> np.ndarray:
    """Compute the MSD of a normalized state trajectory."""

    size = trajectory[0].shape[0]
    positions = _position_grid(size)
    initial_density = probability_from_psi(trajectory[0])
    origin = np.sum(positions * initial_density, dtype=np.float64)

    msd_values = []
    for state in trajectory:
        density = probability_from_psi(state)
        msd = np.sum(((positions - origin) ** 2) * density, dtype=np.float64)
        msd_values.append(np.float64(msd))
    return np.array(msd_values, dtype=np.float64)


def compute_current_series(trajectory: list[np.ndarray], hopping: float) -> np.ndarray:
    """Estimate the mean absolute nearest-neighbour probability current."""

    currents = []
    hopping64 = np.float64(hopping)
    for state in trajectory:
        local_currents = 2.0 * hopping64 * np.imag(np.conjugate(state[:-1]) * state[1:])
        currents.append(np.mean(np.abs(local_currents), dtype=np.float64))
    return np.array(currents, dtype=np.float64)


def compute_transport_metrics(
    trajectory: list[np.ndarray],
    dt: float,
    hopping: float,
    fit_start_step: int,
    fit_end_step: int | None,
) -> dict[str, Any]:
    """Compute transport observables, scaling exponent alpha, and fit quality."""

    msd = compute_mean_square_displacement(trajectory)
    sigma = np.sqrt(np.clip(msd, a_min=np.float64(1.0e-18), a_max=None)).astype(np.float64)
    times = (np.arange(len(trajectory), dtype=np.float64) * np.float64(dt)).astype(np.float64)
    current_series = compute_current_series(trajectory=trajectory, hopping=hopping)

    fit_stop = fit_end_step if fit_end_step is not None else len(times)
    fit_slice = slice(fit_start_step, fit_stop)
    fit_times = times[fit_slice]
    fit_sigma = sigma[fit_slice]

    positive_mask = fit_times > 0.0
    fit_times = fit_times[positive_mask]
    fit_sigma = fit_sigma[positive_mask]

    regression = linear_regression_metrics(
        x=np.log(fit_times),
        y=np.log(np.clip(fit_sigma, a_min=np.float64(1.0e-18), a_max=None)),
    )
    window_metrics = two_window_scaling_metrics(
        x=np.log(fit_times),
        y=np.log(np.clip(fit_sigma, a_min=np.float64(1.0e-18), a_max=None)),
    )

    return {
        "times": times,
        "msd": msd,
        "sigma": sigma,
        "current_series": current_series,
        "mean_current": float(np.mean(current_series, dtype=np.float64)),
        "alpha": float(regression["slope"]),
        "r2": float(regression["r2"]),
        "regression_rmse": float(regression["rmse"]),
        "early_alpha": float(window_metrics["early_alpha"]),
        "late_alpha": float(window_metrics["late_alpha"]),
        "early_r2": float(window_metrics["early_r2"]),
        "late_r2": float(window_metrics["late_r2"]),
        "alpha_window_shift": float(window_metrics["relative_shift"]),
        "scaling_window_stable": bool(window_metrics["stable"]),
    }


def classify_transport_regime(
    alpha: float,
    r2: float,
    unitarity_error_value: float,
    hermitian_error_value: float,
    scaling_window_stable: bool,
    tolerance: float = 1.0e-12,
) -> str:
    """Map continuous metrics into strict atlas classifications."""

    if hermitian_error_value >= tolerance:
        return "INVALID_NONHERMITIAN"
    if unitarity_error_value >= tolerance:
        return "INVALID_NONUNITARY"
    if r2 <= 0.90:
        return "WEAK_FIT"
    if not scaling_window_stable:
        return "WEAK_FIT"
    if alpha >= 0.75:
        return "VALID_BALLISTIC"
    if alpha <= 0.25:
        return "VALID_LOCALIZED"
    if 0.45 < alpha < 0.55 and r2 >= 0.95:
        return "VALID_DIFFUSIVE_CANDIDATE"
    return "WEAK_FIT"

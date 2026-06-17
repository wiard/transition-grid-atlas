"""Kinetic Transition Atlas audit helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from engine.lab import LabConfig
from engine.observables import edge_contact, frame_correlation, ipr, x_mean, x_variance
from validation.statistics import linear_regression_metrics


def late_window_metrics(probability_frames: np.ndarray, burn_in_fraction: float) -> dict[str, float]:
    frames = np.asarray(probability_frames, dtype=np.float64)
    if frames.shape[0] < 2:
        return {"frame_corr_late": 0.0, "x_var_drift_late": 0.0}
    corr = np.array([frame_correlation(frames[i], frames[i + 1]) for i in range(frames.shape[0] - 1)], dtype=np.float64)
    x_vars = np.array([x_variance(frame) for frame in frames], dtype=np.float64)
    burn_in_index = min(max(1, int(round(burn_in_fraction * frames.shape[0]))), frames.shape[0] - 1)
    return {
        "frame_corr_late": float(np.mean(corr[burn_in_index - 1 :], dtype=np.float64)),
        "x_var_drift_late": float(np.max(x_vars[burn_in_index:]) - x_vars[burn_in_index]),
    }


def summarize_kta_audit(
    *,
    probability_frames: np.ndarray,
    times: np.ndarray,
    trace_series: np.ndarray,
    x0: float,
    config: LabConfig,
    coherence_norm_final: float = 0.0,
) -> dict[str, Any]:
    frames = np.asarray(probability_frames, dtype=np.float64)
    positions = np.arange(frames.shape[1], dtype=np.float64)
    x_means = np.array([x_mean(frame) for frame in frames], dtype=np.float64)
    x_vars = np.array([x_variance(frame) for frame in frames], dtype=np.float64)
    ipr_series = np.array([ipr(frame) for frame in frames], dtype=np.float64)
    edge_hits = np.array([edge_contact(frame, edge_width=config.edge_width, threshold=config.edge_threshold) for frame in frames], dtype=bool)
    msd_origin = np.sum(((positions[None, :] - np.float64(x0)) ** 2) * frames, axis=1, dtype=np.float64)
    positive_mask = np.asarray(times, dtype=np.float64) > 0.0
    fit_times = np.asarray(times, dtype=np.float64)[positive_mask]
    fit_sigma = np.sqrt(np.clip(msd_origin[positive_mask], a_min=np.float64(1.0e-18), a_max=None))
    split = max(1, fit_times.size // 2)
    if fit_times.size >= 4:
        regression = linear_regression_metrics(np.log(fit_times[split:]), np.log(fit_sigma[split:]))
        alpha_late = float(regression["slope"])
        r2_late = float(regression["r2"])
    else:
        alpha_late = 0.0
        r2_late = 0.0

    late = late_window_metrics(frames, config.burn_in_fraction)
    edge_hit_any = bool(np.any(edge_hits))
    falldown_candidate = bool(
        not edge_hit_any
        and late["x_var_drift_late"] < config.falldown_variance_epsilon
        and late["frame_corr_late"] > config.frame_corr_threshold
    )
    falldown_score = max(
        0.0,
        late["frame_corr_late"]
        - late["x_var_drift_late"]
        - (1.0 if edge_hit_any else 0.0),
    )
    zeno_indicator = bool(config.gamma > 0.5 and alpha_late < 0.1)

    return {
        "x_mean": x_means,
        "x_var": x_vars,
        "ipr": ipr_series,
        "msd_origin": msd_origin.astype(np.float64, copy=False),
        "edge_hits": edge_hits,
        "edge_hit": edge_hit_any,
        "trace_error": float(np.max(np.abs(np.asarray(trace_series, dtype=np.float64)))),
        "ipr_final": float(ipr_series[-1]),
        "alpha_late": alpha_late,
        "r2_late": r2_late,
        "frame_corr_late": float(late["frame_corr_late"]),
        "x_var_drift_late": float(late["x_var_drift_late"]),
        "falldown_candidate": falldown_candidate,
        "falldown_score": float(falldown_score),
        "coherence_norm_final": float(coherence_norm_final),
        "zeno_indicator": zeno_indicator,
    }

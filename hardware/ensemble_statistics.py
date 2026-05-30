"""Statistical audit utilities for the transition-motor ensemble."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REQUIRED_COLUMNS = {
    "detector_success_gain",
    "transport_gain",
    "noise_action_reduction",
    "noise_leakage_reduction",
    "objective_gain",
    "saturated_knobs",
    "near_bound_knobs",
}


@dataclass(frozen=True)
class EnsembleWinRates:
    n_samples: int
    detector_win_rate: float
    noise_action_win_rate: float
    leakage_win_rate: float
    objective_win_rate: float
    detector_and_noise_win_rate: float
    all_core_metrics_win_rate: float


@dataclass(frozen=True)
class BootstrapCI:
    metric: str
    mean: float
    median: float
    ci_low: float
    ci_high: float
    n_bootstrap: int


@dataclass(frozen=True)
class BoundPressureSummary:
    mean_saturated_knobs: float
    median_saturated_knobs: float
    max_saturated_knobs: float
    mean_near_bound_knobs: float
    median_near_bound_knobs: float
    max_near_bound_knobs: float


@dataclass(frozen=True)
class TradeoffSummary:
    detector_vs_noise_corr: float
    detector_vs_leakage_corr: float
    objective_vs_detector_corr: float


def read_ensemble_csv(path: str | Path) -> list[dict[str, float]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"ensemble CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("ensemble CSV has no header")
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames))
        if missing:
            raise ValueError(f"ensemble CSV missing required columns: {missing}")
        rows: list[dict[str, float]] = []
        for raw_row in reader:
            parsed: dict[str, float] = {}
            for key, value in raw_row.items():
                if value is None or value == "":
                    raise ValueError(f"ensemble CSV contains empty value in column {key}")
                if str(value).lower() in {"true", "false"}:
                    parsed[key] = 1.0 if str(value).lower() == "true" else 0.0
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    if not rows:
        raise ValueError("ensemble CSV contains no data rows")
    return rows


def compute_win_rates(rows: list[dict[str, float]], *, tolerance: float = 0.0) -> EnsembleWinRates:
    if not rows:
        raise ValueError("rows must be non-empty")
    detector_wins = np.array([row["detector_success_gain"] > tolerance for row in rows], dtype=np.float64)
    noise_wins = np.array([row["noise_action_reduction"] > tolerance for row in rows], dtype=np.float64)
    leakage_wins = np.array([row["noise_leakage_reduction"] > tolerance for row in rows], dtype=np.float64)
    objective_wins = np.array([row["objective_gain"] > tolerance for row in rows], dtype=np.float64)
    detector_and_noise = detector_wins * noise_wins
    all_core = detector_wins * noise_wins * objective_wins
    return EnsembleWinRates(
        n_samples=len(rows),
        detector_win_rate=float(np.mean(detector_wins)),
        noise_action_win_rate=float(np.mean(noise_wins)),
        leakage_win_rate=float(np.mean(leakage_wins)),
        objective_win_rate=float(np.mean(objective_wins)),
        detector_and_noise_win_rate=float(np.mean(detector_and_noise)),
        all_core_metrics_win_rate=float(np.mean(all_core)),
    )


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_bootstrap: int = 5000,
    ci: float = 0.95,
    seed: int = 123,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("values must be non-empty")
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sample = rng.choice(array, size=array.size, replace=True)
        boot_means[index] = float(np.mean(sample))
    alpha = (1.0 - ci) / 2.0
    return (
        float(np.quantile(boot_means, alpha)),
        float(np.quantile(boot_means, 1.0 - alpha)),
    )


def compute_bootstrap_summary(
    rows: list[dict[str, float]],
    metrics: list[str],
    *,
    n_bootstrap: int = 5000,
    seed: int = 123,
) -> list[BootstrapCI]:
    if not rows:
        raise ValueError("rows must be non-empty")
    summaries: list[BootstrapCI] = []
    for offset, metric in enumerate(metrics):
        values = np.array([row[metric] for row in rows], dtype=np.float64)
        ci_low, ci_high = bootstrap_mean_ci(values, n_bootstrap=n_bootstrap, seed=seed + offset)
        summaries.append(
            BootstrapCI(
                metric=metric,
                mean=float(np.mean(values)),
                median=float(np.median(values)),
                ci_low=ci_low,
                ci_high=ci_high,
                n_bootstrap=n_bootstrap,
            )
        )
    return summaries


def compute_bound_pressure_summary(rows: list[dict[str, float]]) -> BoundPressureSummary:
    if not rows:
        raise ValueError("rows must be non-empty")
    saturated = np.array([row["saturated_knobs"] for row in rows], dtype=np.float64)
    near = np.array([row["near_bound_knobs"] for row in rows], dtype=np.float64)
    return BoundPressureSummary(
        mean_saturated_knobs=float(np.mean(saturated)),
        median_saturated_knobs=float(np.median(saturated)),
        max_saturated_knobs=float(np.max(saturated)),
        mean_near_bound_knobs=float(np.mean(near)),
        median_near_bound_knobs=float(np.median(near)),
        max_near_bound_knobs=float(np.max(near)),
    )


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.size == 0 or y_array.size == 0 or x_array.size != y_array.size:
        raise ValueError("x and y must be non-empty arrays with equal length")
    if float(np.std(x_array)) <= 1.0e-15 or float(np.std(y_array)) <= 1.0e-15:
        return 0.0
    corr = float(np.corrcoef(x_array, y_array)[0, 1])
    return float(np.clip(corr, -1.0, 1.0))


def compute_tradeoff_summary(rows: list[dict[str, float]]) -> TradeoffSummary:
    if not rows:
        raise ValueError("rows must be non-empty")
        # unreachable but keeps intent explicit
    detector = np.array([row["detector_success_gain"] for row in rows], dtype=np.float64)
    noise = np.array([row["noise_action_reduction"] for row in rows], dtype=np.float64)
    leakage = np.array([row["noise_leakage_reduction"] for row in rows], dtype=np.float64)
    objective = np.array([row["objective_gain"] for row in rows], dtype=np.float64)
    return TradeoffSummary(
        detector_vs_noise_corr=safe_corrcoef(detector, noise),
        detector_vs_leakage_corr=safe_corrcoef(detector, leakage),
        objective_vs_detector_corr=safe_corrcoef(objective, detector),
    )


def plot_ensemble_statistics(rows: list[dict[str, float]], output_path: str | Path) -> Path:
    if not rows:
        raise ValueError("rows must be non-empty")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    detector = np.array([row["detector_success_gain"] for row in rows], dtype=np.float64)
    noise = np.array([row["noise_action_reduction"] for row in rows], dtype=np.float64)
    saturated = np.array([row["saturated_knobs"] for row in rows], dtype=np.float64)
    near = np.array([row["near_bound_knobs"] for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Synthetic Transition Motor Ensemble Statistics", fontsize=13)

    axes[0, 0].hist(detector, bins=min(12, max(5, len(detector) // 2)), color="#2a6f97", alpha=0.85)
    axes[0, 0].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0, 0].set_title("Detector Success Gain")
    axes[0, 0].set_xlabel("gain")
    axes[0, 0].set_ylabel("count")

    axes[0, 1].hist(noise, bins=min(12, max(5, len(noise) // 2)), color="#52b788", alpha=0.85)
    axes[0, 1].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0, 1].set_title("Noise-Action Reduction")
    axes[0, 1].set_xlabel("reduction")
    axes[0, 1].set_ylabel("count")

    axes[1, 0].scatter(detector, noise, color="#bc4749", alpha=0.8)
    axes[1, 0].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1, 0].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1, 0].set_title("Detector Gain vs Noise-Action Reduction")
    axes[1, 0].set_xlabel("detector_success_gain")
    axes[1, 0].set_ylabel("noise_action_reduction")

    bins = np.arange(0, max(float(np.max(saturated)), float(np.max(near))) + 2.0) - 0.5
    axes[1, 1].hist(saturated, bins=bins, alpha=0.7, label="saturated_knobs", color="#6d597a")
    axes[1, 1].hist(near, bins=bins, alpha=0.5, label="near_bound_knobs", color="#f4a261")
    axes[1, 1].set_title("Bound Pressure")
    axes[1, 1].set_xlabel("count")
    axes[1, 1].set_ylabel("samples")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_statistics_json(
    path: str | Path,
    *,
    win_rates: EnsembleWinRates,
    bootstrap: list[BootstrapCI],
    bound_pressure: BoundPressureSummary,
    tradeoffs: TradeoffSummary,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "win_rates": asdict(win_rates),
        "bootstrap": [asdict(item) for item in bootstrap],
        "bound_pressure": asdict(bound_pressure),
        "tradeoffs": asdict(tradeoffs),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output

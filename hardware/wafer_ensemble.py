"""Synthetic wafer ensemble study for transition-dynamics tuning."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hardware.subspaces import diagonal_phase_noise
from hardware.transition_tuner import (
    FixedGrid,
    TransitionTunerConfig,
    _normalize_edge,
    apply_transition_controls,
    build_base_hamiltonian,
    evaluate_candidate,
    fixed_grid_from_dict,
    transition_tuner_config_from_dict,
)


@dataclass(frozen=True)
class EnsembleSampleResult:
    sample_id: int
    baseline_transport_efficiency: float
    best_transport_efficiency: float
    baseline_noise_overlap: float
    best_noise_overlap: float
    baseline_suppression_score: float
    best_suppression_score: float
    baseline_objective: float
    best_objective: float
    transport_gain: float
    noise_overlap_reduction: float
    objective_gain: float
    success: bool


@dataclass(frozen=True)
class EnsembleSummary:
    n_samples: int
    success_rate: float
    mean_transport_gain: float
    median_transport_gain: float
    mean_noise_overlap_reduction: float
    median_noise_overlap_reduction: float
    mean_objective_gain: float
    median_objective_gain: float
    mean_baseline_transport_efficiency: float
    mean_best_transport_efficiency: float
    mean_baseline_noise_overlap: float
    mean_best_noise_overlap: float


def wafer_ensemble_from_dict(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "grid": fixed_grid_from_dict(dict(data["grid"])),
        "fabrication_disorder": {
            "n_samples": int(data["fabrication_disorder"]["n_samples"]),
            "onsite_sigma": float(data["fabrication_disorder"]["onsite_sigma"]),
            "correlation_length_sites": float(data["fabrication_disorder"]["correlation_length_sites"]),
            "seed": int(data["fabrication_disorder"]["seed"]),
        },
        "phase_noise": {
            "n_profiles_per_sample": int(data["phase_noise"]["n_profiles_per_sample"]),
            "profile_sigma": float(data["phase_noise"]["profile_sigma"]),
            "correlation_length_sites": float(data["phase_noise"]["correlation_length_sites"]),
            "seed": int(data["phase_noise"]["seed"]),
        },
        "tuner": transition_tuner_config_from_dict(dict(data["tuner"])),
        "outputs": {
            "csv_path": str(data["outputs"]["csv_path"]),
            "summary_path": str(data["outputs"]["summary_path"]),
        },
    }


def sample_correlated_profile(
    n_sites: int,
    sigma: float,
    correlation_length_sites: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a smooth correlated 1D profile using only numpy."""

    if n_sites <= 0:
        raise ValueError("n_sites must be positive")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    if correlation_length_sites < 0.0:
        raise ValueError("correlation_length_sites must be non-negative")
    if sigma == 0.0:
        return np.zeros(n_sites, dtype=np.float64)

    white = rng.normal(0.0, sigma, size=n_sites).astype(np.float64, copy=False)
    if correlation_length_sites == 0.0:
        return white

    radius = max(1, int(np.ceil(3.0 * correlation_length_sites)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2.0 * correlation_length_sites**2))
    kernel /= np.sum(kernel)
    full = np.convolve(white, kernel, mode="full")
    start = max(0, (full.shape[0] - n_sites) // 2)
    correlated = full[start : start + n_sites]
    std = float(np.std(correlated))
    if std > 1.0e-12:
        correlated *= sigma / std
    return correlated.astype(np.float64, copy=False)


def sample_fabrication_disorder_profiles(
    n_samples: int,
    n_sites: int,
    onsite_sigma: float,
    correlation_length_sites: float,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        sample_correlated_profile(n_sites, onsite_sigma, correlation_length_sites, rng)
        for _ in range(n_samples)
    ]


def sample_phase_noise_profiles(
    n_profiles: int,
    n_sites: int,
    profile_sigma: float,
    correlation_length_sites: float,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        sample_correlated_profile(n_sites, profile_sigma, correlation_length_sites, rng)
        for _ in range(n_profiles)
    ]


def apply_fabrication_disorder(H: np.ndarray, onsite_disorder: np.ndarray) -> np.ndarray:
    """Add fabrication disorder as onsite propagation-constant shifts only."""

    operator = np.asarray(H, dtype=np.complex128)
    disorder = np.asarray(onsite_disorder, dtype=np.float64)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("H must be a square matrix")
    if disorder.shape != (operator.shape[0],):
        raise ValueError("onsite_disorder must match the Hamiltonian dimension")

    out = np.array(operator, copy=True)
    out[np.diag_indices(operator.shape[0])] += disorder.astype(np.complex128, copy=False)
    return 0.5 * (out + np.conjugate(out.T))


def _noise_ops_from_profiles(noise_profiles: list[np.ndarray]) -> list[np.ndarray]:
    return [diagonal_phase_noise(np.asarray(profile, dtype=np.float64)) for profile in noise_profiles]


def run_single_wafer_sample(
    grid: FixedGrid,
    onsite_disorder: np.ndarray,
    noise_profiles: list[np.ndarray],
    tuner_config: TransitionTunerConfig,
    *,
    tolerance: float = 1e-9,
) -> EnsembleSampleResult:
    base_hamiltonian = build_base_hamiltonian(grid)
    disordered_hamiltonian = apply_fabrication_disorder(base_hamiltonian, onsite_disorder)
    noise_ops = _noise_ops_from_profiles(noise_profiles)

    baseline = evaluate_candidate(disordered_hamiltonian, noise_ops, grid, tuner_config)
    best = dict(baseline)

    rng = np.random.default_rng(tuner_config.seed)
    normalized_edges = [_normalize_edge(edge) for edge in grid.edges]

    for _ in range(tuner_config.n_candidates):
        coupling_scales = {
            edge: float(
                rng.uniform(
                    1.0 - tuner_config.max_relative_coupling_delta,
                    1.0 + tuner_config.max_relative_coupling_delta,
                )
            )
            for edge in normalized_edges
        }
        onsite_shifts = rng.uniform(
            -tuner_config.max_onsite_shift,
            tuner_config.max_onsite_shift,
            size=grid.n_sites,
        ).astype(np.float64, copy=False)
        candidate_hamiltonian = apply_transition_controls(
            disordered_hamiltonian,
            grid,
            coupling_scales,
            onsite_shifts,
        )
        candidate = evaluate_candidate(candidate_hamiltonian, noise_ops, grid, tuner_config)
        if candidate["objective"] > best["objective"] + 1.0e-12:
            best = candidate

    transport_gain = float(best["transport_efficiency"] - baseline["transport_efficiency"])
    noise_overlap_reduction = float(baseline["noise_overlap"] - best["noise_overlap"])
    objective_gain = float(best["objective"] - baseline["objective"])
    success = bool(
        objective_gain >= -tolerance
        and noise_overlap_reduction >= -tolerance
    )

    return EnsembleSampleResult(
        sample_id=-1,
        baseline_transport_efficiency=float(baseline["transport_efficiency"]),
        best_transport_efficiency=float(best["transport_efficiency"]),
        baseline_noise_overlap=float(baseline["noise_overlap"]),
        best_noise_overlap=float(best["noise_overlap"]),
        baseline_suppression_score=float(baseline["suppression_score"]),
        best_suppression_score=float(best["suppression_score"]),
        baseline_objective=float(baseline["objective"]),
        best_objective=float(best["objective"]),
        transport_gain=transport_gain,
        noise_overlap_reduction=noise_overlap_reduction,
        objective_gain=objective_gain,
        success=success,
    )


def summarize_ensemble(results: list[EnsembleSampleResult]) -> EnsembleSummary:
    if not results:
        raise ValueError("results must be non-empty")

    def column(name: str) -> np.ndarray:
        return np.array([getattr(result, name) for result in results], dtype=np.float64)

    success_rate = float(np.mean([1.0 if result.success else 0.0 for result in results], dtype=np.float64))
    transport_gain = column("transport_gain")
    overlap_reduction = column("noise_overlap_reduction")
    objective_gain = column("objective_gain")
    baseline_transport = column("baseline_transport_efficiency")
    best_transport = column("best_transport_efficiency")
    baseline_overlap = column("baseline_noise_overlap")
    best_overlap = column("best_noise_overlap")

    return EnsembleSummary(
        n_samples=len(results),
        success_rate=success_rate,
        mean_transport_gain=float(np.mean(transport_gain)),
        median_transport_gain=float(np.median(transport_gain)),
        mean_noise_overlap_reduction=float(np.mean(overlap_reduction)),
        median_noise_overlap_reduction=float(np.median(overlap_reduction)),
        mean_objective_gain=float(np.mean(objective_gain)),
        median_objective_gain=float(np.median(objective_gain)),
        mean_baseline_transport_efficiency=float(np.mean(baseline_transport)),
        mean_best_transport_efficiency=float(np.mean(best_transport)),
        mean_baseline_noise_overlap=float(np.mean(baseline_overlap)),
        mean_best_noise_overlap=float(np.mean(best_overlap)),
    )


def write_ensemble_csv(path: str | Path, results: list[EnsembleSampleResult]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(EnsembleSampleResult.__dataclass_fields__.keys())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    return output


def write_ensemble_summary_json(path: str | Path, summary: EnsembleSummary) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8")
    return output


def run_wafer_ensemble_study(config: dict[str, Any]) -> tuple[list[EnsembleSampleResult], EnsembleSummary, Path, Path]:
    ensemble = wafer_ensemble_from_dict(dict(config["wafer_ensemble"]))
    grid: FixedGrid = ensemble["grid"]
    fabrication = ensemble["fabrication_disorder"]
    phase_noise = ensemble["phase_noise"]
    tuner_config: TransitionTunerConfig = ensemble["tuner"]

    fabrication_profiles = sample_fabrication_disorder_profiles(
        fabrication["n_samples"],
        grid.n_sites,
        fabrication["onsite_sigma"],
        fabrication["correlation_length_sites"],
        fabrication["seed"],
    )

    results: list[EnsembleSampleResult] = []
    for sample_id, onsite_profile in enumerate(fabrication_profiles):
        noise_profiles = sample_phase_noise_profiles(
            phase_noise["n_profiles_per_sample"],
            grid.n_sites,
            phase_noise["profile_sigma"],
            phase_noise["correlation_length_sites"],
            phase_noise["seed"] + sample_id,
        )
        result = run_single_wafer_sample(grid, onsite_profile, noise_profiles, tuner_config)
        results.append(
            EnsembleSampleResult(
                sample_id=sample_id,
                baseline_transport_efficiency=result.baseline_transport_efficiency,
                best_transport_efficiency=result.best_transport_efficiency,
                baseline_noise_overlap=result.baseline_noise_overlap,
                best_noise_overlap=result.best_noise_overlap,
                baseline_suppression_score=result.baseline_suppression_score,
                best_suppression_score=result.best_suppression_score,
                baseline_objective=result.baseline_objective,
                best_objective=result.best_objective,
                transport_gain=result.transport_gain,
                noise_overlap_reduction=result.noise_overlap_reduction,
                objective_gain=result.objective_gain,
                success=result.success,
            )
        )

    summary = summarize_ensemble(results)
    csv_path = write_ensemble_csv(ensemble["outputs"]["csv_path"], results)
    summary_path = write_ensemble_summary_json(ensemble["outputs"]["summary_path"], summary)
    return results, summary, csv_path, summary_path

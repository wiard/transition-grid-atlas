"""Synthetic ensemble validation for the KTA transition motor."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hardware.control_basis import ControlBasis, make_control_basis
from hardware.control_knobs import KnobRegistry, knob_registry_from_dicts
from hardware.motor_audit import count_saturated_knobs, knob_bound_statuses
from hardware.objectives import MotorMetrics, evaluate_motor_metrics
from hardware.subspaces import diagonal_phase_noise
from hardware.transition_motor import (
    TransitionMotorConfig,
    build_control_hamiltonian,
    random_restart_transition_motor_search,
)
from hardware.transition_tuner import FixedGrid, build_base_hamiltonian, fixed_grid_from_dict


@dataclass(frozen=True)
class TransitionMotorEnsembleConfig:
    motor_config: TransitionMotorConfig
    fabrication_disorder: dict[str, float | int]
    phase_noise: dict[str, float | int]
    success_criteria: dict[str, float]
    outputs: dict[str, str]


@dataclass(frozen=True)
class MotorEnsembleSampleResult:
    sample_id: int
    baseline_detector_success: float
    best_detector_success: float
    baseline_detector_final: float
    best_detector_final: float
    baseline_transport_efficiency: float
    best_transport_efficiency: float
    baseline_noise_action_on_info: float
    best_noise_action_on_info: float
    baseline_noise_leakage: float
    best_noise_leakage: float
    baseline_control_cost: float
    best_control_cost: float
    baseline_objective: float
    best_objective: float
    detector_success_gain: float
    transport_gain: float
    noise_action_reduction: float
    noise_leakage_reduction: float
    objective_gain: float
    saturated_knobs: int
    near_bound_knobs: int
    success: bool


@dataclass(frozen=True)
class DetailedMotorEnsembleSample:
    sample: MotorEnsembleSampleResult
    baseline_theta: dict[str, float]
    best_theta: dict[str, float]
    baseline_metrics: MotorMetrics
    best_metrics: MotorMetrics


@dataclass(frozen=True)
class MotorEnsembleSummary:
    n_samples: int
    success_rate: float
    mean_detector_success_gain: float
    median_detector_success_gain: float
    mean_transport_gain: float
    median_transport_gain: float
    mean_noise_action_reduction: float
    median_noise_action_reduction: float
    mean_noise_leakage_reduction: float
    median_noise_leakage_reduction: float
    mean_objective_gain: float
    median_objective_gain: float
    mean_saturated_knobs: float
    mean_near_bound_knobs: float
    mean_baseline_detector_success: float
    mean_best_detector_success: float


def transition_motor_ensemble_from_dict(data: dict[str, Any]) -> TransitionMotorEnsembleConfig:
    block = dict(data["transition_motor_ensemble"])
    grid = fixed_grid_from_dict(dict(block["grid"]))
    registry = knob_registry_from_dicts(list(block["knobs"]))
    optimizer = dict(block["optimizer"])
    motor_config = TransitionMotorConfig(
        grid=grid,
        knob_registry=registry,
        objective_weights={str(k): float(v) for k, v in dict(block["objective_weights"]).items()},
        objective_mode="raw",
        normalization_scales=None,
        selected_operating_mode=None,
        operating_mode_registry=None,
        noise_profiles=[],
        time_min=float(optimizer["time_min"]),
        time_max=float(optimizer["time_max"]),
        n_time_samples=int(optimizer["n_time_samples"]),
        n_transport_modes=int(optimizer["n_transport_modes"]),
        finite_diff_eps=float(optimizer["finite_diff_eps"]),
        optimizer_steps=int(optimizer["optimizer_steps"]),
        optimizer_step_size=float(optimizer["optimizer_step_size"]),
        random_restarts=int(optimizer["random_restarts"]),
        seed=int(optimizer["seed"]),
    )
    return TransitionMotorEnsembleConfig(
        motor_config=motor_config,
        fabrication_disorder={
            "n_samples": int(block["fabrication_disorder"]["n_samples"]),
            "onsite_sigma": float(block["fabrication_disorder"]["onsite_sigma"]),
            "correlation_length_sites": float(block["fabrication_disorder"]["correlation_length_sites"]),
            "seed": int(block["fabrication_disorder"]["seed"]),
        },
        phase_noise={
            "n_profiles_per_sample": int(block["phase_noise"]["n_profiles_per_sample"]),
            "profile_sigma": float(block["phase_noise"]["profile_sigma"]),
            "correlation_length_sites": float(block["phase_noise"]["correlation_length_sites"]),
            "seed": int(block["phase_noise"]["seed"]),
        },
        success_criteria={
            "min_objective_gain": float(block["success_criteria"]["min_objective_gain"]),
            "min_detector_gain": float(block["success_criteria"]["min_detector_gain"]),
            "min_noise_action_reduction": float(block["success_criteria"]["min_noise_action_reduction"]),
        },
        outputs={
            "csv_path": str(block["outputs"]["csv_path"]),
            "summary_path": str(block["outputs"]["summary_path"]),
        },
    )


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
    if correlation_length_sites <= 0.0:
        raise ValueError("correlation_length_sites must be positive")
    if sigma == 0.0:
        return np.zeros(n_sites, dtype=np.float64)

    white = rng.normal(0.0, sigma, size=n_sites).astype(np.float64, copy=False)
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


def detector_distribution_at_time(H: np.ndarray, input_index: int, t: float) -> np.ndarray:
    """Compute detector probabilities using Hermitian eigendecomposition."""

    operator = np.asarray(H, dtype=np.complex128)
    eigenvalues, eigenvectors = np.linalg.eigh(operator)
    phases = np.exp(-1j * eigenvalues * float(t))
    input_state = np.zeros(operator.shape[0], dtype=np.complex128)
    input_state[int(input_index)] = 1.0
    amplitudes = eigenvectors @ (phases * (np.conjugate(eigenvectors).T @ input_state))
    probabilities = np.abs(amplitudes) ** 2
    probabilities /= np.sum(probabilities)
    return probabilities.astype(np.float64, copy=False)


def detector_success_probability(
    H: np.ndarray,
    input_index: int,
    target_indices: list[int],
    times: np.ndarray,
) -> float:
    values = [
        float(np.sum(detector_distribution_at_time(H, input_index, float(t))[target_indices]))
        for t in np.asarray(times, dtype=np.float64)
    ]
    return float(np.clip(np.max(values), 0.0, 1.0))


def detector_final_probability(
    H: np.ndarray,
    input_index: int,
    target_indices: list[int],
    t_final: float,
) -> float:
    distribution = detector_distribution_at_time(H, input_index, float(t_final))
    return float(np.clip(np.sum(distribution[target_indices]), 0.0, 1.0))


def _noise_ops_from_profiles(noise_profiles: list[np.ndarray]) -> list[np.ndarray]:
    return [diagonal_phase_noise(np.asarray(profile, dtype=np.float64)) for profile in noise_profiles]


def run_single_motor_ensemble_sample_detailed(
    grid: FixedGrid,
    registry: KnobRegistry,
    basis: ControlBasis,
    onsite_disorder: np.ndarray,
    noise_profiles: list[np.ndarray],
    motor_config: TransitionMotorConfig,
    success_criteria: dict[str, float],
) -> MotorEnsembleSampleResult:
    times = np.linspace(motor_config.time_min, motor_config.time_max, motor_config.n_time_samples, dtype=np.float64)
    t_final = float(times[-1])
    noise_ops = _noise_ops_from_profiles(noise_profiles)
    disordered_H0 = apply_fabrication_disorder(build_base_hamiltonian(grid), onsite_disorder)

    baseline_theta = registry.defaults()
    baseline_H = build_control_hamiltonian(disordered_H0, grid, baseline_theta, basis, registry)
    baseline_metrics = evaluate_motor_metrics(
        baseline_H,
        noise_ops,
        grid,
        baseline_theta,
        times,
        motor_config.objective_weights,
        n_modes=motor_config.n_transport_modes,
    )

    motor_result = random_restart_transition_motor_search(
        disordered_H0,
        grid,
        noise_ops,
        basis,
        registry,
        motor_config,
    )
    best_H = build_control_hamiltonian(disordered_H0, grid, motor_result.best_theta, basis, registry)

    baseline_detector_success = detector_success_probability(
        baseline_H,
        grid.input_index,
        list(grid.target_indices),
        times,
    )
    best_detector_success = detector_success_probability(
        best_H,
        grid.input_index,
        list(grid.target_indices),
        times,
    )
    baseline_detector_final = detector_final_probability(
        baseline_H,
        grid.input_index,
        list(grid.target_indices),
        t_final,
    )
    best_detector_final = detector_final_probability(
        best_H,
        grid.input_index,
        list(grid.target_indices),
        t_final,
    )

    statuses = knob_bound_statuses(motor_result.best_theta, registry)
    detector_success_gain = float(best_detector_success - baseline_detector_success)
    transport_gain = float(motor_result.best_metrics.transport_efficiency - baseline_metrics.transport_efficiency)
    noise_action_reduction = float(baseline_metrics.noise_action_on_info - motor_result.best_metrics.noise_action_on_info)
    noise_leakage_reduction = float(baseline_metrics.noise_leakage - motor_result.best_metrics.noise_leakage)
    objective_gain = float(motor_result.best_metrics.objective - baseline_metrics.objective)
    success = bool(
        objective_gain >= float(success_criteria["min_objective_gain"])
        and detector_success_gain >= float(success_criteria["min_detector_gain"])
        and noise_action_reduction >= float(success_criteria["min_noise_action_reduction"])
    )

    return DetailedMotorEnsembleSample(
        sample=MotorEnsembleSampleResult(
            sample_id=-1,
            baseline_detector_success=baseline_detector_success,
            best_detector_success=best_detector_success,
            baseline_detector_final=baseline_detector_final,
            best_detector_final=best_detector_final,
            baseline_transport_efficiency=float(baseline_metrics.transport_efficiency),
            best_transport_efficiency=float(motor_result.best_metrics.transport_efficiency),
            baseline_noise_action_on_info=float(baseline_metrics.noise_action_on_info),
            best_noise_action_on_info=float(motor_result.best_metrics.noise_action_on_info),
            baseline_noise_leakage=float(baseline_metrics.noise_leakage),
            best_noise_leakage=float(motor_result.best_metrics.noise_leakage),
            baseline_control_cost=float(baseline_metrics.control_cost),
            best_control_cost=float(motor_result.best_metrics.control_cost),
            baseline_objective=float(baseline_metrics.objective),
            best_objective=float(motor_result.best_metrics.objective),
            detector_success_gain=detector_success_gain,
            transport_gain=transport_gain,
            noise_action_reduction=noise_action_reduction,
            noise_leakage_reduction=noise_leakage_reduction,
            objective_gain=objective_gain,
            saturated_knobs=count_saturated_knobs(statuses),
            near_bound_knobs=sum(1 for item in statuses if item.near_bound),
            success=success,
        ),
        baseline_theta=dict(baseline_theta),
        best_theta=dict(motor_result.best_theta),
        baseline_metrics=baseline_metrics,
        best_metrics=motor_result.best_metrics,
    )


def run_single_motor_ensemble_sample(
    grid: FixedGrid,
    registry: KnobRegistry,
    basis: ControlBasis,
    onsite_disorder: np.ndarray,
    noise_profiles: list[np.ndarray],
    motor_config: TransitionMotorConfig,
    success_criteria: dict[str, float],
) -> MotorEnsembleSampleResult:
    return run_single_motor_ensemble_sample_detailed(
        grid,
        registry,
        basis,
        onsite_disorder,
        noise_profiles,
        motor_config,
        success_criteria,
    ).sample


def summarize_motor_ensemble(results: list[MotorEnsembleSampleResult]) -> MotorEnsembleSummary:
    if not results:
        raise ValueError("results must be non-empty")

    def column(name: str) -> np.ndarray:
        return np.array([getattr(result, name) for result in results], dtype=np.float64)

    return MotorEnsembleSummary(
        n_samples=len(results),
        success_rate=float(np.mean([1.0 if item.success else 0.0 for item in results], dtype=np.float64)),
        mean_detector_success_gain=float(np.mean(column("detector_success_gain"))),
        median_detector_success_gain=float(np.median(column("detector_success_gain"))),
        mean_transport_gain=float(np.mean(column("transport_gain"))),
        median_transport_gain=float(np.median(column("transport_gain"))),
        mean_noise_action_reduction=float(np.mean(column("noise_action_reduction"))),
        median_noise_action_reduction=float(np.median(column("noise_action_reduction"))),
        mean_noise_leakage_reduction=float(np.mean(column("noise_leakage_reduction"))),
        median_noise_leakage_reduction=float(np.median(column("noise_leakage_reduction"))),
        mean_objective_gain=float(np.mean(column("objective_gain"))),
        median_objective_gain=float(np.median(column("objective_gain"))),
        mean_saturated_knobs=float(np.mean(column("saturated_knobs"))),
        mean_near_bound_knobs=float(np.mean(column("near_bound_knobs"))),
        mean_baseline_detector_success=float(np.mean(column("baseline_detector_success"))),
        mean_best_detector_success=float(np.mean(column("best_detector_success"))),
    )


def write_motor_ensemble_csv(path: str | Path, results: list[MotorEnsembleSampleResult]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(MotorEnsembleSampleResult.__dataclass_fields__.keys())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    return output


def write_motor_ensemble_summary_json(path: str | Path, summary: MotorEnsembleSummary) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8")
    return output


def run_transition_motor_ensemble_study(
    config: dict[str, Any],
) -> tuple[list[MotorEnsembleSampleResult], MotorEnsembleSummary, Path, Path]:
    ensemble = transition_motor_ensemble_from_dict(config)
    motor_config = ensemble.motor_config
    grid = motor_config.grid
    registry = motor_config.knob_registry
    basis = make_control_basis(grid)

    fabrication_profiles = sample_fabrication_disorder_profiles(
        int(ensemble.fabrication_disorder["n_samples"]),
        grid.n_sites,
        float(ensemble.fabrication_disorder["onsite_sigma"]),
        float(ensemble.fabrication_disorder["correlation_length_sites"]),
        int(ensemble.fabrication_disorder["seed"]),
    )

    results: list[MotorEnsembleSampleResult] = []
    for sample_id, onsite_profile in enumerate(fabrication_profiles):
        noise_profiles = sample_phase_noise_profiles(
            int(ensemble.phase_noise["n_profiles_per_sample"]),
            grid.n_sites,
            float(ensemble.phase_noise["profile_sigma"]),
            float(ensemble.phase_noise["correlation_length_sites"]),
            int(ensemble.phase_noise["seed"]) + sample_id,
        )
        sample = run_single_motor_ensemble_sample(
            grid,
            registry,
            basis,
            onsite_profile,
            noise_profiles,
            motor_config,
            ensemble.success_criteria,
        )
        results.append(
            MotorEnsembleSampleResult(
                sample_id=sample_id,
                baseline_detector_success=sample.baseline_detector_success,
                best_detector_success=sample.best_detector_success,
                baseline_detector_final=sample.baseline_detector_final,
                best_detector_final=sample.best_detector_final,
                baseline_transport_efficiency=sample.baseline_transport_efficiency,
                best_transport_efficiency=sample.best_transport_efficiency,
                baseline_noise_action_on_info=sample.baseline_noise_action_on_info,
                best_noise_action_on_info=sample.best_noise_action_on_info,
                baseline_noise_leakage=sample.baseline_noise_leakage,
                best_noise_leakage=sample.best_noise_leakage,
                baseline_control_cost=sample.baseline_control_cost,
                best_control_cost=sample.best_control_cost,
                baseline_objective=sample.baseline_objective,
                best_objective=sample.best_objective,
                detector_success_gain=sample.detector_success_gain,
                transport_gain=sample.transport_gain,
                noise_action_reduction=sample.noise_action_reduction,
                noise_leakage_reduction=sample.noise_leakage_reduction,
                objective_gain=sample.objective_gain,
                saturated_knobs=sample.saturated_knobs,
                near_bound_knobs=sample.near_bound_knobs,
                success=sample.success,
            )
        )

    summary = summarize_motor_ensemble(results)
    csv_path = write_motor_ensemble_csv(ensemble.outputs["csv_path"], results)
    summary_path = write_motor_ensemble_summary_json(ensemble.outputs["summary_path"], summary)
    return results, summary, csv_path, summary_path

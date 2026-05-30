"""Algebraic transition-dynamics tuner for hardware-native error suppression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hardware.subspaces import (
    diagonal_phase_noise,
    noise_overlap_with_subspace,
    transport_subspace_from_hamiltonian,
)


def _normalize_edge(edge: tuple[int, int]) -> tuple[int, int]:
    i, j = int(edge[0]), int(edge[1])
    if i == j:
        raise ValueError("self-edges are not allowed in a fixed grid")
    return (min(i, j), max(i, j))


@dataclass(frozen=True)
class FixedGrid:
    n_sites: int
    edges: list[tuple[int, int]]
    base_coupling: float
    input_index: int
    target_indices: list[int]


@dataclass(frozen=True)
class TransitionTunerConfig:
    max_relative_coupling_delta: float
    max_onsite_shift: float
    n_candidates: int
    seed: int
    time_min: float
    time_max: float
    n_time_samples: int
    transport_weight: float
    suppression_weight: float
    control_penalty_weight: float


@dataclass(frozen=True)
class TransitionTuningResult:
    baseline_objective: float
    best_objective: float
    baseline_transport_efficiency: float
    best_transport_efficiency: float
    baseline_noise_overlap: float
    best_noise_overlap: float
    baseline_suppression_score: float
    best_suppression_score: float
    best_coupling_scales: dict[tuple[int, int], float]
    best_onsite_shifts: list[float]


def fixed_grid_from_dict(data: dict[str, object]) -> FixedGrid:
    edges = [_normalize_edge(tuple(edge)) for edge in data["edges"]]
    return FixedGrid(
        n_sites=int(data["n_sites"]),
        edges=edges,
        base_coupling=float(data["base_coupling"]),
        input_index=int(data["input_index"]),
        target_indices=[int(index) for index in data["target_indices"]],
    )


def transition_tuner_config_from_dict(data: dict[str, object]) -> TransitionTunerConfig:
    return TransitionTunerConfig(
        max_relative_coupling_delta=float(data["max_relative_coupling_delta"]),
        max_onsite_shift=float(data["max_onsite_shift"]),
        n_candidates=int(data["n_candidates"]),
        seed=int(data["seed"]),
        time_min=float(data["time_min"]),
        time_max=float(data["time_max"]),
        n_time_samples=int(data["n_time_samples"]),
        transport_weight=float(data["transport_weight"]),
        suppression_weight=float(data["suppression_weight"]),
        control_penalty_weight=float(data["control_penalty_weight"]),
    )


def noise_operators_from_profiles(profiles: list[list[float]]) -> list[np.ndarray]:
    return [diagonal_phase_noise(np.asarray(profile, dtype=np.float64)) for profile in profiles]


def _validate_fixed_grid(grid: FixedGrid) -> None:
    if grid.n_sites <= 0:
        raise ValueError("n_sites must be positive")
    if grid.base_coupling < 0.0:
        raise ValueError("base_coupling must be non-negative")
    if grid.input_index < 0 or grid.input_index >= grid.n_sites:
        raise ValueError("input_index out of range")
    if not grid.target_indices:
        raise ValueError("target_indices must be non-empty")

    seen_edges: set[tuple[int, int]] = set()
    for edge in grid.edges:
        normalized = _normalize_edge(edge)
        if normalized in seen_edges:
            raise ValueError(f"duplicate edge in fixed grid: {normalized}")
        seen_edges.add(normalized)
        if normalized[1] >= grid.n_sites:
            raise ValueError(f"edge index out of range: {normalized}")

    for index in grid.target_indices:
        if index < 0 or index >= grid.n_sites:
            raise ValueError(f"target index out of range: {index}")


def build_base_hamiltonian(grid: FixedGrid) -> np.ndarray:
    _validate_fixed_grid(grid)
    H0 = np.zeros((grid.n_sites, grid.n_sites), dtype=np.complex128)
    for i, j in grid.edges:
        H0[i, j] = float(grid.base_coupling)
        H0[j, i] = float(grid.base_coupling)
    return H0


def apply_transition_controls(
    H0: np.ndarray,
    grid: FixedGrid,
    coupling_scales: dict[tuple[int, int], float],
    onsite_shifts: np.ndarray,
) -> np.ndarray:
    """Modify only existing edges and onsite terms on a fixed grid."""

    _validate_fixed_grid(grid)
    base = np.asarray(H0, dtype=np.complex128)
    if base.shape != (grid.n_sites, grid.n_sites):
        raise ValueError("H0 shape does not match grid")

    shifts = np.asarray(onsite_shifts, dtype=np.float64)
    if shifts.shape != (grid.n_sites,):
        raise ValueError("onsite_shifts must match n_sites")

    allowed_edges = {_normalize_edge(edge) for edge in grid.edges}
    H = np.array(base, copy=True)

    for edge, scale in coupling_scales.items():
        normalized = _normalize_edge(edge)
        if normalized not in allowed_edges:
            raise ValueError(f"coupling scale references non-existent edge: {normalized}")
        if not np.isfinite(scale):
            raise ValueError("coupling scales must be finite")
        i, j = normalized
        H[i, j] = base[i, j] * float(scale)
        H[j, i] = np.conjugate(H[i, j])

    H[np.diag_indices(grid.n_sites)] += shifts.astype(np.complex128, copy=False)
    return 0.5 * (H + np.conjugate(H.T))


def _time_grid(config: TransitionTunerConfig) -> np.ndarray:
    if config.n_time_samples <= 0:
        raise ValueError("n_time_samples must be positive")
    if config.time_max < config.time_min:
        raise ValueError("time_max must be >= time_min")
    return np.linspace(config.time_min, config.time_max, config.n_time_samples, dtype=np.float64)


def transport_efficiency(
    H: np.ndarray,
    input_index: int,
    target_indices: list[int],
    times: np.ndarray,
) -> float:
    operator = np.asarray(H, dtype=np.complex128)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("H must be a square matrix")
    if input_index < 0 or input_index >= operator.shape[0]:
        raise ValueError("input_index out of range")
    if not target_indices:
        raise ValueError("target_indices must be non-empty")

    eigvals, eigvecs = np.linalg.eigh(operator)
    input_state = np.zeros(operator.shape[0], dtype=np.complex128)
    input_state[input_index] = 1.0
    coeffs = np.conjugate(eigvecs.T) @ input_state
    target_set = sorted({int(index) for index in target_indices})
    best_efficiency = 0.0

    for time_value in np.asarray(times, dtype=np.float64):
        phase = np.exp(-1j * eigvals * float(time_value))
        state = eigvecs @ (phase * coeffs)
        probabilities = np.abs(state) ** 2
        efficiency = float(np.sum(probabilities[target_set]))
        best_efficiency = max(best_efficiency, efficiency)

    return float(np.clip(best_efficiency, 0.0, 1.0))


def dynamic_noise_overlap(
    H: np.ndarray,
    noise_ops: list[np.ndarray],
    grid: FixedGrid,
) -> float:
    transport_subspace = transport_subspace_from_hamiltonian(
        np.asarray(H, dtype=np.complex128),
        grid.input_index,
        list(grid.target_indices),
    )
    return noise_overlap_with_subspace(noise_ops, transport_subspace)


def _control_penalty(H: np.ndarray, grid: FixedGrid) -> float:
    delta = np.asarray(H, dtype=np.complex128) - build_base_hamiltonian(grid)
    return float(np.linalg.norm(delta, ord="fro") ** 2 / max(1, grid.n_sites))


def evaluate_candidate(
    H: np.ndarray,
    noise_ops: list[np.ndarray],
    grid: FixedGrid,
    config: TransitionTunerConfig,
) -> dict[str, float]:
    times = _time_grid(config)
    transport = transport_efficiency(H, grid.input_index, list(grid.target_indices), times)
    overlap = dynamic_noise_overlap(H, noise_ops, grid)
    suppression = float(np.clip(1.0 - overlap, 0.0, 1.0))
    penalty = _control_penalty(H, grid)
    objective = (
        float(config.transport_weight) * transport
        + float(config.suppression_weight) * suppression
        - float(config.control_penalty_weight) * penalty
    )
    return {
        "transport_efficiency": transport,
        "noise_overlap": overlap,
        "suppression_score": suppression,
        "dark_subspace_projection": suppression,
        "objective": float(objective),
    }


def random_transition_search(
    grid: FixedGrid,
    noise_ops: list[np.ndarray],
    config: TransitionTunerConfig,
) -> TransitionTuningResult:
    _validate_fixed_grid(grid)
    H0 = build_base_hamiltonian(grid)
    baseline_metrics = evaluate_candidate(H0, noise_ops, grid, config)

    best_metrics = dict(baseline_metrics)
    best_coupling_scales = {_normalize_edge(edge): 1.0 for edge in grid.edges}
    best_onsite_shifts = np.zeros(grid.n_sites, dtype=np.float64)
    rng = np.random.default_rng(config.seed)
    normalized_edges = [_normalize_edge(edge) for edge in grid.edges]

    for _ in range(config.n_candidates):
        coupling_scales = {
            edge: float(rng.uniform(1.0 - config.max_relative_coupling_delta, 1.0 + config.max_relative_coupling_delta))
            for edge in normalized_edges
        }
        onsite_shifts = rng.uniform(
            -config.max_onsite_shift,
            config.max_onsite_shift,
            size=grid.n_sites,
        ).astype(np.float64, copy=False)
        candidate_hamiltonian = apply_transition_controls(H0, grid, coupling_scales, onsite_shifts)
        candidate_metrics = evaluate_candidate(candidate_hamiltonian, noise_ops, grid, config)
        if candidate_metrics["objective"] > best_metrics["objective"] + 1.0e-12:
            best_metrics = candidate_metrics
            best_coupling_scales = dict(coupling_scales)
            best_onsite_shifts = np.array(onsite_shifts, copy=True)

    return TransitionTuningResult(
        baseline_objective=float(baseline_metrics["objective"]),
        best_objective=float(best_metrics["objective"]),
        baseline_transport_efficiency=float(baseline_metrics["transport_efficiency"]),
        best_transport_efficiency=float(best_metrics["transport_efficiency"]),
        baseline_noise_overlap=float(baseline_metrics["noise_overlap"]),
        best_noise_overlap=float(best_metrics["noise_overlap"]),
        baseline_suppression_score=float(baseline_metrics["suppression_score"]),
        best_suppression_score=float(best_metrics["suppression_score"]),
        best_coupling_scales={edge: float(scale) for edge, scale in best_coupling_scales.items()},
        best_onsite_shifts=[float(value) for value in best_onsite_shifts],
    )

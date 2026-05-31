"""Metric decomposition for the KTA transition motor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hardware.subspaces import complement_projector, projector_from_basis, transport_subspace_from_hamiltonian
from hardware.transition_tuner import FixedGrid, transport_efficiency


@dataclass(frozen=True)
class MotorMetrics:
    transport_efficiency: float
    noise_internal: float
    noise_leakage: float
    noise_action_on_info: float
    suppression_score: float
    control_cost: float
    objective: float


@dataclass(frozen=True)
class ObjectiveNormalizationScale:
    transport: float
    noise_action: float
    leakage: float
    control_cost: float

    def __post_init__(self) -> None:
        if min(self.transport, self.noise_action, self.leakage, self.control_cost) <= 0.0:
            raise ValueError("all normalization scales must be positive")


ObjectiveNormalizationScales = ObjectiveNormalizationScale


def projector_from_transport_subspace(
    H: np.ndarray,
    input_index: int,
    target_indices: list[int],
    *,
    n_modes: int = 2,
) -> np.ndarray:
    basis = transport_subspace_from_hamiltonian(H, input_index, target_indices, n_modes=n_modes)
    return projector_from_basis(basis)


def noise_metric_decomposition(
    noise_ops: list[np.ndarray],
    P_info: np.ndarray,
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    P = np.asarray(P_info, dtype=np.complex128)
    Q = complement_projector(P)

    total_noise = 0.0
    internal = 0.0
    leakage = 0.0
    action = 0.0

    for noise_operator in noise_ops:
        N = np.asarray(noise_operator, dtype=np.complex128)
        total_noise += float(np.linalg.norm(N, ord="fro") ** 2)
        internal += float(np.linalg.norm(P @ N @ P, ord="fro") ** 2)
        leakage += float(np.linalg.norm(Q @ N @ P, ord="fro") ** 2)
        action += float(np.linalg.norm(N @ P, ord="fro") ** 2)

    denom = total_noise + float(eps)
    return {
        "noise_internal": internal / denom,
        "noise_leakage": leakage / denom,
        "noise_action_on_info": action / denom,
    }


def control_cost(theta: dict[str, float]) -> float:
    return float(np.sum(np.square(np.array(list(theta.values()), dtype=np.float64))))


def motor_objective(
    *,
    transport_efficiency: float,
    noise_action_on_info: float,
    noise_leakage: float,
    control_cost: float,
    weights: dict[str, float],
) -> float:
    return float(
        float(weights.get("transport", 1.0)) * transport_efficiency
        - float(weights.get("noise_action", 0.0)) * noise_action_on_info
        - float(weights.get("leakage", 0.0)) * noise_leakage
        - float(weights.get("control_cost", 0.0)) * control_cost
    )


def normalized_motor_objective(
    *,
    baseline_metrics: MotorMetrics,
    candidate_metrics: MotorMetrics,
    weights: dict[str, float],
    normalization_scales: ObjectiveNormalizationScale,
) -> float:
    transport_gain = float(candidate_metrics.transport_efficiency - baseline_metrics.transport_efficiency)
    noise_action_reduction = float(baseline_metrics.noise_action_on_info - candidate_metrics.noise_action_on_info)
    leakage_reduction = float(baseline_metrics.noise_leakage - candidate_metrics.noise_leakage)
    control_cost_increase = float(candidate_metrics.control_cost - baseline_metrics.control_cost)
    return float(
        float(weights.get("transport", 1.0)) * transport_gain / normalization_scales.transport
        + float(weights.get("noise_action", 0.0)) * noise_action_reduction / normalization_scales.noise_action
        + float(weights.get("leakage", 0.0)) * leakage_reduction / normalization_scales.leakage
        - float(weights.get("control_cost", 0.0)) * control_cost_increase / normalization_scales.control_cost
    )


def evaluate_motor_metrics(
    H: np.ndarray,
    noise_ops: list[np.ndarray],
    grid: FixedGrid,
    theta: dict[str, float],
    times: np.ndarray,
    weights: dict[str, float],
    *,
    n_modes: int = 2,
) -> MotorMetrics:
    transport = transport_efficiency(H, grid.input_index, list(grid.target_indices), times)
    P_info = projector_from_transport_subspace(
        H,
        grid.input_index,
        list(grid.target_indices),
        n_modes=n_modes,
    )
    noise = noise_metric_decomposition(noise_ops, P_info)
    cost = control_cost(theta)
    suppression = float(np.clip(1.0 - noise["noise_action_on_info"], 0.0, 1.0))
    objective = motor_objective(
        transport_efficiency=transport,
        noise_action_on_info=noise["noise_action_on_info"],
        noise_leakage=noise["noise_leakage"],
        control_cost=cost,
        weights=weights,
    )
    return MotorMetrics(
        transport_efficiency=float(np.clip(transport, 0.0, 1.0)),
        noise_internal=float(noise["noise_internal"]),
        noise_leakage=float(noise["noise_leakage"]),
        noise_action_on_info=float(noise["noise_action_on_info"]),
        suppression_score=suppression,
        control_cost=cost,
        objective=objective,
    )

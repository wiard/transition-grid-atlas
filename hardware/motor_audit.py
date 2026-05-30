"""Bound-pressure and robustness audit for the KTA transition motor."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from hardware.control_knobs import ControlKnob, KnobRegistry
from hardware.objectives import MotorMetrics, evaluate_motor_metrics
from hardware.transition_motor import (
    TransitionMotorConfig,
    TransitionMotorResult,
    build_control_hamiltonian,
    build_default_motor_basis,
    random_restart_transition_motor_search,
)
from hardware.transition_tuner import build_base_hamiltonian


@dataclass(frozen=True)
class KnobBoundStatus:
    name: str
    value: float
    min_value: float
    max_value: float
    relative_position: float
    at_lower_bound: bool
    at_upper_bound: bool
    near_bound: bool


@dataclass(frozen=True)
class KnobAblationResult:
    knob: str
    baseline_objective: float
    best_objective: float
    ablated_objective: float
    objective_loss_from_ablation: float
    ablated_transport_efficiency: float
    ablated_noise_action_on_info: float
    ablated_noise_leakage: float


@dataclass(frozen=True)
class KnobEfficiencyResult:
    knob: str
    value: float
    control_cost_contribution: float
    objective_loss_from_ablation: float
    gain_per_cost: float


@dataclass(frozen=True)
class LimitSweepResult:
    scale: float
    best_objective: float
    objective_improvement: float
    best_transport_efficiency: float
    best_noise_action_on_info: float
    best_noise_leakage: float
    saturated_knobs: int


@dataclass(frozen=True)
class SeedStabilityResult:
    seed: int
    best_objective: float
    best_transport_efficiency: float
    best_noise_action_on_info: float
    best_noise_leakage: float
    saturated_knobs: int


@dataclass(frozen=True)
class TransitionMotorAuditResult:
    motor_result: TransitionMotorResult
    bound_statuses: list[KnobBoundStatus]
    ablations: list[KnobAblationResult]
    efficiencies: list[KnobEfficiencyResult]
    limit_sweep: list[LimitSweepResult]
    seed_results: list[SeedStabilityResult]
    seed_summary: dict[str, float]
    constraint_limited: bool
    recommended_action: str


def build_motor_evaluate_theta_fn(config: TransitionMotorConfig):
    grid = config.grid
    registry = config.knob_registry
    basis = build_default_motor_basis(grid)
    H0 = build_base_hamiltonian(grid)
    times = np.linspace(config.time_min, config.time_max, config.n_time_samples, dtype=np.float64)

    def evaluate_theta(theta: dict[str, float]) -> MotorMetrics:
        clipped = registry.clip_theta(theta)
        H = build_control_hamiltonian(H0, grid, clipped, basis, registry)
        return evaluate_motor_metrics(
            H,
            config.noise_profiles,
            grid,
            clipped,
            times,
            config.objective_weights,
            n_modes=config.n_transport_modes,
        )

    return evaluate_theta


def run_transition_motor_instance(config: TransitionMotorConfig) -> TransitionMotorResult:
    grid = config.grid
    registry = config.knob_registry
    basis = build_default_motor_basis(grid)
    H0 = build_base_hamiltonian(grid)
    return random_restart_transition_motor_search(
        H0,
        grid,
        config.noise_profiles,
        basis,
        registry,
        config,
    )


def knob_bound_statuses(
    theta: dict[str, float],
    registry: KnobRegistry,
    *,
    tolerance_fraction: float = 0.02,
) -> list[KnobBoundStatus]:
    clipped = registry.clip_theta(theta)
    statuses: list[KnobBoundStatus] = []
    for knob in registry.knobs:
        span = knob.max_value - knob.min_value
        tolerance = tolerance_fraction * span
        value = float(clipped[knob.name])
        relative_position = (value - knob.min_value) / span
        at_lower = abs(value - knob.min_value) <= tolerance
        at_upper = abs(value - knob.max_value) <= tolerance
        statuses.append(
            KnobBoundStatus(
                name=knob.name,
                value=value,
                min_value=float(knob.min_value),
                max_value=float(knob.max_value),
                relative_position=float(relative_position),
                at_lower_bound=at_lower,
                at_upper_bound=at_upper,
                near_bound=at_lower or at_upper,
            )
        )
    return statuses


def count_saturated_knobs(statuses: list[KnobBoundStatus]) -> int:
    return sum(1 for status in statuses if status.at_lower_bound or status.at_upper_bound)


def ablate_knobs(
    best_theta: dict[str, float],
    default_theta: dict[str, float],
    evaluate_theta_fn,
) -> list[KnobAblationResult]:
    baseline_metrics = evaluate_theta_fn(default_theta)
    best_metrics = evaluate_theta_fn(best_theta)
    results: list[KnobAblationResult] = []
    for knob in best_theta:
        theta_ablated = dict(best_theta)
        theta_ablated[knob] = float(default_theta[knob])
        ablated_metrics = evaluate_theta_fn(theta_ablated)
        results.append(
            KnobAblationResult(
                knob=knob,
                baseline_objective=float(baseline_metrics.objective),
                best_objective=float(best_metrics.objective),
                ablated_objective=float(ablated_metrics.objective),
                objective_loss_from_ablation=float(best_metrics.objective - ablated_metrics.objective),
                ablated_transport_efficiency=float(ablated_metrics.transport_efficiency),
                ablated_noise_action_on_info=float(ablated_metrics.noise_action_on_info),
                ablated_noise_leakage=float(ablated_metrics.noise_leakage),
            )
        )
    return results


def knob_efficiency_from_ablation(
    best_theta: dict[str, float],
    ablations: list[KnobAblationResult],
    *,
    eps: float = 1e-12,
) -> list[KnobEfficiencyResult]:
    results: list[KnobEfficiencyResult] = []
    for ablation in ablations:
        value = float(best_theta[ablation.knob])
        cost = value * value
        results.append(
            KnobEfficiencyResult(
                knob=ablation.knob,
                value=value,
                control_cost_contribution=float(cost),
                objective_loss_from_ablation=float(ablation.objective_loss_from_ablation),
                gain_per_cost=float(ablation.objective_loss_from_ablation / (cost + eps)),
            )
        )
    return results


def scale_registry_limits(registry: KnobRegistry, scale: float) -> KnobRegistry:
    scaled_knobs = []
    for knob in registry.knobs:
        scaled_knobs.append(
            ControlKnob(
                name=knob.name,
                family=knob.family,
                symbol=knob.symbol,
                basis_name=knob.basis_name,
                min_value=float(knob.default + scale * (knob.min_value - knob.default)),
                max_value=float(knob.default + scale * (knob.max_value - knob.default)),
                default=float(knob.default),
                units=knob.units,
                description=knob.description,
                hardware_meaning=knob.hardware_meaning,
            )
        )
    return KnobRegistry(knobs=tuple(scaled_knobs))


def run_control_limit_sweep(
    config: TransitionMotorConfig,
    scales: list[float],
    run_motor_fn,
) -> list[LimitSweepResult]:
    results: list[LimitSweepResult] = []
    for scale in scales:
        scaled_config = replace(config, knob_registry=scale_registry_limits(config.knob_registry, scale))
        motor_result = run_motor_fn(scaled_config)
        statuses = knob_bound_statuses(motor_result.best_theta, scaled_config.knob_registry)
        results.append(
            LimitSweepResult(
                scale=float(scale),
                best_objective=float(motor_result.best_metrics.objective),
                objective_improvement=float(motor_result.objective_improvement),
                best_transport_efficiency=float(motor_result.best_metrics.transport_efficiency),
                best_noise_action_on_info=float(motor_result.best_metrics.noise_action_on_info),
                best_noise_leakage=float(motor_result.best_metrics.noise_leakage),
                saturated_knobs=count_saturated_knobs(statuses),
            )
        )
    return results


def run_seed_stability_audit(
    config: TransitionMotorConfig,
    seeds: list[int],
    run_motor_fn,
) -> list[SeedStabilityResult]:
    results: list[SeedStabilityResult] = []
    for seed in seeds:
        seeded_config = replace(config, seed=int(seed))
        motor_result = run_motor_fn(seeded_config)
        statuses = knob_bound_statuses(motor_result.best_theta, seeded_config.knob_registry)
        results.append(
            SeedStabilityResult(
                seed=int(seed),
                best_objective=float(motor_result.best_metrics.objective),
                best_transport_efficiency=float(motor_result.best_metrics.transport_efficiency),
                best_noise_action_on_info=float(motor_result.best_metrics.noise_action_on_info),
                best_noise_leakage=float(motor_result.best_metrics.noise_leakage),
                saturated_knobs=count_saturated_knobs(statuses),
            )
        )
    return results


def summarize_seed_stability(results: list[SeedStabilityResult]) -> dict[str, float]:
    if not results:
        raise ValueError("seed stability results cannot be empty")

    def summary(values: list[float], prefix: str) -> dict[str, float]:
        array = np.array(values, dtype=np.float64)
        return {
            f"mean_{prefix}": float(np.mean(array)),
            f"std_{prefix}": float(np.std(array)),
            f"min_{prefix}": float(np.min(array)),
            f"max_{prefix}": float(np.max(array)),
        }

    merged: dict[str, float] = {}
    merged.update(summary([item.best_objective for item in results], "best_objective"))
    merged.update(summary([item.best_transport_efficiency for item in results], "best_transport_efficiency"))
    merged.update(summary([item.best_noise_action_on_info for item in results], "best_noise_action_on_info"))
    merged.update(summary([item.best_noise_leakage for item in results], "best_noise_leakage"))
    merged.update(summary([float(item.saturated_knobs) for item in results], "saturated_knobs"))
    return merged


def recommend_action(
    *,
    constraint_limited: bool,
    limit_sweep: list[LimitSweepResult],
    seed_summary: dict[str, float],
) -> str:
    by_scale = {round(item.scale, 6): item for item in limit_sweep}
    base = by_scale.get(1.0)
    tighter = by_scale.get(0.75)
    wider_candidates = [item for item in limit_sweep if item.scale > 1.0]
    best_wider = max((item.objective_improvement for item in wider_candidates), default=-np.inf)
    seed_std = float(seed_summary.get("std_best_objective", 0.0))

    if seed_std > 0.02:
        return "increase restarts or improve optimizer"
    if constraint_limited and base is not None and best_wider > base.objective_improvement + 0.01:
        return "consider wider physically calibrated ranges"
    if base is not None and tighter is not None and tighter.objective_improvement >= 0.9 * base.objective_improvement:
        return "current behavior robust under tighter actuator limits"
    return "current ranges are informative; refine basis design and hardware calibration before widening controls"


def run_transition_motor_bound_audit(
    config: TransitionMotorConfig,
    *,
    scales: list[float] | None = None,
    seeds: list[int] | None = None,
) -> TransitionMotorAuditResult:
    scales = scales or [0.5, 0.75, 1.0, 1.25, 1.5]
    seeds = seeds or [1, 2, 3, 4, 5]

    motor_result = run_transition_motor_instance(config)
    statuses = knob_bound_statuses(motor_result.best_theta, config.knob_registry)
    evaluate_theta = build_motor_evaluate_theta_fn(config)
    ablations = ablate_knobs(motor_result.best_theta, motor_result.baseline_theta, evaluate_theta)
    efficiencies = knob_efficiency_from_ablation(motor_result.best_theta, ablations)
    limit_sweep = run_control_limit_sweep(config, scales, run_transition_motor_instance)
    seed_results = run_seed_stability_audit(config, seeds, run_transition_motor_instance)
    seed_summary = summarize_seed_stability(seed_results)
    constraint_limited = count_saturated_knobs(statuses) >= 2
    return TransitionMotorAuditResult(
        motor_result=motor_result,
        bound_statuses=statuses,
        ablations=ablations,
        efficiencies=efficiencies,
        limit_sweep=limit_sweep,
        seed_results=seed_results,
        seed_summary=seed_summary,
        constraint_limited=constraint_limited,
        recommended_action=recommend_action(
            constraint_limited=constraint_limited,
            limit_sweep=limit_sweep,
            seed_summary=seed_summary,
        ),
    )

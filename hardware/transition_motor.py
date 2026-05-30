"""Transition Motor control stack for instrumented KTA tuning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hardware.control_basis import ControlBasis, make_control_basis
from hardware.control_knobs import KnobRegistry, knob_registry_from_dicts
from hardware.objectives import MotorMetrics, evaluate_motor_metrics
from hardware.sensitivity import compute_sensitivity_matrix, top_sensitivities
from hardware.transition_tuner import FixedGrid, build_base_hamiltonian, fixed_grid_from_dict, noise_operators_from_profiles


@dataclass(frozen=True)
class TransitionMotorConfig:
    grid: FixedGrid
    knob_registry: KnobRegistry
    objective_weights: dict[str, float]
    noise_profiles: list[np.ndarray]
    time_min: float
    time_max: float
    n_time_samples: int
    n_transport_modes: int
    finite_diff_eps: float
    optimizer_steps: int
    optimizer_step_size: float
    random_restarts: int
    seed: int


@dataclass(frozen=True)
class TransitionMotorResult:
    baseline_theta: dict[str, float]
    best_theta: dict[str, float]
    baseline_metrics: MotorMetrics
    best_metrics: MotorMetrics
    objective_improvement: float
    top_sensitivities: list[dict[str, object]]


def transition_motor_config_from_dict(data: dict[str, object]) -> tuple[TransitionMotorConfig, dict[str, object]]:
    block = dict(data["transition_motor"])
    grid = fixed_grid_from_dict(dict(block["grid"]))
    registry = knob_registry_from_dicts(list(block["knobs"]))
    optimizer = dict(block["optimizer"])
    config = TransitionMotorConfig(
        grid=grid,
        knob_registry=registry,
        objective_weights={str(k): float(v) for k, v in dict(block["objective_weights"]).items()},
        noise_profiles=noise_operators_from_profiles(list(dict(block["noise"])["profiles"])),
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
    outputs = {str(k): v for k, v in dict(block.get("outputs", {})).items()}
    return config, outputs


def build_control_hamiltonian(
    H0: np.ndarray,
    grid: FixedGrid,
    theta: dict[str, float],
    basis: ControlBasis,
    registry: KnobRegistry,
) -> np.ndarray:
    """Apply interpretable controls to H0 without changing the grid topology."""

    clipped = registry.clip_theta(theta)
    base = np.asarray(H0, dtype=np.complex128)
    H = np.array(base, copy=True)
    allowed_edges = {tuple(sorted((int(i), int(j)))) for i, j in grid.edges}

    for knob in registry.knobs:
        value = float(clipped[knob.name])
        if abs(value) <= 1.0e-15:
            continue
        if knob.family == "coherent_coupling":
            edge_map = basis.edge_basis[knob.basis_name]
            for edge, weight in edge_map.items():
                normalized = tuple(sorted(edge))
                if normalized not in allowed_edges:
                    raise ValueError(f"control basis references non-existent edge: {normalized}")
                i, j = normalized
                delta = base[i, j] * value * float(weight)
                H[i, j] += delta
                H[j, i] = np.conjugate(H[i, j])
        elif knob.family == "onsite_phase":
            site_vector = basis.site_basis[knob.basis_name]
            H[np.diag_indices(grid.n_sites)] += value * site_vector.astype(np.complex128, copy=False)
        elif knob.family in {"phase_noise", "objective"}:
            continue
        else:
            raise ValueError(f"unsupported knob family: {knob.family}")

    return 0.5 * (H + np.conjugate(H.T))


def finite_difference_gradient(
    objective_fn,
    theta: dict[str, float],
    registry: KnobRegistry,
    *,
    eps: float,
) -> dict[str, float]:
    gradient: dict[str, float] = {}
    clipped = registry.clip_theta(theta)
    for knob_name in registry.names():
        plus = dict(clipped)
        minus = dict(clipped)
        plus[knob_name] += eps
        minus[knob_name] -= eps
        plus = registry.clip_theta(plus)
        minus = registry.clip_theta(minus)
        delta = plus[knob_name] - minus[knob_name]
        if abs(delta) <= 1.0e-15:
            gradient[knob_name] = 0.0
            continue
        gradient[knob_name] = float((objective_fn(plus) - objective_fn(minus)) / delta)
    return gradient


def projected_gradient_ascent(
    objective_fn,
    theta0: dict[str, float],
    registry: KnobRegistry,
    *,
    n_steps: int,
    step_size: float,
    eps: float,
) -> tuple[dict[str, float], float]:
    current_theta = registry.clip_theta(theta0)
    current_value = float(objective_fn(current_theta))
    best_theta = dict(current_theta)
    best_value = current_value

    for _ in range(n_steps):
        gradient = finite_difference_gradient(objective_fn, current_theta, registry, eps=eps)
        grad_vector = np.array([gradient[name] for name in registry.names()], dtype=np.float64)
        grad_norm = float(np.linalg.norm(grad_vector))
        if grad_norm > 1.0e-12:
            grad_vector /= grad_norm
        candidate = dict(current_theta)
        for index, knob_name in enumerate(registry.names()):
            candidate[knob_name] += step_size * float(grad_vector[index])
        candidate = registry.clip_theta(candidate)
        candidate_value = float(objective_fn(candidate))
        if candidate_value >= current_value - 1.0e-12:
            current_theta = candidate
            current_value = candidate_value
            if candidate_value > best_value + 1.0e-12:
                best_theta = dict(candidate)
                best_value = candidate_value
    return best_theta, best_value


def random_restart_transition_motor_search(
    H0: np.ndarray,
    grid: FixedGrid,
    noise_ops: list[np.ndarray],
    basis: ControlBasis,
    registry: KnobRegistry,
    config: TransitionMotorConfig,
) -> TransitionMotorResult:
    times = np.linspace(config.time_min, config.time_max, config.n_time_samples, dtype=np.float64)
    baseline_theta = registry.defaults()

    def metrics_fn(theta: dict[str, float]) -> MotorMetrics:
        H = build_control_hamiltonian(H0, grid, theta, basis, registry)
        return evaluate_motor_metrics(
            H,
            noise_ops,
            grid,
            registry.clip_theta(theta),
            times,
            config.objective_weights,
            n_modes=config.n_transport_modes,
        )

    def objective_fn(theta: dict[str, float]) -> float:
        return float(metrics_fn(theta).objective)

    baseline_metrics = metrics_fn(baseline_theta)
    best_theta = dict(baseline_theta)
    best_metrics = baseline_metrics
    rng = np.random.default_rng(config.seed)

    restart_count = max(1, config.random_restarts)
    for restart_index in range(restart_count):
        if restart_index == 0:
            theta0 = dict(baseline_theta)
        else:
            theta0 = {
                knob.name: float(rng.uniform(knob.min_value, knob.max_value))
                for knob in registry.knobs
            }
        candidate_theta, _ = projected_gradient_ascent(
            objective_fn,
            theta0,
            registry,
            n_steps=config.optimizer_steps,
            step_size=config.optimizer_step_size,
            eps=config.finite_diff_eps,
        )
        candidate_metrics = metrics_fn(candidate_theta)
        if candidate_metrics.objective > best_metrics.objective + 1.0e-12:
            best_theta = dict(candidate_theta)
            best_metrics = candidate_metrics

    entries = compute_sensitivity_matrix(
        metrics_fn,
        best_theta,
        registry,
        eps=config.finite_diff_eps,
        metric_names=[
            "transport_efficiency",
            "noise_action_on_info",
            "noise_leakage",
            "control_cost",
            "objective",
        ],
    )
    return TransitionMotorResult(
        baseline_theta=dict(baseline_theta),
        best_theta=dict(best_theta),
        baseline_metrics=baseline_metrics,
        best_metrics=best_metrics,
        objective_improvement=float(best_metrics.objective - baseline_metrics.objective),
        top_sensitivities=top_sensitivities(entries, limit=3),
    )


def build_default_motor_basis(grid: FixedGrid) -> ControlBasis:
    return make_control_basis(grid)

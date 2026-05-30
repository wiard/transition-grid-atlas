"""Experimental Quantum & RTT Lab engine.

This module adds a theory switchboard on top of the existing transport atlas
without changing the legacy simulation entry points. The lab focuses on a
coherent Gaussian wave packet and supports four theory modes:

- ``standard_qm``: disorder-free unitary evolution
- ``anderson``: static-disorder unitary evolution
- ``lindblad``: density-matrix dephasing dynamics
- ``rtt``: adaptive unitary evolution with reflexive topology feedback
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import expm

from engine.feedback import feedback_potential
from engine.hamiltonian import build_tight_binding_hamiltonian
from engine.lindblad import coherence_norm, psi_to_rho, rk4_lindblad_step, trace_error as rho_trace_error
from engine.observables import edge_contact, frame_correlation, ipr, probability_from_psi, probability_from_rho, x_mean, x_variance
from engine.trajectories import config_hash, save_trajectory_npz
from engine.wavepacket import gaussian_wavepacket
from validation.hermitian import hermitian_error
from validation.statistics import linear_regression_metrics
from validation.unitarity import unitarity_error


LAB_RESULTS_FIELDS = [
    "run_id",
    "mode",
    "theory_mode",
    "seed",
    "W",
    "gamma",
    "k0",
    "sigma",
    "x0",
    "L",
    "T",
    "trace_error",
    "ipr_final",
    "alpha_late",
    "r2_late",
    "edge_hit",
    "trajectory_path",
    "config_hash",
    "falldown_candidate",
    "falldown_score",
    "frame_corr_late",
    "x_var_drift_late",
    "coherence_norm_final",
    "zeno_indicator",
    "artifact_hash",
]


@dataclass(slots=True)
class LabConfig:
    theory_mode: str
    grid_size: int
    steps: int
    dt: float
    hopping: float
    W: float
    bias: float
    eta: float
    gamma: float
    seed: int
    x0: float
    sigma: float
    k0: float
    feedback_smoothing: float
    burn_in_fraction: float
    falldown_variance_epsilon: float
    falldown_velocity_min: float
    edge_threshold: float
    trace_tolerance: float
    rtt_stagnation_window: int
    rtt_transport_epsilon: float
    rtt_adaptive_strength: float
    edge_width: int
    frame_corr_threshold: float
    rtt_experimental: bool
    topology_policy: str | None
    audit_feedback: str
    render_animate: bool
    render_fps: int


def make_gaussian_packet(L: int, x0: float, sigma: float, k0: float) -> np.ndarray:
    """Generate a normalized 1D Gaussian wave packet with momentum k0."""
    return gaussian_wavepacket(L=L, x0=x0, sigma=sigma, k0=k0)


def parse_lab_config(config: dict[str, Any]) -> LabConfig:
    """Merge legacy simulation/parameter blocks with the new lab section."""

    simulation_cfg = dict(config["simulation"])
    parameter_cfg = dict(config["parameters"])
    lab_cfg = dict(config.get("lab", {}))
    wave_packet_cfg = dict(lab_cfg.get("wave_packet", {}))
    render_cfg = dict(lab_cfg.get("render", {}))

    return LabConfig(
        theory_mode=str(lab_cfg.get("theory_mode", "rtt")).strip().lower() or "rtt",
        grid_size=int(lab_cfg.get("grid_size", simulation_cfg.get("grid_size", 129))),
        steps=int(lab_cfg.get("time_horizon_steps", simulation_cfg.get("t_max_steps", simulation_cfg.get("steps", 384)))),
        dt=float(lab_cfg.get("dt", simulation_cfg.get("dt", 0.25))),
        hopping=float(lab_cfg.get("hopping", simulation_cfg.get("hopping", 1.0))),
        W=float(lab_cfg.get("W", parameter_cfg.get("W", 0.0))),
        bias=float(lab_cfg.get("bias", parameter_cfg.get("bias", 0.0))),
        eta=float(lab_cfg.get("eta", parameter_cfg.get("eta", 0.0))),
        gamma=float(lab_cfg.get("gamma", parameter_cfg.get("gamma", 0.0))),
        seed=int(lab_cfg.get("seed", parameter_cfg.get("seed", 42))),
        x0=float(wave_packet_cfg.get("x0", lab_cfg.get("x0", 64.0))),
        sigma=float(wave_packet_cfg.get("sigma", 4.0)),
        k0=float(wave_packet_cfg.get("k0", 0.785)),
        feedback_smoothing=float(lab_cfg.get("feedback_smoothing", simulation_cfg.get("feedback_smoothing", 1.0))),
        burn_in_fraction=float(lab_cfg.get("burn_in_fraction", 0.5)),
        falldown_variance_epsilon=float(lab_cfg.get("falldown_variance_epsilon", 1.5)),
        falldown_velocity_min=float(lab_cfg.get("falldown_velocity_min", 0.05)),
        edge_threshold=float(lab_cfg.get("edge_threshold", 0.05)),
        trace_tolerance=float(lab_cfg.get("trace_tolerance", 1.0e-6)),
        rtt_stagnation_window=int(lab_cfg.get("rtt_stagnation_window", 8)),
        rtt_transport_epsilon=float(lab_cfg.get("rtt_transport_epsilon", 5.0e-3)),
        rtt_adaptive_strength=float(lab_cfg.get("rtt_adaptive_strength", 0.18)),
        edge_width=int(lab_cfg.get("edge_width", 4)),
        frame_corr_threshold=float(lab_cfg.get("frame_corr_threshold", 0.995)),
        rtt_experimental=bool(lab_cfg.get("experimental", True)),
        topology_policy=lab_cfg.get("topology_policy"),
        audit_feedback=str(lab_cfg.get("audit_feedback", "logged_only")),
        render_animate=bool(render_cfg.get("animate", True)),
        render_fps=int(render_cfg.get("fps", 30)),
    )


def _make_density_matrix(psi: np.ndarray) -> np.ndarray:
    return psi_to_rho(psi)


def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.real(probabilities).astype(np.float64, copy=False), a_min=0.0, a_max=None)
    total = np.sum(clipped, dtype=np.float64)
    if total <= 0.0:
        return np.zeros_like(clipped, dtype=np.float64)
    return (clipped / total).astype(np.float64, copy=False)


def _probabilities_from_state(state: np.ndarray) -> np.ndarray:
    return probability_from_psi(state)


def _probabilities_from_density_matrix(rho: np.ndarray) -> np.ndarray:
    return probability_from_rho(rho)


def _density_trace_error(rho: np.ndarray) -> float:
    return rho_trace_error(rho)


def _state_trace_error(probabilities: np.ndarray) -> float:
    return float(abs(np.sum(probabilities, dtype=np.float64) - np.float64(1.0)))


def _pure_dephasing_rhs(rho: np.ndarray, hamiltonian: np.ndarray, gamma: float) -> np.ndarray:
    commutator = hamiltonian @ rho - rho @ hamiltonian
    dephasing = np.diag(np.diag(rho)) - rho
    return (-1j * commutator + np.float64(gamma) * dephasing).astype(np.complex128, copy=False)


def _rk4_lindblad_step(rho: np.ndarray, hamiltonian: np.ndarray, dt: float, gamma: float) -> np.ndarray:
    dt64 = np.float64(dt)
    k1 = _pure_dephasing_rhs(rho, hamiltonian, gamma)
    k2 = _pure_dephasing_rhs(rho + 0.5 * dt64 * k1, hamiltonian, gamma)
    k3 = _pure_dephasing_rhs(rho + 0.5 * dt64 * k2, hamiltonian, gamma)
    k4 = _pure_dephasing_rhs(rho + dt64 * k3, hamiltonian, gamma)
    next_rho = rho + (dt64 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    next_rho = 0.5 * (next_rho + next_rho.conjugate().T)
    trace = np.trace(next_rho)
    if abs(trace) > 0.0:
        next_rho = next_rho / trace
    return next_rho.astype(np.complex128, copy=False)


def _variance_about_origin(probabilities: np.ndarray, origin: float) -> float:
    positions = np.arange(probabilities.size, dtype=np.float64)
    return float(np.sum(((positions - np.float64(origin)) ** 2) * probabilities, dtype=np.float64))


def _mean_position(probabilities: np.ndarray) -> float:
    return x_mean(probabilities)


def _spatial_variance(probabilities: np.ndarray, x_mean: float) -> float:
    _ = x_mean
    return x_variance(probabilities)


def _inverse_participation_ratio(probabilities: np.ndarray) -> float:
    return ipr(probabilities)


def _edge_contact(probabilities: np.ndarray, threshold: float) -> bool:
    return edge_contact(probabilities, edge_width=4, threshold=threshold)


def _rtt_topology_adjustment(size: int, center: float, strength: float) -> np.ndarray:
    adjustment = np.zeros((size, size), dtype=np.complex128)
    center_index = int(np.clip(round(center), 1, size - 2))
    coupling = np.float64(strength)
    for left in range(max(0, center_index - 2), min(size - 2, center_index + 2)):
        right = left + 2
        if right < size:
            adjustment[left, right] = np.complex128(-coupling)
            adjustment[right, left] = np.complex128(-coupling)
    return adjustment


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _late_alpha(times: np.ndarray, msd_origin: np.ndarray) -> float:
    positive_mask = times > 0.0
    times = times[positive_mask]
    msd_origin = msd_origin[positive_mask]
    if times.size < 4:
        return 0.0
    start_index = max(1, times.size // 2)
    sigma = np.sqrt(np.clip(msd_origin[start_index:], a_min=np.float64(1.0e-18), a_max=None))
    fit_times = np.log(times[start_index:])
    fit_sigma = np.log(sigma)
    regression = linear_regression_metrics(fit_times, fit_sigma)
    return float(regression["slope"])


def _falldown_detected(
    times: np.ndarray,
    x_mean: np.ndarray,
    x_var: np.ndarray,
    edge_hits: np.ndarray,
    config: LabConfig,
) -> bool:
    burn_in_index = min(max(1, int(round(config.burn_in_fraction * times.size))), times.size - 1)
    variance_growth = float(np.max(x_var[burn_in_index:]) - x_var[burn_in_index])
    if np.any(edge_hits[burn_in_index:]):
        return False
    if variance_growth >= config.falldown_variance_epsilon:
        return False
    tail_times = times[burn_in_index:]
    tail_mean = x_mean[burn_in_index:]
    if tail_times.size < 3:
        return False
    regression = linear_regression_metrics(tail_times, tail_mean)
    return bool(abs(float(regression["slope"])) >= config.falldown_velocity_min)


def _build_base_hamiltonian(config: LabConfig, effective_w: float) -> np.ndarray:
    return build_tight_binding_hamiltonian(
        size=config.grid_size,
        hopping=config.hopping,
        disorder_strength=effective_w,
        bias=config.bias,
        seed=config.seed,
    )


def _simulate_unitary_mode(config: LabConfig, theory_mode: str) -> dict[str, Any]:
    effective_w = 0.0 if theory_mode in {"standard_qm", "qm_free"} else config.W
    base_hamiltonian = _build_base_hamiltonian(config, effective_w)
    psi = make_gaussian_packet(config.grid_size, config.x0, config.sigma, config.k0)

    probability_frames: list[np.ndarray] = []
    trace_series: list[float] = []
    step_hamiltonians: list[np.ndarray] = []
    total_operator = np.eye(config.grid_size, dtype=np.complex128)

    probabilities = _probabilities_from_state(psi)
    probability_frames.append(probabilities)
    trace_series.append(_state_trace_error(probabilities))
    transport_history = [_variance_about_origin(probabilities, config.x0)]

    for _ in range(config.steps - 1):
        step_hamiltonian = base_hamiltonian
        if theory_mode == "rtt":
            current_probabilities = _probabilities_from_state(psi)
            step_hamiltonian = step_hamiltonian + feedback_potential(
                state=psi,
                eta=config.eta,
                smoothing=config.feedback_smoothing,
            )

            state_derivative = (-1j * step_hamiltonian @ psi).astype(np.complex128, copy=False)
            transport_now = _variance_about_origin(current_probabilities, config.x0)
            transport_history.append(transport_now)
            if (
                config.topology_policy
                and not config.rtt_experimental
                and len(transport_history) > config.rtt_stagnation_window
            ):
                current_center = _mean_position(current_probabilities)
                delta_transport = transport_history[-1] - transport_history[-1 - config.rtt_stagnation_window]
                derivative_norm = float(np.linalg.norm(state_derivative))
                stagnating = (
                    abs(delta_transport) <= config.rtt_transport_epsilon
                    or derivative_norm <= config.rtt_transport_epsilon
                )
                if stagnating:
                    step_hamiltonian = step_hamiltonian + _rtt_topology_adjustment(
                        size=config.grid_size,
                        center=current_center,
                        strength=config.rtt_adaptive_strength * max(config.eta, 1.0),
                    )
            elif config.audit_feedback == "logged_only":
                # Experimental RTT default: observe and log stagnation indicators
                # without mutating topology yet.
                step_hamiltonian = step_hamiltonian + _rtt_topology_adjustment(
                    size=config.grid_size,
                    center=_mean_position(current_probabilities),
                    strength=0.0,
                )

        step_operator = expm((-1j) * step_hamiltonian * np.float64(config.dt)).astype(np.complex128)
        total_operator = step_operator @ total_operator
        psi = step_operator @ psi
        norm = np.linalg.norm(psi)
        if norm > 0.0:
            psi = psi / np.complex128(norm)

        probabilities = _probabilities_from_state(psi)
        probability_frames.append(probabilities)
        trace_series.append(_state_trace_error(probabilities))
        step_hamiltonians.append(step_hamiltonian.astype(np.complex128, copy=False))

    return {
        "probability_frames": np.array(probability_frames, dtype=np.float64),
        "trace_series": np.array(trace_series, dtype=np.float64),
        "step_hamiltonians": step_hamiltonians or [base_hamiltonian],
        "total_operator": total_operator.astype(np.complex128, copy=False),
        "base_hamiltonian": base_hamiltonian,
    }


def _simulate_lindblad_mode(config: LabConfig) -> dict[str, Any]:
    base_hamiltonian = _build_base_hamiltonian(config, config.W)
    psi0 = make_gaussian_packet(config.grid_size, config.x0, config.sigma, config.k0)
    rho = _make_density_matrix(psi0)

    probability_frames: list[np.ndarray] = []
    trace_series: list[float] = []

    probabilities = _probabilities_from_density_matrix(rho)
    probability_frames.append(probabilities)
    trace_series.append(_density_trace_error(rho))

    for _ in range(config.steps - 1):
        rho = _rk4_lindblad_step(rho, base_hamiltonian, config.dt, config.gamma)
        probabilities = _probabilities_from_density_matrix(rho)
        probability_frames.append(probabilities)
        trace_series.append(_density_trace_error(rho))

    identity = np.eye(config.grid_size, dtype=np.complex128)
    return {
        "probability_frames": np.array(probability_frames, dtype=np.float64),
        "trace_series": np.array(trace_series, dtype=np.float64),
        "step_hamiltonians": [base_hamiltonian],
        "total_operator": identity,
        "base_hamiltonian": base_hamiltonian,
        "density_matrix_final": rho.astype(np.complex128, copy=False),
    }


def run_lab_simulation(
    config: dict[str, Any],
    run_id: str,
    trajectories_dir: Path,
    *,
    lab_config: LabConfig | None = None,
) -> dict[str, Any]:
    """Run the experimental lab and persist a trajectory artifact."""

    lab_config = lab_config or parse_lab_config(config)
    supported_modes = {"standard_qm", "qm_free", "anderson", "lindblad", "rtt"}
    if lab_config.theory_mode not in supported_modes:
        raise ValueError(f"Unsupported lab theory_mode: {lab_config.theory_mode}")
    if lab_config.grid_size < 8:
        raise ValueError("lab grid_size must be at least 8")
    if not (0.0 <= lab_config.x0 < lab_config.grid_size):
        raise ValueError("lab wave_packet.x0 must lie inside the grid")
    if lab_config.sigma <= 0.0:
        raise ValueError("lab wave_packet.sigma must be positive")
    if lab_config.steps < 4:
        raise ValueError("lab time_horizon_steps must be at least 4")

    if lab_config.theory_mode == "lindblad":
        simulation = _simulate_lindblad_mode(lab_config)
        effective_gamma = lab_config.gamma
        effective_w = lab_config.W
        effective_total_operator = np.eye(lab_config.grid_size, dtype=np.complex128)
        total_unitarity_error = 0.0
    else:
        simulation = _simulate_unitary_mode(lab_config, lab_config.theory_mode)
        effective_gamma = 0.0
        effective_w = 0.0 if lab_config.theory_mode in {"standard_qm", "qm_free"} else lab_config.W
        effective_total_operator = simulation["total_operator"]
        total_unitarity_error = unitarity_error(effective_total_operator)

    probability_frames = np.array(simulation["probability_frames"], dtype=np.float64)
    times = np.arange(probability_frames.shape[0], dtype=np.float64) * np.float64(lab_config.dt)
    x_mean = np.array([_mean_position(frame) for frame in probability_frames], dtype=np.float64)
    x_var = np.array([_spatial_variance(frame, mean) for frame, mean in zip(probability_frames, x_mean)], dtype=np.float64)
    msd_origin = np.array([_variance_about_origin(frame, lab_config.x0) for frame in probability_frames], dtype=np.float64)
    ipr = np.array([_inverse_participation_ratio(frame) for frame in probability_frames], dtype=np.float64)
    edge_hits = np.array([_edge_contact(frame, lab_config.edge_threshold) for frame in probability_frames], dtype=bool)
    trace_series = np.array(simulation["trace_series"], dtype=np.float64)
    trace_error = float(np.max(np.abs(trace_series)))
    alpha_late = _late_alpha(times, msd_origin)
    falldown = _falldown_detected(times, x_mean, x_var, edge_hits, lab_config)
    frame_corr = np.array(
        [frame_correlation(probability_frames[index], probability_frames[index + 1]) for index in range(probability_frames.shape[0] - 1)],
        dtype=np.float64,
    )
    burn_in_index = min(max(1, int(round(lab_config.burn_in_fraction * probability_frames.shape[0]))), probability_frames.shape[0] - 1)
    frame_corr_late = float(np.mean(frame_corr[burn_in_index - 1 :], dtype=np.float64)) if frame_corr.size else 0.0
    x_var_drift_late = float(np.max(x_var[burn_in_index:]) - x_var[burn_in_index])
    coherence_norm_final = float(coherence_norm(simulation["density_matrix_final"])) if "density_matrix_final" in simulation else 0.0
    zeno_indicator = bool(lab_config.gamma >= 0.5 and alpha_late <= 0.1)
    falldown_score = max(0.0, frame_corr_late - x_var_drift_late)
    max_hermitian_error = float(
        max(hermitian_error(step_h) for step_h in simulation["step_hamiltonians"])
    )

    trajectories_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = trajectories_dir / f"run_{run_id}.npz"
    metadata = {
        "run_id": run_id,
        "mode": "lab",
        "theory_mode": "qm_free" if lab_config.theory_mode == "standard_qm" else lab_config.theory_mode,
        "L": lab_config.grid_size,
        "T": int(probability_frames.shape[0]),
        "W": float(effective_w),
        "gamma": float(effective_gamma),
        "eta": float(lab_config.eta),
        "bias": float(lab_config.bias),
        "k0": float(lab_config.k0),
        "sigma": float(lab_config.sigma),
        "x0": float(lab_config.x0),
        "seed": int(lab_config.seed),
        "experimental": bool(lab_config.rtt_experimental),
        "topology_policy": lab_config.topology_policy,
        "audit_feedback": lab_config.audit_feedback,
    }
    cfg_hash = save_trajectory_npz(
        artifact_path,
        probabilities=probability_frames.astype(np.float64),
        times=times.astype(np.float64),
        x_mean=x_mean.astype(np.float64),
        x_var=x_var.astype(np.float64),
        metadata=metadata,
        extra_arrays={
            "ipr": ipr.astype(np.float64),
            "msd_origin": msd_origin.astype(np.float64),
            "edge_hits": edge_hits.astype(bool),
            "trace_series": trace_series.astype(np.float64),
            "frame_corr": frame_corr.astype(np.float64),
        },
    )
    artifact_hash = _sha256_file(artifact_path)

    summary_row = {
        "run_id": run_id,
        "mode": "lab",
        "theory_mode": metadata["theory_mode"],
        "seed": int(lab_config.seed),
        "W": float(effective_w),
        "gamma": float(effective_gamma),
        "k0": float(lab_config.k0),
        "sigma": float(lab_config.sigma),
        "x0": float(lab_config.x0),
        "L": int(lab_config.grid_size),
        "T": int(probability_frames.shape[0]),
        "trace_error": trace_error,
        "ipr_final": float(ipr[-1]),
        "alpha_late": float(alpha_late),
        "r2_late": float(linear_regression_metrics(
            np.log(np.clip(times[max(1, times.size // 2):], a_min=np.float64(1.0e-18), a_max=None)),
            np.log(np.sqrt(np.clip(msd_origin[max(1, times.size // 2):], a_min=np.float64(1.0e-18), a_max=None))),
        )["r2"]) if times.size >= 4 else 0.0,
        "edge_hit": bool(np.any(edge_hits)),
        "trajectory_path": str(artifact_path),
        "config_hash": cfg_hash,
        "falldown_candidate": bool(falldown and not np.any(edge_hits)),
        "falldown_score": float(falldown_score),
        "frame_corr_late": float(frame_corr_late),
        "x_var_drift_late": float(x_var_drift_late),
        "coherence_norm_final": float(coherence_norm_final),
        "zeno_indicator": bool(zeno_indicator),
        "artifact_hash": artifact_hash,
    }

    return {
        "run_id": run_id,
        "config": lab_config,
        "probability_frames": probability_frames,
        "times": times,
        "x_mean": x_mean,
        "x_var": x_var,
        "msd_origin": msd_origin,
        "ipr": ipr,
        "edge_hits": edge_hits,
        "trace_series": trace_series,
        "frame_corr": frame_corr,
        "trace_error": trace_error,
        "alpha_late": alpha_late,
        "falldown": falldown,
        "edge_hit": bool(np.any(edge_hits)),
        "artifact_path": artifact_path,
        "artifact_hash": artifact_hash,
        "summary_row": summary_row,
        "unitarity_error": float(total_unitarity_error),
        "hermitian_error": max_hermitian_error,
        "config_hash": cfg_hash,
    }

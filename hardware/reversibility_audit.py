"""Instrumental reversibility audit for the KTA transition motor."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml

from hardware.motor_ensemble import detector_distribution_at_time
from hardware.objective_mode_comparison import _load_yaml, _resolve_config_path
from hardware.transition_motor import (
    build_control_hamiltonian,
    build_default_motor_basis,
    random_restart_transition_motor_search,
    transition_motor_config_from_dict,
)
from hardware.transition_tuner import build_base_hamiltonian


DEFAULT_COHERENT_FIDELITY_MIN = 0.999999
DEFAULT_COHERENT_L2_ERROR_MAX = 1.0e-6


@dataclass(frozen=True)
class ReversibilityMetadata:
    coherent_reversibility_score: float
    coherent_loss_delta: float
    open_reversibility_score: float
    open_loss_delta: float
    dephasing_strength: float
    time_step_multiplier: float
    passed_coherent: bool


@dataclass(frozen=True)
class CoherentReversibilityResult:
    mode_name: str
    time_step_multiplier: float
    n_time_samples: int
    forward_time: float
    fidelity: float
    phase_aligned_l2_error: float
    probability_l1_error: float
    return_probability: float
    loss_delta: float
    passed: bool


@dataclass(frozen=True)
class OpenReversibilityResult:
    mode_name: str
    time_step_multiplier: float
    n_time_samples: int
    forward_time: float
    dephasing_strength: float
    return_fidelity: float
    trace_error: float
    hermiticity_error: float
    return_probability: float
    loss_delta: float
    warning: bool


@dataclass(frozen=True)
class ReversibilityAuditSummary:
    n_modes: int
    coherent_all_passed: bool
    min_coherent_fidelity: float
    max_coherent_l2_error: float
    max_coherent_loss_delta: float
    max_open_loss_delta: float
    time_resolution_sensitive: bool
    recommended_registry_action: str


def unitary_from_hamiltonian(H: np.ndarray, t: float) -> np.ndarray:
    """Compute U(t)=exp(-i H t) for Hermitian H using numpy eigendecomposition."""

    operator = np.asarray(H, dtype=np.complex128)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("H must be a square matrix")
    eigenvalues, eigenvectors = np.linalg.eigh(operator)
    phases = np.exp(-1j * eigenvalues * float(t))
    return eigenvectors @ np.diag(phases) @ np.conjugate(eigenvectors).T


def basis_state(n_sites: int, index: int) -> np.ndarray:
    if n_sites <= 0:
        raise ValueError("n_sites must be positive")
    if index < 0 or index >= n_sites:
        raise ValueError("index out of range")
    psi = np.zeros(n_sites, dtype=np.complex128)
    psi[int(index)] = 1.0
    return psi


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    state = np.asarray(psi, dtype=np.complex128).reshape(-1)
    norm = float(np.linalg.norm(state))
    if norm <= 1.0e-15:
        raise ValueError("state norm must be positive")
    return state / norm


def state_fidelity(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    """|<psi_a|psi_b>|^2 with normalized states."""

    left = _normalize_state(psi_a)
    right = _normalize_state(psi_b)
    overlap = np.vdot(left, right)
    return float(np.clip(abs(overlap) ** 2, 0.0, 1.0))


def phase_aligned_l2_error(psi_ref: np.ndarray, psi: np.ndarray) -> float:
    """L2 error after removing global phase."""

    ref = _normalize_state(psi_ref)
    state = _normalize_state(psi)
    overlap = np.vdot(ref, state)
    if abs(overlap) > 1.0e-15:
        state = state * np.exp(-1j * np.angle(overlap))
    return float(np.linalg.norm(ref - state))


def probability_l1_error(psi_ref: np.ndarray, psi: np.ndarray) -> float:
    ref = np.abs(_normalize_state(psi_ref)) ** 2
    state = np.abs(_normalize_state(psi)) ** 2
    return float(np.sum(np.abs(ref - state)))


def density_from_state(psi: np.ndarray) -> np.ndarray:
    state = _normalize_state(psi)
    return np.outer(state, np.conjugate(state))


def apply_dephasing_channel(rho: np.ndarray, strength: float) -> np.ndarray:
    """rho -> (1-strength) * rho + strength * diag(diag(rho))."""

    if strength < 0.0 or strength > 1.0:
        raise ValueError("strength must be in [0, 1]")
    matrix = np.asarray(rho, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("rho must be square")
    diagonal = np.diag(np.diag(matrix))
    out = (1.0 - float(strength)) * matrix + float(strength) * diagonal
    return 0.5 * (out + np.conjugate(out.T))


def reverse_density_matrix(H: np.ndarray, rho: np.ndarray, t: float) -> np.ndarray:
    """rho_rev = U(-t) rho U(-t)^dagger."""

    U_rev = unitary_from_hamiltonian(H, -float(t))
    matrix = np.asarray(rho, dtype=np.complex128)
    out = U_rev @ matrix @ np.conjugate(U_rev.T)
    return 0.5 * (out + np.conjugate(out.T))


def pure_state_return_fidelity(rho: np.ndarray, psi0: np.ndarray) -> float:
    """<psi0|rho|psi0>"""

    state = _normalize_state(psi0)
    matrix = np.asarray(rho, dtype=np.complex128)
    value = np.vdot(state, matrix @ state)
    return float(np.clip(np.real_if_close(value).item(), 0.0, 1.0))


def adjusted_n_time_samples(
    *,
    time_min: float,
    time_max: float,
    n_time_samples: int,
    time_step_multiplier: float,
) -> int:
    if time_step_multiplier <= 0.0:
        raise ValueError("time_step_multiplier must be positive")
    if time_max <= time_min:
        raise ValueError("time_max must be greater than time_min")
    if n_time_samples < 2:
        raise ValueError("n_time_samples must be at least 2")

    span = float(time_max - time_min)
    base_dt = span / float(n_time_samples - 1)
    new_dt = base_dt * float(time_step_multiplier)
    new_n = int(np.floor((span / new_dt) + 1.0e-12)) + 1
    return max(2, new_n)


def choose_forward_time(
    H: np.ndarray,
    input_index: int,
    target_indices: list[int],
    times: np.ndarray,
    *,
    method: str,
) -> float:
    grid = np.asarray(times, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 1:
        raise ValueError("times must be a non-empty 1D array")
    if method == "final":
        return float(grid[-1])
    if method != "peak_target":
        raise ValueError(f"unsupported forward_time_selection: {method}")

    scores = [
        float(np.sum(detector_distribution_at_time(H, input_index, float(t))[target_indices]))
        for t in grid
    ]
    return float(grid[int(np.argmax(np.asarray(scores, dtype=np.float64)))])


def compute_reversibility_metadata_for_hamiltonian(
    *,
    H: np.ndarray,
    input_index: int,
    target_indices: list[int],
    time_min: float,
    time_max: float,
    n_time_samples: int,
    dephasing_strength: float = 0.05,
    time_step_multiplier: float = 1.0,
    forward_time_selection: str = "peak_target",
    coherent_fidelity_min: float = DEFAULT_COHERENT_FIDELITY_MIN,
    coherent_l2_error_max: float = DEFAULT_COHERENT_L2_ERROR_MAX,
) -> ReversibilityMetadata:
    adjusted_samples = adjusted_n_time_samples(
        time_min=float(time_min),
        time_max=float(time_max),
        n_time_samples=int(n_time_samples),
        time_step_multiplier=float(time_step_multiplier),
    )
    times = np.linspace(float(time_min), float(time_max), adjusted_samples, dtype=np.float64)
    forward_time = choose_forward_time(
        np.asarray(H, dtype=np.complex128),
        int(input_index),
        [int(value) for value in target_indices],
        times,
        method=forward_time_selection,
    )
    psi0 = basis_state(int(np.asarray(H).shape[0]), int(input_index))
    U_fwd = unitary_from_hamiltonian(H, forward_time)
    U_rev = unitary_from_hamiltonian(H, -forward_time)
    psi_rev = U_rev @ (U_fwd @ psi0)
    coherent_score = state_fidelity(psi0, psi_rev)
    coherent_return_probability = float(np.clip(abs(np.vdot(psi0, psi_rev)) ** 2, 0.0, 1.0))
    coherent_loss = float(max(0.0, 1.0 - coherent_return_probability))
    l2_error = phase_aligned_l2_error(psi0, psi_rev)
    passed_coherent = bool(
        coherent_score >= float(coherent_fidelity_min)
        and l2_error <= float(coherent_l2_error_max)
    )

    psi_fwd = U_fwd @ psi0
    rho_fwd = density_from_state(psi_fwd)
    rho_dephased = apply_dephasing_channel(rho_fwd, float(dephasing_strength))
    rho_rev = reverse_density_matrix(H, rho_dephased, forward_time)
    open_score = pure_state_return_fidelity(rho_rev, psi0)
    open_loss = float(max(0.0, 1.0 - open_score))

    return ReversibilityMetadata(
        coherent_reversibility_score=float(coherent_score),
        coherent_loss_delta=coherent_loss,
        open_reversibility_score=float(open_score),
        open_loss_delta=open_loss,
        dephasing_strength=float(dephasing_strength),
        time_step_multiplier=float(time_step_multiplier),
        passed_coherent=passed_coherent,
    )


def compute_reversibility_metadata_for_mode(
    *,
    mode_name: str,
    mode_config: dict[str, Any],
    dephasing_strength: float = 0.05,
    time_step_multiplier: float = 1.0,
) -> ReversibilityMetadata:
    del mode_name
    H_best, motor_config, _ = _mode_hamiltonian(mode_config, time_step_multiplier=float(time_step_multiplier))
    return compute_reversibility_metadata_for_hamiltonian(
        H=H_best,
        input_index=motor_config.grid.input_index,
        target_indices=list(motor_config.grid.target_indices),
        time_min=motor_config.time_min,
        time_max=motor_config.time_max,
        n_time_samples=motor_config.n_time_samples,
        dephasing_strength=float(dephasing_strength),
        time_step_multiplier=float(time_step_multiplier),
    )


def _reversibility_block(config: dict[str, Any]) -> dict[str, Any]:
    if "reversibility_audit" not in config:
        raise ValueError("config must define reversibility_audit")
    return dict(config["reversibility_audit"])


def _load_mode_payloads(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    block = _reversibility_block(config)
    payloads: list[tuple[str, dict[str, Any]]] = []
    for item in list(block["operating_modes"]):
        entry = dict(item)
        path = _resolve_config_path(config, str(entry["config"]))
        payloads.append((str(entry["name"]), _load_yaml(path)))
    return payloads


def _mode_hamiltonian(payload: dict[str, Any], *, time_step_multiplier: float) -> tuple[np.ndarray, Any, int]:
    adjusted_payload = yaml.safe_load(yaml.safe_dump(payload, sort_keys=False))
    block = dict(adjusted_payload["transition_motor"])
    optimizer = dict(block["optimizer"])
    optimizer["n_time_samples"] = adjusted_n_time_samples(
        time_min=float(optimizer["time_min"]),
        time_max=float(optimizer["time_max"]),
        n_time_samples=int(optimizer["n_time_samples"]),
        time_step_multiplier=float(time_step_multiplier),
    )
    block["optimizer"] = optimizer
    adjusted_payload["transition_motor"] = block

    motor_config, _ = transition_motor_config_from_dict(adjusted_payload)
    H0 = build_base_hamiltonian(motor_config.grid)
    basis = build_default_motor_basis(motor_config.grid)
    result = random_restart_transition_motor_search(
        H0,
        motor_config.grid,
        motor_config.noise_profiles,
        basis,
        motor_config.knob_registry,
        motor_config,
    )
    H_best = build_control_hamiltonian(
        H0,
        motor_config.grid,
        result.best_theta,
        basis,
        motor_config.knob_registry,
    )
    return H_best, motor_config, int(motor_config.n_time_samples)


def _coherent_result(
    *,
    mode_name: str,
    H: np.ndarray,
    motor_config: Any,
    time_step_multiplier: float,
    n_time_samples: int,
    forward_time_selection: str,
    thresholds: dict[str, float],
) -> CoherentReversibilityResult:
    times = np.linspace(motor_config.time_min, motor_config.time_max, n_time_samples, dtype=np.float64)
    forward_time = choose_forward_time(
        H,
        motor_config.grid.input_index,
        list(motor_config.grid.target_indices),
        times,
        method=forward_time_selection,
    )
    psi0 = basis_state(motor_config.grid.n_sites, motor_config.grid.input_index)
    U_fwd = unitary_from_hamiltonian(H, forward_time)
    U_rev = unitary_from_hamiltonian(H, -forward_time)
    psi_fwd = U_fwd @ psi0
    psi_rev = U_rev @ psi_fwd
    fidelity = state_fidelity(psi0, psi_rev)
    l2_error = phase_aligned_l2_error(psi0, psi_rev)
    prob_l1 = probability_l1_error(psi0, psi_rev)
    return_probability = float(np.clip(abs(np.vdot(psi0, psi_rev)) ** 2, 0.0, 1.0))
    loss_delta = float(max(0.0, 1.0 - return_probability))
    passed = bool(
        fidelity >= float(thresholds["coherent_fidelity_min"])
        and l2_error <= float(thresholds["coherent_l2_error_max"])
    )
    return CoherentReversibilityResult(
        mode_name=mode_name,
        time_step_multiplier=float(time_step_multiplier),
        n_time_samples=int(n_time_samples),
        forward_time=float(forward_time),
        fidelity=float(fidelity),
        phase_aligned_l2_error=float(l2_error),
        probability_l1_error=float(prob_l1),
        return_probability=float(return_probability),
        loss_delta=float(loss_delta),
        passed=passed,
    )


def _open_results(
    *,
    mode_name: str,
    H: np.ndarray,
    motor_config: Any,
    time_step_multiplier: float,
    n_time_samples: int,
    forward_time: float,
    dephasing_strengths: list[float],
    thresholds: dict[str, float],
) -> list[OpenReversibilityResult]:
    psi0 = basis_state(motor_config.grid.n_sites, motor_config.grid.input_index)
    psi_fwd = unitary_from_hamiltonian(H, forward_time) @ psi0
    rho_fwd = density_from_state(psi_fwd)
    results: list[OpenReversibilityResult] = []
    for strength in dephasing_strengths:
        rho_dephased = apply_dephasing_channel(rho_fwd, float(strength))
        rho_rev = reverse_density_matrix(H, rho_dephased, forward_time)
        return_fidelity = pure_state_return_fidelity(rho_rev, psi0)
        trace_error = float(abs(np.trace(rho_rev) - 1.0))
        hermiticity_error = float(np.linalg.norm(rho_rev - np.conjugate(rho_rev.T)))
        return_probability = float(return_fidelity)
        loss_delta = float(max(0.0, 1.0 - return_probability))
        warning = bool(
            return_fidelity < float(thresholds["open_fidelity_warning"])
            or loss_delta > float(thresholds["loss_delta_warning"])
        )
        results.append(
            OpenReversibilityResult(
                mode_name=mode_name,
                time_step_multiplier=float(time_step_multiplier),
                n_time_samples=int(n_time_samples),
                forward_time=float(forward_time),
                dephasing_strength=float(strength),
                return_fidelity=float(return_fidelity),
                trace_error=trace_error,
                hermiticity_error=hermiticity_error,
                return_probability=return_probability,
                loss_delta=loss_delta,
                warning=warning,
            )
        )
    return results


def _relative_change(values: list[float], baseline_index: int) -> float:
    baseline = float(values[baseline_index])
    changes: list[float] = []
    for index, value in enumerate(values):
        if index == baseline_index:
            continue
        delta = abs(float(value) - baseline)
        if abs(baseline) <= 1.0e-12:
            changes.append(delta)
        else:
            changes.append(float(delta / abs(baseline)))
    return float(max(changes, default=0.0))


def summarize_reversibility_audit(
    coherent_results: list[CoherentReversibilityResult],
    open_results: list[OpenReversibilityResult],
    *,
    time_resolution_relative_change: float,
    loss_delta_warning: float,
    baseline_multiplier: float = 1.0,
) -> ReversibilityAuditSummary:
    if not coherent_results:
        raise ValueError("coherent_results must be non-empty")

    mode_names = sorted({result.mode_name for result in coherent_results})
    coherent_all_passed = all(result.passed for result in coherent_results)
    min_coherent_fidelity = float(min(result.fidelity for result in coherent_results))
    max_coherent_l2_error = float(max(result.phase_aligned_l2_error for result in coherent_results))
    max_coherent_loss_delta = float(max(result.loss_delta for result in coherent_results))
    max_open_loss_delta = float(max((result.loss_delta for result in open_results), default=0.0))

    sensitivity_values: list[float] = []
    for mode_name in mode_names:
        mode_coherent = sorted(
            [item for item in coherent_results if item.mode_name == mode_name],
            key=lambda item: item.time_step_multiplier,
        )
        multipliers = [item.time_step_multiplier for item in mode_coherent]
        if not any(np.isclose(multiplier, baseline_multiplier) for multiplier in multipliers):
            raise ValueError("each mode must include the baseline multiplier")
        baseline_index = next(index for index, value in enumerate(multipliers) if np.isclose(value, baseline_multiplier))
        sensitivity_values.append(_relative_change([item.return_probability for item in mode_coherent], baseline_index))
        sensitivity_values.append(_relative_change([item.loss_delta for item in mode_coherent], baseline_index))

        strengths = sorted({item.dephasing_strength for item in open_results if item.mode_name == mode_name})
        for strength in strengths:
            mode_open = sorted(
                [
                    item
                    for item in open_results
                    if item.mode_name == mode_name and np.isclose(item.dephasing_strength, strength)
                ],
                key=lambda item: item.time_step_multiplier,
            )
            if not mode_open:
                continue
            sensitivity_values.append(_relative_change([item.return_fidelity for item in mode_open], baseline_index))
            sensitivity_values.append(_relative_change([item.loss_delta for item in mode_open], baseline_index))

    time_resolution_sensitive = bool(max(sensitivity_values, default=0.0) > float(time_resolution_relative_change))

    if not coherent_all_passed:
        recommended_registry_action = "Investigate numerical propagation / Hamiltonian Hermiticity before registry promotion."
    elif time_resolution_sensitive:
        recommended_registry_action = "Track time_step_resolution as audited operating-mode metadata."
    elif max_open_loss_delta > float(loss_delta_warning):
        recommended_registry_action = "Open/noisy dynamics are not reversible under coherent inverse; treat as irreversibility due to dephasing."
    else:
        recommended_registry_action = "Do not add time_step_resolution as a registry knob; keep as audit metadata."

    return ReversibilityAuditSummary(
        n_modes=len(mode_names),
        coherent_all_passed=coherent_all_passed,
        min_coherent_fidelity=min_coherent_fidelity,
        max_coherent_l2_error=max_coherent_l2_error,
        max_coherent_loss_delta=max_coherent_loss_delta,
        max_open_loss_delta=max_open_loss_delta,
        time_resolution_sensitive=time_resolution_sensitive,
        recommended_registry_action=recommended_registry_action,
    )


def run_reversibility_audit(
    config: dict[str, Any],
) -> tuple[list[CoherentReversibilityResult], list[OpenReversibilityResult], ReversibilityAuditSummary]:
    block = _reversibility_block(config)
    thresholds = {str(key): float(value) for key, value in dict(block["thresholds"]).items()}
    coherent_results: list[CoherentReversibilityResult] = []
    open_results: list[OpenReversibilityResult] = []

    for mode_name, payload in _load_mode_payloads(config):
        for multiplier in [float(value) for value in list(block["time_step_multipliers"])]:
            H_best, motor_config, n_time_samples = _mode_hamiltonian(payload, time_step_multiplier=multiplier)
            coherent = _coherent_result(
                mode_name=mode_name,
                H=H_best,
                motor_config=motor_config,
                time_step_multiplier=multiplier,
                n_time_samples=n_time_samples,
                forward_time_selection=str(block["forward_time_selection"]),
                thresholds=thresholds,
            )
            coherent_results.append(coherent)
            open_results.extend(
                _open_results(
                    mode_name=mode_name,
                    H=H_best,
                    motor_config=motor_config,
                    time_step_multiplier=multiplier,
                    n_time_samples=n_time_samples,
                    forward_time=coherent.forward_time,
                    dephasing_strengths=[float(value) for value in list(block["dephasing_strengths"])],
                    thresholds=thresholds,
                )
            )

    summary = summarize_reversibility_audit(
        coherent_results,
        open_results,
        time_resolution_relative_change=float(thresholds["time_resolution_relative_change"]),
        loss_delta_warning=float(thresholds["loss_delta_warning"]),
    )
    return coherent_results, open_results, summary


def write_reversibility_csv(
    path: str | Path,
    coherent_results: list[CoherentReversibilityResult],
    open_results: list[OpenReversibilityResult],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    coherent_fields = list(CoherentReversibilityResult.__dataclass_fields__.keys())
    open_fields = list(OpenReversibilityResult.__dataclass_fields__.keys())
    fieldnames = [
        "result_type",
        *coherent_fields,
        *(field for field in open_fields if field not in coherent_fields),
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in coherent_results:
            row = {field: "" for field in fieldnames}
            row["result_type"] = "coherent"
            row.update(asdict(result))
            writer.writerow(row)
        for result in open_results:
            row = {field: "" for field in fieldnames}
            row["result_type"] = "open"
            row.update(asdict(result))
            writer.writerow(row)
    return output


def write_reversibility_summary_json(
    path: str | Path,
    summary: ReversibilityAuditSummary,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8")
    return output


def plot_reversibility_audit(
    coherent_results: list[CoherentReversibilityResult],
    open_results: list[OpenReversibilityResult],
    output_path: str | Path,
) -> Path:
    if not coherent_results:
        raise ValueError("coherent_results must be non-empty")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Instrumental reversibility audit")

    mode_names = sorted({result.mode_name for result in coherent_results})
    for mode_name in mode_names:
        mode_coherent = sorted(
            [result for result in coherent_results if result.mode_name == mode_name],
            key=lambda result: result.time_step_multiplier,
        )
        multipliers = [result.time_step_multiplier for result in mode_coherent]
        axes[0, 0].plot(multipliers, [result.fidelity for result in mode_coherent], marker="o", label=mode_name)
        axes[0, 1].plot(multipliers, [result.loss_delta for result in mode_coherent], marker="o", label=mode_name)

    axes[0, 0].axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[0, 0].set_title("Coherent fidelity vs time-step multiplier")
    axes[0, 0].set_xlabel("time_step_multiplier")
    axes[0, 0].set_ylabel("fidelity")
    axes[0, 0].legend()

    axes[0, 1].axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[0, 1].set_title("Coherent loss delta vs time-step multiplier")
    axes[0, 1].set_xlabel("time_step_multiplier")
    axes[0, 1].set_ylabel("loss_delta")
    axes[0, 1].legend()

    for mode_name in mode_names:
        for multiplier in sorted({result.time_step_multiplier for result in open_results if result.mode_name == mode_name}):
            mode_open = sorted(
                [
                    result
                    for result in open_results
                    if result.mode_name == mode_name and np.isclose(result.time_step_multiplier, multiplier)
                ],
                key=lambda result: result.dephasing_strength,
            )
            label = f"{mode_name} x{multiplier:g}"
            axes[1, 0].plot(
                [result.dephasing_strength for result in mode_open],
                [result.return_fidelity for result in mode_open],
                marker="o",
                label=label,
            )
            axes[1, 1].plot(
                [result.dephasing_strength for result in mode_open],
                [result.loss_delta for result in mode_open],
                marker="o",
                label=label,
            )

    axes[1, 0].set_title("Open return fidelity vs dephasing strength")
    axes[1, 0].set_xlabel("dephasing_strength")
    axes[1, 0].set_ylabel("return_fidelity")
    axes[1, 0].legend(fontsize="small")

    axes[1, 1].set_title("Open loss delta vs dephasing strength")
    axes[1, 1].set_xlabel("dephasing_strength")
    axes[1, 1].set_ylabel("loss_delta")
    axes[1, 1].legend(fontsize="small")

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def load_reversibility_audit_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config["__config_path__"] = str(config_path)
    return config

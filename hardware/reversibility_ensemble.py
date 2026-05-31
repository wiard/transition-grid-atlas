"""Synthetic ensemble validation for reversibility-aware operating modes."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml

from hardware.control_basis import make_control_basis
from hardware.motor_ensemble import (
    DetailedMotorEnsembleSample,
    TransitionMotorEnsembleConfig,
    apply_fabrication_disorder,
    detector_success_probability,
    run_single_motor_ensemble_sample_detailed,
    sample_fabrication_disorder_profiles,
    sample_phase_noise_profiles,
)
from hardware.motor_pareto import ObjectiveWeightSet, transition_motor_pareto_from_dict
from hardware.motor_pareto_audit import (
    _load_base_pareto_config,
    load_audit_config,
    merge_base_and_stress_weight_sets,
)
from hardware.objective_mode_comparison import _load_yaml, _resolve_config_path
from hardware.objective_modes import ObjectiveMode
from hardware.reversibility_audit import (
    adjusted_n_time_samples,
    apply_dephasing_channel,
    basis_state,
    choose_forward_time,
    density_from_state,
    phase_aligned_l2_error,
    pure_state_return_fidelity,
    reverse_density_matrix,
    state_fidelity,
    unitary_from_hamiltonian,
)
from hardware.transition_motor import build_control_hamiltonian
from hardware.transition_tuner import FixedGrid, build_base_hamiltonian, fixed_grid_from_dict


@dataclass(frozen=True)
class ReversibilityEnsembleSampleResult:
    sample_id: int
    mode_name: str
    dephasing_strength: float
    detector_success: float
    coherent_reversibility_score: float
    coherent_loss_delta: float
    open_reversibility_score: float
    open_loss_delta: float
    return_probability: float
    phase_noise_profile_index: int | None
    time_step_multiplier: float


@dataclass(frozen=True)
class ReversibilityModeSummary:
    mode_name: str
    dephasing_strength: float
    n_samples: int
    mean_detector_success: float
    median_detector_success: float
    std_detector_success: float
    mean_reversibility_score: float
    median_reversibility_score: float
    std_reversibility_score: float
    mean_open_loss_delta: float
    median_open_loss_delta: float
    std_open_loss_delta: float
    detector_success_cv: float


@dataclass(frozen=True)
class ReversibilityEnsembleSummary:
    n_samples: int
    operating_modes: list[str]
    dephasing_strengths: list[float]
    best_reversibility_by_dephasing: dict[float, str]
    lowest_open_loss_by_dephasing: dict[float, str]
    best_detector_stability_by_dephasing: dict[float, str]
    detector_reversibility_corr_by_dephasing: dict[float, float]
    detector_loss_corr_by_dephasing: dict[float, float]
    preferred_mode_by_dephasing: dict[float, str]


@dataclass(frozen=True)
class LoadedOperatingMode:
    name: str
    preset_name: str
    ensemble_config: TransitionMotorEnsembleConfig
    weight_set: ObjectiveWeightSet


def _ensemble_block(config: dict[str, Any]) -> dict[str, Any]:
    if "reversibility_ensemble" not in config:
        raise ValueError("config must define reversibility_ensemble")
    return dict(config["reversibility_ensemble"])


def _permissive_success_criteria() -> dict[str, float]:
    return {
        "min_objective_gain": -1.0e12,
        "min_detector_gain": -1.0e12,
        "min_noise_action_reduction": -1.0e12,
    }


def _grid_equals(left: FixedGrid, right: FixedGrid) -> bool:
    return (
        left.n_sites == right.n_sites
        and left.edges == right.edges
        and left.base_coupling == right.base_coupling
        and left.input_index == right.input_index
        and left.target_indices == right.target_indices
    )


def _load_pareto_payload(path: Path) -> dict[str, Any]:
    payload = _load_yaml(path)
    if "transition_motor_pareto_audit" in payload:
        audit_config = load_audit_config(path)
        base_config = _load_base_pareto_config(audit_config)
        merged = merge_base_and_stress_weight_sets(base_config, audit_config)
        merged["__config_path__"] = str(path)
        return merged
    if "transition_motor_pareto" in payload:
        payload["__config_path__"] = str(path)
        return payload
    raise ValueError("operating mode config must define transition_motor_pareto or transition_motor_pareto_audit")


def _load_operating_modes(config: dict[str, Any]) -> tuple[FixedGrid, list[LoadedOperatingMode]]:
    block = _ensemble_block(config)
    reference_grid = fixed_grid_from_dict(dict(block["grid"]))
    loaded_modes: list[LoadedOperatingMode] = []
    for item in list(block["operating_modes"]):
        entry = dict(item)
        config_path = _resolve_config_path(config, str(entry["config"]))
        preset_name = str(entry["preset_name"])
        merged_payload = _load_pareto_payload(config_path)
        parsed = transition_motor_pareto_from_dict(merged_payload)
        if not _grid_equals(parsed.ensemble_config.motor_config.grid, reference_grid):
            raise ValueError(f"operating mode grid mismatch for preset {preset_name}")
        lookup = {weight_set.name: weight_set for weight_set in parsed.weight_sets}
        if preset_name not in lookup:
            raise ValueError(f"preset not found in config: {preset_name}")
        loaded_modes.append(
            LoadedOperatingMode(
                name=str(entry["name"]),
                preset_name=preset_name,
                ensemble_config=parsed.ensemble_config,
                weight_set=lookup[preset_name],
            )
        )
    return reference_grid, loaded_modes


def _motor_config_for_mode(
    mode: LoadedOperatingMode,
    *,
    time_step_multiplier: float,
):
    base_motor = mode.ensemble_config.motor_config
    adjusted_samples = adjusted_n_time_samples(
        time_min=base_motor.time_min,
        time_max=base_motor.time_max,
        n_time_samples=base_motor.n_time_samples,
        time_step_multiplier=time_step_multiplier,
    )
    objective_mode = ObjectiveMode(
        name=mode.weight_set.name,
        mode=mode.weight_set.mode,
        transport_weight=mode.weight_set.transport,
        noise_action_weight=mode.weight_set.noise_action,
        leakage_weight=mode.weight_set.leakage,
        control_cost_weight=mode.weight_set.control_cost,
        normalization=mode.weight_set.normalization,
        description=mode.weight_set.description or f"Ensemble preset {mode.weight_set.name}",
    )
    return replace(
        base_motor,
        objective_weights=objective_mode.weights_dict(),
        objective_mode=objective_mode,
        n_time_samples=adjusted_samples,
    )


def build_mode_hamiltonian_for_sample(
    *,
    mode_name: str,
    base_grid_config: dict[str, Any],
    onsite_disorder: np.ndarray,
    phase_noise_profiles: list[np.ndarray],
    mode_config: LoadedOperatingMode,
    time_step_multiplier: float,
) -> tuple[np.ndarray, DetailedMotorEnsembleSample]:
    reference_grid = fixed_grid_from_dict(dict(base_grid_config))
    grid = mode_config.ensemble_config.motor_config.grid
    if not _grid_equals(grid, reference_grid):
        raise ValueError(f"mode {mode_name} does not match the ensemble grid")
    motor_config = _motor_config_for_mode(mode_config, time_step_multiplier=time_step_multiplier)
    registry = motor_config.knob_registry
    basis = make_control_basis(grid)
    detail = run_single_motor_ensemble_sample_detailed(
        grid,
        registry,
        basis,
        np.asarray(onsite_disorder, dtype=np.float64),
        [np.asarray(profile, dtype=np.float64) for profile in phase_noise_profiles],
        motor_config,
        _permissive_success_criteria(),
    )
    disordered_H0 = apply_fabrication_disorder(build_base_hamiltonian(grid), np.asarray(onsite_disorder, dtype=np.float64))
    best_H = build_control_hamiltonian(
        disordered_H0,
        grid,
        detail.best_theta,
        basis,
        registry,
    )
    return best_H, detail


def run_single_reversibility_ensemble_sample(
    *,
    sample_id: int,
    mode_name: str,
    H: np.ndarray,
    input_index: int,
    target_indices: list[int],
    dephasing_strength: float,
    time_min: float,
    time_max: float,
    n_time_samples: int,
    time_step_multiplier: float,
    forward_time_selection: str,
    phase_noise_profile_index: int | None = None,
) -> ReversibilityEnsembleSampleResult:
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
    psi_fwd = U_fwd @ psi0
    psi_rev = U_rev @ psi_fwd
    coherent_score = state_fidelity(psi0, psi_rev)
    coherent_return_probability = float(np.clip(abs(np.vdot(psi0, psi_rev)) ** 2, 0.0, 1.0))
    coherent_loss = float(max(0.0, 1.0 - coherent_return_probability))
    _ = phase_aligned_l2_error(psi0, psi_rev)

    rho_fwd = density_from_state(psi_fwd)
    rho_dephased = apply_dephasing_channel(rho_fwd, float(dephasing_strength))
    rho_rev = reverse_density_matrix(H, rho_dephased, forward_time)
    open_score = pure_state_return_fidelity(rho_rev, psi0)
    open_loss = float(max(0.0, 1.0 - open_score))
    detector_success = detector_success_probability(
        np.asarray(H, dtype=np.complex128),
        int(input_index),
        [int(value) for value in target_indices],
        times,
    )
    return ReversibilityEnsembleSampleResult(
        sample_id=int(sample_id),
        mode_name=str(mode_name),
        dephasing_strength=float(dephasing_strength),
        detector_success=float(detector_success),
        coherent_reversibility_score=float(coherent_score),
        coherent_loss_delta=coherent_loss,
        open_reversibility_score=float(open_score),
        open_loss_delta=open_loss,
        return_probability=float(open_score),
        phase_noise_profile_index=phase_noise_profile_index,
        time_step_multiplier=float(time_step_multiplier),
    )


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    left = np.asarray(x, dtype=np.float64).reshape(-1)
    right = np.asarray(y, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("x and y must have the same shape")
    if left.size < 2:
        return 0.0
    if np.std(left) <= 1.0e-15 or np.std(right) <= 1.0e-15:
        return 0.0
    corr = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip(corr, -1.0, 1.0))


def _detector_success_cv(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if abs(mean) <= 1.0e-15:
        return 0.0 if std <= 1.0e-15 else float("inf")
    return float(std / abs(mean))


def summarize_reversibility_ensemble(
    results: list[ReversibilityEnsembleSampleResult],
    *,
    detector_tolerance_fraction: float,
) -> tuple[list[ReversibilityModeSummary], ReversibilityEnsembleSummary]:
    if not results:
        raise ValueError("results must be non-empty")
    grouped: dict[tuple[str, float], list[ReversibilityEnsembleSampleResult]] = {}
    for item in results:
        grouped.setdefault((item.mode_name, float(item.dephasing_strength)), []).append(item)

    mode_summaries: list[ReversibilityModeSummary] = []
    for (mode_name, dephasing_strength), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        detector = np.array([item.detector_success for item in group], dtype=np.float64)
        reversibility = np.array([item.open_reversibility_score for item in group], dtype=np.float64)
        open_loss = np.array([item.open_loss_delta for item in group], dtype=np.float64)
        mode_summaries.append(
            ReversibilityModeSummary(
                mode_name=mode_name,
                dephasing_strength=float(dephasing_strength),
                n_samples=len(group),
                mean_detector_success=float(np.mean(detector)),
                median_detector_success=float(np.median(detector)),
                std_detector_success=float(np.std(detector)),
                mean_reversibility_score=float(np.mean(reversibility)),
                median_reversibility_score=float(np.median(reversibility)),
                std_reversibility_score=float(np.std(reversibility)),
                mean_open_loss_delta=float(np.mean(open_loss)),
                median_open_loss_delta=float(np.median(open_loss)),
                std_open_loss_delta=float(np.std(open_loss)),
                detector_success_cv=_detector_success_cv(detector),
            )
        )

    operating_modes = sorted({item.mode_name for item in results})
    dephasing_strengths = sorted({float(item.dephasing_strength) for item in results})
    best_reversibility_by_dephasing: dict[float, str] = {}
    lowest_open_loss_by_dephasing: dict[float, str] = {}
    best_detector_stability_by_dephasing: dict[float, str] = {}
    preferred_mode_by_dephasing: dict[float, str] = {}
    detector_reversibility_corr_by_dephasing: dict[float, float] = {}
    detector_loss_corr_by_dephasing: dict[float, float] = {}

    for dephasing_strength in dephasing_strengths:
        per_dephasing = [item for item in mode_summaries if item.dephasing_strength == dephasing_strength]
        per_result = [item for item in results if float(item.dephasing_strength) == dephasing_strength]
        best_reversibility_by_dephasing[dephasing_strength] = max(
            per_dephasing,
            key=lambda item: item.mean_reversibility_score,
        ).mode_name
        lowest_open_loss_by_dephasing[dephasing_strength] = min(
            per_dephasing,
            key=lambda item: item.mean_open_loss_delta,
        ).mode_name
        best_detector_stability_by_dephasing[dephasing_strength] = min(
            per_dephasing,
            key=lambda item: (item.detector_success_cv, -item.mean_detector_success, item.mean_open_loss_delta),
        ).mode_name
        detector_values = np.array([item.detector_success for item in per_result], dtype=np.float64)
        reversibility_values = np.array([item.open_reversibility_score for item in per_result], dtype=np.float64)
        loss_values = np.array([item.open_loss_delta for item in per_result], dtype=np.float64)
        detector_reversibility_corr_by_dephasing[dephasing_strength] = safe_corrcoef(detector_values, reversibility_values)
        detector_loss_corr_by_dephasing[dephasing_strength] = safe_corrcoef(detector_values, loss_values)

        best_detector = max(item.mean_detector_success for item in per_dephasing)
        tolerance = max(abs(best_detector) * float(detector_tolerance_fraction), 1.0e-12)
        eligible = [
            item
            for item in per_dephasing
            if best_detector - item.mean_detector_success <= tolerance
        ]
        preferred = sorted(
            eligible,
            key=lambda item: (
                -item.mean_reversibility_score,
                item.mean_open_loss_delta,
                item.std_detector_success,
            ),
        )[0]
        preferred_mode_by_dephasing[dephasing_strength] = preferred.mode_name

    summary = ReversibilityEnsembleSummary(
        n_samples=len({item.sample_id for item in results}),
        operating_modes=operating_modes,
        dephasing_strengths=dephasing_strengths,
        best_reversibility_by_dephasing=best_reversibility_by_dephasing,
        lowest_open_loss_by_dephasing=lowest_open_loss_by_dephasing,
        best_detector_stability_by_dephasing=best_detector_stability_by_dephasing,
        detector_reversibility_corr_by_dephasing=detector_reversibility_corr_by_dephasing,
        detector_loss_corr_by_dephasing=detector_loss_corr_by_dephasing,
        preferred_mode_by_dephasing=preferred_mode_by_dephasing,
    )
    return mode_summaries, summary


def run_reversibility_ensemble(
    config: dict[str, Any],
) -> tuple[list[ReversibilityEnsembleSampleResult], ReversibilityEnsembleSummary]:
    block = _ensemble_block(config)
    grid, loaded_modes = _load_operating_modes(config)
    fabrication_block = dict(block["fabrication_disorder"])
    phase_block = dict(block["phase_noise"])
    time_block = dict(block["time"])
    selection_block = dict(block["selection"])

    fabrication_profiles = sample_fabrication_disorder_profiles(
        int(fabrication_block["n_samples"]),
        grid.n_sites,
        float(fabrication_block["onsite_sigma"]),
        float(fabrication_block["correlation_length_sites"]),
        int(fabrication_block["seed"]),
    )
    phase_noise_by_sample = [
        sample_phase_noise_profiles(
            int(phase_block["n_profiles_per_sample"]),
            grid.n_sites,
            float(phase_block["profile_sigma"]),
            float(phase_block["correlation_length_sites"]),
            int(phase_block["seed"]) + sample_id,
        )
        for sample_id in range(len(fabrication_profiles))
    ]

    results: list[ReversibilityEnsembleSampleResult] = []
    for sample_id, onsite_disorder in enumerate(fabrication_profiles):
        phase_noise_profiles = phase_noise_by_sample[sample_id]
        for mode in loaded_modes:
            H_mode, detail = build_mode_hamiltonian_for_sample(
                mode_name=mode.name,
                base_grid_config=dict(block["grid"]),
                onsite_disorder=onsite_disorder,
                phase_noise_profiles=phase_noise_profiles,
                mode_config=mode,
                time_step_multiplier=float(time_block["time_step_multiplier"]),
            )
            motor_config = _motor_config_for_mode(mode, time_step_multiplier=float(time_block["time_step_multiplier"]))
            for dephasing_strength in list(block["dephasing_strengths"]):
                result = run_single_reversibility_ensemble_sample(
                    sample_id=sample_id,
                    mode_name=mode.name,
                    H=H_mode,
                    input_index=grid.input_index,
                    target_indices=list(grid.target_indices),
                    dephasing_strength=float(dephasing_strength),
                    time_min=float(time_block["time_min"]),
                    time_max=float(time_block["time_max"]),
                    n_time_samples=int(motor_config.n_time_samples),
                    time_step_multiplier=float(time_block["time_step_multiplier"]),
                    forward_time_selection=str(time_block["forward_time_selection"]),
                    phase_noise_profile_index=None,
                )
                results.append(replace(result, detector_success=float(detail.sample.best_detector_success)))

    _, summary = summarize_reversibility_ensemble(
        results,
        detector_tolerance_fraction=float(selection_block["detector_tolerance_fraction"]),
    )
    return results, summary


def write_reversibility_ensemble_csv(
    path: str | Path,
    results: list[ReversibilityEnsembleSampleResult],
    mode_summaries: list[ReversibilityModeSummary],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_type",
        "sample_id",
        "mode_name",
        "dephasing_strength",
        "detector_success",
        "coherent_reversibility_score",
        "coherent_loss_delta",
        "open_reversibility_score",
        "open_loss_delta",
        "return_probability",
        "phase_noise_profile_index",
        "time_step_multiplier",
        "n_samples",
        "mean_detector_success",
        "median_detector_success",
        "std_detector_success",
        "mean_reversibility_score",
        "median_reversibility_score",
        "std_reversibility_score",
        "mean_open_loss_delta",
        "median_open_loss_delta",
        "std_open_loss_delta",
        "detector_success_cv",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "row_type": "sample",
                    **asdict(result),
                }
            )
        for summary in mode_summaries:
            writer.writerow(
                {
                    "row_type": "summary",
                    "mode_name": summary.mode_name,
                    "dephasing_strength": summary.dephasing_strength,
                    "n_samples": summary.n_samples,
                    "mean_detector_success": summary.mean_detector_success,
                    "median_detector_success": summary.median_detector_success,
                    "std_detector_success": summary.std_detector_success,
                    "mean_reversibility_score": summary.mean_reversibility_score,
                    "median_reversibility_score": summary.median_reversibility_score,
                    "std_reversibility_score": summary.std_reversibility_score,
                    "mean_open_loss_delta": summary.mean_open_loss_delta,
                    "median_open_loss_delta": summary.median_open_loss_delta,
                    "std_open_loss_delta": summary.std_open_loss_delta,
                    "detector_success_cv": summary.detector_success_cv,
                }
            )
    return output


def write_reversibility_ensemble_summary_json(
    path: str | Path,
    summary: ReversibilityEnsembleSummary,
    mode_summaries: list[ReversibilityModeSummary],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": asdict(summary),
        "mode_summaries": [asdict(item) for item in mode_summaries],
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def plot_reversibility_ensemble(
    mode_summaries: list[ReversibilityModeSummary],
    summary: ReversibilityEnsembleSummary,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Synthetic reversibility ensemble", fontsize=13)

    modes = summary.operating_modes
    dephasing_strengths = summary.dephasing_strengths
    palette = ["#1b4965", "#b23a48", "#3a7d44", "#7b2cbf", "#f4a261", "#264653"]
    colors = {mode: palette[index % len(palette)] for index, mode in enumerate(modes)}

    for mode in modes:
        rows = sorted(
            [item for item in mode_summaries if item.mode_name == mode],
            key=lambda item: item.dephasing_strength,
        )
        x = [item.dephasing_strength for item in rows]
        axes[0, 0].plot(x, [item.mean_reversibility_score for item in rows], marker="o", color=colors[mode], label=mode)
        axes[0, 1].plot(x, [item.mean_open_loss_delta for item in rows], marker="o", color=colors[mode], label=mode)
        axes[1, 0].plot(x, [item.mean_detector_success for item in rows], marker="o", color=colors[mode], label=mode)
        for item in rows:
            axes[1, 1].scatter(
                item.mean_detector_success,
                item.mean_reversibility_score,
                color=colors[mode],
                s=70,
            )
            axes[1, 1].text(
                item.mean_detector_success,
                item.mean_reversibility_score,
                f"{mode}@{item.dephasing_strength:g}",
                fontsize=7,
            )

    axes[0, 0].set_title("Mean reversibility vs dephasing")
    axes[0, 0].set_xlabel("dephasing_strength")
    axes[0, 0].set_ylabel("mean_reversibility_score")

    axes[0, 1].set_title("Mean open loss vs dephasing")
    axes[0, 1].set_xlabel("dephasing_strength")
    axes[0, 1].set_ylabel("mean_open_loss_delta")

    axes[1, 0].set_title("Mean detector success vs dephasing")
    axes[1, 0].set_xlabel("dephasing_strength")
    axes[1, 0].set_ylabel("mean_detector_success")

    axes[1, 1].set_title("Detector success vs reversibility")
    axes[1, 1].set_xlabel("mean_detector_success")
    axes[1, 1].set_ylabel("mean_reversibility_score")

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


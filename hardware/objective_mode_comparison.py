"""Paired synthetic comparison between raw and normalized transition-motor modes."""

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
    run_single_motor_ensemble_sample_detailed,
    sample_fabrication_disorder_profiles,
    sample_phase_noise_profiles,
)
from hardware.transition_motor import TransitionMotorConfig, transition_motor_config_from_dict


@dataclass(frozen=True)
class ObjectiveModeComparisonSample:
    sample_id: int
    raw_detector_success: float
    normalized_detector_success: float
    raw_noise_action: float
    normalized_noise_action: float
    raw_leakage: float
    normalized_leakage: float
    raw_control_cost: float
    normalized_control_cost: float
    raw_objective: float
    normalized_objective: float
    detector_delta: float
    noise_action_delta: float
    leakage_delta: float
    control_cost_delta: float
    objective_delta: float
    normalized_wins_detector: bool
    normalized_wins_noise_action: bool
    normalized_wins_leakage: bool
    normalized_wins_objective: bool
    normalized_wins_control_cost: bool


@dataclass(frozen=True)
class ObjectiveModeComparisonSummary:
    n_samples: int
    detector_win_rate: float
    noise_action_win_rate: float
    leakage_win_rate: float
    objective_win_rate: float
    control_cost_win_rate: float
    all_core_win_rate: float
    mean_detector_delta: float
    median_detector_delta: float
    mean_noise_action_delta: float
    median_noise_action_delta: float
    mean_leakage_delta: float
    median_leakage_delta: float
    mean_control_cost_delta: float
    median_control_cost_delta: float
    mean_objective_delta: float
    median_objective_delta: float


def _resolve_config_path(config: dict[str, Any], value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    config_path = config.get("__config_path__")
    if config_path:
        resolved = (Path(str(config_path)).expanduser().resolve().parent / candidate).resolve()
        if resolved.exists():
            return resolved
    return (Path.cwd() / candidate).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _comparison_block(config: dict[str, Any]) -> dict[str, Any]:
    if "objective_mode_comparison" not in config:
        raise ValueError("config must define objective_mode_comparison")
    return dict(config["objective_mode_comparison"])


def _load_motor_configs(config: dict[str, Any]) -> tuple[TransitionMotorConfig, TransitionMotorConfig]:
    block = _comparison_block(config)
    raw_path = _resolve_config_path(config, str(block["base_raw_config"]))
    normalized_path = _resolve_config_path(config, str(block["base_normalized_config"]))
    raw_config, _ = transition_motor_config_from_dict(_load_yaml(raw_path))
    normalized_config, _ = transition_motor_config_from_dict(_load_yaml(normalized_path))
    if raw_config.objective_mode is not None and raw_config.objective_mode.mode != "raw":
        raise ValueError("base_raw_config must use raw objective semantics")
    if normalized_config.objective_mode is None or normalized_config.objective_mode.mode not in {"normalized", "calibrated"}:
        raise ValueError("base_normalized_config must define a normalized or calibrated objective_mode")
    return raw_config, normalized_config


def _validate_compatible_motor_configs(raw_config: TransitionMotorConfig, normalized_config: TransitionMotorConfig) -> None:
    raw_grid = raw_config.grid
    normalized_grid = normalized_config.grid
    if (
        raw_grid.n_sites != normalized_grid.n_sites
        or raw_grid.edges != normalized_grid.edges
        or raw_grid.input_index != normalized_grid.input_index
        or raw_grid.target_indices != normalized_grid.target_indices
    ):
        raise ValueError("raw and normalized configs must use the same fixed grid")
    if raw_config.knob_registry.names() != normalized_config.knob_registry.names():
        raise ValueError("raw and normalized configs must use the same knob registry")


def _build_sample_inputs(config: dict[str, Any], n_sites: int) -> list[tuple[int, np.ndarray, list[np.ndarray]]]:
    block = _comparison_block(config)
    ensemble = dict(block["ensemble"])
    fabrication_profiles = sample_fabrication_disorder_profiles(
        int(ensemble["n_samples"]),
        n_sites,
        float(ensemble["onsite_sigma"]),
        float(ensemble["fabrication_correlation_length_sites"]),
        int(ensemble["fabrication_seed"]),
    )
    inputs: list[tuple[int, np.ndarray, list[np.ndarray]]] = []
    for sample_id, onsite_profile in enumerate(fabrication_profiles):
        noise_profiles = sample_phase_noise_profiles(
            int(ensemble["n_noise_profiles_per_sample"]),
            n_sites,
            float(ensemble["phase_noise_sigma"]),
            float(ensemble["phase_noise_correlation_length_sites"]),
            int(ensemble["phase_noise_seed"]) + sample_id,
        )
        inputs.append((sample_id, onsite_profile, noise_profiles))
    return inputs


def _permissive_success_criteria() -> dict[str, float]:
    return {
        "min_objective_gain": -1.0e12,
        "min_detector_gain": -1.0e12,
        "min_noise_action_reduction": -1.0e12,
    }


def _run_mode_over_inputs(
    motor_config: TransitionMotorConfig,
    sample_inputs: list[tuple[int, np.ndarray, list[np.ndarray]]],
) -> list[DetailedMotorEnsembleSample]:
    grid = motor_config.grid
    registry = motor_config.knob_registry
    basis = make_control_basis(grid)
    details: list[DetailedMotorEnsembleSample] = []
    for sample_id, onsite_profile, noise_profiles in sample_inputs:
        detail = run_single_motor_ensemble_sample_detailed(
            grid,
            registry,
            basis,
            onsite_profile,
            noise_profiles,
            motor_config,
            _permissive_success_criteria(),
        )
        sample = detail.sample
        details.append(
            DetailedMotorEnsembleSample(
                sample=replace(sample, sample_id=sample_id),
                baseline_theta=dict(detail.baseline_theta),
                best_theta=dict(detail.best_theta),
                baseline_metrics=detail.baseline_metrics,
                best_metrics=detail.best_metrics,
            )
        )
    return details


def _comparison_sample(
    sample_id: int,
    raw_detail: DetailedMotorEnsembleSample,
    normalized_detail: DetailedMotorEnsembleSample,
) -> ObjectiveModeComparisonSample:
    raw_detector_success = float(raw_detail.sample.best_detector_success)
    normalized_detector_success = float(normalized_detail.sample.best_detector_success)
    raw_noise_action = float(raw_detail.best_metrics.noise_action_on_info)
    normalized_noise_action = float(normalized_detail.best_metrics.noise_action_on_info)
    raw_leakage = float(raw_detail.best_metrics.noise_leakage)
    normalized_leakage = float(normalized_detail.best_metrics.noise_leakage)
    raw_control_cost = float(raw_detail.best_metrics.control_cost)
    normalized_control_cost = float(normalized_detail.best_metrics.control_cost)
    raw_objective = float(raw_detail.best_metrics.objective)
    normalized_objective = float(normalized_detail.best_metrics.objective)

    detector_delta = float(normalized_detector_success - raw_detector_success)
    noise_action_delta = float(raw_noise_action - normalized_noise_action)
    leakage_delta = float(raw_leakage - normalized_leakage)
    control_cost_delta = float(raw_control_cost - normalized_control_cost)
    objective_delta = float(normalized_objective - raw_objective)

    return ObjectiveModeComparisonSample(
        sample_id=sample_id,
        raw_detector_success=raw_detector_success,
        normalized_detector_success=normalized_detector_success,
        raw_noise_action=raw_noise_action,
        normalized_noise_action=normalized_noise_action,
        raw_leakage=raw_leakage,
        normalized_leakage=normalized_leakage,
        raw_control_cost=raw_control_cost,
        normalized_control_cost=normalized_control_cost,
        raw_objective=raw_objective,
        normalized_objective=normalized_objective,
        detector_delta=detector_delta,
        noise_action_delta=noise_action_delta,
        leakage_delta=leakage_delta,
        control_cost_delta=control_cost_delta,
        objective_delta=objective_delta,
        normalized_wins_detector=detector_delta > 0.0,
        normalized_wins_noise_action=noise_action_delta > 0.0,
        normalized_wins_leakage=leakage_delta > 0.0,
        normalized_wins_objective=objective_delta > 0.0,
        normalized_wins_control_cost=control_cost_delta > 0.0,
    )


def summarize_objective_mode_comparison(
    samples: list[ObjectiveModeComparisonSample],
) -> ObjectiveModeComparisonSummary:
    if not samples:
        raise ValueError("samples must be non-empty")

    def column(name: str) -> np.ndarray:
        return np.array([getattr(sample, name) for sample in samples], dtype=np.float64)

    return ObjectiveModeComparisonSummary(
        n_samples=len(samples),
        detector_win_rate=float(np.mean(column("normalized_wins_detector"))),
        noise_action_win_rate=float(np.mean(column("normalized_wins_noise_action"))),
        leakage_win_rate=float(np.mean(column("normalized_wins_leakage"))),
        objective_win_rate=float(np.mean(column("normalized_wins_objective"))),
        control_cost_win_rate=float(np.mean(column("normalized_wins_control_cost"))),
        all_core_win_rate=float(
            np.mean(
                [
                    1.0
                    if sample.normalized_wins_detector
                    and sample.normalized_wins_noise_action
                    and sample.normalized_wins_objective
                    else 0.0
                    for sample in samples
                ],
                dtype=np.float64,
            )
        ),
        mean_detector_delta=float(np.mean(column("detector_delta"))),
        median_detector_delta=float(np.median(column("detector_delta"))),
        mean_noise_action_delta=float(np.mean(column("noise_action_delta"))),
        median_noise_action_delta=float(np.median(column("noise_action_delta"))),
        mean_leakage_delta=float(np.mean(column("leakage_delta"))),
        median_leakage_delta=float(np.median(column("leakage_delta"))),
        mean_control_cost_delta=float(np.mean(column("control_cost_delta"))),
        median_control_cost_delta=float(np.median(column("control_cost_delta"))),
        mean_objective_delta=float(np.mean(column("objective_delta"))),
        median_objective_delta=float(np.median(column("objective_delta"))),
    )


def _build_paired_samples(
    raw_details: list[DetailedMotorEnsembleSample],
    normalized_details: list[DetailedMotorEnsembleSample],
) -> list[ObjectiveModeComparisonSample]:
    if len(raw_details) != len(normalized_details):
        raise ValueError("raw and normalized detail lists must have the same length")
    samples: list[ObjectiveModeComparisonSample] = []
    for raw_detail, normalized_detail in zip(raw_details, normalized_details):
        if raw_detail.sample.sample_id != normalized_detail.sample.sample_id:
            raise ValueError("paired raw and normalized samples must share the same sample_id")
        samples.append(
            _comparison_sample(
                int(raw_detail.sample.sample_id),
                raw_detail,
                normalized_detail,
            )
        )
    return samples


def run_objective_mode_comparison(
    config: dict[str, Any],
) -> tuple[list[ObjectiveModeComparisonSample], ObjectiveModeComparisonSummary]:
    raw_config, normalized_config = _load_motor_configs(config)
    _validate_compatible_motor_configs(raw_config, normalized_config)
    sample_inputs = _build_sample_inputs(config, raw_config.grid.n_sites)
    raw_details = _run_mode_over_inputs(raw_config, sample_inputs)
    normalized_details = _run_mode_over_inputs(normalized_config, sample_inputs)
    samples = _build_paired_samples(raw_details, normalized_details)
    return samples, summarize_objective_mode_comparison(samples)


def bootstrap_delta_ci(
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


def _scaled_normalized_config(
    config: dict[str, Any],
    multiplier: float,
) -> dict[str, Any]:
    scaled = deepcopy(config)
    block = dict(scaled["transition_motor"])
    objective_mode = dict(block["objective_mode"])
    normalization = dict(objective_mode["normalization"])
    for key in (
        "transport_scale",
        "noise_action_scale",
        "leakage_scale",
        "control_cost_scale",
    ):
        normalization[key] = float(normalization[key]) * float(multiplier)
    objective_mode["normalization"] = normalization
    block["objective_mode"] = objective_mode
    scaled["transition_motor"] = block
    return scaled


def run_scale_sensitivity(
    config: dict[str, Any],
) -> list[dict[str, float]]:
    block = _comparison_block(config)
    sensitivity = dict(block.get("calibration_sensitivity", {}))
    if not bool(sensitivity.get("enabled", False)):
        return []

    raw_config, normalized_base_config = _load_motor_configs(config)
    _validate_compatible_motor_configs(raw_config, normalized_base_config)
    sample_inputs = _build_sample_inputs(config, raw_config.grid.n_sites)
    raw_details = _run_mode_over_inputs(raw_config, sample_inputs)

    normalized_path = _resolve_config_path(config, str(block["base_normalized_config"]))
    normalized_payload = _load_yaml(normalized_path)

    rows: list[dict[str, float]] = []
    for multiplier in list(sensitivity.get("scale_multipliers", [])):
        scaled_payload = _scaled_normalized_config(normalized_payload, float(multiplier))
        scaled_normalized_config, _ = transition_motor_config_from_dict(scaled_payload)
        normalized_details = _run_mode_over_inputs(scaled_normalized_config, sample_inputs)
        samples = _build_paired_samples(raw_details, normalized_details)
        summary = summarize_objective_mode_comparison(samples)
        rows.append(
            {
                "scale_multiplier": float(multiplier),
                "detector_win_rate": float(summary.detector_win_rate),
                "mean_detector_delta": float(summary.mean_detector_delta),
                "noise_action_win_rate": float(summary.noise_action_win_rate),
                "mean_noise_action_delta": float(summary.mean_noise_action_delta),
                "leakage_win_rate": float(summary.leakage_win_rate),
                "mean_leakage_delta": float(summary.mean_leakage_delta),
                "control_cost_win_rate": float(summary.control_cost_win_rate),
                "mean_control_cost_delta": float(summary.mean_control_cost_delta),
                "objective_win_rate": float(summary.objective_win_rate),
                "mean_objective_delta": float(summary.mean_objective_delta),
            }
        )
    return rows


def write_comparison_csv(
    path: str | Path,
    samples: list[ObjectiveModeComparisonSample],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ObjectiveModeComparisonSample.__dataclass_fields__.keys())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))
    return output


def write_comparison_summary_json(
    path: str | Path,
    summary: ObjectiveModeComparisonSummary,
    *,
    bootstrap_ci: dict[str, tuple[float, float]],
    scale_sensitivity: list[dict[str, float]],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": asdict(summary),
        "bootstrap_ci": {
            key: [float(value[0]), float(value[1])] for key, value in bootstrap_ci.items()
        },
        "scale_sensitivity": scale_sensitivity,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def plot_objective_mode_comparison(
    samples: list[ObjectiveModeComparisonSample],
    output_path: str | Path,
) -> Path:
    if not samples:
        raise ValueError("samples must be non-empty")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    detector = np.array([sample.detector_delta for sample in samples], dtype=np.float64)
    noise = np.array([sample.noise_action_delta for sample in samples], dtype=np.float64)
    control = np.array([sample.control_cost_delta for sample in samples], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Synthetic paired ensemble: raw vs normalized objective modes")

    axes[0, 0].hist(detector, bins=min(12, max(5, len(detector))), color="#2f6b8a", alpha=0.85)
    axes[0, 0].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0, 0].set_title("Detector delta")
    axes[0, 0].set_xlabel("normalized - raw detector success")

    axes[0, 1].hist(noise, bins=min(12, max(5, len(noise))), color="#4f8a3f", alpha=0.85)
    axes[0, 1].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0, 1].set_title("Noise-action delta")
    axes[0, 1].set_xlabel("raw - normalized noise action")

    axes[1, 0].scatter(detector, noise, color="#8a3f6b", alpha=0.85)
    axes[1, 0].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1, 0].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1, 0].set_title("Detector vs noise-action delta")
    axes[1, 0].set_xlabel("detector delta")
    axes[1, 0].set_ylabel("noise-action delta")

    axes[1, 1].hist(control, bins=min(12, max(5, len(control))), color="#b1792a", alpha=0.85)
    axes[1, 1].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1, 1].set_title("Control-cost delta")
    axes[1, 1].set_xlabel("raw - normalized control cost")

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output

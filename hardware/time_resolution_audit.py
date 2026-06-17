"""Time-resolution sensitivity audit for paired transition-motor comparisons."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml

from hardware.objective_mode_comparison import (
    _build_paired_samples,
    _build_sample_inputs,
    _load_yaml,
    _resolve_config_path,
    _run_mode_over_inputs,
    _validate_compatible_motor_configs,
    summarize_objective_mode_comparison,
)
from hardware.transition_motor import transition_motor_config_from_dict


@dataclass(frozen=True)
class TimeResolutionAuditRun:
    time_step_multiplier: float
    n_time_samples: int
    detector_win_rate: float
    mean_detector_delta: float
    noise_action_win_rate: float
    mean_noise_action_delta: float
    leakage_win_rate: float
    mean_leakage_delta: float
    common_balanced_win_rate: float
    mean_common_balanced_delta: float
    raw_metric_win_rate: float
    mean_raw_metric_delta: float
    normalized_metric_win_rate: float
    mean_normalized_metric_delta: float


@dataclass(frozen=True)
class TimeResolutionAuditSummary:
    baseline_multiplier: float
    common_balanced_relative_change_max: float
    detector_delta_absolute_change_max: float
    noise_action_delta_absolute_change_max: float
    leakage_delta_absolute_change_max: float
    sign_stable_common_balanced: bool
    sign_stable_detector_delta: bool
    sign_stable_noise_action_delta: bool
    sign_stable_leakage_delta: bool
    time_resolution_sensitive: bool
    registry_recommendation: str


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


def _time_resolution_block(config: dict[str, Any]) -> dict[str, Any]:
    if "time_resolution_audit" not in config:
        raise ValueError("config must define time_resolution_audit")
    return dict(config["time_resolution_audit"])


def _load_base_comparison_config(config: dict[str, Any]) -> dict[str, Any]:
    block = _time_resolution_block(config)
    comparison_path = _resolve_config_path(config, str(block["base_comparison_config"]))
    comparison_config = _load_yaml(comparison_path)
    comparison_config["__config_path__"] = str(comparison_path)
    return comparison_config


def _resolved_motor_payloads(comparison_config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_payload = comparison_config.get("__raw_config_data__")
    normalized_payload = comparison_config.get("__normalized_config_data__")
    if raw_payload is not None and normalized_payload is not None:
        return deepcopy(dict(raw_payload)), deepcopy(dict(normalized_payload))

    block = dict(comparison_config["objective_mode_comparison"])
    raw_path = _resolve_config_path(comparison_config, str(block["base_raw_config"]))
    normalized_path = _resolve_config_path(comparison_config, str(block["base_normalized_config"]))
    return _load_yaml(raw_path), _load_yaml(normalized_path)


def with_time_resolution_multiplier(
    comparison_config: dict[str, Any],
    multiplier: float,
) -> dict[str, Any]:
    if multiplier <= 0.0:
        raise ValueError("multiplier must be positive")

    base_config = comparison_config
    if "objective_mode_comparison" not in base_config and "time_resolution_audit" in base_config:
        base_config = _load_base_comparison_config(base_config)

    adjusted = deepcopy(base_config)
    raw_payload, normalized_payload = _resolved_motor_payloads(base_config)

    for payload in (raw_payload, normalized_payload):
        block = dict(payload["transition_motor"])
        optimizer = dict(block["optimizer"])
        optimizer["n_time_samples"] = adjusted_n_time_samples(
            time_min=float(optimizer["time_min"]),
            time_max=float(optimizer["time_max"]),
            n_time_samples=int(optimizer["n_time_samples"]),
            time_step_multiplier=float(multiplier),
        )
        block["optimizer"] = optimizer
        payload["transition_motor"] = block

    adjusted["__raw_config_data__"] = raw_payload
    adjusted["__normalized_config_data__"] = normalized_payload
    adjusted["__time_step_multiplier__"] = float(multiplier)
    return adjusted


def _comparison_summary_for_multiplier(
    comparison_config: dict[str, Any],
    sample_inputs: list[tuple[int, np.ndarray, list[np.ndarray]]],
) -> tuple[TimeResolutionAuditRun, int]:
    raw_payload = dict(comparison_config["__raw_config_data__"])
    normalized_payload = dict(comparison_config["__normalized_config_data__"])
    raw_config, _ = transition_motor_config_from_dict(raw_payload)
    normalized_config, _ = transition_motor_config_from_dict(normalized_payload)
    _validate_compatible_motor_configs(raw_config, normalized_config)

    raw_details = _run_mode_over_inputs(raw_config, sample_inputs)
    normalized_details = _run_mode_over_inputs(normalized_config, sample_inputs)
    samples = _build_paired_samples(
        raw_details,
        normalized_details,
        raw_config=raw_config,
        normalized_config=normalized_config,
    )
    summary = summarize_objective_mode_comparison(samples)
    run = TimeResolutionAuditRun(
        time_step_multiplier=float(comparison_config["__time_step_multiplier__"]),
        n_time_samples=int(raw_config.n_time_samples),
        detector_win_rate=float(summary.detector_win_rate),
        mean_detector_delta=float(summary.mean_detector_delta),
        noise_action_win_rate=float(summary.noise_action_win_rate),
        mean_noise_action_delta=float(summary.mean_noise_action_delta),
        leakage_win_rate=float(summary.leakage_win_rate),
        mean_leakage_delta=float(summary.mean_leakage_delta),
        common_balanced_win_rate=float(summary.common_balanced_win_rate),
        mean_common_balanced_delta=float(summary.mean_common_balanced_delta),
        raw_metric_win_rate=float(summary.raw_metric_win_rate),
        mean_raw_metric_delta=float(summary.mean_raw_metric_delta),
        normalized_metric_win_rate=float(summary.normalized_metric_win_rate),
        mean_normalized_metric_delta=float(summary.mean_normalized_metric_delta),
    )
    return run, int(summary.n_samples)


def _metric_sign(value: float, *, tol: float = 1.0e-12) -> int:
    if abs(value) <= tol:
        return 0
    return 1 if value > 0.0 else -1


def summarize_time_resolution_audit(
    runs: list[TimeResolutionAuditRun],
    *,
    sensitivity_thresholds: dict[str, float],
    baseline_multiplier: float = 1.0,
) -> TimeResolutionAuditSummary:
    if not runs:
        raise ValueError("runs must be non-empty")

    baseline_matches = [run for run in runs if np.isclose(run.time_step_multiplier, baseline_multiplier)]
    if len(baseline_matches) != 1:
        raise ValueError("runs must include exactly one baseline multiplier")
    baseline = baseline_matches[0]

    def absolute_change(field: str) -> float:
        base_value = float(getattr(baseline, field))
        return float(max(abs(float(getattr(run, field)) - base_value) for run in runs))

    def relative_change(field: str) -> float:
        base_value = float(getattr(baseline, field))
        changes: list[float] = []
        for run in runs:
            if np.isclose(run.time_step_multiplier, baseline_multiplier):
                continue
            delta = abs(float(getattr(run, field)) - base_value)
            if abs(base_value) <= 1.0e-12:
                changes.append(0.0 if delta <= 1.0e-12 else float("inf"))
            else:
                changes.append(float(delta / abs(base_value)))
        return float(max(changes, default=0.0))

    def sign_stable(field: str) -> bool:
        signs = {_metric_sign(float(getattr(run, field))) for run in runs}
        return len(signs) <= 1

    common_balanced_relative_change_max = relative_change("mean_common_balanced_delta")
    detector_delta_absolute_change_max = absolute_change("mean_detector_delta")
    noise_action_delta_absolute_change_max = absolute_change("mean_noise_action_delta")
    leakage_delta_absolute_change_max = absolute_change("mean_leakage_delta")
    sign_stable_common_balanced = sign_stable("mean_common_balanced_delta")
    sign_stable_detector_delta = sign_stable("mean_detector_delta")
    sign_stable_noise_action_delta = sign_stable("mean_noise_action_delta")
    sign_stable_leakage_delta = sign_stable("mean_leakage_delta")

    time_resolution_sensitive = bool(
        common_balanced_relative_change_max > float(sensitivity_thresholds["common_balanced_relative_change"])
        or detector_delta_absolute_change_max > float(sensitivity_thresholds["detector_delta_absolute_change"])
        or noise_action_delta_absolute_change_max > float(sensitivity_thresholds["noise_action_delta_absolute_change"])
        or leakage_delta_absolute_change_max > float(sensitivity_thresholds["leakage_delta_absolute_change"])
        or not sign_stable_common_balanced
        or not sign_stable_detector_delta
        or not sign_stable_noise_action_delta
        or not sign_stable_leakage_delta
    )

    registry_recommendation = (
        "Add time_step_resolution as audited operating-mode metadata or registry parameter."
        if time_resolution_sensitive
        else "Do not promote time_step_resolution to registry knob yet; current metrics are resolution-stable."
    )

    return TimeResolutionAuditSummary(
        baseline_multiplier=float(baseline_multiplier),
        common_balanced_relative_change_max=common_balanced_relative_change_max,
        detector_delta_absolute_change_max=detector_delta_absolute_change_max,
        noise_action_delta_absolute_change_max=noise_action_delta_absolute_change_max,
        leakage_delta_absolute_change_max=leakage_delta_absolute_change_max,
        sign_stable_common_balanced=sign_stable_common_balanced,
        sign_stable_detector_delta=sign_stable_detector_delta,
        sign_stable_noise_action_delta=sign_stable_noise_action_delta,
        sign_stable_leakage_delta=sign_stable_leakage_delta,
        time_resolution_sensitive=time_resolution_sensitive,
        registry_recommendation=registry_recommendation,
    )


def run_time_resolution_audit(
    config: dict[str, Any],
) -> tuple[list[TimeResolutionAuditRun], TimeResolutionAuditSummary]:
    comparison_config = _load_base_comparison_config(config)
    raw_payload, _ = _resolved_motor_payloads(comparison_config)
    raw_config, _ = transition_motor_config_from_dict(raw_payload)
    sample_inputs = _build_sample_inputs(comparison_config, raw_config.grid.n_sites)

    block = _time_resolution_block(config)
    runs: list[TimeResolutionAuditRun] = []
    for multiplier in [float(value) for value in list(block["time_step_multipliers"])]:
        adjusted_config = with_time_resolution_multiplier(comparison_config, multiplier)
        run, _ = _comparison_summary_for_multiplier(adjusted_config, sample_inputs)
        runs.append(run)

    summary = summarize_time_resolution_audit(
        runs,
        sensitivity_thresholds={str(k): float(v) for k, v in dict(block["sensitivity_thresholds"]).items()},
        baseline_multiplier=1.0,
    )
    return runs, summary


def write_time_resolution_csv(
    path: str | Path,
    runs: list[TimeResolutionAuditRun],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(TimeResolutionAuditRun.__dataclass_fields__.keys())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            writer.writerow(asdict(run))
    return output


def write_time_resolution_summary_json(
    path: str | Path,
    summary: TimeResolutionAuditSummary,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8")
    return output


def plot_time_resolution_audit(
    runs: list[TimeResolutionAuditRun],
    output_path: str | Path,
) -> Path:
    if not runs:
        raise ValueError("runs must be non-empty")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    multipliers = np.array([run.time_step_multiplier for run in runs], dtype=np.float64)
    detector = np.array([run.mean_detector_delta for run in runs], dtype=np.float64)
    noise = np.array([run.mean_noise_action_delta for run in runs], dtype=np.float64)
    leakage = np.array([run.mean_leakage_delta for run in runs], dtype=np.float64)
    common = np.array([run.mean_common_balanced_delta for run in runs], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Time-resolution sensitivity audit, synthetic validation")

    series = (
        (axes[0, 0], detector, "Mean detector delta", "#2f6b8a"),
        (axes[0, 1], noise, "Mean noise-action delta", "#4f8a3f"),
        (axes[1, 0], leakage, "Mean leakage delta", "#8a3f6b"),
        (axes[1, 1], common, "Mean common balanced delta", "#b1792a"),
    )
    for axis, values, title, color in series:
        axis.plot(multipliers, values, marker="o", linewidth=2.0, color=color)
        axis.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_title(title)
        axis.set_xlabel("time_step_multiplier")
        axis.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def load_time_resolution_audit_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config["__config_path__"] = str(config_path)
    return config

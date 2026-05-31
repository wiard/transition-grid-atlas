"""Stress audit for Pareto weight sweeps in the KTA transition motor."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from hardware.control_basis import make_control_basis
from hardware.ensemble_statistics import bootstrap_mean_ci
from hardware.motor_audit import knob_bound_statuses
from hardware.motor_ensemble import (
    DetailedMotorEnsembleSample,
    run_single_motor_ensemble_sample_detailed,
    sample_fabrication_disorder_profiles,
    sample_phase_noise_profiles,
)
from hardware.motor_pareto import (
    ObjectiveWeightSet,
    ParetoRunResult,
    load_weight_sets,
    mark_pareto_front,
    validate_weight_set,
)
from hardware.transition_motor import TransitionMotorConfig
from hardware.transition_tuner import fixed_grid_from_dict
from hardware.control_knobs import knob_registry_from_dicts


@dataclass(frozen=True)
class PresetConfidenceInterval:
    preset_name: str
    metric: str
    mean: float
    ci_low: float
    ci_high: float
    n_bootstrap: int


@dataclass(frozen=True)
class KnobProfileSummary:
    preset_name: str
    knob: str
    mean_value: float
    mean_abs_value: float
    saturation_rate: float
    near_bound_rate: float


@dataclass(frozen=True)
class ObjectiveScaleSummary:
    transport_scale: float
    noise_action_scale: float
    leakage_scale: float
    control_cost_scale: float
    raw_scale_ratio: float


@dataclass(frozen=True)
class ParetoAuditRunResult:
    name: str
    weights: ObjectiveWeightSet
    n_samples: int
    success_rate: float
    mean_detector_success_gain: float
    mean_noise_action_reduction: float
    mean_noise_leakage_reduction: float
    mean_objective_gain: float
    mean_normalized_objective_gain: float
    mean_best_control_cost: float
    mean_saturated_knobs: float
    mean_near_bound_knobs: float
    mean_transport_term_gain: float
    mean_noise_action_term_gain: float
    mean_noise_leakage_term_gain: float
    mean_control_cost_penalty: float
    detector_gain_ci_low: float
    detector_gain_ci_high: float
    noise_action_ci_low: float
    noise_action_ci_high: float
    leakage_ci_low: float
    leakage_ci_high: float
    objective_ci_low: float
    objective_ci_high: float
    normalized_objective_ci_low: float
    normalized_objective_ci_high: float
    mean_theta_l2: float
    knob_profiles: list[KnobProfileSummary]
    is_pareto_optimal: bool = False


@dataclass(frozen=True)
class ParetoStressAuditSummary:
    n_weight_sets: int
    n_samples_per_weight_set: int
    pareto_optimal_names: list[str]
    best_detector_name: str
    best_noise_action_name: str
    best_leakage_name: str
    best_cost_name: str
    best_normalized_name: str
    normalization_recommended: bool
    regime_assessment: str
    mean_pairwise_theta_distance: float
    min_pairwise_theta_distance: float
    objective_scale_summary: ObjectiveScaleSummary


@dataclass(frozen=True)
class TransitionMotorParetoAuditConfig:
    motor_config: TransitionMotorConfig
    fabrication_disorder: dict[str, float | int]
    phase_noise: dict[str, float | int]
    success_criteria: dict[str, float]
    weight_sets: list[ObjectiveWeightSet]
    maximize: list[str]
    minimize: list[str]
    bootstrap_n: int
    bootstrap_seed: int
    outputs: dict[str, str]


def transition_motor_pareto_audit_from_dict(data: dict[str, Any]) -> TransitionMotorParetoAuditConfig:
    block = dict(data["transition_motor_pareto_audit"])
    grid = fixed_grid_from_dict(dict(block["grid"]))
    registry = knob_registry_from_dicts(list(block["knobs"]))
    optimizer = dict(block["optimizer"])
    motor_config = TransitionMotorConfig(
        grid=grid,
        knob_registry=registry,
        objective_weights={
            "transport": 1.0,
            "noise_action": 0.5,
            "leakage": 0.25,
            "control_cost": 0.01,
        },
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
    weight_sets = load_weight_sets(block)
    for item in weight_sets:
        validate_weight_set(item)
    audit = dict(block.get("audit", {}))
    return TransitionMotorParetoAuditConfig(
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
        weight_sets=weight_sets,
        maximize=[str(item) for item in list(block["pareto"]["maximize"])],
        minimize=[str(item) for item in list(block["pareto"]["minimize"])],
        bootstrap_n=int(audit.get("n_bootstrap", 2000)),
        bootstrap_seed=int(audit.get("seed", 321)),
        outputs={
            "csv_path": str(block["outputs"]["csv_path"]),
            "summary_path": str(block["outputs"]["summary_path"]),
            "plot_path": str(block["outputs"]["plot_path"]),
        },
    )


def compute_objective_scale_summary(
    detailed_by_preset: dict[str, list[DetailedMotorEnsembleSample]],
    *,
    quantile: float = 0.9,
    eps: float = 1e-12,
) -> ObjectiveScaleSummary:
    transport = np.array(
        [detail.sample.transport_gain for details in detailed_by_preset.values() for detail in details],
        dtype=np.float64,
    )
    noise = np.array(
        [detail.sample.noise_action_reduction for details in detailed_by_preset.values() for detail in details],
        dtype=np.float64,
    )
    leakage = np.array(
        [detail.sample.noise_leakage_reduction for details in detailed_by_preset.values() for detail in details],
        dtype=np.float64,
    )
    cost = np.array(
        [
            detail.sample.best_control_cost - detail.sample.baseline_control_cost
            for details in detailed_by_preset.values()
            for detail in details
        ],
        dtype=np.float64,
    )

    def robust_scale(values: np.ndarray) -> float:
        scale = float(np.quantile(np.abs(values), quantile))
        return max(scale, eps)

    transport_scale = robust_scale(transport)
    noise_scale = robust_scale(noise)
    leakage_scale = robust_scale(leakage)
    cost_scale = robust_scale(cost)
    raw_scale_ratio = float(max(transport_scale, noise_scale, leakage_scale, cost_scale) / min(transport_scale, noise_scale, leakage_scale, cost_scale))
    return ObjectiveScaleSummary(
        transport_scale=transport_scale,
        noise_action_scale=noise_scale,
        leakage_scale=leakage_scale,
        control_cost_scale=cost_scale,
        raw_scale_ratio=raw_scale_ratio,
    )


def normalized_objective_gain(
    detail: DetailedMotorEnsembleSample,
    weight_set: ObjectiveWeightSet,
    scales: ObjectiveScaleSummary,
) -> float:
    control_cost_delta = detail.sample.best_control_cost - detail.sample.baseline_control_cost
    return float(
        weight_set.transport * detail.sample.transport_gain / scales.transport_scale
        + weight_set.noise_action * detail.sample.noise_action_reduction / scales.noise_action_scale
        + weight_set.leakage * detail.sample.noise_leakage_reduction / scales.leakage_scale
        - weight_set.control_cost * control_cost_delta / scales.control_cost_scale
    )


def summarize_knob_profiles(
    preset_name: str,
    details: list[DetailedMotorEnsembleSample],
    registry,
) -> list[KnobProfileSummary]:
    profiles: list[KnobProfileSummary] = []
    for knob in registry.knobs:
        values = np.array([detail.best_theta[knob.name] for detail in details], dtype=np.float64)
        statuses = [knob_bound_statuses(detail.best_theta, registry) for detail in details]
        knob_status = [
            next(item for item in status_list if item.name == knob.name)
            for status_list in statuses
        ]
        profiles.append(
            KnobProfileSummary(
                preset_name=preset_name,
                knob=knob.name,
                mean_value=float(np.mean(values)),
                mean_abs_value=float(np.mean(np.abs(values))),
                saturation_rate=float(np.mean([1.0 if item.at_lower_bound or item.at_upper_bound else 0.0 for item in knob_status])),
                near_bound_rate=float(np.mean([1.0 if item.near_bound else 0.0 for item in knob_status])),
            )
        )
    return profiles


def _ci_for_metric(
    preset_name: str,
    metric: str,
    values: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> PresetConfidenceInterval:
    ci_low, ci_high = bootstrap_mean_ci(values, n_bootstrap=n_bootstrap, seed=seed)
    return PresetConfidenceInterval(
        preset_name=preset_name,
        metric=metric,
        mean=float(np.mean(values)),
        ci_low=ci_low,
        ci_high=ci_high,
        n_bootstrap=n_bootstrap,
    )


def _build_audit_result(
    summary: ParetoRunResult,
    weight_set: ObjectiveWeightSet,
    details: list[DetailedMotorEnsembleSample],
    registry,
    scales: ObjectiveScaleSummary,
    *,
    n_bootstrap: int,
    seed: int,
) -> ParetoAuditRunResult:
    detector_values = np.array([detail.sample.detector_success_gain for detail in details], dtype=np.float64)
    noise_values = np.array([detail.sample.noise_action_reduction for detail in details], dtype=np.float64)
    leakage_values = np.array([detail.sample.noise_leakage_reduction for detail in details], dtype=np.float64)
    objective_values = np.array([detail.sample.objective_gain for detail in details], dtype=np.float64)
    normalized_values = np.array(
        [normalized_objective_gain(detail, weight_set, scales) for detail in details],
        dtype=np.float64,
    )
    control_cost_delta = np.array(
        [detail.sample.best_control_cost - detail.sample.baseline_control_cost for detail in details],
        dtype=np.float64,
    )
    theta_l2 = np.array(
        [float(np.linalg.norm(np.array(list(detail.best_theta.values()), dtype=np.float64))) for detail in details],
        dtype=np.float64,
    )

    detector_ci = _ci_for_metric(summary.name, "detector_success_gain", detector_values, n_bootstrap=n_bootstrap, seed=seed)
    noise_ci = _ci_for_metric(summary.name, "noise_action_reduction", noise_values, n_bootstrap=n_bootstrap, seed=seed + 1)
    leakage_ci = _ci_for_metric(summary.name, "noise_leakage_reduction", leakage_values, n_bootstrap=n_bootstrap, seed=seed + 2)
    objective_ci = _ci_for_metric(summary.name, "objective_gain", objective_values, n_bootstrap=n_bootstrap, seed=seed + 3)
    normalized_ci = _ci_for_metric(summary.name, "normalized_objective_gain", normalized_values, n_bootstrap=n_bootstrap, seed=seed + 4)

    return ParetoAuditRunResult(
        name=summary.name,
        weights=weight_set,
        n_samples=summary.n_samples,
        success_rate=summary.success_rate,
        mean_detector_success_gain=summary.mean_detector_success_gain,
        mean_noise_action_reduction=summary.mean_noise_action_reduction,
        mean_noise_leakage_reduction=summary.mean_noise_leakage_reduction,
        mean_objective_gain=summary.mean_objective_gain,
        mean_normalized_objective_gain=float(np.mean(normalized_values)),
        mean_best_control_cost=summary.mean_best_control_cost,
        mean_saturated_knobs=summary.mean_saturated_knobs,
        mean_near_bound_knobs=summary.mean_near_bound_knobs,
        mean_transport_term_gain=float(np.mean(weight_set.transport * np.array([detail.sample.transport_gain for detail in details], dtype=np.float64))),
        mean_noise_action_term_gain=float(np.mean(weight_set.noise_action * noise_values)),
        mean_noise_leakage_term_gain=float(np.mean(weight_set.leakage * leakage_values)),
        mean_control_cost_penalty=float(np.mean(weight_set.control_cost * control_cost_delta)),
        detector_gain_ci_low=detector_ci.ci_low,
        detector_gain_ci_high=detector_ci.ci_high,
        noise_action_ci_low=noise_ci.ci_low,
        noise_action_ci_high=noise_ci.ci_high,
        leakage_ci_low=leakage_ci.ci_low,
        leakage_ci_high=leakage_ci.ci_high,
        objective_ci_low=objective_ci.ci_low,
        objective_ci_high=objective_ci.ci_high,
        normalized_objective_ci_low=normalized_ci.ci_low,
        normalized_objective_ci_high=normalized_ci.ci_high,
        mean_theta_l2=float(np.mean(theta_l2)),
        knob_profiles=summarize_knob_profiles(summary.name, details, registry),
    )


def pairwise_theta_distance_summary(audit_results: list[ParetoAuditRunResult]) -> tuple[float, float]:
    if len(audit_results) < 2:
        return 0.0, 0.0
    profile_vectors = {
        result.name: np.array([profile.mean_value for profile in result.knob_profiles], dtype=np.float64)
        for result in audit_results
    }
    distances: list[float] = []
    names = list(profile_vectors)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            distances.append(float(np.linalg.norm(profile_vectors[left_name] - profile_vectors[right_name])))
    return float(np.mean(distances)), float(np.min(distances))


def assess_regime_shape(
    audit_results: list[ParetoAuditRunResult],
    scale_summary: ObjectiveScaleSummary,
    mean_pairwise_theta_distance: float,
) -> tuple[bool, str]:
    detector_span = float(
        max(result.mean_detector_success_gain for result in audit_results)
        - min(result.mean_detector_success_gain for result in audit_results)
    )
    noise_span = float(
        max(result.mean_noise_action_reduction for result in audit_results)
        - min(result.mean_noise_action_reduction for result in audit_results)
    )
    normalization_recommended = bool(scale_summary.raw_scale_ratio >= 4.0 or detector_span < 0.0025)
    if detector_span < 0.0025 and mean_pairwise_theta_distance < 0.08:
        assessment = "constraint_shaped_family"
    elif noise_span > 0.0010 or mean_pairwise_theta_distance >= 0.08:
        assessment = "partially_separated_regimes"
    else:
        assessment = "mildly_separated_family"
    return normalization_recommended, assessment


def run_transition_motor_pareto_audit(
    config: dict[str, Any],
) -> tuple[list[ParetoAuditRunResult], ParetoStressAuditSummary, Path, Path, Path]:
    parsed = transition_motor_pareto_audit_from_dict(config)
    motor_config = parsed.motor_config
    grid = motor_config.grid
    registry = motor_config.knob_registry
    basis = make_control_basis(grid)

    fabrication_profiles = sample_fabrication_disorder_profiles(
        int(parsed.fabrication_disorder["n_samples"]),
        grid.n_sites,
        float(parsed.fabrication_disorder["onsite_sigma"]),
        float(parsed.fabrication_disorder["correlation_length_sites"]),
        int(parsed.fabrication_disorder["seed"]),
    )

    detailed_by_preset: dict[str, list[DetailedMotorEnsembleSample]] = {}
    raw_results: list[ParetoRunResult] = []
    for weight_set in parsed.weight_sets:
        preset_motor = replace(
            motor_config,
            objective_weights={
                "transport": weight_set.transport,
                "noise_action": weight_set.noise_action,
                "leakage": weight_set.leakage,
                "control_cost": weight_set.control_cost,
            },
        )
        details: list[DetailedMotorEnsembleSample] = []
        for sample_id, onsite_profile in enumerate(fabrication_profiles):
            noise_profiles = sample_phase_noise_profiles(
                int(parsed.phase_noise["n_profiles_per_sample"]),
                grid.n_sites,
                float(parsed.phase_noise["profile_sigma"]),
                float(parsed.phase_noise["correlation_length_sites"]),
                int(parsed.phase_noise["seed"]) + sample_id,
            )
            detail = run_single_motor_ensemble_sample_detailed(
                grid,
                registry,
                basis,
                onsite_profile,
                noise_profiles,
                preset_motor,
                parsed.success_criteria,
            )
            details.append(replace(detail, sample=replace(detail.sample, sample_id=sample_id)))
        detailed_by_preset[weight_set.name] = details
        raw_results.append(
            ParetoRunResult(
                name=weight_set.name,
                weights=weight_set,
                n_samples=len(details),
                success_rate=float(np.mean([1.0 if detail.sample.success else 0.0 for detail in details], dtype=np.float64)),
                detector_win_rate=float(np.mean([1.0 if detail.sample.detector_success_gain > 0.0 else 0.0 for detail in details], dtype=np.float64)),
                noise_action_win_rate=float(np.mean([1.0 if detail.sample.noise_action_reduction > 0.0 else 0.0 for detail in details], dtype=np.float64)),
                leakage_win_rate=float(np.mean([1.0 if detail.sample.noise_leakage_reduction > 0.0 else 0.0 for detail in details], dtype=np.float64)),
                objective_win_rate=float(np.mean([1.0 if detail.sample.objective_gain > 0.0 else 0.0 for detail in details], dtype=np.float64)),
                mean_detector_success_gain=float(np.mean([detail.sample.detector_success_gain for detail in details])),
                median_detector_success_gain=float(np.median([detail.sample.detector_success_gain for detail in details])),
                mean_noise_action_reduction=float(np.mean([detail.sample.noise_action_reduction for detail in details])),
                median_noise_action_reduction=float(np.median([detail.sample.noise_action_reduction for detail in details])),
                mean_noise_leakage_reduction=float(np.mean([detail.sample.noise_leakage_reduction for detail in details])),
                median_noise_leakage_reduction=float(np.median([detail.sample.noise_leakage_reduction for detail in details])),
                mean_objective_gain=float(np.mean([detail.sample.objective_gain for detail in details])),
                median_objective_gain=float(np.median([detail.sample.objective_gain for detail in details])),
                mean_best_control_cost=float(np.mean([detail.sample.best_control_cost for detail in details])),
                mean_saturated_knobs=float(np.mean([detail.sample.saturated_knobs for detail in details])),
                mean_near_bound_knobs=float(np.mean([detail.sample.near_bound_knobs for detail in details])),
            )
        )

    scale_summary = compute_objective_scale_summary(detailed_by_preset)
    audit_results = [
        _build_audit_result(
            summary,
            summary.weights,
            detailed_by_preset[summary.name],
            registry,
            scale_summary,
            n_bootstrap=parsed.bootstrap_n,
            seed=parsed.bootstrap_seed + index * 17,
        )
        for index, summary in enumerate(raw_results)
    ]

    marked = mark_pareto_front(
        [
            ParetoRunResult(
                name=result.name,
                weights=result.weights,
                n_samples=result.n_samples,
                success_rate=result.success_rate,
                detector_win_rate=0.0,
                noise_action_win_rate=0.0,
                leakage_win_rate=0.0,
                objective_win_rate=0.0,
                mean_detector_success_gain=result.mean_detector_success_gain,
                median_detector_success_gain=result.mean_detector_success_gain,
                mean_noise_action_reduction=result.mean_noise_action_reduction,
                median_noise_action_reduction=result.mean_noise_action_reduction,
                mean_noise_leakage_reduction=result.mean_noise_leakage_reduction,
                median_noise_leakage_reduction=result.mean_noise_leakage_reduction,
                mean_objective_gain=result.mean_objective_gain,
                median_objective_gain=result.mean_objective_gain,
                mean_best_control_cost=result.mean_best_control_cost,
                mean_saturated_knobs=result.mean_saturated_knobs,
                mean_near_bound_knobs=result.mean_near_bound_knobs,
            )
            for result in audit_results
        ],
        maximize=parsed.maximize,
        minimize=parsed.minimize,
    )
    pareto_lookup = {item.name: item.is_pareto_optimal for item in marked}
    final_results = [replace(result, is_pareto_optimal=pareto_lookup[result.name]) for result in audit_results]

    mean_pairwise_theta_distance, min_pairwise_theta_distance = pairwise_theta_distance_summary(final_results)
    normalization_recommended, regime_assessment = assess_regime_shape(
        final_results,
        scale_summary,
        mean_pairwise_theta_distance,
    )
    summary = ParetoStressAuditSummary(
        n_weight_sets=len(final_results),
        n_samples_per_weight_set=final_results[0].n_samples if final_results else 0,
        pareto_optimal_names=[result.name for result in final_results if result.is_pareto_optimal],
        best_detector_name=max(final_results, key=lambda item: item.mean_detector_success_gain).name,
        best_noise_action_name=max(final_results, key=lambda item: item.mean_noise_action_reduction).name,
        best_leakage_name=max(final_results, key=lambda item: item.mean_noise_leakage_reduction).name,
        best_cost_name=min(final_results, key=lambda item: item.mean_best_control_cost).name,
        best_normalized_name=max(final_results, key=lambda item: item.mean_normalized_objective_gain).name,
        normalization_recommended=normalization_recommended,
        regime_assessment=regime_assessment,
        mean_pairwise_theta_distance=mean_pairwise_theta_distance,
        min_pairwise_theta_distance=min_pairwise_theta_distance,
        objective_scale_summary=scale_summary,
    )

    csv_path = write_pareto_audit_csv(parsed.outputs["csv_path"], final_results)
    summary_path = write_pareto_audit_summary_json(parsed.outputs["summary_path"], final_results, summary)
    plot_path = plot_pareto_audit(final_results, parsed.outputs["plot_path"])
    return final_results, summary, csv_path, summary_path, plot_path


def write_pareto_audit_csv(path: str | Path, results: list[ParetoAuditRunResult]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    knob_names = sorted({profile.knob for result in results for profile in result.knob_profiles})
    fieldnames = [
        "name",
        "transport",
        "noise_action",
        "leakage",
        "control_cost_weight",
        "n_samples",
        "success_rate",
        "mean_detector_success_gain",
        "mean_noise_action_reduction",
        "mean_noise_leakage_reduction",
        "mean_objective_gain",
        "mean_normalized_objective_gain",
        "mean_best_control_cost",
        "mean_saturated_knobs",
        "mean_near_bound_knobs",
        "mean_transport_term_gain",
        "mean_noise_action_term_gain",
        "mean_noise_leakage_term_gain",
        "mean_control_cost_penalty",
        "detector_gain_ci_low",
        "detector_gain_ci_high",
        "noise_action_ci_low",
        "noise_action_ci_high",
        "leakage_ci_low",
        "leakage_ci_high",
        "objective_ci_low",
        "objective_ci_high",
        "normalized_objective_ci_low",
        "normalized_objective_ci_high",
        "mean_theta_l2",
        "is_pareto_optimal",
    ]
    for knob_name in knob_names:
        fieldnames.extend(
            [
                f"{knob_name}_mean_value",
                f"{knob_name}_mean_abs_value",
                f"{knob_name}_saturation_rate",
                f"{knob_name}_near_bound_rate",
            ]
        )

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "name": result.name,
                "transport": result.weights.transport,
                "noise_action": result.weights.noise_action,
                "leakage": result.weights.leakage,
                "control_cost_weight": result.weights.control_cost,
                "n_samples": result.n_samples,
                "success_rate": result.success_rate,
                "mean_detector_success_gain": result.mean_detector_success_gain,
                "mean_noise_action_reduction": result.mean_noise_action_reduction,
                "mean_noise_leakage_reduction": result.mean_noise_leakage_reduction,
                "mean_objective_gain": result.mean_objective_gain,
                "mean_normalized_objective_gain": result.mean_normalized_objective_gain,
                "mean_best_control_cost": result.mean_best_control_cost,
                "mean_saturated_knobs": result.mean_saturated_knobs,
                "mean_near_bound_knobs": result.mean_near_bound_knobs,
                "mean_transport_term_gain": result.mean_transport_term_gain,
                "mean_noise_action_term_gain": result.mean_noise_action_term_gain,
                "mean_noise_leakage_term_gain": result.mean_noise_leakage_term_gain,
                "mean_control_cost_penalty": result.mean_control_cost_penalty,
                "detector_gain_ci_low": result.detector_gain_ci_low,
                "detector_gain_ci_high": result.detector_gain_ci_high,
                "noise_action_ci_low": result.noise_action_ci_low,
                "noise_action_ci_high": result.noise_action_ci_high,
                "leakage_ci_low": result.leakage_ci_low,
                "leakage_ci_high": result.leakage_ci_high,
                "objective_ci_low": result.objective_ci_low,
                "objective_ci_high": result.objective_ci_high,
                "normalized_objective_ci_low": result.normalized_objective_ci_low,
                "normalized_objective_ci_high": result.normalized_objective_ci_high,
                "mean_theta_l2": result.mean_theta_l2,
                "is_pareto_optimal": result.is_pareto_optimal,
            }
            profile_lookup = {profile.knob: profile for profile in result.knob_profiles}
            for knob_name in knob_names:
                profile = profile_lookup[knob_name]
                row[f"{knob_name}_mean_value"] = profile.mean_value
                row[f"{knob_name}_mean_abs_value"] = profile.mean_abs_value
                row[f"{knob_name}_saturation_rate"] = profile.saturation_rate
                row[f"{knob_name}_near_bound_rate"] = profile.near_bound_rate
            writer.writerow(row)
    return output


def write_pareto_audit_summary_json(
    path: str | Path,
    results: list[ParetoAuditRunResult],
    summary: ParetoStressAuditSummary,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": [{**asdict(result), "weights": asdict(result.weights)} for result in results],
        "summary": asdict(summary),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def plot_pareto_audit(results: list[ParetoAuditRunResult], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Synthetic Transition Motor Pareto Stress Audit", fontsize=13)

    knob_names = [profile.knob for profile in results[0].knob_profiles]
    theta_heatmap = np.array(
        [
            [next(profile.mean_abs_value for profile in result.knob_profiles if profile.knob == knob) for knob in knob_names]
            for result in results
        ],
        dtype=np.float64,
    )

    for result in results:
        marker = "D" if result.is_pareto_optimal else "o"
        axes[0, 0].errorbar(
            result.mean_detector_success_gain,
            result.mean_noise_action_reduction,
            xerr=[[result.mean_detector_success_gain - result.detector_gain_ci_low], [result.detector_gain_ci_high - result.mean_detector_success_gain]],
            yerr=[[result.mean_noise_action_reduction - result.noise_action_ci_low], [result.noise_action_ci_high - result.mean_noise_action_reduction]],
            fmt=marker,
            capsize=3,
        )
        axes[0, 0].text(result.mean_detector_success_gain, result.mean_noise_action_reduction, result.name, fontsize=8)
        axes[0, 1].scatter(result.mean_noise_leakage_reduction, result.mean_best_control_cost, marker=marker, s=90)
        axes[0, 1].text(result.mean_noise_leakage_reduction, result.mean_best_control_cost, result.name, fontsize=8)

    axes[0, 0].set_title("Detector Gain vs Noise-Action Reduction")
    axes[0, 0].set_xlabel("mean_detector_success_gain")
    axes[0, 0].set_ylabel("mean_noise_action_reduction")
    axes[0, 0].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0, 0].axhline(0.0, color="black", linestyle="--", linewidth=1.0)

    axes[0, 1].set_title("Leakage Reduction vs Control Cost")
    axes[0, 1].set_xlabel("mean_noise_leakage_reduction")
    axes[0, 1].set_ylabel("mean_best_control_cost")
    axes[0, 1].axvline(0.0, color="black", linestyle="--", linewidth=1.0)

    names = [result.name for result in results]
    indices = np.arange(len(results))
    normalized_means = np.array([result.mean_normalized_objective_gain for result in results], dtype=np.float64)
    normalized_err_low = normalized_means - np.array([result.normalized_objective_ci_low for result in results], dtype=np.float64)
    normalized_err_high = np.array([result.normalized_objective_ci_high for result in results], dtype=np.float64) - normalized_means
    axes[1, 0].bar(indices, normalized_means, color="#5c7aea")
    axes[1, 0].errorbar(indices, normalized_means, yerr=np.vstack([normalized_err_low, normalized_err_high]), fmt="none", ecolor="black", capsize=3)
    axes[1, 0].set_title("Normalized Objective Gain")
    axes[1, 0].set_xticks(indices, names, rotation=30, ha="right")
    axes[1, 0].axhline(0.0, color="black", linestyle="--", linewidth=1.0)

    image = axes[1, 1].imshow(theta_heatmap, aspect="auto", cmap="viridis")
    axes[1, 1].set_title("Mean |theta| by Preset and Knob")
    axes[1, 1].set_xticks(np.arange(len(knob_names)), knob_names, rotation=30, ha="right")
    axes[1, 1].set_yticks(np.arange(len(names)), names)
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output

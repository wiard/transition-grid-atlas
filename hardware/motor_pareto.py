"""Objective-weight sweep and Pareto analysis for the transition motor."""

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
from hardware.ensemble_statistics import compute_win_rates
from hardware.motor_ensemble import (
    DetailedMotorEnsembleSample,
    MotorEnsembleSampleResult,
    run_single_motor_ensemble_sample,
    run_single_motor_ensemble_sample_detailed,
    sample_fabrication_disorder_profiles,
    sample_phase_noise_profiles,
    transition_motor_ensemble_from_dict,
)


@dataclass(frozen=True)
class ObjectiveWeightSet:
    name: str
    transport: float
    noise_action: float
    leakage: float
    control_cost: float


@dataclass(frozen=True)
class ParetoRunResult:
    name: str
    weights: ObjectiveWeightSet
    n_samples: int
    success_rate: float
    detector_win_rate: float
    noise_action_win_rate: float
    leakage_win_rate: float
    objective_win_rate: float
    mean_detector_success_gain: float
    median_detector_success_gain: float
    mean_noise_action_reduction: float
    median_noise_action_reduction: float
    mean_noise_leakage_reduction: float
    median_noise_leakage_reduction: float
    mean_objective_gain: float
    median_objective_gain: float
    mean_best_control_cost: float
    mean_saturated_knobs: float
    mean_near_bound_knobs: float
    is_pareto_optimal: bool = False


@dataclass(frozen=True)
class ParetoSweepSummary:
    n_weight_sets: int
    pareto_optimal_names: list[str]
    best_detector_name: str
    best_noise_action_name: str
    best_leakage_name: str
    best_balanced_name: str


@dataclass(frozen=True)
class TransitionMotorParetoConfig:
    ensemble_config: TransitionMotorEnsembleConfig
    weight_sets: list[ObjectiveWeightSet]
    maximize: list[str]
    minimize: list[str]
    outputs: dict[str, str]


def validate_weight_set(weight_set: ObjectiveWeightSet) -> None:
    if not weight_set.name.strip():
        raise ValueError("weight set name cannot be empty")
    weights = [
        float(weight_set.transport),
        float(weight_set.noise_action),
        float(weight_set.leakage),
        float(weight_set.control_cost),
    ]
    if any(value < 0.0 for value in weights):
        raise ValueError("objective weights must be non-negative")
    if weight_set.transport <= 0.0 and weight_set.noise_action <= 0.0 and weight_set.leakage <= 0.0:
        raise ValueError("at least one performance weight must be positive")


def load_weight_sets(config: dict[str, Any]) -> list[ObjectiveWeightSet]:
    weight_sets = [
        ObjectiveWeightSet(
            name=str(item["name"]),
            transport=float(item["transport"]),
            noise_action=float(item["noise_action"]),
            leakage=float(item["leakage"]),
            control_cost=float(item["control_cost"]),
        )
        for item in list(config["weight_sets"])
    ]
    for weight_set in weight_sets:
        validate_weight_set(weight_set)
    return weight_sets


def transition_motor_pareto_from_dict(data: dict[str, Any]) -> TransitionMotorParetoConfig:
    block = dict(data["transition_motor_pareto"])
    ensemble_like = {
        "transition_motor_ensemble": {
            "grid": block["grid"],
            "knobs": block["knobs"],
            "fabrication_disorder": block["fabrication_disorder"],
            "phase_noise": block["phase_noise"],
            "objective_weights": {
                "transport": 1.0,
                "noise_action": 0.5,
                "leakage": 0.25,
                "control_cost": 0.01,
            },
            "optimizer": block["optimizer"],
            "success_criteria": block["success_criteria"],
            "outputs": {
                "csv_path": str(block["outputs"]["csv_path"]),
                "summary_path": str(block["outputs"]["summary_path"]),
            },
        }
    }
    ensemble_config = transition_motor_ensemble_from_dict(ensemble_like)
    return TransitionMotorParetoConfig(
        ensemble_config=ensemble_config,
        weight_sets=load_weight_sets(block),
        maximize=[str(item) for item in list(block["pareto"]["maximize"])],
        minimize=[str(item) for item in list(block["pareto"]["minimize"])],
        outputs={
            "csv_path": str(block["outputs"]["csv_path"]),
            "summary_path": str(block["outputs"]["summary_path"]),
            "plot_path": str(block["outputs"]["plot_path"]),
        },
    )


def dominates(
    a: ParetoRunResult,
    b: ParetoRunResult,
    *,
    maximize: list[str],
    minimize: list[str],
    eps: float = 1e-12,
) -> bool:
    at_least_as_good = True
    strictly_better = False

    for metric in maximize:
        a_val = float(getattr(a, metric))
        b_val = float(getattr(b, metric))
        if a_val < b_val - eps:
            at_least_as_good = False
            break
        if a_val > b_val + eps:
            strictly_better = True

    if at_least_as_good:
        for metric in minimize:
            a_val = float(getattr(a, metric))
            b_val = float(getattr(b, metric))
            if a_val > b_val + eps:
                at_least_as_good = False
                break
            if a_val < b_val - eps:
                strictly_better = True

    return at_least_as_good and strictly_better


def mark_pareto_front(
    results: list[ParetoRunResult],
    *,
    maximize: list[str],
    minimize: list[str],
) -> list[ParetoRunResult]:
    marked: list[ParetoRunResult] = []
    for candidate in results:
        dominated = any(
            dominates(other, candidate, maximize=maximize, minimize=minimize)
            for other in results
            if other.name != candidate.name
        )
        marked.append(replace(candidate, is_pareto_optimal=not dominated))
    return marked


def normalize_metric_values(
    results: list[ParetoRunResult],
    metric: str,
    *,
    higher_is_better: bool,
    eps: float = 1e-12,
) -> dict[str, float]:
    values = np.array([float(getattr(result, metric)) for result in results], dtype=np.float64)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo <= eps:
        return {result.name: 0.5 for result in results}
    normalized: dict[str, float] = {}
    for result in results:
        value = float(getattr(result, metric))
        score = (value - lo) / (hi - lo)
        normalized[result.name] = float(score if higher_is_better else 1.0 - score)
    return normalized


def balanced_score(
    result: ParetoRunResult,
    normalized: dict[str, dict[str, float]],
) -> float:
    metrics = [
        "mean_detector_success_gain",
        "mean_noise_action_reduction",
        "mean_noise_leakage_reduction",
        "success_rate",
        "mean_best_control_cost",
        "mean_saturated_knobs",
    ]
    return float(np.mean([normalized[metric][result.name] for metric in metrics]))


def choose_best_balanced(results: list[ParetoRunResult]) -> str:
    normalized = {
        "mean_detector_success_gain": normalize_metric_values(results, "mean_detector_success_gain", higher_is_better=True),
        "mean_noise_action_reduction": normalize_metric_values(results, "mean_noise_action_reduction", higher_is_better=True),
        "mean_noise_leakage_reduction": normalize_metric_values(results, "mean_noise_leakage_reduction", higher_is_better=True),
        "success_rate": normalize_metric_values(results, "success_rate", higher_is_better=True),
        "mean_best_control_cost": normalize_metric_values(results, "mean_best_control_cost", higher_is_better=False),
        "mean_saturated_knobs": normalize_metric_values(results, "mean_saturated_knobs", higher_is_better=False),
    }
    return max(results, key=lambda item: balanced_score(item, normalized)).name


def _summarize_samples(name: str, weight_set: ObjectiveWeightSet, samples: list[MotorEnsembleSampleResult]) -> ParetoRunResult:
    rows = [
        {
            "detector_success_gain": sample.detector_success_gain,
            "transport_gain": sample.transport_gain,
            "noise_action_reduction": sample.noise_action_reduction,
            "noise_leakage_reduction": sample.noise_leakage_reduction,
            "objective_gain": sample.objective_gain,
            "saturated_knobs": float(sample.saturated_knobs),
            "near_bound_knobs": float(sample.near_bound_knobs),
        }
        for sample in samples
    ]
    win_rates = compute_win_rates(rows)
    return ParetoRunResult(
        name=name,
        weights=weight_set,
        n_samples=len(samples),
        success_rate=float(np.mean([1.0 if sample.success else 0.0 for sample in samples], dtype=np.float64)),
        detector_win_rate=win_rates.detector_win_rate,
        noise_action_win_rate=win_rates.noise_action_win_rate,
        leakage_win_rate=win_rates.leakage_win_rate,
        objective_win_rate=win_rates.objective_win_rate,
        mean_detector_success_gain=float(np.mean([sample.detector_success_gain for sample in samples])),
        median_detector_success_gain=float(np.median([sample.detector_success_gain for sample in samples])),
        mean_noise_action_reduction=float(np.mean([sample.noise_action_reduction for sample in samples])),
        median_noise_action_reduction=float(np.median([sample.noise_action_reduction for sample in samples])),
        mean_noise_leakage_reduction=float(np.mean([sample.noise_leakage_reduction for sample in samples])),
        median_noise_leakage_reduction=float(np.median([sample.noise_leakage_reduction for sample in samples])),
        mean_objective_gain=float(np.mean([sample.objective_gain for sample in samples])),
        median_objective_gain=float(np.median([sample.objective_gain for sample in samples])),
        mean_best_control_cost=float(np.mean([sample.best_control_cost for sample in samples])),
        mean_saturated_knobs=float(np.mean([sample.saturated_knobs for sample in samples])),
        mean_near_bound_knobs=float(np.mean([sample.near_bound_knobs for sample in samples])),
    )


def _sample_row_from_detail(detail: DetailedMotorEnsembleSample) -> dict[str, object]:
    return {
        "sample_id": int(detail.sample.sample_id),
        "detector_success_gain": float(detail.sample.detector_success_gain),
        "transport_gain": float(detail.sample.transport_gain),
        "noise_action_reduction": float(detail.sample.noise_action_reduction),
        "noise_leakage_reduction": float(detail.sample.noise_leakage_reduction),
        "objective_gain": float(detail.sample.objective_gain),
        "baseline_detector_success": float(detail.sample.baseline_detector_success),
        "best_detector_success": float(detail.sample.best_detector_success),
        "baseline_transport_efficiency": float(detail.sample.baseline_transport_efficiency),
        "best_transport_efficiency": float(detail.sample.best_transport_efficiency),
        "baseline_noise_action_on_info": float(detail.sample.baseline_noise_action_on_info),
        "best_noise_action_on_info": float(detail.sample.best_noise_action_on_info),
        "baseline_noise_leakage": float(detail.sample.baseline_noise_leakage),
        "best_noise_leakage": float(detail.sample.best_noise_leakage),
        "baseline_control_cost": float(detail.sample.baseline_control_cost),
        "best_control_cost": float(detail.sample.best_control_cost),
        "saturated_knobs": float(detail.sample.saturated_knobs),
        "near_bound_knobs": float(detail.sample.near_bound_knobs),
        "success": float(1.0 if detail.sample.success else 0.0),
        "best_theta": dict(detail.best_theta),
    }


def run_pareto_weight_sweep_with_samples(
    config: dict[str, Any],
) -> tuple[list[ParetoRunResult], ParetoSweepSummary, dict[str, list[dict[str, object]]]]:
    parsed = transition_motor_pareto_from_dict(config)
    ensemble = parsed.ensemble_config
    base_motor = ensemble.motor_config
    grid = base_motor.grid
    registry = base_motor.knob_registry
    basis = make_control_basis(grid)

    fabrication_profiles = sample_fabrication_disorder_profiles(
        int(ensemble.fabrication_disorder["n_samples"]),
        grid.n_sites,
        float(ensemble.fabrication_disorder["onsite_sigma"]),
        float(ensemble.fabrication_disorder["correlation_length_sites"]),
        int(ensemble.fabrication_disorder["seed"]),
    )
    phase_noise_seed = int(ensemble.phase_noise["seed"])
    phase_noise_per_sample = int(ensemble.phase_noise["n_profiles_per_sample"])
    phase_noise_sigma = float(ensemble.phase_noise["profile_sigma"])
    phase_noise_corr = float(ensemble.phase_noise["correlation_length_sites"])

    results: list[ParetoRunResult] = []
    per_preset_sample_rows: dict[str, list[dict[str, object]]] = {}
    for weight_set in parsed.weight_sets:
        motor_config = replace(
            base_motor,
            objective_weights={
                "transport": weight_set.transport,
                "noise_action": weight_set.noise_action,
                "leakage": weight_set.leakage,
                "control_cost": weight_set.control_cost,
            },
        )
        samples: list[MotorEnsembleSampleResult] = []
        sample_rows: list[dict[str, object]] = []
        for sample_id, onsite_profile in enumerate(fabrication_profiles):
            noise_profiles = sample_phase_noise_profiles(
                phase_noise_per_sample,
                grid.n_sites,
                phase_noise_sigma,
                phase_noise_corr,
                phase_noise_seed + sample_id,
            )
            detail = run_single_motor_ensemble_sample_detailed(
                grid,
                registry,
                basis,
                onsite_profile,
                noise_profiles,
                motor_config,
                ensemble.success_criteria,
            )
            sample = replace(detail.sample, sample_id=sample_id)
            samples.append(sample)
            sample_rows.append(_sample_row_from_detail(replace(detail, sample=sample)))
        results.append(_summarize_samples(weight_set.name, weight_set, samples))
        per_preset_sample_rows[weight_set.name] = sample_rows

    marked = mark_pareto_front(results, maximize=parsed.maximize, minimize=parsed.minimize)
    summary = ParetoSweepSummary(
        n_weight_sets=len(marked),
        pareto_optimal_names=[item.name for item in marked if item.is_pareto_optimal],
        best_detector_name=max(marked, key=lambda item: item.mean_detector_success_gain).name,
        best_noise_action_name=max(marked, key=lambda item: item.mean_noise_action_reduction).name,
        best_leakage_name=max(marked, key=lambda item: item.mean_noise_leakage_reduction).name,
        best_balanced_name=choose_best_balanced(marked),
    )
    return marked, summary, per_preset_sample_rows


def run_pareto_weight_sweep(config: dict[str, Any]) -> tuple[list[ParetoRunResult], ParetoSweepSummary]:
    results, summary, _ = run_pareto_weight_sweep_with_samples(config)
    return results, summary


def write_pareto_csv(path: str | Path, results: list[ParetoRunResult]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "transport",
        "noise_action",
        "leakage",
        "control_cost_weight",
        "n_samples",
        "success_rate",
        "detector_win_rate",
        "noise_action_win_rate",
        "leakage_win_rate",
        "objective_win_rate",
        "mean_detector_success_gain",
        "median_detector_success_gain",
        "mean_noise_action_reduction",
        "median_noise_action_reduction",
        "mean_noise_leakage_reduction",
        "median_noise_leakage_reduction",
        "mean_objective_gain",
        "median_objective_gain",
        "mean_best_control_cost",
        "mean_saturated_knobs",
        "mean_near_bound_knobs",
        "is_pareto_optimal",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "name": result.name,
                    "transport": result.weights.transport,
                    "noise_action": result.weights.noise_action,
                    "leakage": result.weights.leakage,
                    "control_cost_weight": result.weights.control_cost,
                    "n_samples": result.n_samples,
                    "success_rate": result.success_rate,
                    "detector_win_rate": result.detector_win_rate,
                    "noise_action_win_rate": result.noise_action_win_rate,
                    "leakage_win_rate": result.leakage_win_rate,
                    "objective_win_rate": result.objective_win_rate,
                    "mean_detector_success_gain": result.mean_detector_success_gain,
                    "median_detector_success_gain": result.median_detector_success_gain,
                    "mean_noise_action_reduction": result.mean_noise_action_reduction,
                    "median_noise_action_reduction": result.median_noise_action_reduction,
                    "mean_noise_leakage_reduction": result.mean_noise_leakage_reduction,
                    "median_noise_leakage_reduction": result.median_noise_leakage_reduction,
                    "mean_objective_gain": result.mean_objective_gain,
                    "median_objective_gain": result.median_objective_gain,
                    "mean_best_control_cost": result.mean_best_control_cost,
                    "mean_saturated_knobs": result.mean_saturated_knobs,
                    "mean_near_bound_knobs": result.mean_near_bound_knobs,
                    "is_pareto_optimal": result.is_pareto_optimal,
                }
            )
    return output


def write_pareto_summary_json(
    path: str | Path,
    results: list[ParetoRunResult],
    summary: ParetoSweepSummary,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": [{**asdict(result), "weights": asdict(result.weights)} for result in results],
        "summary": asdict(summary),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def plot_pareto_results(results: list[ParetoRunResult], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Synthetic Transition Motor Pareto Weight Sweep", fontsize=13)

    def style_for(result: ParetoRunResult) -> tuple[str, int]:
        return ("D", 90) if result.is_pareto_optimal else ("o", 60)

    for result in results:
        marker, size = style_for(result)
        axes[0, 0].scatter(result.mean_detector_success_gain, result.mean_noise_action_reduction, marker=marker, s=size)
        axes[0, 0].text(result.mean_detector_success_gain, result.mean_noise_action_reduction, result.name, fontsize=8)
        axes[0, 1].scatter(result.mean_detector_success_gain, result.mean_noise_leakage_reduction, marker=marker, s=size)
        axes[0, 1].text(result.mean_detector_success_gain, result.mean_noise_leakage_reduction, result.name, fontsize=8)

    axes[0, 0].set_title("Detector Gain vs Noise-Action Reduction")
    axes[0, 0].set_xlabel("mean_detector_success_gain")
    axes[0, 0].set_ylabel("mean_noise_action_reduction")
    axes[0, 0].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0, 0].axhline(0.0, color="black", linestyle="--", linewidth=1.0)

    axes[0, 1].set_title("Detector Gain vs Leakage Reduction")
    axes[0, 1].set_xlabel("mean_detector_success_gain")
    axes[0, 1].set_ylabel("mean_noise_leakage_reduction")
    axes[0, 1].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0, 1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)

    names = [result.name for result in results]
    indices = np.arange(len(results))
    axes[1, 0].bar(indices, [result.success_rate for result in results], color="#2a9d8f")
    axes[1, 0].set_title("Success Rate by Weight Set")
    axes[1, 0].set_xticks(indices, names, rotation=30, ha="right")
    axes[1, 0].set_ylabel("success_rate")

    width = 0.38
    axes[1, 1].bar(indices - width / 2.0, [result.mean_best_control_cost for result in results], width=width, label="mean_best_control_cost", color="#6d597a")
    axes[1, 1].bar(indices + width / 2.0, [result.mean_saturated_knobs for result in results], width=width, label="mean_saturated_knobs", color="#f4a261")
    axes[1, 1].set_title("Control Cost and Bound Pressure")
    axes[1, 1].set_xticks(indices, names, rotation=30, ha="right")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output

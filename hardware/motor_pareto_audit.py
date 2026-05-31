"""Pareto stress, normalization, and regime-separation audit for the transition motor."""

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

from hardware.control_knobs import knob_registry_from_dicts
from hardware.motor_pareto import (
    ParetoRunResult,
    ObjectiveWeightSet,
    run_pareto_weight_sweep_with_samples,
)


@dataclass(frozen=True)
class ObjectiveComponentScale:
    preset_name: str
    mean_detector_component: float
    mean_noise_action_component: float
    mean_leakage_component: float
    mean_control_cost_component: float
    dominant_component: str
    detector_to_noise_ratio: float
    detector_to_leakage_ratio: float


@dataclass(frozen=True)
class PresetCI:
    preset_name: str
    metric: str
    mean: float
    median: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class KnobProfile:
    preset_name: str
    mean_theta: dict[str, float]
    std_theta: dict[str, float]
    fraction_at_upper_bound: dict[str, float]
    fraction_at_lower_bound: dict[str, float]


@dataclass(frozen=True)
class RegimeDistance:
    preset_a: str
    preset_b: str
    distance: float


@dataclass(frozen=True)
class ParetoAuditSummary:
    n_presets: int
    base_preset_names: list[str]
    stress_preset_names: list[str]
    dominant_component_overall: str
    component_scaling_issue: bool
    mean_pairwise_regime_distance: float
    closest_presets: tuple[str, str]
    most_separated_presets: tuple[str, str]
    compressed_frontier: bool


@dataclass(frozen=True)
class ParetoAuditResult:
    pareto_results: list[ParetoRunResult]
    component_scales: list[ObjectiveComponentScale]
    normalized_scores: dict[str, float]
    confidence_intervals: list[PresetCI]
    knob_profiles: list[KnobProfile]
    regime_distances: list[RegimeDistance]
    summary: ParetoAuditSummary
    best_normalized_score_name: str
    n_samples_per_preset: int
    bootstrap_n: int
    bootstrap_ci: float
    csv_path: str | None = None
    summary_path: str | None = None
    plot_path: str | None = None


def load_audit_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if "transition_motor_pareto_audit" not in payload:
        raise ValueError("audit config must define transition_motor_pareto_audit")
    block = payload["transition_motor_pareto_audit"]
    if "base_config" not in block:
        raise ValueError("transition_motor_pareto_audit.base_config is required")
    payload["__config_path__"] = str(config_path)
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_base_config_path(audit_path: Path, base_config_value: str) -> Path:
    candidate = Path(base_config_value).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()
    relative_to_audit = (audit_path.parent / candidate).resolve()
    if relative_to_audit.exists():
        return relative_to_audit
    relative_to_cwd = (Path.cwd() / candidate).resolve()
    if relative_to_cwd.exists():
        return relative_to_cwd
    raise FileNotFoundError(f"base_config not found: {base_config_value}")


def _load_base_pareto_config(audit_config: dict[str, Any]) -> dict[str, Any]:
    audit_path = Path(str(audit_config["__config_path__"]))
    block = dict(audit_config["transition_motor_pareto_audit"])
    base_path = _resolve_base_config_path(audit_path, str(block["base_config"]))
    base_payload = _load_yaml(base_path)
    if "transition_motor_pareto" not in base_payload:
        raise ValueError("base Pareto config must define transition_motor_pareto")
    return base_payload


def merge_base_and_stress_weight_sets(base_config: dict[str, Any], audit_config: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base_config)
    base_block = dict(merged["transition_motor_pareto"])
    audit_block = dict(audit_config["transition_motor_pareto_audit"])
    base_weight_sets = [deepcopy(item) for item in list(base_block["weight_sets"])]
    stress_weight_sets = [deepcopy(item) for item in list(audit_block["stress_weight_sets"])]

    seen_names: set[str] = set()
    ordered_weight_sets: list[dict[str, Any]] = []
    for item in [*base_weight_sets, *stress_weight_sets]:
        name = str(item["name"])
        if name in seen_names:
            raise ValueError(f"duplicate preset name: {name}")
        seen_names.add(name)
        ordered_weight_sets.append(item)

    base_block["weight_sets"] = ordered_weight_sets
    if "outputs" in audit_block:
        base_block["outputs"] = deepcopy(dict(audit_block["outputs"]))
    merged["transition_motor_pareto"] = base_block
    return merged


def compute_objective_components_for_result(result: ParetoRunResult) -> dict[str, float]:
    if result.objective_mode_type in {"normalized", "calibrated"} and result.normalization is not None:
        return {
            "detector_component": float(
                result.weights.transport * result.mean_detector_success_gain / result.normalization.transport_scale
            ),
            "noise_action_component": float(
                -result.weights.noise_action * result.mean_noise_action_reduction / result.normalization.noise_action_scale
            ),
            "leakage_component": float(
                -result.weights.leakage * result.mean_noise_leakage_reduction / result.normalization.leakage_scale
            ),
            "control_cost_component": float(
                -result.weights.control_cost * result.mean_best_control_cost / result.normalization.control_cost_scale
            ),
        }
    return {
        "detector_component": float(result.weights.transport * result.mean_detector_success_gain),
        "noise_action_component": float(-result.weights.noise_action * result.mean_noise_action_reduction),
        "leakage_component": float(-result.weights.leakage * result.mean_noise_leakage_reduction),
        "control_cost_component": float(-result.weights.control_cost * result.mean_best_control_cost),
    }


def compute_component_scale(
    result: ParetoRunResult,
    *,
    eps: float = 1e-12,
) -> ObjectiveComponentScale:
    components = compute_objective_components_for_result(result)
    dominant_component = max(components.items(), key=lambda item: abs(item[1]))[0].replace("_component", "")
    return ObjectiveComponentScale(
        preset_name=result.name,
        mean_detector_component=components["detector_component"],
        mean_noise_action_component=components["noise_action_component"],
        mean_leakage_component=components["leakage_component"],
        mean_control_cost_component=components["control_cost_component"],
        dominant_component=dominant_component,
        detector_to_noise_ratio=float(abs(components["detector_component"]) / (abs(components["noise_action_component"]) + eps)),
        detector_to_leakage_ratio=float(abs(components["detector_component"]) / (abs(components["leakage_component"]) + eps)),
    )


def minmax_normalize_values(
    values: dict[str, float],
    *,
    higher_is_better: bool = True,
    eps: float = 1e-12,
) -> dict[str, float]:
    numeric = np.array(list(values.values()), dtype=np.float64)
    lo = float(np.min(numeric))
    hi = float(np.max(numeric))
    if hi - lo <= eps:
        return {name: 0.5 for name in values}
    normalized: dict[str, float] = {}
    for name, value in values.items():
        score = float((value - lo) / (hi - lo))
        normalized[name] = score if higher_is_better else float(1.0 - score)
    return normalized


def normalized_metric_vectors(results: list[ParetoRunResult]) -> dict[str, dict[str, float]]:
    metrics = {
        "detector_gain_norm": minmax_normalize_values(
            {result.name: result.mean_detector_success_gain for result in results},
            higher_is_better=True,
        ),
        "noise_action_norm": minmax_normalize_values(
            {result.name: result.mean_noise_action_reduction for result in results},
            higher_is_better=True,
        ),
        "leakage_norm": minmax_normalize_values(
            {result.name: result.mean_noise_leakage_reduction for result in results},
            higher_is_better=True,
        ),
        "success_rate_norm": minmax_normalize_values(
            {result.name: result.success_rate for result in results},
            higher_is_better=True,
        ),
        "control_cost_norm": minmax_normalize_values(
            {result.name: result.mean_best_control_cost for result in results},
            higher_is_better=False,
        ),
        "saturated_knobs_norm": minmax_normalize_values(
            {result.name: result.mean_saturated_knobs for result in results},
            higher_is_better=False,
        ),
    }
    vectors: dict[str, dict[str, float]] = {}
    for result in results:
        vectors[result.name] = {metric: values[result.name] for metric, values in metrics.items()}
    return vectors


def normalized_balanced_scores(results: list[ParetoRunResult]) -> dict[str, float]:
    vectors = normalized_metric_vectors(results)
    return {
        name: float(np.mean(list(metric_values.values()), dtype=np.float64))
        for name, metric_values in vectors.items()
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    ci: float,
    seed: int,
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


def compute_preset_confidence_intervals(
    per_preset_sample_rows: dict[str, list[dict[str, float | object]]],
    *,
    metrics: list[str],
    n_bootstrap: int,
    ci: float,
    seed: int,
) -> list[PresetCI]:
    intervals: list[PresetCI] = []
    for preset_index, preset_name in enumerate(sorted(per_preset_sample_rows)):
        rows = per_preset_sample_rows[preset_name]
        if not rows:
            raise ValueError(f"preset {preset_name} has no sample rows")
        for metric_index, metric in enumerate(metrics):
            values = np.array([float(row[metric]) for row in rows], dtype=np.float64)
            ci_low, ci_high = bootstrap_mean_ci(
                values,
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed + preset_index * 101 + metric_index,
            )
            intervals.append(
                PresetCI(
                    preset_name=preset_name,
                    metric=metric,
                    mean=float(np.mean(values)),
                    median=float(np.median(values)),
                    ci_low=ci_low,
                    ci_high=ci_high,
                )
            )
    return intervals


def compute_knob_profiles(
    per_preset_sample_rows: dict[str, list[dict[str, object]]],
    knob_names: list[str],
    registry,
    *,
    bound_tolerance_fraction: float = 0.02,
) -> list[KnobProfile]:
    profiles: list[KnobProfile] = []
    knob_lookup = {knob.name: knob for knob in registry.knobs}
    for knob_name in knob_names:
        if knob_name not in knob_lookup:
            raise ValueError(f"unknown knob in profile request: {knob_name}")

    for preset_name, rows in per_preset_sample_rows.items():
        if not rows:
            raise ValueError(f"preset {preset_name} has no sample rows")
        mean_theta: dict[str, float] = {}
        std_theta: dict[str, float] = {}
        fraction_at_upper_bound: dict[str, float] = {}
        fraction_at_lower_bound: dict[str, float] = {}
        for knob_name in knob_names:
            values: list[float] = []
            upper_hits: list[float] = []
            lower_hits: list[float] = []
            knob = knob_lookup[knob_name]
            span = float(knob.max_value - knob.min_value)
            tolerance = bound_tolerance_fraction * span
            for row in rows:
                theta = row.get("best_theta")
                if not isinstance(theta, dict) or knob_name not in theta:
                    raise ValueError(f"missing best_theta[{knob_name}] for preset {preset_name}")
                value = float(theta[knob_name])
                values.append(value)
                upper_hits.append(1.0 if value >= knob.max_value - tolerance else 0.0)
                lower_hits.append(1.0 if value <= knob.min_value + tolerance else 0.0)
            mean_theta[knob_name] = float(np.mean(values))
            std_theta[knob_name] = float(np.std(values))
            fraction_at_upper_bound[knob_name] = float(np.mean(upper_hits))
            fraction_at_lower_bound[knob_name] = float(np.mean(lower_hits))
        profiles.append(
            KnobProfile(
                preset_name=preset_name,
                mean_theta=mean_theta,
                std_theta=std_theta,
                fraction_at_upper_bound=fraction_at_upper_bound,
                fraction_at_lower_bound=fraction_at_lower_bound,
            )
        )
    return profiles


def regime_metric_vectors(results: list[ParetoRunResult]) -> dict[str, np.ndarray]:
    normalized = normalized_metric_vectors(results)
    return {
        name: np.array(
            [
                metrics["detector_gain_norm"],
                metrics["noise_action_norm"],
                metrics["leakage_norm"],
                metrics["success_rate_norm"],
                metrics["control_cost_norm"],
                metrics["saturated_knobs_norm"],
            ],
            dtype=np.float64,
        )
        for name, metrics in normalized.items()
    }


def pairwise_regime_distances(vectors: dict[str, np.ndarray]) -> list[RegimeDistance]:
    names = sorted(vectors)
    distances: list[RegimeDistance] = []
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            distance = float(np.linalg.norm(vectors[left_name] - vectors[right_name]))
            distances.append(RegimeDistance(left_name, right_name, distance))
    return distances


def summarize_regime_distances(distances: list[RegimeDistance]) -> dict[str, object]:
    if not distances:
        raise ValueError("distances must be non-empty")
    closest = min(distances, key=lambda item: item.distance)
    furthest = max(distances, key=lambda item: item.distance)
    return {
        "mean_pairwise_regime_distance": float(np.mean([item.distance for item in distances], dtype=np.float64)),
        "closest_presets": (closest.preset_a, closest.preset_b),
        "most_separated_presets": (furthest.preset_a, furthest.preset_b),
    }


def is_compressed_frontier(
    mean_pairwise_distance: float,
    *,
    threshold: float = 0.15,
) -> bool:
    return bool(mean_pairwise_distance < threshold)


def _dominant_component_overall(component_scales: list[ObjectiveComponentScale]) -> tuple[str, bool]:
    aggregates = {
        "detector": float(np.mean([abs(item.mean_detector_component) for item in component_scales], dtype=np.float64)),
        "noise_action": float(np.mean([abs(item.mean_noise_action_component) for item in component_scales], dtype=np.float64)),
        "leakage": float(np.mean([abs(item.mean_leakage_component) for item in component_scales], dtype=np.float64)),
        "control_cost": float(np.mean([abs(item.mean_control_cost_component) for item in component_scales], dtype=np.float64)),
    }
    dominant = max(aggregates.items(), key=lambda item: item[1])[0]
    nonzero = [value for value in aggregates.values() if value > 1.0e-12]
    scaling_issue = False
    if len(nonzero) >= 2:
        scaling_issue = max(nonzero) / min(nonzero) >= 4.0
    return dominant, scaling_issue


def run_pareto_audit(config_path: str | Path) -> ParetoAuditResult:
    audit_config = load_audit_config(config_path)
    base_config = _load_base_pareto_config(audit_config)
    merged_config = merge_base_and_stress_weight_sets(base_config, audit_config)

    base_names = [str(item["name"]) for item in list(base_config["transition_motor_pareto"]["weight_sets"])]
    stress_names = [str(item["name"]) for item in list(audit_config["transition_motor_pareto_audit"]["stress_weight_sets"])]

    results, _, per_preset_sample_rows = run_pareto_weight_sweep_with_samples(merged_config)
    registry = knob_registry_from_dicts(list(merged_config["transition_motor_pareto"]["knobs"]))
    knob_names = registry.names()

    component_scales = [compute_component_scale(result) for result in results]
    normalized_scores = normalized_balanced_scores(results)
    best_normalized_score_name = max(normalized_scores.items(), key=lambda item: item[1])[0]

    bootstrap_block = dict(audit_config["transition_motor_pareto_audit"]["bootstrap"])
    ci_metrics = [
        "detector_success_gain",
        "noise_action_reduction",
        "noise_leakage_reduction",
        "objective_gain",
    ]
    confidence_intervals = compute_preset_confidence_intervals(
        per_preset_sample_rows,
        metrics=ci_metrics,
        n_bootstrap=int(bootstrap_block["n_bootstrap"]),
        ci=float(bootstrap_block["ci"]),
        seed=int(bootstrap_block["seed"]),
    )
    knob_profiles = compute_knob_profiles(per_preset_sample_rows, knob_names, registry)
    regime_distances = pairwise_regime_distances(regime_metric_vectors(results))
    regime_summary = summarize_regime_distances(regime_distances)
    dominant_component_overall, component_scaling_issue = _dominant_component_overall(component_scales)
    compressed = is_compressed_frontier(float(regime_summary["mean_pairwise_regime_distance"]))

    summary = ParetoAuditSummary(
        n_presets=len(results),
        base_preset_names=base_names,
        stress_preset_names=stress_names,
        dominant_component_overall=dominant_component_overall,
        component_scaling_issue=component_scaling_issue,
        mean_pairwise_regime_distance=float(regime_summary["mean_pairwise_regime_distance"]),
        closest_presets=tuple(regime_summary["closest_presets"]),
        most_separated_presets=tuple(regime_summary["most_separated_presets"]),
        compressed_frontier=compressed,
    )
    return ParetoAuditResult(
        pareto_results=results,
        component_scales=component_scales,
        normalized_scores=normalized_scores,
        confidence_intervals=confidence_intervals,
        knob_profiles=knob_profiles,
        regime_distances=regime_distances,
        summary=summary,
        best_normalized_score_name=best_normalized_score_name,
        n_samples_per_preset=results[0].n_samples if results else 0,
        bootstrap_n=int(bootstrap_block["n_bootstrap"]),
        bootstrap_ci=float(bootstrap_block["ci"]),
        csv_path=str(audit_config["transition_motor_pareto_audit"]["outputs"]["csv_path"]),
        summary_path=str(audit_config["transition_motor_pareto_audit"]["outputs"]["summary_path"]),
        plot_path=str(audit_config["transition_motor_pareto_audit"]["outputs"]["plot_path"]),
    )


def write_pareto_audit_csv(
    path: str | Path,
    audit_result: ParetoAuditResult,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    component_lookup = {item.preset_name: item for item in audit_result.component_scales}
    rows = []
    for result in audit_result.pareto_results:
        component = component_lookup[result.name]
        rows.append(
            {
                "name": result.name,
                "mode": result.objective_mode_type,
                "normalization_scales": None
                if result.normalization is None
                else {
                    "transport_scale": result.normalization.transport_scale,
                    "noise_action_scale": result.normalization.noise_action_scale,
                    "leakage_scale": result.normalization.leakage_scale,
                    "control_cost_scale": result.normalization.control_cost_scale,
                },
                "success_rate": result.success_rate,
                "mean_detector_success_gain": result.mean_detector_success_gain,
                "mean_noise_action_reduction": result.mean_noise_action_reduction,
                "mean_noise_leakage_reduction": result.mean_noise_leakage_reduction,
                "mean_best_control_cost": result.mean_best_control_cost,
                "mean_saturated_knobs": result.mean_saturated_knobs,
                "pareto_optimal": result.is_pareto_optimal,
                "normalized_balanced_score": audit_result.normalized_scores[result.name],
                "dominant_component": component.dominant_component,
                "detector_to_noise_ratio": component.detector_to_noise_ratio,
                "detector_to_leakage_ratio": component.detector_to_leakage_ratio,
            }
        )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_pareto_audit_summary_json(
    path: str | Path,
    audit_result: ParetoAuditResult,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": asdict(audit_result.summary),
        "component_scales": [asdict(item) for item in audit_result.component_scales],
        "normalized_scores": audit_result.normalized_scores,
        "confidence_intervals": [asdict(item) for item in audit_result.confidence_intervals],
        "knob_profiles": [asdict(item) for item in audit_result.knob_profiles],
        "regime_distances": [asdict(item) for item in audit_result.regime_distances],
        "best_normalized_score_name": audit_result.best_normalized_score_name,
        "n_samples_per_preset": audit_result.n_samples_per_preset,
        "bootstrap_n": audit_result.bootstrap_n,
        "bootstrap_ci": audit_result.bootstrap_ci,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def plot_pareto_audit(
    audit_result: ParetoAuditResult,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    results = audit_result.pareto_results
    base_names = set(audit_result.summary.base_preset_names)
    stress_names = set(audit_result.summary.stress_preset_names)
    component_lookup = {item.preset_name: item for item in audit_result.component_scales}
    knob_lookup = {item.preset_name: item for item in audit_result.knob_profiles}
    knob_names = list(audit_result.knob_profiles[0].mean_theta.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Synthetic Transition Motor Pareto Audit", fontsize=13)

    for result in results:
        is_stress = result.name in stress_names
        color = "#d1495b" if is_stress else "#2a6f97"
        marker = "D" if result.is_pareto_optimal else "o"
        axes[0, 0].scatter(result.mean_detector_success_gain, result.mean_noise_action_reduction, color=color, marker=marker, s=85)
        axes[0, 0].text(result.mean_detector_success_gain, result.mean_noise_action_reduction, result.name, fontsize=8)
        axes[0, 1].scatter(result.mean_detector_success_gain, result.mean_noise_leakage_reduction, color=color, marker=marker, s=85)
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

    heatmap = np.array(
        [[knob_lookup[result.name].mean_theta[knob_name] for knob_name in knob_names] for result in results],
        dtype=np.float64,
    )
    image = axes[1, 0].imshow(heatmap, aspect="auto", cmap="coolwarm")
    axes[1, 0].set_title("Knob-Profile Heatmap (mean theta)")
    axes[1, 0].set_xticks(np.arange(len(knob_names)), knob_names, rotation=30, ha="right")
    axes[1, 0].set_yticks(np.arange(len(results)), [result.name for result in results])
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)

    indices = np.arange(len(results))
    width = 0.2
    axes[1, 1].bar(indices - 1.5 * width, [component_lookup[result.name].mean_detector_component for result in results], width=width, label="detector", color="#2a9d8f")
    axes[1, 1].bar(indices - 0.5 * width, [component_lookup[result.name].mean_noise_action_component for result in results], width=width, label="noise", color="#e76f51")
    axes[1, 1].bar(indices + 0.5 * width, [component_lookup[result.name].mean_leakage_component for result in results], width=width, label="leakage", color="#f4a261")
    axes[1, 1].bar(indices + 1.5 * width, [component_lookup[result.name].mean_control_cost_component for result in results], width=width, label="cost", color="#6d597a")
    axes[1, 1].set_title("Objective Component Dominance")
    axes[1, 1].set_xticks(indices, [result.name for result in results], rotation=30, ha="right")
    axes[1, 1].axhline(0.0, color="black", linewidth=1.0)
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output

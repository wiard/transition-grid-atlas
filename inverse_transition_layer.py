"""Inverse transition analysis for ordered sweep paths.

This module treats each fixed-parameter sweep slice as an ordered transition
path and compares that path with its exact inverse. The goal is operational:
measure symmetry, directional dominance, path selection, and emergent
regularities without making claims about physical law.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from explorer.parameter_sweep import run_parameter_sweep


TransitionPath = tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class InversePair:
    forward_path: str
    inverse_path: str
    forward_score: float
    inverse_score: float
    dominance: float
    annihilation_score: float


def generate_inverse_path(path: Sequence[Any]) -> list[Any]:
    """Return the exact reverse ordering of a path."""

    return list(reversed(path))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _clip_unit_interval(value: float) -> float:
    return float(np.clip(np.float64(value), np.float64(0.0), np.float64(1.0)))


def evaluate_state_coherence(result: dict[str, Any]) -> float:
    """Map fit quality into a bounded coherence score."""

    return _clip_unit_interval(float(result["r2"]))


def evaluate_state_stability(result: dict[str, Any]) -> float:
    """Map stability diagnostics into a bounded stability score."""

    stable_component = 1.0 if _as_bool(result["scaling_window_stable"]) else 0.0
    shift_component = 1.0 / (1.0 + max(float(result["alpha_window_shift"]), 0.0))
    validity_component = 1.0 if str(result["status"]).startswith("VALID") else 0.25 if str(result["status"]) == "WEAK_FIT" else 0.0
    return _clip_unit_interval(0.45 * stable_component + 0.35 * shift_component + 0.20 * validity_component)


def evaluate_state_selection_score(result: dict[str, Any]) -> float:
    """Map existing atlas diagnostics into a bounded path-selection score."""

    validity_component = 1.0 if str(result["status"]).startswith("VALID") else 0.25 if str(result["status"]) == "WEAK_FIT" else 0.0
    coherence_component = evaluate_state_coherence(result)
    stability_component = evaluate_state_stability(result)
    total_error = max(float(result["unitarity_error"]) + float(result["hermitian_error"]), 0.0)
    error_component = 1.0 / (1.0 + min(total_error * 1.0e12, 1.0e6))
    return _clip_unit_interval(
        0.40 * validity_component
        + 0.30 * coherence_component
        + 0.20 * stability_component
        + 0.10 * error_component
    )


def evaluate_path_metrics(path: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Evaluate coherence, stability, and directional selection for a path."""

    if not path:
        return {"coherence": 0.0, "stability": 0.0, "selection_score": 0.0, "score": 0.0}

    coherence_values = np.array([evaluate_state_coherence(node) for node in path], dtype=np.float64)
    stability_values = np.array([evaluate_state_stability(node) for node in path], dtype=np.float64)
    selection_values = np.array([evaluate_state_selection_score(node) for node in path], dtype=np.float64)

    coherence = float(np.mean(coherence_values, dtype=np.float64))
    stability = float(np.mean(stability_values, dtype=np.float64))

    if selection_values.size < 2:
        directional_component = 0.5
    else:
        end_to_start_delta = float(selection_values[-1] - selection_values[0])
        directional_component = _clip_unit_interval(0.5 + end_to_start_delta)

    selection_score = _clip_unit_interval(0.45 * float(np.mean(selection_values, dtype=np.float64)) + 0.55 * directional_component)
    score = _clip_unit_interval(0.20 * coherence + 0.20 * stability + 0.60 * selection_score)

    return {
        "coherence": coherence,
        "stability": stability,
        "selection_score": selection_score,
        "score": score,
    }


def _build_slice_key(result: dict[str, Any], slice_keys: Sequence[str]) -> tuple[float, ...]:
    params = result["parameters"]
    return tuple(float(params[key]) for key in slice_keys)


def generate_transition_paths(
    results: Sequence[dict[str, Any]],
    path_axis: str = "W",
    slice_keys: Sequence[str] = ("bias", "eta", "gamma"),
) -> list[TransitionPath]:
    """Generate ordered transition paths from sweep results."""

    grouped: dict[tuple[float, ...], list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(_build_slice_key(result, slice_keys), []).append(result)

    paths: list[TransitionPath] = []
    for slice_key in sorted(grouped):
        ordered = tuple(sorted(grouped[slice_key], key=lambda item: float(item["parameters"][path_axis])))
        if len(ordered) >= 2:
            paths.append(ordered)
    return paths


def _format_path(path: Sequence[dict[str, Any]], path_axis: str) -> str:
    labels = []
    for node in path:
        params = node["parameters"]
        labels.append(
            f"{path_axis}={float(params[path_axis]):.2f}:{node['status']}:alpha={float(node['alpha']):.3f}"
        )
    return " -> ".join(labels)


def build_inverse_pairs(
    results: Sequence[dict[str, Any]],
    path_axis: str = "W",
    slice_keys: Sequence[str] = ("bias", "eta", "gamma"),
) -> list[InversePair]:
    """Build forward/inverse path pairs and compare their directional scores."""

    pairs: list[InversePair] = []
    for path in generate_transition_paths(results=results, path_axis=path_axis, slice_keys=slice_keys):
        inverse_path = tuple(generate_inverse_path(path))
        forward_metrics = evaluate_path_metrics(path)
        inverse_metrics = evaluate_path_metrics(inverse_path)
        dominance = float(forward_metrics["score"] - inverse_metrics["score"])
        annihilation_score = float(1.0 - abs(dominance))
        pairs.append(
            InversePair(
                forward_path=_format_path(path, path_axis=path_axis),
                inverse_path=_format_path(inverse_path, path_axis=path_axis),
                forward_score=float(forward_metrics["score"]),
                inverse_score=float(inverse_metrics["score"]),
                dominance=dominance,
                annihilation_score=annihilation_score,
            )
        )
    return pairs


def extract_emergent_laws(
    pairs: Sequence[InversePair],
    dominance_threshold: float = 0.25,
    annihilation_threshold: float = 0.90,
) -> dict[str, Any]:
    """Report only regularities with sufficient directional dominance."""

    symmetric_pairs = [pair for pair in pairs if abs(pair.dominance) <= dominance_threshold]
    annihilated_pairs = [pair for pair in pairs if pair.annihilation_score >= annihilation_threshold]
    dominant_pairs = [pair for pair in pairs if abs(pair.dominance) > dominance_threshold]

    if dominant_pairs:
        mean_dominance = float(np.mean([pair.dominance for pair in dominant_pairs], dtype=np.float64))
    else:
        mean_dominance = 0.0

    if mean_dominance > 0.0:
        dominant_direction = "forward"
    elif mean_dominance < 0.0:
        dominant_direction = "inverse"
    else:
        dominant_direction = "symmetric"

    reported_laws = []
    for pair in dominant_pairs:
        direction = "forward" if pair.dominance > 0.0 else "inverse"
        reported_laws.append(
            {
                "direction": direction,
                "dominance": pair.dominance,
                "annihilation_score": pair.annihilation_score,
                "forward_path": pair.forward_path,
                "inverse_path": pair.inverse_path,
                "summary": (
                    f"{direction} directional dominance observed with dominance={pair.dominance:.3f} "
                    f"and annihilation_score={pair.annihilation_score:.3f}"
                ),
            }
        )

    average_annihilation_score = float(
        np.mean([pair.annihilation_score for pair in pairs], dtype=np.float64)
    ) if pairs else 0.0

    return {
        "symmetrical_paths": len(symmetric_pairs),
        "annihilated_paths": len(annihilated_pairs),
        "dominant_paths": len(dominant_pairs),
        "dominant_direction": dominant_direction,
        "average_annihilation_score": average_annihilation_score,
        "reported_laws": reported_laws,
        "dominance_threshold": float(dominance_threshold),
        "annihilation_threshold": float(annihilation_threshold),
    }


def write_inverse_analysis_csv(path: Path, pairs: Sequence[InversePair]) -> None:
    """Persist inverse pair metrics to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "forward_path",
                "inverse_path",
                "forward_score",
                "inverse_score",
                "dominance",
                "annihilation_score",
            ],
        )
        writer.writeheader()
        for pair in pairs:
            writer.writerow(
                {
                    "forward_path": pair.forward_path,
                    "inverse_path": pair.inverse_path,
                    "forward_score": f"{pair.forward_score:.6f}",
                    "inverse_score": f"{pair.inverse_score:.6f}",
                    "dominance": f"{pair.dominance:.6f}",
                    "annihilation_score": f"{pair.annihilation_score:.6f}",
                }
            )


def save_inverse_transition_map(path: Path, pairs: Sequence[InversePair]) -> None:
    """Save a compact visual summary of forward/inverse comparison scores."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    if not pairs:
        ax.text(0.5, 0.5, "No inverse transition pairs generated", ha="center", va="center")
        ax.set_axis_off()
    else:
        x = np.arange(len(pairs), dtype=np.float64)
        ax.plot(x, [pair.forward_score for pair in pairs], marker="o", linewidth=1.5, label="forward score")
        ax.plot(x, [pair.inverse_score for pair in pairs], marker="s", linewidth=1.5, label="inverse score")
        ax.plot(x, [pair.annihilation_score for pair in pairs], marker="^", linewidth=1.5, label="annihilation score")
        ax.set_xlabel("Path index")
        ax.set_ylabel("Score")
        ax.set_title("Inverse transition comparison map")
        ax.set_xticks(x)
        ax.set_xticklabels([str(index + 1) for index in range(len(pairs))], rotation=45, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_inverse_transition_analysis(
    config: dict[str, Any],
    csv_output_path: Path,
    plot_output_path: Path,
) -> dict[str, Any]:
    """Execute the inverse transition experiment from a fresh sweep."""

    inverse_cfg = config.get("inverse_analysis", {})
    path_axis = str(inverse_cfg.get("path_axis", "W"))
    slice_keys = tuple(inverse_cfg.get("slice_keys", ["bias", "eta", "gamma"]))
    dominance_threshold = float(inverse_cfg.get("dominance_threshold", 0.25))
    annihilation_threshold = float(inverse_cfg.get("annihilation_threshold", 0.90))

    results = run_parameter_sweep(config=config)
    pairs = build_inverse_pairs(results=results, path_axis=path_axis, slice_keys=slice_keys)
    summary = extract_emergent_laws(
        pairs=pairs,
        dominance_threshold=dominance_threshold,
        annihilation_threshold=annihilation_threshold,
    )

    write_inverse_analysis_csv(csv_output_path, pairs)
    save_inverse_transition_map(plot_output_path, pairs)

    return {
        "results": results,
        "pairs": pairs,
        "summary": summary,
        "path_axis": path_axis,
        "slice_keys": slice_keys,
        "csv_output_path": csv_output_path,
        "plot_output_path": plot_output_path,
    }

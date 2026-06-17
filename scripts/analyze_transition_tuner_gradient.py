from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.transition_tuner import (
    fixed_grid_from_dict,
    noise_operators_from_profiles,
    random_transition_search,
    transition_tuner_config_from_dict,
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "hardware" / "transition_tuner_etch_gradient.yaml"
DEFAULT_PLOT = PROJECT_ROOT / "results" / "renders" / "transition_tuner_noise_overlap_comparison.png"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def render_noise_overlap_plot(output_path: Path, *, baseline: float, best: float) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["Baseline noise overlap", "Best noise overlap"]
    values = [baseline, best]
    colors = ["#6c8ebf", "#2a9d8f"]
    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.set_ylabel("Noise overlap")
    ax.set_title("Transition tuner noise-overlap comparison")
    ax.set_ylim(0.0, max(values) * 1.25 if max(values) > 0 else 1.0)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.01 * max(1.0, max(values)),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def analyze_transition_tuner(config_path: Path, plot_path: Path) -> int:
    payload = load_yaml(config_path)
    tuner_block = dict(payload.get("transition_tuner", {}))
    if not tuner_block:
        raise ValueError("Config must contain a transition_tuner block.")

    grid = fixed_grid_from_dict(dict(tuner_block.get("grid", {})))
    noise_profiles = list(dict(tuner_block.get("noise", {})).get("profiles", []))
    if not noise_profiles:
        raise ValueError("Config must contain transition_tuner.noise.profiles.")
    search = transition_tuner_config_from_dict(dict(tuner_block.get("search", {})))
    noise_ops = noise_operators_from_profiles(noise_profiles)
    result = random_transition_search(grid, noise_ops, search)

    threshold_ok = (
        result.best_suppression_score >= 0.65 and result.best_transport_efficiency >= 0.7
    )
    plot = render_noise_overlap_plot(
        plot_path,
        baseline=result.baseline_noise_overlap,
        best=result.best_noise_overlap,
    )

    print("Transition tuner gradient-noise analysis")
    print(f"config = {config_path}")
    print("noise_profile = non_homogeneous_etch_gradient")
    print("control_family = onsite_shifts_only")
    print(f"baseline_transport_efficiency = {result.baseline_transport_efficiency:.6f}")
    print(f"best_transport_efficiency = {result.best_transport_efficiency:.6f}")
    print(f"baseline_noise_overlap = {result.baseline_noise_overlap:.6f}")
    print(f"best_noise_overlap = {result.best_noise_overlap:.6f}")
    print(f"baseline_suppression_score = {result.baseline_suppression_score:.6f}")
    print(f"best_suppression_score = {result.best_suppression_score:.6f}")
    print(f"baseline_objective = {result.baseline_objective:.6f}")
    print(f"best_objective = {result.best_objective:.6f}")
    print(f"threshold_target = suppression_score>=0.65 and transport_efficiency>=0.7")
    print(f"threshold_passed = {threshold_ok}")
    print(f"plot_artifact = {plot}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze transition-tuner behavior under etch-gradient noise.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to transition tuner YAML config")
    parser.add_argument("--out", default=str(DEFAULT_PLOT), help="Output PNG path for the overlap comparison plot")
    args = parser.parse_args()
    return analyze_transition_tuner(Path(args.config).expanduser().resolve(), Path(args.out).expanduser().resolve())


if __name__ == "__main__":
    raise SystemExit(main())

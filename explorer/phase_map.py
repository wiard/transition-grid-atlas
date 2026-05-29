"""Phase map generation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def build_phase_matrix(
    results: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    metric_key: str,
    valid_only: bool = True,
) -> tuple[np.ndarray, list[float], list[float]]:
    """Collapse high-dimensional sweep results into a 2D matrix."""

    filtered = results
    if valid_only:
        filtered = [result for result in results if result["status"].startswith("VALID")]

    x_values = sorted({float(result["parameters"][x_key]) for result in filtered or results})
    y_values = sorted({float(result["parameters"][y_key]) for result in filtered or results})
    matrix = np.full((len(y_values), len(x_values)), np.nan, dtype=np.float64)

    for y_index, y_value in enumerate(y_values):
        for x_index, x_value in enumerate(x_values):
            cell_values = [
                float(result[metric_key])
                for result in filtered
                if float(result["parameters"][x_key]) == x_value
                and float(result["parameters"][y_key]) == y_value
            ]
            if cell_values:
                matrix[y_index, x_index] = float(np.mean(np.array(cell_values, dtype=np.float64)))

    return matrix, x_values, y_values


def save_phase_map_plot(
    matrix: np.ndarray,
    x_values: list[float],
    y_values: list[float],
    x_label: str,
    y_label: str,
    title: str,
    colorbar_label: str,
    output_path: str | Path,
) -> None:
    """Save a clean non-interactive phase map plot."""

    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
    )
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels([f"{value:.2f}" for value in x_values], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_yticklabels([f"{value:.2f}" for value in y_values])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

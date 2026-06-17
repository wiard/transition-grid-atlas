"""Interpretable low-dimensional control bases for the transition motor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hardware.transition_tuner import FixedGrid


def _normalize_edge(edge: tuple[int, int]) -> tuple[int, int]:
    i, j = int(edge[0]), int(edge[1])
    return (min(i, j), max(i, j))


@dataclass(frozen=True)
class ControlBasis:
    site_basis: dict[str, np.ndarray]
    edge_basis: dict[str, dict[tuple[int, int], float]]


def normalized_vector(v: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    vector = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= eps:
        return np.zeros_like(vector, dtype=np.float64)
    return vector / norm


def _window_vector(n_sites: int, center_indices: list[int], *, width: float = 1.5) -> np.ndarray:
    x = np.arange(n_sites, dtype=np.float64)
    window = np.zeros(n_sites, dtype=np.float64)
    for center in center_indices:
        window += np.exp(-((x - float(center)) ** 2) / (2.0 * width**2))
    return normalized_vector(window)


def make_site_basis(grid: FixedGrid) -> dict[str, np.ndarray]:
    n_sites = int(grid.n_sites)
    x = np.linspace(-1.0, 1.0, n_sites, dtype=np.float64)
    center_profile = 1.0 - x**2
    boundary_profile = np.abs(x)

    site_basis = {
        "uniform": normalized_vector(np.ones(n_sites, dtype=np.float64)),
        "linear_gradient": normalized_vector(x),
        "input_window": _window_vector(n_sites, [grid.input_index]),
        "target_window": _window_vector(n_sites, list(grid.target_indices)),
        "center_bowl": normalized_vector(center_profile),
        "boundary_suppression": normalized_vector(boundary_profile),
    }
    return site_basis


def make_edge_basis(grid: FixedGrid) -> dict[str, dict[tuple[int, int], float]]:
    n_sites = max(1, int(grid.n_sites) - 1)
    target_center = float(np.mean(grid.target_indices))
    overall_center = 0.5 * float(grid.n_sites - 1)
    normalized_edges = [_normalize_edge(edge) for edge in grid.edges]

    def midpoint(edge: tuple[int, int]) -> float:
        return 0.5 * float(edge[0] + edge[1])

    edge_basis: dict[str, dict[tuple[int, int], float]] = {
        "uniform_edges": {edge: 1.0 for edge in normalized_edges},
        "input_to_target_gradient": {},
        "target_corridor": {},
        "boundary_edges": {},
        "center_edges": {},
    }

    input_center = float(grid.input_index)
    corridor_center = 0.5 * (input_center + target_center)
    corridor_width = max(1.0, abs(target_center - input_center) / 3.0)

    for edge in normalized_edges:
        mid = midpoint(edge)
        normalized_mid = (mid - input_center) / max(1.0, target_center - input_center if target_center != input_center else n_sites)
        distance_to_boundary = min(mid, float(grid.n_sites - 1) - mid) / max(1.0, overall_center)
        distance_to_center = abs(mid - overall_center) / max(1.0, overall_center)

        edge_basis["input_to_target_gradient"][edge] = float(np.clip(normalized_mid, -1.0, 1.0))
        edge_basis["target_corridor"][edge] = float(np.exp(-((mid - corridor_center) ** 2) / (2.0 * corridor_width**2)))
        edge_basis["boundary_edges"][edge] = float(np.clip(1.0 - distance_to_boundary, 0.0, 1.0))
        edge_basis["center_edges"][edge] = float(np.clip(1.0 - distance_to_center, 0.0, 1.0))

    return edge_basis


def make_control_basis(grid: FixedGrid) -> ControlBasis:
    return ControlBasis(
        site_basis=make_site_basis(grid),
        edge_basis=make_edge_basis(grid),
    )

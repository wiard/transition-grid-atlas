"""Parameter sweep routines for Transition Grid Atlas."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

from engine import run_transport_simulation


def run_parameter_sweep(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Run a deterministic Cartesian sweep over W, bias, eta, and gamma."""

    sweep_cfg = config["sweep"]
    seed = int(config["parameters"]["seed"])
    results: list[dict[str, Any]] = []

    for W, bias, eta, gamma in product(
        sweep_cfg["W_values"],
        sweep_cfg["bias_values"],
        sweep_cfg["eta_values"],
        sweep_cfg["gamma_values"],
    ):
        result = run_transport_simulation(
            config=config,
            parameters={
                "W": float(W),
                "bias": float(bias),
                "eta": float(eta),
                "gamma": float(gamma),
                "seed": seed,
            },
        )
        results.append(result)

    return results


def detect_phase_boundary_zones(config: dict[str, Any], results: list[dict[str, Any]]) -> list[dict[str, float | str]]:
    """Detect sharp alpha cliffs that resemble ballistic-to-localized boundaries.

    A boundary zone is recorded when adjacent W-samples at fixed bias/eta/gamma
    show a sharp drop from ballistic-like alpha to localized-like alpha.
    """

    sweep_cfg = config["sweep"]
    detect_cfg = sweep_cfg.get("boundary_detection", {})
    high_alpha = np.float64(detect_cfg.get("high_alpha_threshold", 0.8))
    low_alpha = np.float64(detect_cfg.get("low_alpha_threshold", 0.2))

    grouped: dict[tuple[np.float64, np.float64, np.float64], list[dict[str, Any]]] = {}
    for result in results:
        params = result["parameters"]
        key = (
            np.float64(params["bias"]),
            np.float64(params["eta"]),
            np.float64(params["gamma"]),
        )
        grouped.setdefault(key, []).append(result)

    zones: list[dict[str, float | str]] = []
    for (bias, eta, gamma), group_results in grouped.items():
        ordered = sorted(group_results, key=lambda item: np.float64(item["parameters"]["W"]))
        for left_index, left in enumerate(ordered[:-1]):
            left_alpha = np.float64(left["alpha"])
            if left_alpha <= high_alpha:
                continue

            right_index = None
            for candidate_index in range(left_index + 1, len(ordered)):
                if np.float64(ordered[candidate_index]["alpha"]) < low_alpha:
                    right_index = candidate_index
                    break

            if right_index is None:
                continue

            right = ordered[right_index]
            left_w = np.float64(left["parameters"]["W"])
            right_w = np.float64(right["parameters"]["W"])
            right_alpha = np.float64(right["alpha"])

            steepest_gradient = np.float64(0.0)
            steepest_left_w = left_w
            steepest_right_w = right_w
            for prev_result, next_result in zip(ordered[left_index:right_index], ordered[left_index + 1 : right_index + 1]):
                prev_w = np.float64(prev_result["parameters"]["W"])
                next_w = np.float64(next_result["parameters"]["W"])
                prev_alpha = np.float64(prev_result["alpha"])
                next_alpha = np.float64(next_result["alpha"])
                delta_w = max(next_w - prev_w, np.float64(1.0e-12))
                local_gradient = (prev_alpha - next_alpha) / delta_w
                if local_gradient > steepest_gradient:
                    steepest_gradient = local_gradient
                    steepest_left_w = prev_w
                    steepest_right_w = next_w

            if steepest_gradient > 0.0:
                zones.append(
                    {
                        "status": "PHASE_BOUNDARY_ZONE",
                        "bias": float(bias),
                        "eta": float(eta),
                        "gamma": float(gamma),
                        "left_W": float(left_w),
                        "right_W": float(right_w),
                        "midpoint_W": float(np.float64(0.5) * (left_w + right_w)),
                        "left_alpha": float(left_alpha),
                        "right_alpha": float(right_alpha),
                        "gradient_magnitude": float(steepest_gradient),
                        "local_gradient_left_W": float(steepest_left_w),
                        "local_gradient_right_W": float(steepest_right_w),
                        "left_status": str(left["status"]),
                        "right_status": str(right["status"]),
                    }
                )
                break

    zones.sort(key=lambda item: (-float(item["gradient_magnitude"]), float(item["midpoint_W"])))
    return zones

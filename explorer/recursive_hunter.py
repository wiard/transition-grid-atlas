"""Recursive search for target transport exponents."""

from __future__ import annotations

from typing import Any

import numpy as np

from engine import run_transport_simulation


def _score_result(
    result: dict[str, Any],
    target_alpha: float,
    boundary_anchor: dict[str, float | str] | None = None,
) -> float:
    alpha_distance = abs(result["alpha"] - target_alpha)
    penalty = 0.0
    if result["status"] == "INVALID_NONHERMITIAN":
        penalty += 1_000.0
    if result["status"] == "INVALID_NONUNITARY":
        penalty += 500.0
    if result["status"] == "WEAK_FIT":
        penalty += 100.0
    if result["r2"] < 0.95:
        penalty += 250.0 * (0.95 - result["r2"])
    if not result["scaling_window_stable"]:
        penalty += 150.0
    if boundary_anchor is not None:
        anchor_w = float(boundary_anchor.get("local_midpoint_W", boundary_anchor["midpoint_W"]))
        if result["status"] == "VALID_BALLISTIC":
            penalty += 25.0
        if result["status"] == "VALID_LOCALIZED":
            penalty += 25.0
        if result["status"] == "VALID_DIFFUSIVE_CANDIDATE":
            penalty -= 50.0
        params = result["parameters"]
        penalty += 1.50 * abs(float(params["W"]) - anchor_w)
        penalty += 10.0 * abs(float(params["bias"]) - float(boundary_anchor["bias"]))
        penalty += 2.0 * abs(float(params["eta"]) - float(boundary_anchor["eta"]))
        penalty += 4.0 * abs(float(params["gamma"]) - float(boundary_anchor["gamma"]))
        penalty += 20.0 * alpha_distance
    return float(alpha_distance + penalty)


def normalize_boundary_zone(zone: dict[str, float | str]) -> dict[str, float | str]:
    """Accept both raw sweep zones and flattened ledger rows."""

    if "gradient_magnitude" in zone:
        return zone
    left_w = np.float64(zone["boundary_left_W"])
    right_w = np.float64(zone["boundary_right_W"])
    local_left_w = np.float64(zone.get("boundary_local_left_W", left_w))
    local_right_w = np.float64(zone.get("boundary_local_right_W", right_w))
    return {
        "status": zone.get("status", "PHASE_BOUNDARY_ZONE"),
        "bias": float(zone["bias"]),
        "eta": float(zone["eta"]),
        "gamma": float(zone["gamma"]),
        "left_W": float(left_w),
        "right_W": float(right_w),
        "midpoint_W": float(np.float64(0.5) * (left_w + right_w)),
        "local_left_W": float(local_left_w),
        "local_right_W": float(local_right_w),
        "local_midpoint_W": float(np.float64(0.5) * (local_left_w + local_right_w)),
        "left_alpha": float(zone["boundary_left_alpha"]),
        "right_alpha": float(zone["boundary_right_alpha"]),
        "gradient_magnitude": float(zone["boundary_gradient_magnitude"]),
        "left_status": zone.get("boundary_left_status", ""),
        "right_status": zone.get("boundary_right_status", ""),
    }


def select_boundary_anchor(
    boundary_zones: list[dict[str, float | str]],
    target_alpha: float,
) -> dict[str, float | str] | None:
    """Choose the most informative boundary zone for recursive focus."""

    if not boundary_zones:
        return None
    normalized = [normalize_boundary_zone(zone) for zone in boundary_zones]
    ranked = sorted(
        normalized,
        key=lambda zone: (
            -float(zone["gradient_magnitude"]),
            abs(np.float64(0.5) * (np.float64(zone["left_alpha"]) + np.float64(zone["right_alpha"])) - target_alpha),
            float(zone["midpoint_W"]),
        ),
    )
    return ranked[0]


def run_recursive_hunter(
    config: dict[str, Any],
    boundary_zones: list[dict[str, float | str]] | None = None,
) -> dict[str, Any]:
    """Zoom into regions that resemble diffusive scaling."""

    hunter_cfg = config["hunter"]
    seed = int(config["parameters"]["seed"])
    rng = np.random.default_rng(seed)
    target_alpha = float(hunter_cfg["target_alpha"])
    rounds = int(hunter_cfg["rounds"])
    samples_per_round = int(hunter_cfg["samples_per_round"])
    shrink_factor = float(hunter_cfg["shrink_factor"])

    center = {key: float(value) for key, value in hunter_cfg["search_center"].items()}
    span = {key: float(value) for key, value in hunter_cfg["search_span"].items()}
    boundary_anchor = select_boundary_anchor(boundary_zones or [], target_alpha=target_alpha)

    if boundary_anchor is not None:
        center.update(
            {
                "W": float(boundary_anchor.get("local_midpoint_W", boundary_anchor["midpoint_W"])),
                "bias": float(boundary_anchor["bias"]),
                "eta": float(boundary_anchor["eta"]),
                "gamma": float(boundary_anchor["gamma"]),
            }
        )
        span["W"] = max(
            np.float64(0.5),
            min(
                np.float64(span["W"]),
                2.0 * (np.float64(boundary_anchor.get("local_right_W", boundary_anchor["right_W"])) - np.float64(boundary_anchor.get("local_left_W", boundary_anchor["left_W"]))),
            ),
        )
        span["bias"] = min(np.float64(span["bias"]), np.float64(0.08))
        span["eta"] = min(np.float64(span["eta"]), np.float64(0.35))
        span["gamma"] = min(np.float64(span["gamma"]), np.float64(0.08))

    all_results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None

    for round_index in range(rounds):
        round_results = []
        for _ in range(samples_per_round):
            sampled_parameters = {}
            for key in ("W", "bias", "eta", "gamma"):
                if span[key] == 0.0:
                    sampled_parameters[key] = center[key]
                else:
                    low = center[key] - 0.5 * span[key]
                    high = center[key] + 0.5 * span[key]
                    sampled_parameters[key] = float(rng.uniform(low, high))
                if key in {"W", "eta", "gamma"}:
                    sampled_parameters[key] = max(0.0, sampled_parameters[key])
            sampled_parameters["seed"] = seed

            result = run_transport_simulation(config=config, parameters=sampled_parameters)
            result["round"] = round_index + 1
            result["score"] = _score_result(result, target_alpha=target_alpha, boundary_anchor=boundary_anchor)
            round_results.append(result)
            all_results.append(result)

        round_best = min(round_results, key=lambda item: item["score"])
        if best_result is None or round_best["score"] < best_result["score"]:
            best_result = round_best

        for key in ("W", "bias", "eta", "gamma"):
            center[key] = float(round_best["parameters"][key])
            span[key] *= shrink_factor

    return {
        "best_result": best_result,
        "all_results": all_results,
        "boundary_anchor": boundary_anchor,
    }

"""Objective modes and normalization helpers for the KTA transition motor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


ObjectiveModeName = Literal["raw", "normalized", "calibrated"]


@dataclass(frozen=True)
class ObjectiveNormalizationScale:
    transport_scale: float
    noise_action_scale: float
    leakage_scale: float
    control_cost_scale: float

    def __post_init__(self) -> None:
        values = (
            float(self.transport_scale),
            float(self.noise_action_scale),
            float(self.leakage_scale),
            float(self.control_cost_scale),
        )
        if min(values) <= 0.0:
            raise ValueError("normalization scales must be positive")

    @property
    def transport(self) -> float:
        return float(self.transport_scale)

    @property
    def noise_action(self) -> float:
        return float(self.noise_action_scale)

    @property
    def leakage(self) -> float:
        return float(self.leakage_scale)

    @property
    def control_cost(self) -> float:
        return float(self.control_cost_scale)


@dataclass(frozen=True)
class ObjectiveMode:
    name: str
    mode: ObjectiveModeName
    transport_weight: float
    noise_action_weight: float
    leakage_weight: float
    control_cost_weight: float
    normalization: ObjectiveNormalizationScale | None
    description: str

    def __post_init__(self) -> None:
        validate_objective_mode(self)

    @property
    def transport(self) -> float:
        return float(self.transport_weight)

    @property
    def noise_action(self) -> float:
        return float(self.noise_action_weight)

    @property
    def leakage(self) -> float:
        return float(self.leakage_weight)

    @property
    def control_cost(self) -> float:
        return float(self.control_cost_weight)

    def weights_dict(self) -> dict[str, float]:
        return {
            "transport": float(self.transport_weight),
            "noise_action": float(self.noise_action_weight),
            "leakage": float(self.leakage_weight),
            "control_cost": float(self.control_cost_weight),
        }


@dataclass(frozen=True)
class ObjectiveModeRegistry:
    modes: tuple[ObjectiveMode, ...]
    default_mode: str

    def __post_init__(self) -> None:
        names = [mode.name for mode in self.modes]
        if len(names) != len(set(names)):
            raise ValueError("duplicate objective mode names are not allowed")
        if self.default_mode not in names:
            raise ValueError("default_mode must reference a known objective mode")

    def names(self) -> list[str]:
        return [mode.name for mode in self.modes]

    def get(self, name: str) -> ObjectiveMode:
        for mode in self.modes:
            if mode.name == name:
                return mode
        raise ValueError(f"unknown objective mode: {name}")


def validate_objective_mode(mode: ObjectiveMode) -> None:
    if not str(mode.name).strip():
        raise ValueError("objective mode name cannot be empty")
    if mode.mode not in {"raw", "normalized", "calibrated"}:
        raise ValueError("objective mode must be 'raw', 'normalized' or 'calibrated'")
    weights = (
        float(mode.transport_weight),
        float(mode.noise_action_weight),
        float(mode.leakage_weight),
        float(mode.control_cost_weight),
    )
    if min(weights) < 0.0:
        raise ValueError("objective mode weights must be non-negative")
    if mode.transport_weight <= 0.0 and mode.noise_action_weight <= 0.0 and mode.leakage_weight <= 0.0:
        raise ValueError("at least one performance weight must be positive")
    if mode.mode == "raw":
        return
    if mode.normalization is None:
        raise ValueError("normalized and calibrated objective modes require normalization scales")


def raw_motor_objective_from_mode(
    *,
    transport_efficiency: float,
    noise_action_on_info: float,
    noise_leakage: float,
    control_cost: float,
    mode: ObjectiveMode,
) -> float:
    return float(
        mode.transport_weight * float(transport_efficiency)
        - mode.noise_action_weight * float(noise_action_on_info)
        - mode.leakage_weight * float(noise_leakage)
        - mode.control_cost_weight * float(control_cost)
    )


def normalized_motor_objective_from_mode(
    *,
    transport_efficiency: float,
    noise_action_on_info: float,
    noise_leakage: float,
    control_cost: float,
    mode: ObjectiveMode,
) -> float:
    if mode.normalization is None:
        raise ValueError("normalized objective mode requires normalization scales")
    scales = mode.normalization
    return float(
        mode.transport_weight * float(transport_efficiency) / scales.transport_scale
        - mode.noise_action_weight * float(noise_action_on_info) / scales.noise_action_scale
        - mode.leakage_weight * float(noise_leakage) / scales.leakage_scale
        - mode.control_cost_weight * float(control_cost) / scales.control_cost_scale
    )


def evaluate_objective_mode(
    *,
    transport_efficiency: float,
    noise_action_on_info: float,
    noise_leakage: float,
    control_cost: float,
    mode: ObjectiveMode,
) -> float:
    if mode.mode == "raw":
        return raw_motor_objective_from_mode(
            transport_efficiency=transport_efficiency,
            noise_action_on_info=noise_action_on_info,
            noise_leakage=noise_leakage,
            control_cost=control_cost,
            mode=mode,
        )
    if mode.mode in {"normalized", "calibrated"}:
        return normalized_motor_objective_from_mode(
            transport_efficiency=transport_efficiency,
            noise_action_on_info=noise_action_on_info,
            noise_leakage=noise_leakage,
            control_cost=control_cost,
            mode=mode,
        )
    raise ValueError(f"unsupported objective mode: {mode.mode}")


def estimate_normalization_scales_from_rows(
    rows: list[dict[str, float]],
    *,
    eps: float = 1e-12,
) -> ObjectiveNormalizationScale:
    required = {
        "detector_success_gain",
        "noise_action_reduction",
        "noise_leakage_reduction",
        "best_control_cost",
    }
    if not rows:
        raise ValueError("rows must be non-empty")
    missing = [name for name in required if any(name not in row for row in rows)]
    if missing:
        raise ValueError(f"missing normalization-scale columns: {sorted(set(missing))}")
    return ObjectiveNormalizationScale(
        transport_scale=float(np.mean([abs(float(row["detector_success_gain"])) for row in rows]) + eps),
        noise_action_scale=float(np.mean([abs(float(row["noise_action_reduction"])) for row in rows]) + eps),
        leakage_scale=float(np.mean([abs(float(row["noise_leakage_reduction"])) for row in rows]) + eps),
        control_cost_scale=float(np.mean([abs(float(row["best_control_cost"])) for row in rows]) + eps),
    )


def default_objective_modes(
    scales: ObjectiveNormalizationScale | None = None,
) -> list[ObjectiveMode]:
    safe_scales = scales or ObjectiveNormalizationScale(
        transport_scale=1.0,
        noise_action_scale=1.0,
        leakage_scale=1.0,
        control_cost_scale=1.0,
    )
    return [
        ObjectiveMode(
            name="raw_detector_max",
            mode="raw",
            transport_weight=1.5,
            noise_action_weight=0.25,
            leakage_weight=0.10,
            control_cost_weight=0.01,
            normalization=None,
            description="Legacy detector-focused raw objective.",
        ),
        ObjectiveMode(
            name="raw_balanced",
            mode="raw",
            transport_weight=1.0,
            noise_action_weight=0.5,
            leakage_weight=0.25,
            control_cost_weight=0.01,
            normalization=None,
            description="Legacy balanced raw objective.",
        ),
        ObjectiveMode(
            name="normalized_balanced",
            mode="normalized",
            transport_weight=1.0,
            noise_action_weight=1.0,
            leakage_weight=1.0,
            control_cost_weight=1.0,
            normalization=safe_scales,
            description="Equal-footing normalized objective across detector, noise, leakage and cost.",
        ),
        ObjectiveMode(
            name="normalized_noise",
            mode="normalized",
            transport_weight=1.0,
            noise_action_weight=2.0,
            leakage_weight=1.0,
            control_cost_weight=0.05,
            normalization=safe_scales,
            description="Normalized mode emphasizing noise-action reduction while preserving detector output.",
        ),
        ObjectiveMode(
            name="normalized_leakage_guard",
            mode="normalized",
            transport_weight=1.0,
            noise_action_weight=1.0,
            leakage_weight=2.0,
            control_cost_weight=0.05,
            normalization=safe_scales,
            description="Normalized leakage-guard mode for suppressing subspace escape.",
        ),
        ObjectiveMode(
            name="normalized_cost_sensitive",
            mode="normalized",
            transport_weight=1.0,
            noise_action_weight=1.0,
            leakage_weight=0.5,
            control_cost_weight=0.20,
            normalization=safe_scales,
            description="Normalized cost-sensitive mode that penalizes actuator effort more strongly.",
        ),
    ]


def _normalization_from_dict(data: dict[str, object] | None) -> ObjectiveNormalizationScale | None:
    if data is None:
        return None
    block = dict(data)
    return ObjectiveNormalizationScale(
        transport_scale=float(block["transport_scale"]),
        noise_action_scale=float(block["noise_action_scale"]),
        leakage_scale=float(block["leakage_scale"]),
        control_cost_scale=float(block["control_cost_scale"]),
    )


def _mode_from_dict(block: dict[str, Any]) -> ObjectiveMode:
    weights_block = dict(block.get("weights", {}))
    return ObjectiveMode(
        name=str(block["name"]),
        mode=str(block["mode"]),
        transport_weight=float(weights_block.get("transport", block.get("transport"))),
        noise_action_weight=float(weights_block.get("noise_action", block.get("noise_action"))),
        leakage_weight=float(weights_block.get("leakage", block.get("leakage"))),
        control_cost_weight=float(weights_block.get("control_cost", block.get("control_cost"))),
        normalization=_normalization_from_dict(block.get("normalization")),
        description=str(block.get("description", "")),
    )


def objective_mode_from_config(config: dict[str, Any]) -> ObjectiveMode | None:
    block = dict(config)
    if "objective_mode" not in block:
        return None
    return _mode_from_dict(dict(block["objective_mode"]))


def objective_mode_registry_from_dict(data: dict[str, object]) -> ObjectiveModeRegistry:
    registry_block = dict(data)
    return ObjectiveModeRegistry(
        modes=tuple(_mode_from_dict(dict(item)) for item in list(registry_block["modes"])),
        default_mode=str(registry_block["default_mode"]),
    )


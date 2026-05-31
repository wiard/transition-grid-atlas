"""Operating mode registry for the KTA transition motor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ObjectiveMode = Literal["raw", "normalized"]


@dataclass(frozen=True)
class OperatingMode:
    name: str
    objective_mode: ObjectiveMode
    transport: float
    noise_action: float
    leakage: float
    control_cost: float
    description: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("operating mode name cannot be empty")
        if self.objective_mode not in {"raw", "normalized"}:
            raise ValueError("objective_mode must be 'raw' or 'normalized'")
        if min(self.transport, self.noise_action, self.leakage, self.control_cost) < 0.0:
            raise ValueError("operating mode weights must be non-negative")
        if self.transport <= 0.0 and self.noise_action <= 0.0 and self.leakage <= 0.0:
            raise ValueError("at least one performance weight must be positive")

    def weights_dict(self) -> dict[str, float]:
        return {
            "transport": float(self.transport),
            "noise_action": float(self.noise_action),
            "leakage": float(self.leakage),
            "control_cost": float(self.control_cost),
        }


@dataclass(frozen=True)
class OperatingModeRegistry:
    modes: tuple[OperatingMode, ...]
    default_mode: str

    def __post_init__(self) -> None:
        names = [mode.name for mode in self.modes]
        if len(names) != len(set(names)):
            raise ValueError("duplicate operating mode names are not allowed")
        if self.default_mode not in names:
            raise ValueError("default_mode must reference a known operating mode")

    def names(self) -> list[str]:
        return [mode.name for mode in self.modes]

    def get(self, name: str) -> OperatingMode:
        for mode in self.modes:
            if mode.name == name:
                return mode
        raise ValueError(f"unknown operating mode: {name}")


def operating_mode_registry_from_dict(data: dict[str, object]) -> OperatingModeRegistry:
    registry_block = dict(data)
    return OperatingModeRegistry(
        modes=tuple(
            OperatingMode(
                name=str(item["name"]),
                objective_mode=str(item["objective_mode"]),
                transport=float(item["transport"]),
                noise_action=float(item["noise_action"]),
                leakage=float(item["leakage"]),
                control_cost=float(item["control_cost"]),
                description=str(item.get("description", "")),
            )
            for item in list(registry_block["modes"])
        ),
        default_mode=str(registry_block["default_mode"]),
    )

"""Control registry for the KTA transition motor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


KnobFamily = Literal[
    "coherent_coupling",
    "onsite_phase",
    "phase_noise",
    "objective",
]


@dataclass(frozen=True)
class ControlKnob:
    name: str
    family: KnobFamily
    symbol: str
    basis_name: str
    min_value: float
    max_value: float
    default: float
    units: str
    description: str
    hardware_meaning: str

    def __post_init__(self) -> None:
        if self.max_value <= self.min_value:
            raise ValueError("max_value must be greater than min_value")
        if not (self.min_value <= self.default <= self.max_value):
            raise ValueError("default must lie within the knob range")


@dataclass(frozen=True)
class KnobRegistry:
    knobs: tuple[ControlKnob, ...]

    def __post_init__(self) -> None:
        names = [knob.name for knob in self.knobs]
        if len(names) != len(set(names)):
            raise ValueError("duplicate knob names are not allowed")

    def defaults(self) -> dict[str, float]:
        return {knob.name: float(knob.default) for knob in self.knobs}

    def validate_theta(self, theta: dict[str, float]) -> None:
        allowed_names = {knob.name for knob in self.knobs}
        unknown = sorted(set(theta) - allowed_names)
        if unknown:
            raise ValueError(f"unknown knobs in theta: {unknown}")
        for knob in self.knobs:
            value = float(theta.get(knob.name, knob.default))
            if value < knob.min_value or value > knob.max_value:
                raise ValueError(f"theta value out of range for knob {knob.name}")

    def clip_theta(self, theta: dict[str, float]) -> dict[str, float]:
        allowed_names = {knob.name for knob in self.knobs}
        unknown = sorted(set(theta) - allowed_names)
        if unknown:
            raise ValueError(f"unknown knobs in theta: {unknown}")
        clipped = {}
        for knob in self.knobs:
            value = float(theta.get(knob.name, knob.default))
            clipped[knob.name] = min(knob.max_value, max(knob.min_value, value))
        return clipped

    def names(self) -> list[str]:
        return [knob.name for knob in self.knobs]


def knob_registry_from_dicts(items: list[dict[str, object]]) -> KnobRegistry:
    return KnobRegistry(
        knobs=tuple(
            ControlKnob(
                name=str(item["name"]),
                family=str(item["family"]),
                symbol=str(item["symbol"]),
                basis_name=str(item["basis_name"]),
                min_value=float(item["min_value"]),
                max_value=float(item["max_value"]),
                default=float(item["default"]),
                units=str(item["units"]),
                description=str(item["description"]),
                hardware_meaning=str(item["hardware_meaning"]),
            )
            for item in items
        )
    )

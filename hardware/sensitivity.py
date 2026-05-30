"""Finite-difference sensitivity atlas for the transition motor."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from hardware.control_knobs import KnobRegistry


@dataclass(frozen=True)
class SensitivityEntry:
    metric: str
    knob: str
    central_difference: float
    plus_value: float
    minus_value: float


def compute_sensitivity_matrix(
    metrics_fn,
    theta0: dict[str, float],
    registry: KnobRegistry,
    *,
    eps: float,
    metric_names: list[str],
) -> list[SensitivityEntry]:
    entries: list[SensitivityEntry] = []
    clipped = registry.clip_theta(theta0)

    for knob_name in registry.names():
        theta_plus = dict(clipped)
        theta_minus = dict(clipped)
        theta_plus[knob_name] += eps
        theta_minus[knob_name] -= eps
        theta_plus = registry.clip_theta(theta_plus)
        theta_minus = registry.clip_theta(theta_minus)

        plus_metrics = metrics_fn(theta_plus)
        minus_metrics = metrics_fn(theta_minus)
        delta = theta_plus[knob_name] - theta_minus[knob_name]

        for metric_name in metric_names:
            plus_value = float(getattr(plus_metrics, metric_name))
            minus_value = float(getattr(minus_metrics, metric_name))
            if abs(delta) <= 1.0e-15:
                central = 0.0
            else:
                central = (plus_value - minus_value) / delta
            entries.append(
                SensitivityEntry(
                    metric=metric_name,
                    knob=knob_name,
                    central_difference=float(central),
                    plus_value=plus_value,
                    minus_value=minus_value,
                )
            )
    return entries


def top_sensitivities(
    entries: list[SensitivityEntry],
    *,
    limit: int = 10,
) -> list[dict[str, object]]:
    sorted_entries = sorted(entries, key=lambda entry: abs(entry.central_difference), reverse=True)
    return [
        {
            "metric": entry.metric,
            "knob": entry.knob,
            "central_difference": entry.central_difference,
        }
        for entry in sorted_entries[:limit]
    ]


def write_sensitivity_csv(
    path: str | Path,
    entries: list[SensitivityEntry],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SensitivityEntry.__dataclass_fields__.keys()))
        writer.writeheader()
        for entry in entries:
            writer.writerow(asdict(entry))
    return output

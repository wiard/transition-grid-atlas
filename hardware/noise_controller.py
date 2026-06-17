"""Effective dephasing mappings for photonic control hardware."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class NoiseController:
    actuator_type: Literal["thermo_optic", "electro_optic", "stochastic_phase_modulator"]
    max_phase_rms_rad: float
    bandwidth_hz: float
    correlation_time_ps: float
    max_power_mw: float | None = None


def noise_controller_from_dict(data: dict[str, object]) -> NoiseController:
    return NoiseController(
        actuator_type=str(data["actuator_type"]),
        max_phase_rms_rad=float(data["max_phase_rms_rad"]),
        bandwidth_hz=float(data["bandwidth_hz"]),
        correlation_time_ps=float(data["correlation_time_ps"]),
        max_power_mw=None if data.get("max_power_mw") is None else float(data["max_power_mw"]),
    )


def effective_gamma_from_controller(controller: NoiseController) -> float:
    """Convert stochastic phase-control settings to an effective ``gamma``.

    This is a calibrated effective-model mapping. Thermo-optic and electro-optic
    phase shifters are not dephasing channels by themselves; they only induce an
    effective Lindblad-style dephasing rate when used in a stochastic,
    time-varying, or ensemble-averaged control mode.
    """

    if controller.max_phase_rms_rad < 0.0:
        raise ValueError("max_phase_rms_rad must be non-negative")
    if controller.bandwidth_hz < 0.0:
        raise ValueError("bandwidth_hz must be non-negative")
    if controller.correlation_time_ps < 0.0:
        raise ValueError("correlation_time_ps must be non-negative")
    if controller.max_power_mw is not None and controller.max_power_mw < 0.0:
        raise ValueError("max_power_mw must be non-negative when provided")

    actuator_factor = {
        "thermo_optic": 0.6,
        "electro_optic": 0.8,
        "stochastic_phase_modulator": 1.0,
    }[controller.actuator_type]
    power_factor = 1.0
    if controller.max_power_mw is not None:
        power_factor = min(1.0, controller.max_power_mw / 20.0)

    phase_variance = controller.max_phase_rms_rad**2
    bandwidth_correlation_scale = controller.bandwidth_hz * controller.correlation_time_ps * 1.0e-6
    gamma_eff = actuator_factor * power_factor * phase_variance * bandwidth_correlation_scale
    return float(max(0.0, gamma_eff))

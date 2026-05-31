"""Backward-compatible aliases for older transition-motor operating-mode imports."""

from __future__ import annotations

from hardware.objective_modes import (
    ObjectiveMode as OperatingMode,
    ObjectiveModeKind as ObjectiveMode,
    ObjectiveModeRegistry as OperatingModeRegistry,
    objective_mode_registry_from_dict as operating_mode_registry_from_dict,
)


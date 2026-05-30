"""Theory switchboard entry points for the Experimental Quantum & RTT Lab."""

from __future__ import annotations

from typing import Any

from engine.lab import LAB_RESULTS_FIELDS, LabConfig, parse_lab_config, run_lab_simulation


MODE_ALIASES = {
    "qm_free": "qm_free",
    "standard_qm": "qm_free",
    "anderson": "anderson",
    "lindblad": "lindblad",
    "rtt": "rtt",
}


def normalize_theory_mode(mode: str) -> str:
    """Normalize CLI/config aliases into canonical switchboard names."""

    normalized = str(mode).strip().lower()
    if normalized not in MODE_ALIASES:
        raise ValueError(f"Unsupported theory mode: {mode}")
    return MODE_ALIASES[normalized]


def build_lab_config(config: dict[str, Any], mode_override: str | None = None, gamma_override: float | None = None) -> LabConfig:
    """Parse and override the base lab config for CLI-driven runs."""

    parsed = parse_lab_config(config)
    if mode_override is not None:
        parsed.theory_mode = normalize_theory_mode(mode_override)
    else:
        parsed.theory_mode = normalize_theory_mode(parsed.theory_mode)
    if gamma_override is not None:
        parsed.gamma = float(gamma_override)
    return parsed


__all__ = [
    "LAB_RESULTS_FIELDS",
    "LabConfig",
    "build_lab_config",
    "normalize_theory_mode",
    "parse_lab_config",
    "run_lab_simulation",
]

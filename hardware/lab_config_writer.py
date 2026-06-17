from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_SIGMA = 4.0
DEFAULT_K0 = 0.5
DEFAULT_TOTAL_TIME = 60.0
DEFAULT_DT = 0.05
DEFAULT_EDGE_MARGIN = 15
DEFAULT_FIT_WINDOW_LATE = [0.6, 0.9]


def write_lab_config_from_hardware_mapping(
    output_path: str | Path,
    *,
    W_eff: float,
    gamma_eff: float,
    n_sites: int,
    target_indices: list[int],
    metadata: dict[str, Any],
) -> Path:
    """Write a lab override config from a photonic hardware mapping.

    The emitted file intentionally follows the existing alternate lab schema
    already supported by the KTA loader (`system`, `wavepacket`, `solver`,
    `audit`) so the hardware route does not introduce a new solver contract.
    """

    if W_eff < 0:
        raise ValueError("W_eff must be non-negative.")
    if gamma_eff < 0:
        raise ValueError("gamma_eff must be non-negative.")
    if n_sites <= 0:
        raise ValueError("n_sites must be positive.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "lab": {
            "mode": "lindblad",
        },
        "system": {
            "mode": "lindblad",
            "L": int(n_sites),
            "disorder_strength": float(W_eff),
            "gamma": float(gamma_eff),
        },
        "wavepacket": {
            "x0": float(n_sites / 2.0),
            "sigma0": float(DEFAULT_SIGMA),
            "k0": float(DEFAULT_K0),
        },
        "solver": {
            "T": float(DEFAULT_TOTAL_TIME),
            "dt": float(DEFAULT_DT),
            "method": "RK4",
        },
        "audit": {
            "edge_margin": int(DEFAULT_EDGE_MARGIN),
            "fit_window_late": list(DEFAULT_FIT_WINDOW_LATE),
        },
        "hardware_mapping": {
            "source": "photonic_wafer",
            "W_eff": float(W_eff),
            "gamma_eff": float(gamma_eff),
            "target_indices": [int(index) for index in target_indices],
            "calibration_status": "phenomenological",
            **metadata,
        },
    }

    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    return output

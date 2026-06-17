"""Sanity checks for photonic wafer hardware mappings."""

from __future__ import annotations

import numpy as np

from hardware.photonic_wafer import PhotonicWaferConfig


def validate_coupling_range(coupling_j: float) -> None:
    if not np.isfinite(coupling_j) or coupling_j < 0.0:
        raise ValueError("coupling_j must be finite and non-negative")


def validate_disorder_range(W_eff: float) -> None:
    if not np.isfinite(W_eff) or W_eff < 0.0:
        raise ValueError("W_eff must be finite and non-negative")


def validate_gamma_range(gamma_eff: float) -> None:
    if not np.isfinite(gamma_eff) or gamma_eff < 0.0:
        raise ValueError("gamma_eff must be finite and non-negative")


def validate_wafer_config(config: PhotonicWaferConfig) -> None:
    if config.layout not in {"1d_chain", "2d_square", "2d_triangular"}:
        raise ValueError(f"unsupported wafer layout: {config.layout}")
    if config.n_rows <= 0 or config.n_cols <= 0:
        raise ValueError("wafer lattice dimensions must be positive")
    if config.n_sites <= 0:
        raise ValueError("wafer must contain at least one site")
    if config.waveguide_pitch_um < 0.0:
        raise ValueError("waveguide_pitch_um must be non-negative")
    if config.length_mm < 0.0:
        raise ValueError("length_mm must be non-negative")
    validate_coupling_range(config.coupling_j)
    if config.fabrication_width_sigma_nm < 0.0:
        raise ValueError("fabrication_width_sigma_nm must be non-negative")
    if config.refractive_index_sigma < 0.0:
        raise ValueError("refractive_index_sigma must be non-negative")
    if config.propagation_loss_db_per_cm < 0.0:
        raise ValueError("propagation_loss_db_per_cm must be non-negative")

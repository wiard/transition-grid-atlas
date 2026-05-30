"""Photonic wafer configuration and phenomenological disorder mappings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class PhotonicWaferConfig:
    layout: Literal["1d_chain", "2d_square", "2d_triangular"]
    n_rows: int
    n_cols: int
    waveguide_pitch_um: float
    length_mm: float

    beta0: float
    coupling_j: float

    fabrication_width_sigma_nm: float
    refractive_index_sigma: float
    propagation_loss_db_per_cm: float = 0.0

    seed: int = 0

    @property
    def n_sites(self) -> int:
        return int(self.n_rows * self.n_cols)


def photonic_wafer_from_dict(data: dict[str, object]) -> PhotonicWaferConfig:
    return PhotonicWaferConfig(
        layout=str(data["layout"]),
        n_rows=int(data["n_rows"]),
        n_cols=int(data["n_cols"]),
        waveguide_pitch_um=float(data["waveguide_pitch_um"]),
        length_mm=float(data["length_mm"]),
        beta0=float(data["beta0"]),
        coupling_j=float(data["coupling_j"]),
        fabrication_width_sigma_nm=float(data["fabrication_width_sigma_nm"]),
        refractive_index_sigma=float(data["refractive_index_sigma"]),
        propagation_loss_db_per_cm=float(data.get("propagation_loss_db_per_cm", 0.0)),
        seed=int(data.get("seed", 0)),
    )


def _beta_sigma(config: PhotonicWaferConfig) -> float:
    width_term = float(config.fabrication_width_sigma_nm) * 1.0e-3
    index_term = abs(float(config.beta0)) * float(config.refractive_index_sigma)
    return float(np.sqrt(width_term**2 + index_term**2))


def disorder_strength_from_fabrication(config: PhotonicWaferConfig) -> float:
    """Map fabrication spread to an effective dimensionless disorder ``W``.

    This is a calibrated phenomenological mapping, not a universal law.
    The intent is to carry wafer-scale fabrication imperfection into the
    existing KTA parameter space without changing the solver.
    """

    coupling_ref = max(abs(float(config.coupling_j)), 1.0e-12)
    return float(_beta_sigma(config) / coupling_ref)


def onsite_disorder_vector(config: PhotonicWaferConfig) -> np.ndarray:
    """Generate effective onsite propagation constants ``beta_i``.

    The resulting vector is ``beta0 + delta_beta_i``, where ``delta_beta_i``
    is a random fabrication-induced perturbation with phenomenological spread.
    """

    rng = np.random.default_rng(config.seed)
    delta_beta = rng.normal(0.0, _beta_sigma(config), size=config.n_sites)
    return (float(config.beta0) + delta_beta).astype(np.float64, copy=False)

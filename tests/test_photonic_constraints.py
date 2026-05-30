from __future__ import annotations

import unittest

from hardware.constraints import (
    validate_coupling_range,
    validate_disorder_range,
    validate_gamma_range,
    validate_wafer_config,
)
from hardware.photonic_wafer import PhotonicWaferConfig


class PhotonicConstraintTests(unittest.TestCase):
    def build_valid_config(self) -> PhotonicWaferConfig:
        return PhotonicWaferConfig(
            layout="2d_square",
            n_rows=4,
            n_cols=5,
            waveguide_pitch_um=12.0,
            length_mm=10.0,
            beta0=1.0,
            coupling_j=0.8,
            fabrication_width_sigma_nm=4.0,
            refractive_index_sigma=1.0e-4,
            propagation_loss_db_per_cm=0.1,
            seed=1,
        )

    def test_invalid_wafer_config_fails_on_negative_pitch_length_and_coupling(self):
        with self.assertRaises(ValueError):
            validate_wafer_config(
                PhotonicWaferConfig(
                    layout="2d_square",
                    n_rows=4,
                    n_cols=5,
                    waveguide_pitch_um=-1.0,
                    length_mm=10.0,
                    beta0=1.0,
                    coupling_j=0.8,
                    fabrication_width_sigma_nm=4.0,
                    refractive_index_sigma=1.0e-4,
                )
            )
        with self.assertRaises(ValueError):
            validate_wafer_config(
                PhotonicWaferConfig(
                    layout="2d_square",
                    n_rows=4,
                    n_cols=5,
                    waveguide_pitch_um=12.0,
                    length_mm=-1.0,
                    beta0=1.0,
                    coupling_j=0.8,
                    fabrication_width_sigma_nm=4.0,
                    refractive_index_sigma=1.0e-4,
                )
            )
        with self.assertRaises(ValueError):
            validate_coupling_range(-0.1)

    def test_range_validators_accept_nonnegative_inputs(self):
        validate_wafer_config(self.build_valid_config())
        validate_disorder_range(0.0)
        validate_gamma_range(0.1)

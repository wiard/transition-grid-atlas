from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from hardware.constraints import validate_wafer_config
from hardware.photonic_wafer import (
    PhotonicWaferConfig,
    disorder_strength_from_fabrication,
    photonic_wafer_from_dict,
)
from run import run_hardware_map_mode


class PhotonicWaferTests(unittest.TestCase):
    def build_config(self, *, width_sigma_nm: float = 5.0) -> PhotonicWaferConfig:
        return PhotonicWaferConfig(
            layout="1d_chain",
            n_rows=1,
            n_cols=200,
            waveguide_pitch_um=10.0,
            length_mm=20.0,
            beta0=1.0,
            coupling_j=1.0,
            fabrication_width_sigma_nm=width_sigma_nm,
            refractive_index_sigma=0.0001,
            propagation_loss_db_per_cm=0.0,
            seed=42,
        )

    def test_photonic_wafer_config_accepts_valid_1d_chain(self):
        config = photonic_wafer_from_dict(
            {
                "layout": "1d_chain",
                "n_rows": 1,
                "n_cols": 200,
                "waveguide_pitch_um": 10.0,
                "length_mm": 20.0,
                "beta0": 1.0,
                "coupling_j": 1.0,
                "fabrication_width_sigma_nm": 5.0,
                "refractive_index_sigma": 0.0001,
                "seed": 42,
            }
        )
        validate_wafer_config(config)
        self.assertEqual(config.n_sites, 200)

    def test_disorder_strength_is_nonnegative_and_monotonic(self):
        low = disorder_strength_from_fabrication(self.build_config(width_sigma_nm=2.0))
        high = disorder_strength_from_fabrication(self.build_config(width_sigma_nm=8.0))
        self.assertGreaterEqual(low, 0.0)
        self.assertGreater(high, low)

    def test_hardware_map_route_reports_mapping_without_lab_artifact(self):
        config = {
            "hardware": {
                "layout": "1d_chain",
                "n_rows": 1,
                "n_cols": 200,
                "waveguide_pitch_um": 10.0,
                "length_mm": 20.0,
                "beta0": 1.0,
                "coupling_j": 1.0,
                "fabrication_width_sigma_nm": 5.0,
                "refractive_index_sigma": 0.0001,
                "propagation_loss_db_per_cm": 0.0,
                "seed": 42,
            },
            "noise_controller": {
                "actuator_type": "stochastic_phase_modulator",
                "max_phase_rms_rad": 0.05,
                "bandwidth_hz": 1_000_000.0,
                "correlation_time_ps": 10.0,
                "max_power_mw": 20.0,
            },
            "mapping": {"target_indices": [180, 181, 182]},
        }
        stream = io.StringIO()
        with redirect_stdout(stream):
            exit_code = run_hardware_map_mode(config)
        self.assertEqual(exit_code, 0)
        output = stream.getvalue()
        self.assertIn("Photonic wafer mapping", output)
        self.assertIn("recommended_kta_command", output)
        self.assertNotIn("trajectory_artifact", output)

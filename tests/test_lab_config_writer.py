from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import yaml

from hardware.lab_config_writer import write_lab_config_from_hardware_mapping
from run import load_config, run_hardware_map_mode


class LabConfigWriterTests(unittest.TestCase):
    def build_mapping_config(self) -> dict[str, object]:
        return {
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
            "mapping": {
                "target_indices": [180, 181, 182, 183, 184],
                "calibration_note": "Phenomenological first-pass mapping from wafer disorder and stochastic phase modulation to KTA W and gamma.",
            },
        }

    def test_writer_creates_yaml_with_mapping_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "generated.yaml"
            written = write_lab_config_from_hardware_mapping(
                output_path,
                W_eff=0.005001,
                gamma_eff=0.025,
                n_sites=200,
                target_indices=[180, 181, 182],
                metadata={"layout": "1d_chain", "n_sites": 200},
            )
            self.assertTrue(written.exists())
            payload = yaml.safe_load(written.read_text(encoding="utf-8"))
            self.assertAlmostEqual(payload["system"]["disorder_strength"], 0.005001)
            self.assertAlmostEqual(payload["system"]["gamma"], 0.025)
            self.assertEqual(payload["hardware_mapping"]["source"], "photonic_wafer")
            self.assertEqual(payload["hardware_mapping"]["target_indices"], [180, 181, 182])

    def test_writer_rejects_negative_gamma(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "generated.yaml"
            with self.assertRaises(ValueError):
                write_lab_config_from_hardware_mapping(
                    output_path,
                    W_eff=0.005001,
                    gamma_eff=-0.1,
                    n_sites=200,
                    target_indices=[],
                    metadata={},
                )

    def test_hardware_map_emit_lab_config_reports_generated_file_and_command(self):
        config = self.build_mapping_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_config = Path(tmpdir) / "photonic_wafer_demo_generated.yaml"
            stream = io.StringIO()
            with redirect_stdout(stream):
                exit_code = run_hardware_map_mode(
                    config,
                    emit_lab_config=True,
                    out_config=str(out_config),
                )
            self.assertEqual(exit_code, 0)
            self.assertTrue(out_config.exists())
            output = stream.getvalue()
            self.assertIn("generated_lab_config =", output)
            resolved_out_config = str(out_config.resolve())
            self.assertIn(resolved_out_config, output)
            self.assertIn("run_command =", output)
            self.assertIn(
                f"python run.py --config {resolved_out_config} lab --mode lindblad",
                output,
            )

    def test_generated_config_loads_and_contains_hardware_mapping_block(self):
        config = self.build_mapping_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_config = Path(tmpdir) / "photonic_wafer_demo_generated.yaml"
            run_hardware_map_mode(
                config,
                emit_lab_config=True,
                out_config=str(out_config),
            )
            merged = load_config(out_config)
            self.assertIn("hardware_mapping", merged)
            self.assertAlmostEqual(merged["system"]["gamma"], 0.025)
            self.assertEqual(merged["hardware_mapping"]["calibration_status"], "phenomenological")

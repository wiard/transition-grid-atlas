from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from hardware.reversibility_audit import (
    CoherentReversibilityResult,
    OpenReversibilityResult,
    ReversibilityMetadata,
    adjusted_n_time_samples,
    apply_dephasing_channel,
    basis_state,
    choose_forward_time,
    compute_reversibility_metadata_for_hamiltonian,
    density_from_state,
    phase_aligned_l2_error,
    plot_reversibility_audit,
    probability_l1_error,
    pure_state_return_fidelity,
    reverse_density_matrix,
    run_reversibility_audit,
    state_fidelity,
    unitary_from_hamiltonian,
    write_reversibility_csv,
    write_reversibility_summary_json,
)


class ReversibilityAuditTests(unittest.TestCase):
    def _base_transition_motor_config(self, *, normalized: bool) -> dict[str, object]:
        block: dict[str, object] = {
            "transition_motor": {
                "grid": {
                    "n_sites": 4,
                    "edges": [[0, 1], [1, 2], [2, 3]],
                    "base_coupling": 1.0,
                    "input_index": 0,
                    "target_indices": [2, 3],
                },
                "knobs": [
                    {
                        "name": "path_coupling_boost",
                        "family": "coherent_coupling",
                        "symbol": "a_path",
                        "basis_name": "target_corridor",
                        "min_value": -0.1,
                        "max_value": 0.1,
                        "default": 0.0,
                        "units": "arb",
                        "description": "path",
                        "hardware_meaning": "path",
                    }
                ],
                "noise": {
                    "profiles": [
                        [0.0, 0.1, 0.1, 0.0],
                    ]
                },
                "optimizer": {
                    "time_min": 0.0,
                    "time_max": 6.0,
                    "n_time_samples": 16,
                    "n_transport_modes": 2,
                    "finite_diff_eps": 1.0e-4,
                    "optimizer_steps": 2,
                    "optimizer_step_size": 0.04,
                    "random_restarts": 1,
                    "seed": 5,
                },
                "outputs": {
                    "sensitivity_csv_path": "outputs/test_reversibility_sensitivity.csv",
                },
            }
        }
        if normalized:
            block["transition_motor"]["objective_mode"] = {
                "name": "normalized_noise",
                "mode": "normalized",
                "weights": {
                    "transport": 1.0,
                    "noise_action": 1.0,
                    "leakage": 0.5,
                    "control_cost": 0.05,
                },
                "normalization": {
                    "transport_scale": 0.05,
                    "noise_action_scale": 0.002,
                    "leakage_scale": 0.001,
                    "control_cost_scale": 0.04,
                },
                "description": "normalized",
            }
        else:
            block["transition_motor"]["objective_weights"] = {
                "transport": 1.0,
                "noise_action": 0.5,
                "leakage": 0.25,
                "control_cost": 0.01,
            }
        return block

    def _write_config_bundle(self, root: Path) -> dict[str, object]:
        raw_path = root / "raw.yaml"
        normalized_path = root / "normalized.yaml"
        raw_path.write_text(yaml.safe_dump(self._base_transition_motor_config(normalized=False), sort_keys=False), encoding="utf-8")
        normalized_path.write_text(yaml.safe_dump(self._base_transition_motor_config(normalized=True), sort_keys=False), encoding="utf-8")
        return {
            "reversibility_audit": {
                "operating_modes": [
                    {"name": "raw", "config": str(raw_path)},
                    {"name": "normalized_noise", "config": str(normalized_path)},
                ],
                "forward_time_selection": "peak_target",
                "time_step_multipliers": [1.0, 2.0],
                "dephasing_strengths": [0.0, 0.1],
                "thresholds": {
                    "coherent_fidelity_min": 0.999999,
                    "coherent_l2_error_max": 1.0e-6,
                    "open_fidelity_warning": 0.99,
                    "loss_delta_warning": 0.01,
                    "time_resolution_relative_change": 0.05,
                },
                "outputs": {
                    "csv_path": str(root / "reversibility.csv"),
                    "summary_path": str(root / "reversibility.json"),
                    "plot_path": str(root / "reversibility.png"),
                },
            },
            "__config_path__": str(root / "reversibility.yaml"),
        }

    def test_unitary_from_hamiltonian_is_unitary(self):
        H = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        U = unitary_from_hamiltonian(H, 0.7)
        identity = np.eye(2, dtype=np.complex128)
        self.assertTrue(np.allclose(np.conjugate(U.T) @ U, identity, atol=1.0e-12))

    def test_u_minus_t_times_u_t_is_identity(self):
        H = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        identity = np.eye(2, dtype=np.complex128)
        self.assertTrue(np.allclose(unitary_from_hamiltonian(H, -0.7) @ unitary_from_hamiltonian(H, 0.7), identity, atol=1.0e-12))

    def test_basis_state_and_fidelity(self):
        psi = basis_state(3, 1)
        self.assertAlmostEqual(float(np.linalg.norm(psi)), 1.0)
        self.assertAlmostEqual(state_fidelity(psi, psi), 1.0)

    def test_reversibility_metadata_dataclass(self):
        metadata = ReversibilityMetadata(
            coherent_reversibility_score=1.0,
            coherent_loss_delta=0.0,
            open_reversibility_score=0.98,
            open_loss_delta=0.02,
            dephasing_strength=0.05,
            time_step_multiplier=1.0,
            passed_coherent=True,
        )
        self.assertTrue(metadata.passed_coherent)
        self.assertAlmostEqual(metadata.open_loss_delta, 0.02)

    def test_phase_aligned_l2_error_ignores_global_phase(self):
        psi = np.array([1.0, 0.0], dtype=np.complex128)
        phased = np.exp(1j * 0.7) * psi
        self.assertAlmostEqual(phase_aligned_l2_error(psi, phased), 0.0, places=12)

    def test_probability_l1_error_ignores_global_phase(self):
        psi = np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
        phased = np.exp(1j * 0.3) * psi
        self.assertAlmostEqual(probability_l1_error(psi, phased), 0.0, places=12)

    def test_density_from_state_trace_one(self):
        psi = np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
        rho = density_from_state(psi)
        self.assertAlmostEqual(float(np.real(np.trace(rho))), 1.0)

    def test_dephasing_strength_zero_and_one(self):
        psi = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
        rho = density_from_state(psi)
        rho_zero = apply_dephasing_channel(rho, 0.0)
        rho_one = apply_dephasing_channel(rho, 1.0)
        self.assertTrue(np.allclose(rho_zero, rho))
        self.assertAlmostEqual(abs(rho_one[0, 1]), 0.0, places=12)
        self.assertAlmostEqual(abs(rho_one[1, 0]), 0.0, places=12)

    def test_dephasing_preserves_trace_and_hermiticity(self):
        psi = np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
        rho = density_from_state(psi)
        rho_d = apply_dephasing_channel(rho, 0.4)
        self.assertAlmostEqual(float(np.real(np.trace(rho_d))), 1.0)
        self.assertTrue(np.allclose(rho_d, np.conjugate(rho_d.T)))

    def test_open_return_fidelity_drops_with_dephasing(self):
        H = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        psi0 = basis_state(2, 0)
        psi_fwd = unitary_from_hamiltonian(H, np.pi / 4.0) @ psi0
        rho = density_from_state(psi_fwd)
        rho_rev_clean = reverse_density_matrix(H, apply_dephasing_channel(rho, 0.0), np.pi / 4.0)
        rho_rev_noisy = reverse_density_matrix(H, apply_dephasing_channel(rho, 1.0), np.pi / 4.0)
        self.assertGreaterEqual(
            pure_state_return_fidelity(rho_rev_clean, psi0),
            pure_state_return_fidelity(rho_rev_noisy, psi0),
        )

    def test_compute_reversibility_metadata_for_hamiltonian(self):
        H = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        clean = compute_reversibility_metadata_for_hamiltonian(
            H=H,
            input_index=0,
            target_indices=[1],
            time_min=0.0,
            time_max=3.0,
            n_time_samples=50,
            dephasing_strength=0.0,
        )
        noisy = compute_reversibility_metadata_for_hamiltonian(
            H=H,
            input_index=0,
            target_indices=[1],
            time_min=0.0,
            time_max=3.0,
            n_time_samples=50,
            dephasing_strength=0.1,
        )
        self.assertGreaterEqual(clean.coherent_reversibility_score, 0.999999)
        self.assertLessEqual(clean.coherent_loss_delta, 1.0e-9)
        self.assertGreater(noisy.open_loss_delta, clean.open_loss_delta)
        self.assertAlmostEqual(noisy.dephasing_strength, 0.1)

    def test_adjusted_n_time_samples(self):
        self.assertEqual(
            adjusted_n_time_samples(time_min=0.0, time_max=20.0, n_time_samples=100, time_step_multiplier=1.0),
            100,
        )
        self.assertGreater(
            adjusted_n_time_samples(time_min=0.0, time_max=20.0, n_time_samples=100, time_step_multiplier=0.5),
            100,
        )
        self.assertLess(
            adjusted_n_time_samples(time_min=0.0, time_max=20.0, n_time_samples=100, time_step_multiplier=2.0),
            100,
        )

    def test_choose_forward_time_in_range(self):
        H = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        times = np.linspace(0.0, 3.0, 20)
        chosen = choose_forward_time(H, 0, [1], times, method="peak_target")
        self.assertGreaterEqual(chosen, float(times[0]))
        self.assertLessEqual(chosen, float(times[-1]))

    def test_choose_forward_time_final_returns_last_time(self):
        H = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        times = np.linspace(0.0, 3.0, 20)
        self.assertEqual(choose_forward_time(H, 0, [1], times, method="final"), float(times[-1]))

    def test_csv_json_and_plot_writers(self):
        coherent = [
            CoherentReversibilityResult("raw", 1.0, 20, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0, True),
            CoherentReversibilityResult("normalized_noise", 2.0, 10, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0, True),
        ]
        open_results = [
            OpenReversibilityResult("raw", 1.0, 20, 2.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, False),
            OpenReversibilityResult("raw", 1.0, 20, 2.0, 0.1, 0.95, 0.0, 0.0, 0.95, 0.05, True),
        ]
        summary_payload = {
            "n_modes": 2,
            "coherent_all_passed": True,
            "min_coherent_fidelity": 1.0,
            "max_coherent_l2_error": 0.0,
            "max_coherent_loss_delta": 0.0,
            "max_open_loss_delta": 0.05,
            "time_resolution_sensitive": False,
            "recommended_registry_action": "Do not add time_step_resolution as a registry knob; keep as audit metadata.",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = write_reversibility_csv(root / "audit.csv", coherent, open_results)
            from hardware.reversibility_audit import ReversibilityAuditSummary
            summary = ReversibilityAuditSummary(**summary_payload)
            summary_path = write_reversibility_summary_json(root / "audit.json", summary)
            plot_path = plot_reversibility_audit(coherent, open_results, root / "audit.png")

            self.assertTrue(csv_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(plot_path.exists())
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertIn("time_resolution_sensitive", payload)

    def test_tiny_audit_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = self._write_config_bundle(root)
            coherent_results, open_results, summary = run_reversibility_audit(config)
            self.assertEqual(summary.n_modes, 2)
            self.assertEqual(len(coherent_results), 4)
            self.assertEqual(len(open_results), 8)
            self.assertTrue(all(result.n_time_samples >= 2 for result in coherent_results))


if __name__ == "__main__":
    unittest.main()

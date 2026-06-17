from __future__ import annotations

import unittest

from hardware.noise_controller import NoiseController, effective_gamma_from_controller


class NoiseControllerTests(unittest.TestCase):
    def test_noise_controller_gamma_is_nonnegative(self):
        controller = NoiseController(
            actuator_type="stochastic_phase_modulator",
            max_phase_rms_rad=0.05,
            bandwidth_hz=1_000_000.0,
            correlation_time_ps=10.0,
            max_power_mw=20.0,
        )
        gamma_eff = effective_gamma_from_controller(controller)
        self.assertGreaterEqual(gamma_eff, 0.0)

    def test_higher_phase_rms_gives_higher_gamma(self):
        low = NoiseController(
            actuator_type="stochastic_phase_modulator",
            max_phase_rms_rad=0.03,
            bandwidth_hz=1_000_000.0,
            correlation_time_ps=10.0,
            max_power_mw=20.0,
        )
        high = NoiseController(
            actuator_type="stochastic_phase_modulator",
            max_phase_rms_rad=0.08,
            bandwidth_hz=1_000_000.0,
            correlation_time_ps=10.0,
            max_power_mw=20.0,
        )
        self.assertGreater(effective_gamma_from_controller(high), effective_gamma_from_controller(low))

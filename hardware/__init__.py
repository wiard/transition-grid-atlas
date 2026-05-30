"""Hardware-to-model mapping helpers for photonic wafer KTA workflows."""

from hardware.constraints import (
    validate_coupling_range,
    validate_disorder_range,
    validate_gamma_range,
    validate_wafer_config,
)
from hardware.detector_mapping import (
    detector_probabilities_from_trajectory,
    transport_efficiency_to_targets,
)
from hardware.noise_controller import (
    NoiseController,
    effective_gamma_from_controller,
    noise_controller_from_dict,
)
from hardware.photonic_wafer import (
    PhotonicWaferConfig,
    disorder_strength_from_fabrication,
    onsite_disorder_vector,
    photonic_wafer_from_dict,
)

__all__ = [
    "NoiseController",
    "PhotonicWaferConfig",
    "detector_probabilities_from_trajectory",
    "disorder_strength_from_fabrication",
    "effective_gamma_from_controller",
    "noise_controller_from_dict",
    "onsite_disorder_vector",
    "photonic_wafer_from_dict",
    "transport_efficiency_to_targets",
    "validate_coupling_range",
    "validate_disorder_range",
    "validate_gamma_range",
    "validate_wafer_config",
]

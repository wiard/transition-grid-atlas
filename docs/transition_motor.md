# Transition Motor Instrumentation

## Purpose

The transition motor makes the second KTA layer explicit: interpretable
controls on the effective Hamiltonian of a fixed photonic grid.

## Fixed-grid assumption

The physical topology is fixed. No new edges are created.

## Hamiltonian control model

`H(theta) = H0 + sum(theta_k * V_k)`

Each knob activates a basis operator over existing edges or onsite terms.

## Knob registry

Each knob has:

- name
- family
- symbol
- basis
- range
- units
- hardware meaning
- affected metrics

## Control bases

Site bases and edge bases provide interpretable low-dimensional controls rather
than opaque free vectors.

## Metrics

- transport_efficiency
- noise_internal
- noise_leakage
- noise_action_on_info
- suppression_score
- control_cost
- objective

## Objective modes

The transition motor now supports two objective interpretations:

- `raw`: optimize the legacy scalar objective directly from absolute transport,
  noise-action, leakage and control cost.
- `normalized`: optimize gain and reduction terms relative to the default knob
  state, divided by audit-calibrated normalization scales so detector gain does
  not silently dominate smaller noise/leakage terms.

Normalized mode does not alter the Hamiltonian model or fixed-grid physics. It
only changes how the optimizer scores candidate controls.

## Operating mode registry

An operating mode registry lets the operator select named motor regimes without
rewriting YAML weights by hand. The demo registry exposes:

- `default_normalized`
- `detector_mode`
- `noise_mode`
- `leakage_mode`
- `cost_mode`
- `balanced_mode`
- `legacy_balanced_raw`

The active mode can be selected in config or via:

```bash
python run.py transition-motor \
  --config configs/hardware/transition_motor_demo.yaml \
  --operating-mode detector_mode
```

## Sensitivity atlas

Finite-difference sensitivities show which knobs affect which metrics most
strongly around the current operating point.

## Optimization

The transition motor uses projected finite-difference ascent with bounded knob
ranges and deterministic random restarts.

## Scientific caution

This is hardware-native error suppression, not full quantum error correction.
No syndrome extraction. No recovery operations. No Steane/surface-code
implementation.

## Next step

Run the transition motor over synthetic wafer ensembles and compare the
optimized controls against detector-output success probabilities and measured or
high-fidelity simulated wafer data.

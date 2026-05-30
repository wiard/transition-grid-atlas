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

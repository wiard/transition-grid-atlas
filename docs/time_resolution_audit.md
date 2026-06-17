# Time-Resolution Sensitivity Audit

## Purpose

This audit treats time as an operational numerical parameter: the resolution of the time grid used in Hamiltonian transport evaluation.

## What it tests

The audit varies time_step_multiplier over 0.5, 1.0 and 2.0 and measures whether detector_delta, noise_action_delta, leakage_delta and common_balanced_delta are stable.

## What it does not test

This audit does not make claims about the fundamental nature of time.

## Interpretation

If metrics remain stable, time resolution is not promoted to an operating-mode registry knob.

If metrics change beyond thresholds, time_step_resolution should be tracked as audited operating-mode metadata or registry parameter.

## Caution

For time-independent Hamiltonians, sensitivity may reflect numerical sampling of max-over-time detector success rather than a physical control-cadence effect.

## KTA governance

Data first, interpretation second.
Fixed grid remains fixed.
No full-QEC claim.

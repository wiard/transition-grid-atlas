# Objective Mode Comparison

## Purpose

Compare raw and normalized transition-motor objective modes over identical synthetic wafer/noise samples.

## Why paired comparison matters

Both modes must see the same samples to make delta metrics meaningful.

## Primary paired deltas

- detector_delta
- noise_action_delta
- leakage_delta
- control_cost_delta

Positive detector_delta means normalized improves detector success.
Positive noise_action_delta means normalized reduces noise action.
Positive leakage_delta means normalized reduces leakage.
Positive control_cost_delta means normalized is cheaper.

## Why direct objective_delta is not cross-mode comparable

Raw and normalized objectives use different scales. Their direct numerical difference is not a valid scientific claim.

The legacy `objective_delta` field remains available for debugging, but it is always labeled `not_cross_mode_comparable`.

## Raw metric delta

Evaluates both raw and normalized samples under the raw metric.

## Normalized metric delta

Evaluates both raw and normalized samples under the normalized metric.

## Common balanced reporting score

Evaluates both modes under one shared score:

- detector_success / transport_scale
- noise_action / noise_action_scale
- leakage / leakage_scale
- control_cost / control_cost_scale

This shared reporting score separates physical paired outcomes from objective-scale artifacts.

## Uniform vs component-wise scale sensitivity

Uniformly multiplying all scales changes objective magnitude but not the relative geometry.
Component-wise perturbation tests actual calibration sensitivity.

## Interpretation

Normalized mode should be interpreted as a suppression-oriented operating mode when it improves noise-action/leakage at the cost of detector-output.

## Scientific caution

Synthetic paired ensemble validation only.
Not experimental validation.
Not full QEC.
Normalization scales remain calibration parameters.

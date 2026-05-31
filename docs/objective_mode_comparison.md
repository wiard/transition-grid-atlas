# Objective Mode Comparison

## Purpose

Compare raw and normalized transition-motor objective modes over identical synthetic wafer/noise samples.

## Why paired comparison matters

Both modes must see the same samples to make delta metrics meaningful.

## Metrics

- detector_delta
- noise_action_delta
- leakage_delta
- control_cost_delta
- objective_delta

## Interpretation of deltas

Positive detector_delta means normalized improves detector success.
Positive noise_action_delta means normalized reduces noise action.
Positive leakage_delta means normalized reduces leakage.
Positive control_cost_delta means normalized is cheaper.
Positive objective_delta means normalized scores higher under its comparison metric.

## Calibration sensitivity

Normalized scales are varied to test whether the result depends strongly on arbitrary scale choices.

## Scientific caution

Synthetic validation only.
Not experimental validation.
Not full QEC.
Normalization scales remain calibration parameters.

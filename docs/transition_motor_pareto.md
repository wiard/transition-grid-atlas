# Transition Motor Pareto Sweep

## Purpose
Explore objective-weight trade-offs for the KTA transition motor.

## Why
The current motor strongly improves detector-output, while noise-action and leakage are trade-off metrics. A Pareto sweep shows how tuning priorities change the outcome.

## Weight sets
- detector_max
- balanced
- noise_suppression
- leakage_guard
- cost_sensitive
- aggressive_balanced

## Metrics
- detector success gain
- noise-action reduction
- leakage reduction
- objective gain
- success rate
- control cost
- saturated knobs

## Pareto dominance
A weight set is Pareto-optimal if no other set is at least as good on all selected metrics and strictly better on one.

## Caution
Synthetic ensemble only.
No experimental validation.
No full QEC.
No syndrome extraction or recovery.

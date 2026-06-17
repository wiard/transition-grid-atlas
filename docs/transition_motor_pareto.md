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

## Reversibility as Operating-Mode Metadata

- coherent_reversibility_score is a sanity check for U(-t)U(t)
- open_reversibility_score measures return under dephasing
- open_loss_delta quantifies dephasing-induced irreversibility
- these metrics do not imply time travel or fundamental time reversal
- they can be used as hardware-native error-suppression metadata
- in detector-equivalent modes, higher reversibility_score can act as a tie-breaker

## Caution
Synthetic ensemble only.
No experimental validation.
No full QEC.
No syndrome extraction or recovery.

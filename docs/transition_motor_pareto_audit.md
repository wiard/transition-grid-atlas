# Transition Motor Pareto Audit

## Purpose
Determine whether Pareto presets represent distinct transition-motor operating regimes or a compressed frontier.

## Background
The first Pareto sweep showed all presets as Pareto-optimal, with small metric differences.

## Objective component scale
Explains detector, noise-action, leakage and control-cost component magnitudes and whether one term dominates the objective landscape.

## Normalized scores
Explains analysis-only min-max normalization. This does not change the motor objective used during optimization; it only makes cross-preset comparison more legible.

## Stress presets
Explains `ultra_detector`, `ultra_noise`, `ultra_leakage`, `ultra_cost` and `balanced_normalized`.

## Confidence intervals
Per-preset bootstrap confidence intervals for detector gain, noise-action reduction, leakage reduction and objective gain.

## Knob profiles
Mean theta, std theta and bound fractions per preset.

## Regime separation
Pairwise distances between normalized preset metric vectors reveal whether presets land in similar or genuinely distinct operating regions.

## Interpretation
Compressed frontier means the current motor and bounds produce similar outcomes across presets.
Separated regimes mean objective weights genuinely steer different transition-motor behavior.

## Reversibility as Operating-Mode Metadata

- coherent_reversibility_score is a sanity check for U(-t)U(t)
- open_reversibility_score measures return under dephasing
- open_loss_delta quantifies dephasing-induced irreversibility
- these metrics do not imply time travel or fundamental time reversal
- they can be used as hardware-native error-suppression metadata
- in detector-equivalent modes, higher reversibility_score can act as a tie-breaker

## Caution
Synthetic ensemble only.
Not experimental validation.
Not full QEC.
No syndrome extraction or recovery.

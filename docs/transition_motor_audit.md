# Transition Motor Bound-Pressure Audit

## Purpose
The transition motor exposes knobs; this audit determines whether the optimum is interior or limited by actuator bounds.

## Bound pressure
The audit marks each knob by relative position inside its allowed range and detects near-bound or saturated controls.

## Ablation
Each knob is reset to its default while the others stay at the tuned operating point. The resulting objective loss measures how strongly that knob contributes to the tuned solution.

## Gain per control cost
Knobs are ranked by objective contribution per `theta^2`, so we can separate expensive controls from efficient ones.

## Control-limit sweep
The audit reruns the motor under tighter and wider actuator ranges to test whether the current optimum is robust or mainly range-limited.

## Seed stability
The audit reruns the optimizer across multiple seeds and reports mean, spread and extrema of the tuned objective and key transport/noise metrics.

## Scientific caution
A bound-saturated optimum is not wrong, but it is not an unconstrained optimum. It indicates that physical actuator limits are part of the result.

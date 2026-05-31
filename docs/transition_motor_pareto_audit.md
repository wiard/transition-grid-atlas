# Transition Motor Pareto Stress Audit

## Purpose
Audit whether the current Pareto frontier shows genuinely different transition-motor regimes or mostly a narrow constraint-shaped family.

## Why
The first Pareto sweep kept all presets on the frontier and showed only small separations in detector gain, success rate and control pressure. This audit adds stress presets and normalized comparison layers so those trade-offs become easier to inspect.

## What it adds
- extreme detector, noise, leakage and cost presets
- bootstrap confidence intervals per preset
- knob-profile summaries per preset
- objective-term scale analysis
- a normalized objective-gain comparison for audit purposes

## Objective normalization
The motor itself still optimizes its configured raw objective. The audit separately computes normalized objective gains from transport gain, noise-action reduction, leakage reduction and control-cost increase so term-scale imbalance becomes visible without changing solver contracts.

## Regime interpretation
A sweep can remain Pareto-wide when detector gains are clustered, success rates are flat and knob profiles stay close together. That suggests a constraint-shaped family rather than cleanly separated operating regimes.

## Scientific caution
This is synthetic ensemble analysis on a fixed grid.
No experimental validation.
No full QEC.
No syndrome extraction or recovery.

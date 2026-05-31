# Reversibility Audit

## Purpose

This audit treats time reversal only as instrumental reversibility.

It tests whether applying U(-t) after U(t) restores the initial state within numerical tolerances.

## What it tests

- coherent unitary reversibility
- dephasing-induced irreversibility
- time-resolution sensitivity
- operating-mode differences

## What it does not test

- time travel
- fundamental nature of time
- full quantum error correction
- experimental validation

## Metrics

- fidelity
- phase_aligned_l2_error
- probability_l1_error
- return_probability
- loss_delta
- trace_error
- hermiticity_error

## Interpretation

For Hermitian time-independent H, coherent reversibility should pass.
For dephasing/noisy dynamics, coherent inverse does not generally undo the noise.

## Governance

Data first, interpretation second.
Fixed grid remains fixed.
No new edges.
No full-QEC claim.

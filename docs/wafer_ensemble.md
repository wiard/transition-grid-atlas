# Synthetic Wafer Ensemble Study

## Purpose

This study tests whether the transition-dynamics tuner remains beneficial over
an ensemble of sampled photonic wafer imperfections and stochastic phase-noise
profiles, rather than only on a single hand-picked demo instance.

## Relation to the transition tuner

The ensemble layer sits above the existing transition tuner. It does not alter
the core solver, Lindblad integrator, audit rules, or fixed-grid topology.

## Fixed physical grid assumption

The physical grid remains fixed for every sample:

- same number of sites
- same edge topology
- same input and target sectors

Only the effective onsite fabrication disorder and transition controls vary.

## Fabrication disorder

Fabrication disorder is modeled as onsite propagation-constant perturbation.
It changes only the Hamiltonian diagonal and does not add edges or modify the
off-diagonal topology.

## Phase-noise profiles

Phase noise is modeled as diagonal noise operators sampled from correlated
profiles. This is a synthetic wafer-level proxy, not an experimental
calibration result.

## Dynamic H-dependent transport subspace

The ensemble inherits the tuner's H-dependent transport subspace. That means
the reported `noise_overlap` remains control-dependent by construction and is
not a fixed geometric projector metric.

## Metrics

Per sample, the study compares:

- baseline transport efficiency
- tuned transport efficiency
- baseline noise overlap
- tuned noise overlap
- baseline objective
- tuned objective

Derived gains:

- `transport_gain`
- `noise_overlap_reduction`
- `objective_gain`

## Success criterion

A sample is marked successful when:

- `objective_gain >= -tolerance`
- `noise_overlap_reduction >= -tolerance`

This is a modest robustness criterion that tolerates trade-offs while still
requiring non-degrading overlap behavior.

## Scientific caution

- This is a synthetic study, not experimental validation.
- It supports hardware-native error-suppression hypotheses.
- It does **not** implement full quantum error correction.

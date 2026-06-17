# Reversibility Ensemble Study

## Purpose
Evaluate whether reversibility_score remains informative across synthetic wafer disorder, phase-noise ensembles, operating modes and dephasing strengths.

## What it tests
- open_reversibility_score per mode
- open_loss_delta per mode
- detector_success stability
- correlation between detector output and reversibility
- preferred mode selection under detector tolerance

## What it does not test
- time travel
- fundamental time reversal
- full quantum error correction
- experimental validation

## Interpretation
High reversibility_score means a mode is more recoverable under the tested dephasing model.
Low open_loss_delta means lower dephasing-induced irreversibility.
If detector-equivalent modes differ in reversibility, the higher reversibility mode is preferred.

## Governance
Fixed grid remains fixed.
No new edges.
Data first, interpretation second.


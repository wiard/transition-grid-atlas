# Transition-Dynamics Tuner

The KTA transition tuner is **not** full quantum error correction.

It explores **hardware-native error suppression** on a fixed photonic grid:

- the physical lattice stays fixed
- no new edges are added
- the tunable layer is transition dynamics
- only effective coupling scales and onsite controls are searched

Phase noise is modeled through diagonal noise operators over the grid sites.

The information-carrying sector is represented by a projector onto modes tied to
the input channel and target detector sector. The tuner then searches controls
that reduce the projection of phase noise onto those information-carrying
transport modes while preserving transport efficiency toward the target sector.

This is a first-pass algebraic model. It does **not** implement syndrome
extraction, recovery operations, or a Steane/surface-code construction.

It is best interpreted as a bridge layer:

- fixed hardware topology
- tunable transition dynamics
- measurable suppression proxies
- detector-facing transport objectives

Experimental calibration is still required before these effective controls can
be treated as quantitative hardware prescriptions.

## Subspace definition and metric validity

Two different subspace viewpoints matter here:

- **Fixed geometric subspace**: the span of the input mode and target detector
  sector. For fixed noise operators, the geometric overlap
  `||U† N U||` is invariant under transition tuning because neither `U` nor `N`
  changes.
- **Dynamic transport subspace**: an `H`-dependent transport sector extracted
  from eigenmodes of the tuned Hamiltonian that have strong combined overlap
  with the input and target sectors. Under this definition, the reported
  `noise_overlap` can change with the transition controls because the transport
  subspace itself changes with `H`.

KTA currently uses the **dynamic transport subspace** for the transition tuner.
This keeps the metric valid as a control-dependent suppression proxy while
remaining far more modest than a full QEC construction.

# Transition Motor Objective Modes

## Purpose

The Transition Motor now supports explicit objective modes so the operator can
choose whether the motor behaves like a legacy detector-weighted optimizer or a
normalized multi-metric optimizer.

## Modes

- `raw`
  Preserves the legacy scalar objective over absolute transport, noise-action,
  leakage and control cost.
- `normalized`
  Scores gains and reductions relative to the default knob state, divided by
  normalization scales so detector output no longer dominates by numerical
  magnitude alone.
- `calibrated`
  Uses the same normalized scoring path, but is intended for audit-derived
  scales and tuned weights anchored to prior synthetic Pareto studies.

## Normalization scales

`normalization_scales` define the numerical footing of:

- transport gain
- noise-action reduction
- leakage reduction
- control-cost increase

These scales do not change the Hamiltonian model or the fixed photonic grid.
They only change how the optimizer scores candidate controls.

## Config pattern

Use the legacy raw demo:

```bash
python run.py transition-motor \
  --config configs/hardware/transition_motor_demo.yaml
```

Use the normalized demo:

```bash
python run.py transition-motor \
  --config configs/hardware/transition_motor_normalized_demo.yaml
```

Override the selected registry mode:

```bash
python run.py transition-motor \
  --config configs/hardware/transition_motor_normalized_demo.yaml \
  --objective-mode calibrated_mode
```

## Scientific caution

This is hardware-native error suppression through fixed-grid Hamiltonian
control. It is not experimental validation and it does not implement full
quantum error correction.

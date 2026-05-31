# Transition Motor Objective Modes

## Purpose

The Transition Motor now supports explicit objective modes so the operator can
choose whether the motor behaves like a legacy detector-weighted optimizer or a
normalized multi-metric optimizer.

## Modes

- `raw`
  Uses unscaled detector, noise-action, leakage and control-cost components.
- `normalized`
  Divides each absolute objective component by a calibration scale before
  applying weights, so detector terms no longer dominate by numerical
  magnitude alone.
- `calibrated`
  Uses the same normalized scoring path, but is reserved for scales estimated
  from larger synthetic ensembles, high-fidelity simulations or measured
  hardware data.

## Normalization scales

`normalization` defines the numerical footing of:

- transport efficiency
- noise-action on information modes
- leakage
- control cost

These scales do not change the Hamiltonian model or the fixed photonic grid.
They only change how the optimizer scores candidate controls.

## Operating modes

- `raw_detector_max`
- `raw_balanced`
- `normalized_balanced`
- `normalized_noise`
- `normalized_leakage_guard`
- `normalized_cost_sensitive`

These are operating-mode templates. Their scales are calibration parameters,
not universal physical constants.

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

Override the selected normalized objective mode:

```bash
python run.py transition-motor \
  --config configs/hardware/transition_motor_normalized_demo.yaml \
  --objective-mode normalized_cost_sensitive
```

## Scientific caution

This is synthetic Hamiltonian-control tuning on a fixed photonic grid. It is
not experimental validation and it does not implement full quantum error
correction.

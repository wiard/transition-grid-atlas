# Transition Grid Atlas

**Subtitle:** A unitary testbench for emergent transport laws

Transition Grid Atlas is a reproducible scientific software instrument for
investigating emergent transport laws in tight-binding and feedback-coupled
grid systems.

It is **not** a proof of a new law of physics.

Its job is more modest and more rigorous:

1. **Simulation**
   Build a Hamiltonian, evolve a localized state, and collect observables.
2. **Validation**
   Check Hermiticity, unitarity, and statistical fit quality before any result
   is classified.
3. **Exploration**
   Run parameter sweeps, Monte Carlo ensembles, and recursive target searches.
4. **Atlas generation**
   Persist results, write reports, and generate non-interactive plots.

The project uses `float64` explicitly for all physical real-valued arrays and
`complex128` for quantum state arrays.

## Project layout

```text
transition-grid-atlas/
├── run.py
├── config.yaml
├── requirements.txt
├── engine/
├── validation/
├── explorer/
├── atlas/
│   ├── results/
│   ├── plots/
│   └── reports/
└── docs/
    └── README.md
```

## Installation

```bash
cd ~/transition-grid-atlas
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Core commands

### 1. Single validated run

```bash
python run.py single
```

Printed metrics:

- `alpha`
- `R²`
- `unitarity_error`
- `hermitian_error`
- `status`

This also appends a row to:

- `atlas/results/master_results.csv`

And overwrites:

- `atlas/reports/latest_report.md`

### 2. Monte Carlo ensemble

```bash
python run.py montecarlo --runs 100
```

Outputs:

- ensemble mean and standard deviation for `alpha`
- ensemble mean `R²`
- CSV saved in `atlas/results/`
- histogram PNGs saved in `atlas/plots/`

### 3. Parameter sweep

```bash
python run.py sweep
```

The sweep scans four parameters:

- `W` disorder strength
- `bias` field slope
- `eta` feedback coupling
- `gamma` dissipation strength

Outputs:

- sweep CSV
- collapsed 2D phase-map CSV
- phase-map PNG

### 4. Recursive hunter

```bash
python run.py hunter
```

The recursive hunter searches for regions where:

- `alpha ≈ 0.50`
- `R² > 0.90`
- `unitarity_error < 1e-12`

It saves:

- candidate CSV
- candidate plot
- a fresh atlas report

### 5. Inverse transition analysis

```bash
python run.py inverse
```

This experimental mode:

- generates ordered sweep paths across the configured path axis
- builds an exact inverse path for every forward path
- compares forward score, inverse score, and annihilation score
- reports only directional regularities that exceed the configured dominance threshold

Outputs:

- `outputs/inverse_analysis.csv`
- `outputs/inverse_transition_map.png`
- `atlas/reports/latest_report.md`

## Status labels

Each run is classified into exactly one of these categories:

- `VALID_BALLISTIC`
- `VALID_DIFFUSIVE_CANDIDATE`
- `VALID_LOCALIZED`
- `INVALID_NONUNITARY`
- `INVALID_NONHERMITIAN`
- `WEAK_FIT`

The ordering is strict:

1. Hermiticity failure invalidates the run.
2. Unitarity failure invalidates the run.
3. Weak regression quality downgrades the run to `WEAK_FIT`.
4. Only then is the transport exponent mapped to ballistic, diffusive, or localized.

## Scientific caution

Transition Grid Atlas can help investigate whether a chosen model displays
ballistic, diffusive-like, or localized transport signatures over a fitted
window.

It does **not** prove:

- a new law of physics
- that a numerical fit implies a universal physical law
- that dissipation-free diffusive behavior in a small model is a real material property

The instrument only reports reproducible, validated outputs from the model that
was actually run.

## Typical workflow

1. Start with `python run.py single`.
2. Confirm the run is Hermitian and unitary.
3. Use `python run.py montecarlo --runs 100` to gauge statistical robustness.
4. Use `python run.py sweep` to map the wider landscape.
5. Use `python run.py hunter` to zoom toward diffusive candidates.

## Files written by the atlas

### Persistent tables

- `atlas/results/master_results.csv`
- `atlas/results/montecarlo_*.csv`
- `atlas/results/sweep_*.csv`
- `atlas/results/hunter_*.csv`

### Plots

- `atlas/plots/montecarlo_alpha_*.png`
- `atlas/plots/montecarlo_r2_*.png`
- `atlas/plots/phase_map_*.png`
- `atlas/plots/hunter_candidates_*.png`

### Reports

- `atlas/reports/latest_report.md`
- `outputs/inverse_analysis.csv`
- `outputs/inverse_transition_map.png`

## Reproducibility notes

- All array-valued physical computations use explicit double precision.
- The disorder field is seeded.
- No plot is generated from a run that has not already passed through the
  validation layer.
- Monte Carlo and hunter modes save the actual sampled parameters used.

## Minimal verification

```bash
python -m unittest discover -s tests -v
python run.py single
python run.py montecarlo --runs 10
python run.py sweep
python run.py hunter
```

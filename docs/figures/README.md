# KTA Figures

Deze map bevat geborgde figuren voor de Kinetic Transition Atlas-documentatie.

## Three-layer architecture

Bestanden:

- `kta_three_layer_architecture.png`
- `kta_three_layer_architecture.pdf`
- TikZ-bron: `../tikz/kta_three_layer_architecture.tex`
- Lean/mathlib companion: `../lean/KTAThreeLayerArchitecture.lean`

Betekenis:

Deze figuur vat de KTA-laagarchitectuur samen:

1. Fysiek Grid — fixed-grid topologie.
2. Transition Motor — Hamiltonian control.
3. Audit & Registry — tests, confidence intervals, Pareto, operating modes.

Governance:

Het fysieke grid blijft vast. De Transition Motor stuurt alleen de effectieve Hamiltoniaan.

## Reversibility ensemble

Bestanden:

- `kta_reversibility_ensemble.png`
- `kta_reversibility_ensemble.pdf`
- TikZ-bron: `../tikz/kta_reversibility_ensemble.tex`
- Lean/mathlib companion: `../lean/KTAReversibilityEnsemble.lean`

Betekenis:

Deze figuur vat de Reversibility Ensemble-resultaten samen:

- cost_sensitive blijft de beste recoverability-mode zodra dephasing meespeelt.
- normalized_noise_strong is kwetsbaarder onder open-system loss.
- coherente inverse-propagatie blijft een sanity check.
- open-system verlies ontstaat door dephasing, niet door een claim over fundamentele tijdsinversie.

Wetenschappelijke voorzichtigheid:

Deze figuren zijn documentatie- en interpretatie-artifacts. Ze vormen geen experimentele validatie, geen full-QEC claim en geen claim over tijdreizen of fundamentele tijd.

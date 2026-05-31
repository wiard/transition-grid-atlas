import Mathlib

/-!
# KTA three-layer architecture (formal companion)

This Lean/mathlib file is not a physics solver.  It formalizes the governance
invariants behind the KTA architecture diagram:

* Layer 1: the photonic grid is fixed.
* Layer 2: the Transition Motor may tune Hamiltonian controls.
* Layer 3: Audit & Registry records metrics and operating modes.

The main invariant is that transition tuning may change effective Hamiltonian
parameters, but it may not mutate the physical grid topology.
-/

namespace KTA

/-- The three conceptual layers in the KTA architecture. -/
inductive Layer where
  | physicalGrid
  | transitionMotor
  | auditRegistry
  deriving DecidableEq, Repr

/-- Objective modes are operating regimes, not proof of full QEC. -/
inductive ObjectiveMode where
  | raw
  | normalized
  | calibrated
  deriving DecidableEq, Repr

/-- A minimal fixed photonic grid: sites and edges are the hardware topology. -/
structure FixedGrid where
  sites : Finset Nat
  edges : Finset (Nat × Nat)
  input : Nat
  targets : Finset Nat
  deriving Repr

/-- Transition controls live in layer 2. They tune the effective Hamiltonian. -/
structure MotorControls where
  pathCouplingBoost : ℝ
  globalCouplingScale : ℝ
  onsitePhaseGradient : ℝ
  targetDetuning : ℝ
  boundaryDetuning : ℝ
  objectiveMode : ObjectiveMode
  deriving Repr

/-- Audit data live in layer 3. They judge, but do not mutate, the grid. -/
structure AuditRecord where
  detectorDelta : ℝ
  noiseActionDelta : ℝ
  leakageDelta : ℝ
  controlCostDelta : ℝ
  commonBalancedDelta : ℝ
  directObjectiveDeltaStatus : String
  deriving Repr

/-- A full KTA state: fixed grid, motor controls, and audit record. -/
structure KTAState where
  grid : FixedGrid
  controls : MotorControls
  audit : AuditRecord
  deriving Repr

/-- A transition-motor action is topology-preserving iff sites and edges are unchanged. -/
def FixedGridInvariant (before after : KTAState) : Prop :=
  after.grid.sites = before.grid.sites ∧
  after.grid.edges = before.grid.edges

/-- If the fixed-grid invariant holds, then no new physical edges were created. -/
theorem no_new_edges_of_fixed_grid
    {before after : KTAState}
    (h : FixedGridInvariant before after) :
    after.grid.edges = before.grid.edges :=
  h.2

/-- If the fixed-grid invariant holds, then the physical site set is unchanged. -/
theorem no_new_sites_of_fixed_grid
    {before after : KTAState}
    (h : FixedGridInvariant before after) :
    after.grid.sites = before.grid.sites :=
  h.1

/-- The audit layer is read-only with respect to the physical grid. -/
def AuditReadOnly (before after : KTAState) : Prop :=
  after.grid = before.grid

/-- A read-only audit preserves the set of physical edges. -/
theorem audit_read_only_preserves_edges
    {before after : KTAState}
    (h : AuditReadOnly before after) :
    after.grid.edges = before.grid.edges := by
  rw [h]

/-- Direct objective subtraction is only comparable within the same objective scale. -/
def ComparableObjectiveScale (a b : ObjectiveMode) : Prop :=
  a = b

/-- Raw and normalized objective values are not directly comparable as one scalar delta. -/
theorem raw_normalized_not_comparable :
    ¬ ComparableObjectiveScale ObjectiveMode.raw ObjectiveMode.normalized := by
  intro h
  cases h

/-- A common balanced reporting score is an explicit shared metric. -/
structure CommonBalancedScore where
  detectorTerm : ℝ
  noiseActionTerm : ℝ
  leakageTerm : ℝ
  controlCostTerm : ℝ
  deriving Repr

/-- The shared score used for cross-mode reporting. -/
def CommonBalancedScore.value (s : CommonBalancedScore) : ℝ :=
  s.detectorTerm - s.noiseActionTerm - s.leakageTerm - s.controlCostTerm

/-- A positive common-balanced delta means the normalized mode wins under the shared score. -/
def NormalizedWinsCommonBalanced (raw normalized : CommonBalancedScore) : Prop :=
  raw.value < normalized.value

/-- KTA claims should be metric-backed before being interpreted phenomenologically. -/
structure MetricBackedClaim where
  hasPairedSamples : Prop
  hasBootstrapCI : Prop
  hasWinRate : Prop
  hasArtifactHygiene : Prop

/-- Minimal audit readiness condition. -/
def AuditReady (c : MetricBackedClaim) : Prop :=
  c.hasPairedSamples ∧ c.hasBootstrapCI ∧ c.hasWinRate ∧ c.hasArtifactHygiene

/-- Audit readiness implies paired samples are present. -/
theorem audit_ready_has_paired_samples {c : MetricBackedClaim}
    (h : AuditReady c) : c.hasPairedSamples :=
  h.1

/-- Audit readiness implies artifact hygiene is part of the claim. -/
theorem audit_ready_has_artifact_hygiene {c : MetricBackedClaim}
    (h : AuditReady c) : c.hasArtifactHygiene :=
  h.2.2.2

end KTA

import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-!
# KTA Reversibility Ensemble Companion

This Lean/mathlib companion formalizes the audit-level ordering used in the
KTA Reversibility Ensemble. It does not prove any physical claim. It only
records the governance logic:

* higher open reversibility score is better;
* lower open loss delta is better;
* detector-equivalent modes may be tie-broken by reversibility;
* this is an instrumental audit of inverse propagation, not a claim about
  fundamental time reversal.
-/

namespace KTA
namespace ReversibilityEnsemble

/-- Minimal metadata for an operating mode in a reversibility ensemble. -/
structure ModeMetrics where
  detectorSuccess : ℝ
  reversibilityScore : ℝ
  openLossDelta : ℝ
  controlCost : ℝ
  deriving Repr

/-- Recovery quality: higher reversibility and lower open loss. -/
def BetterRecovery (a b : ModeMetrics) : Prop :=
  a.reversibilityScore ≥ b.reversibilityScore ∧
  a.openLossDelta ≤ b.openLossDelta

/-- A mode is coherent-reversibility clean when inverse propagation has no loss. -/
def CoherentClean (m : ModeMetrics) : Prop :=
  m.reversibilityScore = 1 ∧ m.openLossDelta = 0

/-- Detector-equivalence within an absolute tolerance. -/
def DetectorEquivalent (tol : ℝ) (a b : ModeMetrics) : Prop :=
  |a.detectorSuccess - b.detectorSuccess| ≤ tol

/-- Operating-mode preference used as a tie-breaker, not as a physics theorem. -/
def PreferByReversibility (tol : ℝ) (a b : ModeMetrics) : Prop :=
  DetectorEquivalent tol a b ∧ BetterRecovery a b

theorem betterRecovery_refl (a : ModeMetrics) : BetterRecovery a a := by
  exact ⟨le_rfl, le_rfl⟩

theorem betterRecovery_trans {a b c : ModeMetrics} :
    BetterRecovery a b → BetterRecovery b c → BetterRecovery a c := by
  intro hab hbc
  exact ⟨le_trans hab.1 hbc.1, le_trans hab.2 hbc.2⟩

theorem prefer_implies_detector_equiv {tol : ℝ} {a b : ModeMetrics} :
    PreferByReversibility tol a b → DetectorEquivalent tol a b := by
  intro h
  exact h.1

theorem prefer_implies_better_recovery {tol : ℝ} {a b : ModeMetrics} :
    PreferByReversibility tol a b → BetterRecovery a b := by
  intro h
  exact h.2

/-- If a mode has score at most one, its open loss proxy `1 - score` is nonnegative. -/
theorem loss_nonnegative_from_score_bound {score : ℝ} (h : score ≤ 1) :
    0 ≤ 1 - score := by
  linarith

/-- If two detector-equivalent modes are compared, reversibility can be used as an audit tie-breaker. -/
theorem detector_equiv_and_better_recovery_gives_preference
    {tol : ℝ} {a b : ModeMetrics}
    (hD : DetectorEquivalent tol a b) (hR : BetterRecovery a b) :
    PreferByReversibility tol a b := by
  exact ⟨hD, hR⟩

end ReversibilityEnsemble
end KTA

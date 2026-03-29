/-
  Eval.lean — Top-level rule evaluator.

  Runs every registered rule against the snapshot and collects results.
  To add a new rule, import it here and add it to `evaluateAll`.
-/

import LyfeRules.Types
import LyfeRules.AcidBase

open LyfeRules

namespace LyfeRules.Eval

/-- Run all registered rules against the snapshot.
    New rules are added here — no Rust changes required. -/
def evaluateAll (snap : Snapshot) : EvalResult :=
  let mut rules : Array ReactionRule := #[]
  let mut diags : Array String := #[]

  -- Acid-base neutralisation
  match AcidBase.evaluate snap with
  | some rule =>
    rules := rules.push rule
    diags := diags.push s!"H⁺+OH⁻ reaction active in {rule.applicableTileIds.size} tiles"
  | none =>
    diags := diags.push "H⁺+OH⁻ reaction: no co-located reactants"

  -- Future rules go here:
  -- match SomeOtherRule.evaluate snap with ...

  { rules, diagnostics := diags }

end LyfeRules.Eval

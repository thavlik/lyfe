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
  let rules := AcidBase.evaluate snap
  let diags := if rules.isEmpty then
    #["acid/base evaluator: no active rules"]
  else
    rules.map fun rule =>
      s!"{rule.reactionName} active in {rule.applicableTileIds.size} tiles"

  -- Future rules go here:
  -- match SomeOtherRule.evaluate snap with ...

  { rules, diagnostics := diags }

end LyfeRules.Eval

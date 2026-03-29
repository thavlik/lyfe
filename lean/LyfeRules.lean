/-
  LyfeRules — Lean 4 rule engine for the Lyfe simulation.

  This library receives a coarse-grid simulation snapshot (JSON on stdin)
  and emits validated reaction rules (JSON on stdout).

  New rules are added here, not in Rust.  The Rust side is rule-agnostic:
  it serialises the snapshot, invokes this binary, and applies whatever
  rules come back.
-/

import LyfeRules.Types
import LyfeRules.AcidBase
import LyfeRules.Eval

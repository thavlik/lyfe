/-
  Main.lean — CLI entry point for the Lean rule engine.

  Protocol:
    1. Read JSON snapshot from stdin  (one object, no newline framing)
    2. Parse into `Snapshot`
    3. Run all rules via `Eval.evaluateAll`
    4. Write JSON `EvalResult` to stdout
    5. Exit 0 on success, non-zero on parse/eval error

  The Rust `lean_bridge` module spawns this binary and communicates
  via stdin/stdout pipes.
-/

import LyfeRules
import Lean.Data.Json

open Lean (Json FromJson ToJson)
open LyfeRules

def main : IO Unit := do
  -- Read entire stdin
  let input ← IO.stdin.readToEnd

  -- Parse JSON
  let json ← match Json.parse input with
    | .ok j    => pure j
    | .error e =>
      IO.eprintln s!"JSON parse error: {e}"
      IO.Process.exit 1

  -- Deserialise snapshot
  let snap ← match @FromJson.fromJson? Snapshot _ json with
    | .ok s    => pure s
    | .error e =>
      IO.eprintln s!"Snapshot deserialisation error: {e}"
      IO.Process.exit 1

  -- Evaluate rules
  let result := Eval.evaluateAll snap

  -- Serialise and write to stdout
  let outJson := ToJson.toJson result
  IO.println outJson.compress

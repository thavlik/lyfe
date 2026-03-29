import Lake
open Lake DSL

package «lyfe-rules» where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

lean_lib LyfeRules where

@[default_target]
lean_exe «lyfe-rules» where
  root := `Main
  supportInterpreter := true

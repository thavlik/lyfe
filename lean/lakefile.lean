import Lake
open Lake DSL

package «lyfe-rules» where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

@[default_target]
lean_exe «lyfe-rules» where
  root := `Main
  supportInterpreter := true

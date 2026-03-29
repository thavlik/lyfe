/-
  Types.lean — Shared data types for the rule engine.

  These mirror the JSON schema used for communication with the Rust side.
  Field names use snake_case in JSON for Rust compatibility.
-/

import Lean.Data.Json

open Lean (Json ToJson FromJson)

namespace LyfeRules

/-! ## Input types (snapshot from Rust → Lean) -/

/-- A single species concentration measurement in a tile. -/
structure SpeciesAmount where
  name     : String
  molarity : Float
  deriving Inhabited, Repr

instance : FromJson SpeciesAmount where
  fromJson? j := do
    let name ← j.getObjValAs? String "name"
    let molarity ← j.getObjValAs? Float "molarity"
    return { name, molarity }

/-- A coarse tile from the simulation snapshot. -/
structure TileSnapshot where
  tileId          : Nat
  fluidFraction   : Float
  meanTemperature : Float
  species         : Array SpeciesAmount
  deriving Inhabited, Repr

instance : FromJson TileSnapshot where
  fromJson? j := do
    let tileId ← j.getObjValAs? Nat "tile_id"
    let fluidFraction ← j.getObjValAs? Float "fluid_fraction"
    let meanTemperature ← j.getObjValAs? Float "mean_temperature"
    let species ← j.getObjValAs? (Array SpeciesAmount) "species"
    return { tileId, fluidFraction, meanTemperature, species }

/-- The simulation snapshot sent by Rust. -/
structure Snapshot where
  simTime      : Float
  speciesNames : Array String
  tiles        : Array TileSnapshot
  deriving Inhabited, Repr

instance : FromJson Snapshot where
  fromJson? j := do
    let simTime ← j.getObjValAs? Float "sim_time"
    let speciesNames ← j.getObjValAs? (Array String) "species_names"
    let tiles ← j.getObjValAs? (Array TileSnapshot) "tiles"
    return { simTime, speciesNames, tiles }

/-! ## Output types (rules from Lean → Rust) -/

/-- A reaction rule emitted by the Lean rule engine. -/
structure ReactionRule where
  reactionName        : String
  reactantA           : String
  reactantB           : String
  productA            : String
  productB            : String
  /-- Physical rate constant (L·mol⁻¹·s⁻¹) — for reference / Arrhenius. -/
  rateConstant        : Float
  /-- Effective rate for GPU simulation, pre-scaled for visual stability. -/
  effectiveRate       : Float
  /-- ΔH° in J/mol.  Negative ⟹ exothermic ⟹ heats the fluid. -/
  enthalpyDelta       : Float
  /-- ΔG° in J/mol.  Negative ⟹ spontaneous. -/
  gibbsFreeEnergy     : Float
  /-- ΔS° in J/(mol·K). -/
  entropyDelta        : Float
  /-- Activation energy Eₐ in J/mol. -/
  activationEnergy    : Float
  isReversible        : Bool
  applicableTileIds   : Array Nat
  deriving Inhabited, Repr

instance : ToJson ReactionRule where
  toJson r := .mkObj [
    ("reaction_name",                ToJson.toJson r.reactionName),
    ("reactant_a",                   ToJson.toJson r.reactantA),
    ("reactant_b",                   ToJson.toJson r.reactantB),
    ("product_a",                    ToJson.toJson r.productA),
    ("product_b",                    ToJson.toJson r.productB),
    ("rate_constant",                ToJson.toJson r.rateConstant),
    ("effective_rate",               ToJson.toJson r.effectiveRate),
    ("enthalpy_delta",               ToJson.toJson r.enthalpyDelta),
    ("gibbs_free_energy",            ToJson.toJson r.gibbsFreeEnergy),
    ("entropy_delta",                ToJson.toJson r.entropyDelta),
    ("activation_energy",            ToJson.toJson r.activationEnergy),
    ("is_reversible",                ToJson.toJson r.isReversible),
    ("applicable_tile_ids",          ToJson.toJson r.applicableTileIds)
  ]

/-- The complete evaluation result sent back to Rust. -/
structure EvalResult where
  rules       : Array ReactionRule
  diagnostics : Array String
  deriving Inhabited, Repr

instance : ToJson EvalResult where
  toJson r := .mkObj [
    ("rules",       ToJson.toJson r.rules),
    ("diagnostics", ToJson.toJson r.diagnostics)
  ]

end LyfeRules

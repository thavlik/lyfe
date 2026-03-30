/-
  Enzyme.lean — Generic catalyst-gated rule helpers plus the hexokinase demo.

  The primitive here models reactions that:
  - consume up to two reactants,
  - produce up to two products,
  - require a catalyst/enzyme species to be present,
  - do not consume the catalyst.
-/

import LyfeRules.Types
import LyfeRules.AcidBase

open LyfeRules

namespace LyfeRules.Enzyme

/-- Hexokinase turnover number ($k_{cat}$) near room temperature in s⁻¹. -/
def hexokinaseTurnoverNumber : Float := 95.0

/-- Michaelis constant for glucose in mol/L. -/
def hexokinaseKmGlucose : Float := 4.7e-5

/-- Michaelis constant for ATP in mol/L. -/
def hexokinaseKmAtp : Float := 3.7e-4

/-- Effective rate before fluidsim applies its global kinetics scale. -/
def hexokinaseEffectiveRate : Float := 5.0e-3

/-- Keep the demo thermally neutral until a better parameterization is added. -/
def enzymeThermalDelta : Float := 0.0

def catalyzedTileIds (
  snap : Snapshot
) (
  reactantA : String
) (
  reactantB : String
) (
  catalyst : String
) : Array Nat :=
  LyfeRules.AcidBase.applicableTileIds snap #[reactantA, reactantB, catalyst]

def makeCatalyzedRule (
  reactionName : String
) (
  reactantA : String
) (
  reactantB : String
) (
  productA : String
) (
  productB : String
) (
  catalyst : String
) (
  kineticModel : String
) (
  rateConstant : Float
) (
  effectiveRate : Float
) (
  kmReactantA : Option Float
) (
  kmReactantB : Option Float
) (
  enthalpyDelta : Float
) (
  gibbsFreeEnergy : Float
) (
  entropyDelta : Float
) (
  activationEnergy : Float
) (
  tileIds : Array Nat
) : ReactionRule := {
  reactionName := reactionName
  reactantA := reactantA
  reactantB := reactantB
  productA := productA
  productB := productB
  catalyst := some catalyst
  kineticModel := kineticModel
  rateConstant := rateConstant
  effectiveRate := effectiveRate
  kmReactantA := kmReactantA
  kmReactantB := kmReactantB
  enthalpyDelta := enthalpyDelta
  gibbsFreeEnergy := gibbsFreeEnergy
  entropyDelta := entropyDelta
  activationEnergy := activationEnergy
  isReversible := false
  applicableTileIds := tileIds
}

/-- Evaluate catalyst-gated reactions against the snapshot. -/
def evaluate (snap : Snapshot) : Array ReactionRule :=
  let hexokinaseTiles := catalyzedTileIds snap "Glucose" "ATP" "Hexokinase"
  let rules : Array ReactionRule := #[]
  let rules :=
    if hexokinaseTiles.isEmpty then rules else rules.push <| makeCatalyzedRule
      "hexokinase_phosphorylation"
      "Glucose"
      "ATP"
      "G6P"
      "ADP"
      "Hexokinase"
      "michaelis_menten"
      hexokinaseTurnoverNumber
      hexokinaseEffectiveRate
      (some hexokinaseKmGlucose)
      (some hexokinaseKmAtp)
      enzymeThermalDelta
      enzymeThermalDelta
      enzymeThermalDelta
      enzymeThermalDelta
      hexokinaseTiles
  rules

end LyfeRules.Enzyme
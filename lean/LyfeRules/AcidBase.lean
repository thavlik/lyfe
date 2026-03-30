/-
  AcidBase.lean — H⁺ + OH⁻ → H₂O neutralisation rule.

  ## Thermodynamics (NIST / standard tables)

  H⁺(aq) + OH⁻(aq) → H₂O(l)

  | Property | Value               | Source                        |
  |----------|---------------------|-------------------------------|
  | ΔH°      | −55 800 J/mol       | Standard enthalpy of neutral. |
  | ΔG°      | −79 900 J/mol       | ΔG°f(H₂O) − ΔG°f(OH⁻)       |
  | ΔS°      | +80.7 J/(mol·K)     | (ΔH° − ΔG°) / 298.15 K      |
  | Eₐ       | ≈ 11 000 J/mol      | Near-barrierless proton xfer  |
  | k(298 K) | 1.4 × 10¹¹ L/(mol·s)| Eigen & De Maeyer (1958)     |

  Verification:  ΔG° = ΔH° − T·ΔS°
    −55 800 − 298.15 × 80.7 ≈ −55 800 − 24 061 ≈ −79 861 ≈ −79 900 ✓

  Because ΔH° < 0 (exothermic), cells where this reaction proceeds will
  heat up over time.  The GPU shader computes:
    dT = −ΔH · extent / Cₚ   (with ΔH negative ⟹ dT > 0)
-/

import LyfeRules.Types

open LyfeRules

namespace LyfeRules.AcidBase

/-- Physical rate constant at 298 K  (L·mol⁻¹·s⁻¹). -/
def waterFormationRateConstant : Float := 1.4e11

/-- Effective rate used by the GPU shader.
    Scaled way down so the reaction is visible over ~10 s of sim-time
    rather than being instantaneous. -/
def waterFormationEffectiveRate : Float := 1.0

/-- ΔH° in J/mol (negative ⟹ exothermic ⟹ releases heat). -/
def waterFormationEnthalpyDelta : Float := -55800.0

/-- ΔG° in J/mol (negative ⟹ spontaneous). -/
def waterFormationGibbsFreeEnergy : Float := -79900.0

/-- ΔS° in J/(mol·K).
    Positive because OH⁻(aq) has an abnormally negative standard entropy
    due to strong solvent ordering; removing it *increases* system entropy. -/
def waterFormationEntropyDelta : Float := 80.7

/-- Activation energy in J/mol.  Nearly barrierless proton transfer. -/
def waterFormationActivationEnergy : Float := 11000.0

/-- Weak-acid dissociation constant for acetic acid at 298 K. -/
def aceticAcidKa298K : Float := 1.8e-5

/-- Forward dissociation rate for CH3COOH → H+ + CH3COO-. -/
def aceticDissociationRate : Float := 1.0e-4

/-- Reverse recombination rate for H+ + CH3COO- → CH3COOH. -/
def aceticRecombinationRate : Float := aceticDissociationRate / aceticAcidKa298K

/-- Direct strong-base neutralisation of acetic acid by hydroxide. -/
def aceticNeutralizationRateConstant : Float := 1.4e11

/-- Effective neutralisation rate used by the GPU shader. -/
def aceticNeutralizationEffectiveRate : Float := 1.0

/-- Acetic-acid buffer equilibration is treated as thermally neutral here. -/
def bufferThermalDelta : Float := 0.0

/-- Minimum molarity for a species to be considered "present" in a tile. -/
def minMolarity : Float := 1e-6

/-- Check whether a tile has both H⁺ and OH⁻ above the threshold. -/
def tileHasSpecies (tile : TileSnapshot) (speciesName : String) : Bool :=
  tile.species.any fun s => s.name == speciesName && s.molarity > minMolarity

def tileHasAllSpecies (tile : TileSnapshot) (speciesNames : Array String) : Bool :=
  speciesNames.all (tileHasSpecies tile)

def applicableTileIds (snap : Snapshot) (speciesNames : Array String) : Array Nat :=
  let matchingTiles := snap.tiles.filter fun tile =>
    tile.fluidFraction > 0.0 && tileHasAllSpecies tile speciesNames
  matchingTiles.map fun tile => tile.tileId

def makeRule (
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
  rateConstant : Float
) (
  effectiveRate : Float
) (
  enthalpyDelta : Float
) (
  gibbsFreeEnergy : Float
) (
  entropyDelta : Float
) (
  activationEnergy : Float
) (
  isReversible : Bool
) (
  tileIds : Array Nat
) : ReactionRule := {
  reactionName := reactionName
  reactantA := reactantA
  reactantB := reactantB
  productA := productA
  productB := productB
  catalyst := none
  kineticModel := "mass_action"
  rateConstant := rateConstant
  effectiveRate := effectiveRate
  kmReactantA := none
  kmReactantB := none
  enthalpyDelta := enthalpyDelta
  gibbsFreeEnergy := gibbsFreeEnergy
  entropyDelta := entropyDelta
  activationEnergy := activationEnergy
  isReversible := isReversible
  applicableTileIds := tileIds
}

def pushRuleIfAny (
  rules : Array ReactionRule
) (
  tileIds : Array Nat
) (
  rule : ReactionRule
) : Array ReactionRule :=
  if tileIds.isEmpty then rules else rules.push rule

/-- Evaluate the acid-base and acetate buffer rules against the full snapshot. -/
def evaluate (snap : Snapshot) : Array ReactionRule :=
  let waterTiles := applicableTileIds snap #["H+", "OH-"]
  let dissociationTiles := applicableTileIds snap #["CH3COOH"]
  let recombinationTiles := applicableTileIds snap #["H+", "CH3COO-"]
  let neutralizationTiles := applicableTileIds snap #["CH3COOH", "OH-"]

  let rules : Array ReactionRule := #[]
  let rules := pushRuleIfAny rules waterTiles <| makeRule
    "water_formation"
    "H+"
    "OH-"
    ""
    ""
    waterFormationRateConstant
    waterFormationEffectiveRate
    waterFormationEnthalpyDelta
    waterFormationGibbsFreeEnergy
    waterFormationEntropyDelta
    waterFormationActivationEnergy
    false
    waterTiles
  let rules := pushRuleIfAny rules dissociationTiles <| makeRule
    "acetic_acid_dissociation"
    "CH3COOH"
    ""
    "H+"
    "CH3COO-"
    aceticDissociationRate
    aceticDissociationRate
    bufferThermalDelta
    bufferThermalDelta
    bufferThermalDelta
    bufferThermalDelta
    true
    dissociationTiles
  let rules := pushRuleIfAny rules recombinationTiles <| makeRule
    "acetic_acid_recombination"
    "H+"
    "CH3COO-"
    "CH3COOH"
    ""
    aceticRecombinationRate
    aceticRecombinationRate
    bufferThermalDelta
    bufferThermalDelta
    bufferThermalDelta
    bufferThermalDelta
    true
    recombinationTiles
  let rules := pushRuleIfAny rules neutralizationTiles <| makeRule
    "acetic_acid_neutralization"
    "CH3COOH"
    "OH-"
    "CH3COO-"
    ""
    aceticNeutralizationRateConstant
    aceticNeutralizationEffectiveRate
    waterFormationEnthalpyDelta
    waterFormationGibbsFreeEnergy
    waterFormationEntropyDelta
    waterFormationActivationEnergy
    false
    neutralizationTiles
  rules

end LyfeRules.AcidBase

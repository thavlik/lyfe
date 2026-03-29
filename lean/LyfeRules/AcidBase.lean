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
def rateConstant : Float := 1.4e11

/-- Effective rate used by the GPU shader.
    Scaled way down so the reaction is visible over ~10 s of sim-time
    rather than being instantaneous. -/
def effectiveRate : Float := 1.0

/-- ΔH° in J/mol (negative ⟹ exothermic ⟹ releases heat). -/
def enthalpyDelta : Float := -55800.0

/-- ΔG° in J/mol (negative ⟹ spontaneous). -/
def gibbsFreeEnergy : Float := -79900.0

/-- ΔS° in J/(mol·K).
    Positive because OH⁻(aq) has an abnormally negative standard entropy
    due to strong solvent ordering; removing it *increases* system entropy. -/
def entropyDelta : Float := 80.7

/-- Activation energy in J/mol.  Nearly barrierless proton transfer. -/
def activationEnergy : Float := 11000.0

/-- Minimum molarity for a species to be considered "present" in a tile. -/
def minMolarity : Float := 1e-6

/-- Check whether a tile has both H⁺ and OH⁻ above the threshold. -/
def tileHasBothReactants (tile : TileSnapshot) : Bool :=
  let hasH  := tile.species.any fun s => s.name == "H+"  && s.molarity > minMolarity
  let hasOH := tile.species.any fun s => s.name == "OH-" && s.molarity > minMolarity
  hasH && hasOH

/-- Evaluate the acid-base rule against the full snapshot.
    Returns a `ReactionRule` if any tiles contain both reactants,
    or `none` otherwise. -/
def evaluate (snap : Snapshot) : Option ReactionRule :=
  let applicableTiles := snap.tiles.filter fun t =>
    t.fluidFraction > 0.0 && tileHasBothReactants t
  if applicableTiles.isEmpty then
    none
  else
    let tileIds := applicableTiles.map fun t => t.tileId
    some {
      reactionName      := "water_formation"
      reactantA         := "H+"
      reactantB         := "OH-"
      rateConstant      := rateConstant
      effectiveRate     := effectiveRate
      enthalpyDelta     := enthalpyDelta
      gibbsFreeEnergy   := gibbsFreeEnergy
      entropyDelta      := entropyDelta
      activationEnergy  := activationEnergy
      isReversible      := false
      applicableTileIds := tileIds
    }

end LyfeRules.AcidBase

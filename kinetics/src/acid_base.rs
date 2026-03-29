//! Acid-base neutralization evaluator (Rust fallback).
//!
//! This is a development fallback for when the Lean binary is not available.
//! The canonical rule definitions live in `lean/LyfeRules/AcidBase.lean`.
//! This evaluator duplicates that logic in Rust for convenience.
//!
//! ## Thermodynamics
//!
//! H⁺(aq) + OH⁻(aq) → H₂O(l)
//! - ΔH° = −55,800 J/mol  (exothermic → heats solution)
//! - ΔG° = −79,900 J/mol  (spontaneous)
//! - ΔS° = +80.7 J/(mol·K) (positive; OH⁻ has abnormally negative S°)

use crate::{
    KineticsConfig, KineticsError, SemanticSnapshot, SemanticUpdate,
    engine::RuleEvaluator,
    update::{ReactionDirective, ReactionId},
};

/// Well-known reaction ID for H⁺ + OH⁻ → H₂O.
pub const REACTION_H_OH_WATER: ReactionId = ReactionId(1);
/// Well-known reaction ID for acetic acid dissociation.
pub const REACTION_ACETIC_DISSOCIATION: ReactionId = ReactionId(2);
/// Well-known reaction ID for acetate recombination.
pub const REACTION_ACETATE_RECOMBINATION: ReactionId = ReactionId(3);
/// Well-known reaction ID for direct neutralization of acetic acid by hydroxide.
pub const REACTION_ACETIC_NEUTRALIZATION: ReactionId = ReactionId(4);

/// Rate constant for H⁺ + OH⁻ → H₂O at 298 K (L·mol⁻¹·s⁻¹).
/// This is one of the fastest known reactions.
const RATE_CONSTANT_298K: f64 = 1.4e11;

/// Effective rate delivered by the rule source before fluidsim applies its
/// global reaction-rate multiplier.
const EFFECTIVE_RATE: f64 = 1.0;

/// Standard enthalpy of neutralization ΔH° (J/mol), exothermic → negative.
const ENTHALPY_DELTA_J_PER_MOL: f64 = -55_800.0;

/// Standard Gibbs free energy change ΔG° (J/mol), spontaneous → negative.
const GIBBS_FREE_ENERGY_J_PER_MOL: f64 = -79_900.0;

/// Standard entropy change ΔS° (J/(mol·K)).
/// Positive because OH⁻(aq) has unusually negative standard entropy.
const ENTROPY_DELTA_J_PER_MOL_K: f64 = 80.7;

/// Activation energy (J/mol). Nearly barrierless proton transfer.
const ACTIVATION_ENERGY_J_PER_MOL: f64 = 11_000.0;

/// Acetic acid dissociation constant at 298 K.
const ACETIC_ACID_KA_298K: f64 = 1.8e-5;
/// Forward dissociation rate for CH3COOH -> H+ + CH3COO-.
const ACETIC_DISSOCIATION_RATE: f64 = 1.0e-4;
/// Reverse recombination rate for H+ + CH3COO- -> CH3COOH.
const ACETIC_RECOMBINATION_RATE: f64 =
    ACETIC_DISSOCIATION_RATE / ACETIC_ACID_KA_298K;
/// Buffer equilibrium reactions are modeled as thermally neutral in the demo.
const BUFFER_THERMAL_DELTA: f64 = 0.0;
/// Direct proton transfer from acetic acid to hydroxide is fast.
const ACETIC_NEUTRALIZATION_RATE_CONSTANT: f64 = RATE_CONSTANT_298K;
/// Effective rate for CH3COOH + OH- -> CH3COO-.
const ACETIC_NEUTRALIZATION_EFFECTIVE_RATE: f64 = EFFECTIVE_RATE;

/// An evaluator that derives H⁺ + OH⁻ → H₂O reaction directives
/// from the semantic snapshot.
#[derive(Debug, Clone, Default)]
pub struct AcidBaseEvaluator {
    evaluation_count: u64,
}

impl AcidBaseEvaluator {
    pub fn new() -> Self {
        Self { evaluation_count: 0 }
    }

    fn species_index(snapshot: &SemanticSnapshot, name: &str) -> Option<usize> {
        snapshot
            .species_table
            .species_names
            .iter()
            .position(|species_name| species_name == name)
    }

    fn tiles_with_species(
        snapshot: &SemanticSnapshot,
        required_species_ids: &[crate::SpeciesId],
        min_molarity: f64,
    ) -> Vec<u32> {
        let mut tiles = Vec::new();

        for tile in &snapshot.tiles {
            if !tile.has_fluid() {
                continue;
            }

            let all_present = required_species_ids.iter().all(|species_id| {
                tile.species_mean_molarity.iter().any(|amount| {
                    amount.species_id == *species_id && amount.value > min_molarity
                })
            });

            if all_present {
                tiles.push(tile.tile_id);
            }
        }

        tiles
    }
}

impl RuleEvaluator for AcidBaseEvaluator {
    fn name(&self) -> &str {
        "acid-base"
    }

    fn evaluate(
        &mut self,
        snapshot: &SemanticSnapshot,
        config: &KineticsConfig,
    ) -> Result<SemanticUpdate, KineticsError> {
        self.evaluation_count += 1;

        let valid_from = snapshot.sim_time_seconds;
        let valid_until = valid_from + config.min_evaluation_interval;

        let mut update = SemanticUpdate::noop(valid_from, valid_until);

        const MIN_MOLARITY: f64 = 1e-8;

        if let (Some(h_idx), Some(oh_idx)) = (
            Self::species_index(snapshot, "H+"),
            Self::species_index(snapshot, "OH-"),
        ) {
            let applicable_tiles = Self::tiles_with_species(
                snapshot,
                &[
                    snapshot.species_table.species_ids[h_idx],
                    snapshot.species_table.species_ids[oh_idx],
                ],
                MIN_MOLARITY,
            );

            if !applicable_tiles.is_empty() {
                update.reaction_directives.push(ReactionDirective {
                    reaction_id: REACTION_H_OH_WATER,
                    reaction_name: "water_formation".to_string(),
                    reactant_a: "H+".to_string(),
                    reactant_b: "OH-".to_string(),
                    product_a: String::new(),
                    product_b: String::new(),
                    applicable_tile_ids: applicable_tiles,
                    max_extent_per_second: f64::MAX,
                    rate_constant: RATE_CONSTANT_298K,
                    effective_rate: EFFECTIVE_RATE,
                    enthalpy_delta_j_per_mol: Some(ENTHALPY_DELTA_J_PER_MOL),
                    gibbs_free_energy_j_per_mol: Some(GIBBS_FREE_ENERGY_J_PER_MOL),
                    entropy_delta_j_per_mol_k: Some(ENTROPY_DELTA_J_PER_MOL_K),
                    activation_energy_j_per_mol: Some(ACTIVATION_ENERGY_J_PER_MOL),
                    is_reversible: false,
                });
            }
        }

        if let Some(acid_idx) = Self::species_index(snapshot, "CH3COOH") {
            let applicable_tiles = Self::tiles_with_species(
                snapshot,
                &[snapshot.species_table.species_ids[acid_idx]],
                MIN_MOLARITY,
            );

            if !applicable_tiles.is_empty() {
                update.reaction_directives.push(ReactionDirective {
                    reaction_id: REACTION_ACETIC_DISSOCIATION,
                    reaction_name: "acetic_acid_dissociation".to_string(),
                    reactant_a: "CH3COOH".to_string(),
                    reactant_b: String::new(),
                    product_a: "H+".to_string(),
                    product_b: "CH3COO-".to_string(),
                    applicable_tile_ids: applicable_tiles,
                    max_extent_per_second: f64::MAX,
                    rate_constant: ACETIC_DISSOCIATION_RATE,
                    effective_rate: ACETIC_DISSOCIATION_RATE,
                    enthalpy_delta_j_per_mol: Some(BUFFER_THERMAL_DELTA),
                    gibbs_free_energy_j_per_mol: Some(BUFFER_THERMAL_DELTA),
                    entropy_delta_j_per_mol_k: Some(BUFFER_THERMAL_DELTA),
                    activation_energy_j_per_mol: Some(BUFFER_THERMAL_DELTA),
                    is_reversible: true,
                });
            }
        }

        if let (Some(h_idx), Some(acetate_idx)) = (
            Self::species_index(snapshot, "H+"),
            Self::species_index(snapshot, "CH3COO-"),
        ) {
            let applicable_tiles = Self::tiles_with_species(
                snapshot,
                &[
                    snapshot.species_table.species_ids[h_idx],
                    snapshot.species_table.species_ids[acetate_idx],
                ],
                MIN_MOLARITY,
            );

            if !applicable_tiles.is_empty() {
                update.reaction_directives.push(ReactionDirective {
                    reaction_id: REACTION_ACETATE_RECOMBINATION,
                    reaction_name: "acetic_acid_recombination".to_string(),
                    reactant_a: "H+".to_string(),
                    reactant_b: "CH3COO-".to_string(),
                    product_a: "CH3COOH".to_string(),
                    product_b: String::new(),
                    applicable_tile_ids: applicable_tiles,
                    max_extent_per_second: f64::MAX,
                    rate_constant: ACETIC_RECOMBINATION_RATE,
                    effective_rate: ACETIC_RECOMBINATION_RATE,
                    enthalpy_delta_j_per_mol: Some(BUFFER_THERMAL_DELTA),
                    gibbs_free_energy_j_per_mol: Some(BUFFER_THERMAL_DELTA),
                    entropy_delta_j_per_mol_k: Some(BUFFER_THERMAL_DELTA),
                    activation_energy_j_per_mol: Some(BUFFER_THERMAL_DELTA),
                    is_reversible: true,
                });
            }
        }

        if let (Some(acid_idx), Some(oh_idx)) = (
            Self::species_index(snapshot, "CH3COOH"),
            Self::species_index(snapshot, "OH-"),
        ) {
            let applicable_tiles = Self::tiles_with_species(
                snapshot,
                &[
                    snapshot.species_table.species_ids[acid_idx],
                    snapshot.species_table.species_ids[oh_idx],
                ],
                MIN_MOLARITY,
            );

            if !applicable_tiles.is_empty() {
                update.reaction_directives.push(ReactionDirective {
                    reaction_id: REACTION_ACETIC_NEUTRALIZATION,
                    reaction_name: "acetic_acid_neutralization".to_string(),
                    reactant_a: "CH3COOH".to_string(),
                    reactant_b: "OH-".to_string(),
                    product_a: "CH3COO-".to_string(),
                    product_b: String::new(),
                    applicable_tile_ids: applicable_tiles,
                    max_extent_per_second: f64::MAX,
                    rate_constant: ACETIC_NEUTRALIZATION_RATE_CONSTANT,
                    effective_rate: ACETIC_NEUTRALIZATION_EFFECTIVE_RATE,
                    enthalpy_delta_j_per_mol: Some(ENTHALPY_DELTA_J_PER_MOL),
                    gibbs_free_energy_j_per_mol: Some(GIBBS_FREE_ENERGY_J_PER_MOL),
                    entropy_delta_j_per_mol_k: Some(ENTROPY_DELTA_J_PER_MOL_K),
                    activation_energy_j_per_mol: Some(ACTIVATION_ENERGY_J_PER_MOL),
                    is_reversible: false,
                });
            }
        }

        if update.reaction_directives.is_empty() {
            return Ok(update);
        }

        if config.verbose {
            log::debug!(
                "AcidBaseEvaluator: emitted {} reaction directives across {} tiles",
                update.reaction_directives.len(),
                snapshot.tile_count(),
            );
        }

        Ok(update)
    }

    fn is_available(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::{
        SemanticSnapshot, SpeciesAmount, SpeciesTableSnapshot, TileFlags,
    };
    use crate::SpeciesId;

    fn make_snapshot_with_h_oh() -> SemanticSnapshot {
        let mut snap = SemanticSnapshot::new(2, 2, 64, 128, 128);
        snap.species_table = SpeciesTableSnapshot {
            species_ids: vec![SpeciesId(1), SpeciesId(2)],
            species_names: vec!["H+".into(), "OH-".into()],
            diffusion_coefficients: vec![1.0, 1.0],
        };
        // Tile 0: both present
        snap.tiles[0].flags = TileFlags::HAS_FLUID;
        snap.tiles[0].species_mean_molarity = vec![
            SpeciesAmount::new(SpeciesId(1), 0.5),
            SpeciesAmount::new(SpeciesId(2), 0.3),
        ];
        // Tile 1: only H+
        snap.tiles[1].flags = TileFlags::HAS_FLUID;
        snap.tiles[1].species_mean_molarity = vec![
            SpeciesAmount::new(SpeciesId(1), 0.5),
        ];
        snap
    }

    fn make_snapshot_with_buffer_species() -> SemanticSnapshot {
        let mut snap = SemanticSnapshot::new(2, 2, 64, 128, 128);
        snap.species_table = SpeciesTableSnapshot {
            species_ids: vec![SpeciesId(1), SpeciesId(2), SpeciesId(3)],
            species_names: vec!["CH3COOH".into(), "H+".into(), "CH3COO-".into()],
            diffusion_coefficients: vec![1.0, 1.0, 1.0],
        };
        snap.tiles[0].flags = TileFlags::HAS_FLUID;
        snap.tiles[0].species_mean_molarity = vec![
            SpeciesAmount::new(SpeciesId(1), 0.35),
            SpeciesAmount::new(SpeciesId(2), 1.8e-5),
            SpeciesAmount::new(SpeciesId(3), 0.35),
        ];
        snap
    }

    fn make_snapshot_with_buffer_and_base_species() -> SemanticSnapshot {
        let mut snap = SemanticSnapshot::new(2, 2, 64, 128, 128);
        snap.species_table = SpeciesTableSnapshot {
            species_ids: vec![SpeciesId(1), SpeciesId(2), SpeciesId(3), SpeciesId(4)],
            species_names: vec![
                "CH3COOH".into(),
                "H+".into(),
                "CH3COO-".into(),
                "OH-".into(),
            ],
            diffusion_coefficients: vec![1.0, 1.0, 1.0, 1.0],
        };
        snap.tiles[0].flags = TileFlags::HAS_FLUID;
        snap.tiles[0].species_mean_molarity = vec![
            SpeciesAmount::new(SpeciesId(1), 0.35),
            SpeciesAmount::new(SpeciesId(2), 1.8e-5),
            SpeciesAmount::new(SpeciesId(3), 0.35),
            SpeciesAmount::new(SpeciesId(4), 0.1),
        ];
        snap
    }

    #[test]
    fn emits_directive_for_collocated_reactants() {
        let mut eval = AcidBaseEvaluator::new();
        let snap = make_snapshot_with_h_oh();
        let config = KineticsConfig::default();
        let update = eval.evaluate(&snap, &config).unwrap();

        assert_eq!(update.reaction_directives.len(), 1);
        let d = &update.reaction_directives[0];
        assert_eq!(d.reaction_id, REACTION_H_OH_WATER);
        assert_eq!(d.applicable_tile_ids, vec![0]); // only tile 0
        assert!((d.rate_constant - RATE_CONSTANT_298K).abs() < 1.0);
    }

    #[test]
    fn noop_when_species_missing() {
        let mut eval = AcidBaseEvaluator::new();
        let snap = SemanticSnapshot::new(2, 2, 64, 128, 128);
        let config = KineticsConfig::default();
        let update = eval.evaluate(&snap, &config).unwrap();
        assert!(update.is_noop());
    }

    #[test]
    fn emits_buffer_equilibrium_pair_when_acetate_system_present() {
        let mut eval = AcidBaseEvaluator::new();
        let snap = make_snapshot_with_buffer_species();
        let config = KineticsConfig::default();
        let update = eval.evaluate(&snap, &config).unwrap();

        assert_eq!(update.reaction_directives.len(), 2);
        let dissociation = update
            .reaction_directives
            .iter()
            .find(|directive| directive.reaction_id == REACTION_ACETIC_DISSOCIATION)
            .unwrap();
        assert_eq!(dissociation.reactant_a, "CH3COOH");
        assert_eq!(dissociation.reactant_b, "");
        assert_eq!(dissociation.product_a, "H+");
        assert_eq!(dissociation.product_b, "CH3COO-");

        let recombination = update
            .reaction_directives
            .iter()
            .find(|directive| directive.reaction_id == REACTION_ACETATE_RECOMBINATION)
            .unwrap();
        assert_eq!(recombination.reactant_a, "H+");
        assert_eq!(recombination.reactant_b, "CH3COO-");
        assert_eq!(recombination.product_a, "CH3COOH");
        assert_eq!(recombination.product_b, "");
    }

    #[test]
    fn emits_direct_neutralization_when_base_reaches_acetic_acid() {
        let mut eval = AcidBaseEvaluator::new();
        let snap = make_snapshot_with_buffer_and_base_species();
        let config = KineticsConfig::default();
        let update = eval.evaluate(&snap, &config).unwrap();

        let neutralization = update
            .reaction_directives
            .iter()
            .find(|directive| directive.reaction_id == REACTION_ACETIC_NEUTRALIZATION)
            .unwrap();
        assert_eq!(neutralization.reactant_a, "CH3COOH");
        assert_eq!(neutralization.reactant_b, "OH-");
        assert_eq!(neutralization.product_a, "CH3COO-");
        assert_eq!(neutralization.product_b, "");
    }
}

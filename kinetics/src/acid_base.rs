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

        // Locate H⁺ and OH⁻ in the species table
        let h_idx = snapshot.species_table.species_names.iter()
            .position(|n| n == "H+");
        let oh_idx = snapshot.species_table.species_names.iter()
            .position(|n| n == "OH-");

        let (h_idx, oh_idx) = match (h_idx, oh_idx) {
            (Some(h), Some(oh)) => (h, oh),
            _ => {
                // Species not present — nothing to react
                return Ok(SemanticUpdate::noop(valid_from, valid_until));
            }
        };

        let h_species_id = snapshot.species_table.species_ids[h_idx];
        let oh_species_id = snapshot.species_table.species_ids[oh_idx];

        // Find tiles where both H⁺ and OH⁻ are present above a threshold
        const MIN_MOLARITY: f64 = 1e-6;
        let mut applicable_tiles: Vec<u32> = Vec::new();

        for tile in &snapshot.tiles {
            if !tile.has_fluid() {
                continue;
            }
            let h_present = tile.species_mean_molarity.iter()
                .any(|a| a.species_id == h_species_id && a.value > MIN_MOLARITY);
            let oh_present = tile.species_mean_molarity.iter()
                .any(|a| a.species_id == oh_species_id && a.value > MIN_MOLARITY);
            if h_present && oh_present {
                applicable_tiles.push(tile.tile_id);
            }
        }

        if applicable_tiles.is_empty() {
            return Ok(SemanticUpdate::noop(valid_from, valid_until));
        }

        if config.verbose {
            log::debug!(
                "AcidBaseEvaluator: H⁺+OH⁻ reaction active in {} / {} tiles",
                applicable_tiles.len(),
                snapshot.tile_count(),
            );
        }

        let directive = ReactionDirective {
            reaction_id: REACTION_H_OH_WATER,
            reaction_name: "water_formation".to_string(),
            reactant_a: "H+".to_string(),
            reactant_b: "OH-".to_string(),
            applicable_tile_ids: applicable_tiles,
            max_extent_per_second: f64::MAX,
            rate_constant: RATE_CONSTANT_298K,
            effective_rate: EFFECTIVE_RATE,
            enthalpy_delta_j_per_mol: Some(ENTHALPY_DELTA_J_PER_MOL),
            gibbs_free_energy_j_per_mol: Some(GIBBS_FREE_ENERGY_J_PER_MOL),
            entropy_delta_j_per_mol_k: Some(ENTROPY_DELTA_J_PER_MOL_K),
            activation_energy_j_per_mol: Some(ACTIVATION_ENERGY_J_PER_MOL),
            is_reversible: false,
        };

        let mut update = SemanticUpdate::noop(valid_from, valid_until);
        update.reaction_directives.push(directive);
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
}

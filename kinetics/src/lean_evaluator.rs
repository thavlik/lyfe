//! Lean-backed rule evaluator.
//!
//! Serializes a `SemanticSnapshot` to JSON, invokes the Lean `lyfe-rules`
//! binary via `LeanBridge`, and deserializes the resulting reaction rules
//! into `ReactionDirective`s.

use serde::{Deserialize, Serialize};

use crate::{
    KineticsConfig, KineticsError, SemanticSnapshot, SemanticUpdate,
    engine::RuleEvaluator,
    lean_bridge::LeanBridge,
    update::{ReactionDirective, ReactionId},
};

// ---------------------------------------------------------------------------
// JSON types sent TO Lean
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct LeanSpeciesAmount {
    name: String,
    molarity: f64,
}

#[derive(Serialize)]
struct LeanTile {
    tile_id: u32,
    fluid_fraction: f64,
    mean_temperature: f64,
    species: Vec<LeanSpeciesAmount>,
}

#[derive(Serialize)]
struct LeanSnapshot {
    sim_time: f64,
    species_names: Vec<String>,
    tiles: Vec<LeanTile>,
}

// ---------------------------------------------------------------------------
// JSON types received FROM Lean
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct LeanReactionRule {
    reaction_name: String,
    reactant_a: String,
    #[serde(default)]
    reactant_b: String,
    #[serde(default)]
    product_a: String,
    #[serde(default)]
    product_b: String,
    rate_constant: f64,
    effective_rate: f64,
    #[serde(rename = "enthalpy_delta")]
    enthalpy_delta: f64,
    #[serde(rename = "gibbs_free_energy")]
    gibbs_free_energy: f64,
    #[serde(rename = "entropy_delta")]
    entropy_delta: f64,
    #[serde(rename = "activation_energy")]
    activation_energy: f64,
    is_reversible: bool,
    applicable_tile_ids: Vec<u32>,
}

#[derive(Deserialize)]
struct LeanEvalResult {
    rules: Vec<LeanReactionRule>,
    diagnostics: Vec<String>,
}

// ---------------------------------------------------------------------------
// LeanEvaluator
// ---------------------------------------------------------------------------

/// Rule evaluator backed by the Lean `lyfe-rules` binary.
pub struct LeanEvaluator {
    bridge: LeanBridge,
    evaluation_count: u64,
}

impl LeanEvaluator {
    /// Create a new evaluator from a discovered bridge.
    pub fn new(bridge: LeanBridge) -> Self {
        Self {
            bridge,
            evaluation_count: 0,
        }
    }

    /// Try to discover the Lean binary and create an evaluator.
    pub fn discover() -> Result<Self, KineticsError> {
        let bridge = LeanBridge::discover()?;
        Ok(Self::new(bridge))
    }

    /// Access the underlying bridge.
    pub fn bridge(&self) -> &LeanBridge {
        &self.bridge
    }

    /// Build the compact JSON snapshot that Lean receives.
    fn build_lean_snapshot(snapshot: &SemanticSnapshot) -> LeanSnapshot {
        let species_names = snapshot.species_table.species_names.clone();

        let tiles = snapshot
            .tiles
            .iter()
            .map(|t| {
                let species = t
                    .species_mean_molarity
                    .iter()
                    .filter_map(|a| {
                        // Look up the name by matching species_id → index
                        let idx = snapshot
                            .species_table
                            .species_ids
                            .iter()
                            .position(|&id| id == a.species_id)?;
                        Some(LeanSpeciesAmount {
                            name: species_names[idx].clone(),
                            molarity: a.value,
                        })
                    })
                    .collect();

                LeanTile {
                    tile_id: t.tile_id,
                    fluid_fraction: t.fluid_fraction as f64,
                    mean_temperature: t.mean_temperature_kelvin,
                    species,
                }
            })
            .collect();

        LeanSnapshot {
            sim_time: snapshot.sim_time_seconds,
            species_names,
            tiles,
        }
    }

    /// Convert a Lean rule into a `ReactionDirective`.
    fn to_directive(rule: LeanReactionRule, index: u32) -> ReactionDirective {
        ReactionDirective {
            reaction_id: ReactionId::new(index),
            reaction_name: rule.reaction_name,
            reactant_a: rule.reactant_a,
            reactant_b: rule.reactant_b,
            product_a: rule.product_a,
            product_b: rule.product_b,
            applicable_tile_ids: rule.applicable_tile_ids,
            max_extent_per_second: f64::MAX,
            rate_constant: rule.rate_constant,
            effective_rate: rule.effective_rate,
            enthalpy_delta_j_per_mol: Some(rule.enthalpy_delta),
            gibbs_free_energy_j_per_mol: Some(rule.gibbs_free_energy),
            entropy_delta_j_per_mol_k: Some(rule.entropy_delta),
            activation_energy_j_per_mol: Some(rule.activation_energy),
            is_reversible: rule.is_reversible,
        }
    }
}

impl RuleEvaluator for LeanEvaluator {
    fn name(&self) -> &str {
        "lean"
    }

    fn evaluate(
        &mut self,
        snapshot: &SemanticSnapshot,
        config: &KineticsConfig,
    ) -> Result<SemanticUpdate, KineticsError> {
        self.evaluation_count += 1;

        let valid_from = snapshot.sim_time_seconds;
        let valid_until = valid_from + config.min_evaluation_interval;

        // Serialize snapshot → JSON
        let lean_snapshot = Self::build_lean_snapshot(snapshot);
        let input_json = serde_json::to_string(&lean_snapshot).map_err(|e| {
            KineticsError::LeanError(format!("Failed to serialize snapshot: {}", e))
        })?;

        // Call Lean binary
        let output_json = self.bridge.evaluate_json(&input_json)?;

        // Deserialize response
        let result: LeanEvalResult =
            serde_json::from_str(&output_json).map_err(|e| {
                KineticsError::LeanError(format!(
                    "Failed to parse Lean output: {}. Raw: {}",
                    e,
                    &output_json[..output_json.len().min(200)]
                ))
            })?;

        // Log diagnostics from Lean
        for diag in &result.diagnostics {
            log::info!("[lean] {}", diag);
        }

        // Convert to SemanticUpdate
        let mut update = SemanticUpdate::noop(valid_from, valid_until);
        for (i, rule) in result.rules.into_iter().enumerate() {
            update
                .reaction_directives
                .push(Self::to_directive(rule, i as u32));
        }

        if config.verbose {
            log::debug!(
                "LeanEvaluator: {} reaction directives from Lean",
                update.reaction_directives.len()
            );
        }

        Ok(update)
    }

    fn is_available(&self) -> bool {
        true
    }
}

impl std::fmt::Debug for LeanEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LeanEvaluator")
            .field("bridge", &self.bridge)
            .field("evaluation_count", &self.evaluation_count)
            .finish()
    }
}

//! Integration with the kinetics crate.
//!
//! This module manages the connection between `fluidsim` and `kinetics`.
//! It handles:
//! - Once-per-second semantic evaluation timing
//! - Building semantic snapshots from simulation state
//! - Applying semantic updates to simulation parameters
//!
//! ## Update Cadence
//!
//! `fluidsim` invokes `kinetics` once per second of simulated time:
//! 1. Normal fine-grid simulation runs every frame/substep
//! 2. Once 1.0 simulated second has elapsed, build a semantic snapshot
//! 3. Call `kinetics.evaluate(&snapshot)`
//! 4. Apply the returned coefficients/directives
//! 5. Continue simulation with updated parameters
//!
//! ## Failure Behavior
//!
//! The system tolerates:
//! - No-op updates (kinetics returns empty update)
//! - Evaluation errors (continue with prior parameters)

use crate::gpu::GpuReactionRule;
use crate::semantic::{SemanticConfig, SemanticSnapshotBuilder, SemanticSnapshotInput};
use crate::solid::MaterialRegistry;
use crate::species::SpeciesRegistry;
use anyhow::Result;
use kinetics::{
    KineticsConfig, KineticsEngine, KineticsError, ReactionKineticsModel, SemanticUpdate,
};

/// State for kinetics integration.
///
/// This struct manages the timing and state for semantic updates.
pub struct KineticsIntegration {
    /// The kinetics engine
    engine: KineticsEngine,

    /// Semantic snapshot builder
    snapshot_builder: SemanticSnapshotBuilder,

    /// Configuration for semantic snapshots
    semantic_config: SemanticConfig,

    /// Time accumulator since last evaluation (simulated seconds)
    time_since_last_evaluation: f64,

    /// Interval between evaluations (simulated seconds)
    evaluation_interval: f64,

    /// Last returned semantic update (cached)
    last_update: Option<SemanticUpdate>,

    /// Last snapshot time
    last_snapshot_time: f64,

    /// Statistics
    stats: IntegrationStats,
}

/// Statistics about kinetics integration.
#[derive(Debug, Clone, Default)]
pub struct IntegrationStats {
    /// Total number of snapshots generated
    pub snapshots_generated: u64,
    /// Total number of evaluations attempted
    pub evaluations_attempted: u64,
    /// Total number of successful evaluations
    pub evaluations_succeeded: u64,
    /// Total number of failed evaluations
    pub evaluations_failed: u64,
    /// Total number of updates applied
    pub updates_applied: u64,
    /// Average snapshot generation time (CPU ms)
    pub avg_snapshot_time_ms: f64,
    /// Average evaluation time (CPU ms)
    pub avg_evaluation_time_ms: f64,
}

impl KineticsIntegration {
    /// Create a new integration with default configuration.
    pub fn new() -> Result<Self, KineticsError> {
        Self::with_config(KineticsConfig::default(), SemanticConfig::default())
    }

    /// Create a new integration with the given configurations.
    pub fn with_config(
        kinetics_config: KineticsConfig,
        semantic_config: SemanticConfig,
    ) -> Result<Self, KineticsError> {
        let engine = KineticsEngine::new(kinetics_config.clone())?;
        let snapshot_builder = SemanticSnapshotBuilder::new(semantic_config.clone());

        Ok(Self {
            engine,
            snapshot_builder,
            semantic_config,
            time_since_last_evaluation: 0.0,
            evaluation_interval: kinetics_config.min_evaluation_interval,
            last_update: None,
            last_snapshot_time: 0.0,
            stats: IntegrationStats::default(),
        })
    }

    /// Create a minimal no-op integration for testing.
    pub fn noop() -> Result<Self, KineticsError> {
        Self::with_config(KineticsConfig::noop(), SemanticConfig::default())
    }

    /// Update the time accumulator after a simulation step.
    ///
    /// Returns `true` if a semantic evaluation should be performed.
    pub fn accumulate_time(&mut self, dt: f64) -> bool {
        self.time_since_last_evaluation += dt;
        self.time_since_last_evaluation >= self.evaluation_interval
    }

    /// Check if a semantic evaluation is due.
    pub fn should_evaluate(&self) -> bool {
        self.time_since_last_evaluation >= self.evaluation_interval
    }

    /// Perform a semantic evaluation cycle.
    ///
    /// This is the main entry point called by `Simulation::step()` once per second.
    /// It builds a snapshot, evaluates it, and returns the update.
    pub fn evaluate(
        &mut self,
        input: SemanticSnapshotInput<'_>,
    ) -> Result<&SemanticUpdate, KineticsError> {
        // Build snapshot
        let snapshot_start = std::time::Instant::now();
        let dt_window = input.sim_time - self.last_snapshot_time;

        let snapshot = self.snapshot_builder.build(SemanticSnapshotInput {
            fine_width: input.fine_width,
            fine_height: input.fine_height,
            sim_time: input.sim_time,
            dt_window,
            concentrations: input.concentrations,
            solid_mask: input.solid_mask,
            material_ids: input.material_ids,
            temperatures: input.temperatures,
            species_registry: input.species_registry,
            material_registry: input.material_registry,
        });

        let snapshot_time = snapshot_start.elapsed().as_secs_f64() * 1000.0;
        self.stats.snapshots_generated += 1;
        self.update_avg_snapshot_time(snapshot_time);

        log::debug!(
            "Generated semantic snapshot: {} tiles, {:.2}ms",
            snapshot.tile_count(),
            snapshot_time
        );

        // Evaluate
        self.stats.evaluations_attempted += 1;
        let eval_start = std::time::Instant::now();

        let result = self.engine.evaluate(&snapshot);

        let eval_time = eval_start.elapsed().as_secs_f64() * 1000.0;
        self.update_avg_evaluation_time(eval_time);

        match result {
            Ok(update) => {
                self.stats.evaluations_succeeded += 1;

                log::debug!(
                    "Kinetics evaluation: {} updates, {:.2}ms",
                    update.update_count(),
                    eval_time
                );

                // Reset accumulator
                self.time_since_last_evaluation = 0.0;
                self.last_snapshot_time = input.sim_time;

                // Cache the update
                self.last_update = Some(update);

                Ok(self.last_update.as_ref().unwrap())
            }
            Err(e) => {
                self.stats.evaluations_failed += 1;
                log::warn!("Kinetics evaluation failed: {}", e);

                // On error, return last good update or a default no-op
                if let Some(ref update) = self.last_update {
                    Ok(update)
                } else {
                    // Create a fallback no-op update
                    self.last_update = Some(SemanticUpdate::noop(
                        input.sim_time,
                        input.sim_time + self.evaluation_interval,
                    ));
                    Ok(self.last_update.as_ref().unwrap())
                }
            }
        }
    }

    /// Get the last semantic update (if any).
    pub fn last_update(&self) -> Option<&SemanticUpdate> {
        self.last_update.as_ref()
    }

    /// Get the current evaluation interval.
    pub fn evaluation_interval(&self) -> f64 {
        self.evaluation_interval
    }

    /// Set the evaluation interval.
    pub fn set_evaluation_interval(&mut self, interval: f64) {
        self.evaluation_interval = interval.max(0.1); // Minimum 100ms
    }

    /// Get the time since last evaluation.
    pub fn time_since_last_evaluation(&self) -> f64 {
        self.time_since_last_evaluation
    }

    /// Get the underlying kinetics engine.
    pub fn engine(&self) -> &KineticsEngine {
        &self.engine
    }

    /// Get mutable access to the kinetics engine.
    pub fn engine_mut(&mut self) -> &mut KineticsEngine {
        &mut self.engine
    }

    /// Get integration statistics.
    pub fn stats(&self) -> &IntegrationStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = IntegrationStats::default();
        self.engine.reset_stats();
    }

    /// Get the semantic configuration.
    pub fn semantic_config(&self) -> &SemanticConfig {
        &self.semantic_config
    }

    /// Update semantic configuration.
    pub fn set_semantic_config(&mut self, config: SemanticConfig) {
        self.semantic_config = config.clone();
        self.snapshot_builder = SemanticSnapshotBuilder::new(config);
    }

    // --- Private helpers ---

    fn update_avg_snapshot_time(&mut self, time_ms: f64) {
        let n = self.stats.snapshots_generated as f64;
        self.stats.avg_snapshot_time_ms =
            (self.stats.avg_snapshot_time_ms * (n - 1.0) + time_ms) / n;
    }

    fn update_avg_evaluation_time(&mut self, time_ms: f64) {
        let n = self.stats.evaluations_attempted as f64;
        self.stats.avg_evaluation_time_ms =
            (self.stats.avg_evaluation_time_ms * (n - 1.0) + time_ms) / n;
    }
}

impl std::fmt::Debug for KineticsIntegration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KineticsIntegration")
            .field("evaluation_interval", &self.evaluation_interval)
            .field(
                "time_since_last_evaluation",
                &self.time_since_last_evaluation,
            )
            .field("has_update", &self.last_update.is_some())
            .field("stats", &self.stats)
            .finish()
    }
}

/// Apply a semantic update to simulation parameters.
///
/// This struct encapsulates the logic for converting semantic updates
/// into changes to simulation state.
pub struct SemanticUpdateApplicator {
    /// Whether to log applied updates
    verbose: bool,
    /// Global multiplier for kinetics-provided effective reaction rates.
    reaction_rate_scale: f32,
}

impl SemanticUpdateApplicator {
    /// Create a new applicator.
    pub fn new(reaction_rate_scale: f32) -> Self {
        Self {
            verbose: false,
            reaction_rate_scale,
        }
    }

    /// Enable verbose logging.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Apply a semantic update to simulation state.
    ///
    /// Returns GPU reaction rules extracted from reaction directives.
    pub fn apply(
        &self,
        update: &SemanticUpdate,
        species_registry: &SpeciesRegistry,
        _material_registry: &mut MaterialRegistry,
    ) -> (u32, Vec<GpuReactionRule>) {
        let mut gpu_rules: Vec<GpuReactionRule> = Vec::new();

        if update.is_noop() {
            if self.verbose {
                log::trace!("Applying no-op semantic update");
            }
            return (0, gpu_rules);
        }

        let mut applied_count = 0u32;

        // Apply global modifiers
        if (update.global_diffusion_multiplier - 1.0).abs() > 1e-9 {
            log::info!(
                "Global diffusion multiplier: {:.4}",
                update.global_diffusion_multiplier
            );
            applied_count += 1;
        }

        if update.global_temperature_adjustment.abs() > 1e-9 {
            log::info!(
                "Global temperature adjustment: {:.2}K",
                update.global_temperature_adjustment
            );
            applied_count += 1;
        }

        // Apply tile updates
        for tile_update in &update.tile_updates {
            if !tile_update.is_noop() {
                if self.verbose {
                    log::debug!("Tile {} update: {:?}", tile_update.tile_id, tile_update);
                }
                applied_count += 1;
            }
        }

        // Apply boundary updates
        for boundary_update in &update.boundary_updates {
            if !boundary_update.is_noop() {
                if self.verbose {
                    log::debug!(
                        "Boundary {} update: {:?}",
                        boundary_update.boundary_id,
                        boundary_update
                    );
                }
                applied_count += 1;
            }
        }

        // Convert reaction directives to GPU rules
        for directive in &update.reaction_directives {
            if self.verbose {
                log::debug!("Reaction directive: {:?}", directive);
            }

            // Map reaction ID to reactant species indices.
            // Currently we support the well-known H⁺ + OH⁻ → H₂O reaction.
            let rule = self.reaction_directive_to_gpu_rule(directive, species_registry);
            if let Some(r) = rule {
                gpu_rules.push(r);
            }
            applied_count += 1;
        }

        // Log diagnostics
        for diagnostic in &update.diagnostics {
            match diagnostic.level {
                kinetics::diagnostics::DiagnosticLevel::Error => {
                    log::error!("Kinetics: {}", diagnostic);
                }
                kinetics::diagnostics::DiagnosticLevel::Warning => {
                    log::warn!("Kinetics: {}", diagnostic);
                }
                kinetics::diagnostics::DiagnosticLevel::Info => {
                    log::info!("Kinetics: {}", diagnostic);
                }
                kinetics::diagnostics::DiagnosticLevel::Debug => {
                    log::debug!("Kinetics: {}", diagnostic);
                }
            }
        }

        log::debug!(
            "Applied {} semantic updates, {} GPU reaction rules",
            applied_count,
            gpu_rules.len()
        );
        (applied_count, gpu_rules)
    }

    /// Convert a kinetics ReactionDirective into a GPU-friendly ReactionRule.
    ///
    /// Returns None if the species mapping fails.
    fn reaction_directive_to_gpu_rule(
        &self,
        directive: &kinetics::ReactionDirective,
        species_registry: &SpeciesRegistry,
    ) -> Option<GpuReactionRule> {
        let lookup_optional_species = |name: &str| -> Option<Option<u32>> {
            if name.is_empty() {
                Some(None)
            } else {
                species_registry
                    .index_of_name(name)
                    .map(|idx| Some(idx as u32))
            }
        };

        let a_idx = species_registry.index_of_name(&directive.reactant_a);
        let b_idx = lookup_optional_species(&directive.reactant_b);
        let product_a_idx = lookup_optional_species(&directive.product_a);
        let product_b_idx = lookup_optional_species(&directive.product_b);
        let catalyst_idx = match directive.catalyst.as_deref() {
            Some(name) => species_registry
                .index_of_name(name)
                .map(|idx| Some(idx as u32)),
            None => Some(None),
        };

        let (a_idx, b_idx, product_a_idx, product_b_idx, catalyst_idx) = match (
            a_idx,
            b_idx,
            product_a_idx,
            product_b_idx,
            catalyst_idx,
        ) {
            (Some(a), Some(b), Some(product_a), Some(product_b), Some(catalyst)) => {
                (a as u32, b, product_a, product_b, catalyst)
            }
            _ => {
                log::warn!(
                    "Reaction '{}': species mapping failed for reactants/products/catalyst ('{}', '{}') -> ('{}', '{}') catalyst={:?}, skipping",
                    directive.reaction_name,
                    directive.reactant_a,
                    directive.reactant_b,
                    directive.product_a,
                    directive.product_b,
                    directive.catalyst,
                );
                return None;
            }
        };

        let (kinetic_model, rate, km_reactant_a, km_reactant_b) = match directive.kinetics_model {
            ReactionKineticsModel::MassAction => (
                0u32,
                (directive.effective_rate as f32) * self.reaction_rate_scale,
                0.0,
                0.0,
            ),
            ReactionKineticsModel::MichaelisMenten => {
                let Some(mm) = directive.michaelis_menten.as_ref() else {
                    log::warn!(
                        "Reaction '{}': Michaelis-Menten kinetics requested without parameters, skipping",
                        directive.reaction_name,
                    );
                    return None;
                };
                (
                    1u32,
                    (directive.rate_constant as f32)
                        * (directive.effective_rate as f32)
                        * self.reaction_rate_scale,
                    mm.km_reactant_a as f32,
                    mm.km_reactant_b.unwrap_or(0.0) as f32,
                )
            }
        };

        let enthalpy = directive.enthalpy_delta_j_per_mol.unwrap_or(0.0) as f32;
        let entropy = directive.entropy_delta_j_per_mol_k.unwrap_or(0.0) as f32;

        Some(GpuReactionRule::new(crate::gpu::GpuReactionRuleConfig {
            reactant_a_index: a_idx,
            reactant_b_index: b_idx,
            product_a_index: product_a_idx,
            product_b_index: product_b_idx,
            catalyst_index: catalyst_idx,
            kinetic_model,
            rate,
            km_reactant_a,
            km_reactant_b,
            enthalpy,
            entropy,
        }))
    }
}

impl Default for SemanticUpdateApplicator {
    fn default() -> Self {
        Self::new(1.0)
    }
}

//! The main kinetics engine API.
//!
//! The `KineticsEngine` is the primary interface for semantic evaluation.
//! It receives `SemanticSnapshot` inputs and produces `SemanticUpdate` outputs.

use crate::{
    KineticsConfig, KineticsError, SemanticSnapshot, SemanticUpdate,
    noop::NoopEvaluator,
    acid_base::AcidBaseEvaluator,
    lean_evaluator::LeanEvaluator,
    diagnostics::KineticsDiagnostic,
};

/// The rule evaluator trait for pluggable rule backends.
pub trait RuleEvaluator: Send + Sync {
    /// Get the name of this evaluator.
    fn name(&self) -> &str;
    
    /// Evaluate a semantic snapshot and produce an update.
    fn evaluate(
        &mut self,
        snapshot: &SemanticSnapshot,
        config: &KineticsConfig,
    ) -> Result<SemanticUpdate, KineticsError>;
    
    /// Check if this evaluator is available/functional.
    fn is_available(&self) -> bool;
}

/// The main kinetics engine.
///
/// This is the primary API for `fluidsim` to interact with the kinetics layer.
/// It manages the rule evaluation backend and provides a clean interface
/// that hides implementation details.
pub struct KineticsEngine {
    /// Configuration
    config: KineticsConfig,
    
    /// The rule evaluator backend
    evaluator: Box<dyn RuleEvaluator>,
    
    /// Statistics for monitoring
    stats: EngineStats,
    
    /// Last evaluation time (simulated)
    last_evaluation_time: f64,
}

/// Statistics about engine operation.
#[derive(Debug, Clone, Default)]
pub struct EngineStats {
    /// Total number of evaluations performed
    pub evaluation_count: u64,
    /// Total time spent in evaluation (CPU seconds)
    pub total_evaluation_time_seconds: f64,
    /// Average evaluation time (CPU seconds)
    pub avg_evaluation_time_seconds: f64,
    /// Number of no-op updates returned
    pub noop_count: u64,
    /// Number of errors encountered
    pub error_count: u64,
    /// Total updates produced
    pub total_updates_produced: u64,
}

impl KineticsEngine {
    /// Create a new kinetics engine with the given configuration.
    pub fn new(config: KineticsConfig) -> Result<Self, KineticsError> {
        config.validate().map_err(KineticsError::ConfigError)?;
        
        // Select the evaluator: Lean first, then Rust fallback, then noop
        let evaluator: Box<dyn RuleEvaluator> = if config.enable_lean {
            match LeanEvaluator::discover() {
                Ok(lean_eval) => {
                    log::info!(
                        "Kinetics engine using Lean rules ({})",
                        lean_eval.bridge().binary_path().display()
                    );
                    Box::new(lean_eval)
                }
                Err(e) => {
                    log::warn!("Lean binary not found: {}", e);
                    if config.enable_reaction_rules {
                        log::info!("Falling back to Rust acid-base evaluator");
                        Box::new(AcidBaseEvaluator::new())
                    } else {
                        log::info!("Kinetics engine using no-op evaluator");
                        Box::new(NoopEvaluator::new())
                    }
                }
            }
        } else if config.enable_reaction_rules {
            log::info!("Kinetics engine using acid-base evaluator");
            Box::new(AcidBaseEvaluator::new())
        } else {
            log::info!("Kinetics engine using no-op evaluator");
            Box::new(NoopEvaluator::new())
        };
        
        Ok(Self {
            config,
            evaluator,
            stats: EngineStats::default(),
            last_evaluation_time: 0.0,
        })
    }

    /// Create a kinetics engine with a custom evaluator.
    pub fn with_evaluator(
        config: KineticsConfig,
        evaluator: Box<dyn RuleEvaluator>,
    ) -> Result<Self, KineticsError> {
        config.validate().map_err(KineticsError::ConfigError)?;
        log::info!("Kinetics engine using custom evaluator: {}", evaluator.name());
        Ok(Self {
            config,
            evaluator,
            stats: EngineStats::default(),
            last_evaluation_time: 0.0,
        })
    }
    
    /// Create a minimal no-op engine for testing.
    pub fn noop() -> Result<Self, KineticsError> {
        Self::new(KineticsConfig::noop())
    }
    
    /// Evaluate a semantic snapshot and produce an update.
    ///
    /// This is the main entry point for `fluidsim` to invoke kinetics.
    /// It should be called once per second of simulated time.
    pub fn evaluate(&mut self, snapshot: &SemanticSnapshot) -> Result<SemanticUpdate, KineticsError> {
        // Validate snapshot
        snapshot.validate().map_err(KineticsError::SnapshotError)?;
        
        // Check if we should skip this evaluation (rate limiting)
        let time_since_last = snapshot.sim_time_seconds - self.last_evaluation_time;
        if time_since_last < self.config.min_evaluation_interval * 0.9 {
            // Return a no-op to avoid over-evaluation
            log::trace!(
                "Skipping evaluation: only {:.3}s since last (min: {:.3}s)",
                time_since_last,
                self.config.min_evaluation_interval
            );
            return Ok(SemanticUpdate::noop(
                snapshot.sim_time_seconds,
                snapshot.sim_time_seconds + self.config.min_evaluation_interval,
            ));
        }
        
        // Perform evaluation with timing
        let start = std::time::Instant::now();
        let result = self.evaluator.evaluate(snapshot, &self.config);
        let elapsed = start.elapsed().as_secs_f64();
        
        // Update statistics
        self.stats.evaluation_count += 1;
        self.stats.total_evaluation_time_seconds += elapsed;
        self.stats.avg_evaluation_time_seconds = 
            self.stats.total_evaluation_time_seconds / self.stats.evaluation_count as f64;
        
        match &result {
            Ok(update) => {
                self.last_evaluation_time = snapshot.sim_time_seconds;
                self.stats.total_updates_produced += update.update_count() as u64;
                if update.is_noop() {
                    self.stats.noop_count += 1;
                }
                
                if self.config.verbose {
                    log::debug!(
                        "Kinetics evaluation completed in {:.3}ms: {} updates",
                        elapsed * 1000.0,
                        update.update_count()
                    );
                }
            }
            Err(e) => {
                self.stats.error_count += 1;
                log::error!("Kinetics evaluation failed: {}", e);
            }
        }
        
        result
    }
    
    /// Get the current configuration.
    pub fn config(&self) -> &KineticsConfig {
        &self.config
    }
    
    /// Update the configuration.
    pub fn set_config(&mut self, config: KineticsConfig) -> Result<(), KineticsError> {
        config.validate().map_err(KineticsError::ConfigError)?;
        self.config = config;
        Ok(())
    }
    
    /// Get engine statistics.
    pub fn stats(&self) -> &EngineStats {
        &self.stats
    }
    
    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = EngineStats::default();
    }
    
    /// Check if the evaluator is Lean-backed.
    pub fn lean_available(&self) -> bool {
        self.evaluator.name() == "lean"
    }
    
    /// Get the evaluator name.
    pub fn evaluator_name(&self) -> &str {
        self.evaluator.name()
    }
    
    /// Check if the evaluator is available.
    pub fn is_available(&self) -> bool {
        self.evaluator.is_available()
    }
    
    /// Get the last evaluation time.
    pub fn last_evaluation_time(&self) -> f64 {
        self.last_evaluation_time
    }
    
    /// Manually trigger a diagnostic check.
    pub fn run_diagnostics(&self, snapshot: &SemanticSnapshot) -> Vec<KineticsDiagnostic> {
        let mut diagnostics = Vec::new();
        
        // Check for conservation issues
        if self.config.enable_conservation_checks {
            // TODO: Implement actual conservation checking
            diagnostics.push(KineticsDiagnostic::info(
                "Conservation check placeholder".to_string(),
            ));
        }
        
        // Check for numerical issues
        for tile in &snapshot.tiles {
            if tile.mean_temperature_kelvin < 0.0 {
                diagnostics.push(KineticsDiagnostic::error(
                    format!("Tile {} has negative temperature", tile.tile_id),
                ));
            }
            for amount in &tile.species_mean_molarity {
                if amount.value < 0.0 {
                    diagnostics.push(KineticsDiagnostic::warning(
                        format!("Tile {} has negative concentration for species {:?}", 
                            tile.tile_id, amount.species_id),
                    ));
                }
            }
        }
        
        diagnostics
    }
}

impl std::fmt::Debug for KineticsEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KineticsEngine")
            .field("config", &self.config)
            .field("evaluator", &self.evaluator.name())
            .field("lean_available", &self.lean_available())
            .field("stats", &self.stats)
            .finish()
    }
}

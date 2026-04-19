//! Configuration for the kinetics engine.

use std::path::PathBuf;

/// Configuration for the kinetics engine.
#[derive(Debug, Clone)]
pub struct KineticsConfig {
    /// Enable Lean-backed rules. Required whenever any semantic rules are enabled.
    pub enable_lean: bool,

    /// Path to Lean library/rules (if compiled separately)
    pub lean_library_path: Option<PathBuf>,

    /// Enable verbose logging for debugging
    pub verbose: bool,

    /// Maximum time for a single evaluation (seconds, CPU time)
    pub evaluation_timeout_seconds: f64,

    // --- Rule toggles ---
    /// Enable diffusion rate modification rules
    pub enable_diffusion_rules: bool,

    /// Enable reaction permission rules
    pub enable_reaction_rules: bool,

    /// Enable thermal source/sink rules
    pub enable_thermal_rules: bool,

    /// Enable miscibility/interaction rules
    pub enable_miscibility_rules: bool,

    /// Enable boundary permeability rules
    pub enable_boundary_rules: bool,

    // --- Performance tuning ---
    /// Minimum time between evaluations (seconds, simulated time)
    pub min_evaluation_interval: f64,

    /// Maximum tiles to process per evaluation (0 = no limit)
    pub max_tiles_per_evaluation: usize,

    /// Skip tiles below this activity threshold
    pub tile_activity_threshold: f64,

    // --- Conservation settings ---
    /// Enable conservation checking
    pub enable_conservation_checks: bool,

    /// Tolerance for conservation violations (relative)
    pub conservation_tolerance: f64,
}

impl Default for KineticsConfig {
    fn default() -> Self {
        Self {
            enable_lean: true,
            lean_library_path: None,
            verbose: false,
            evaluation_timeout_seconds: 0.1, // 100ms max per evaluation

            enable_diffusion_rules: true,
            enable_reaction_rules: true,
            enable_thermal_rules: true,
            enable_miscibility_rules: true,
            enable_boundary_rules: true,

            min_evaluation_interval: 1.0, // Once per simulated second
            max_tiles_per_evaluation: 0,  // No limit
            tile_activity_threshold: 0.0, // Process all tiles

            enable_conservation_checks: true,
            conservation_tolerance: 1e-6,
        }
    }
}

impl KineticsConfig {
    /// Create a new default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a minimal no-op configuration.
    pub fn noop() -> Self {
        Self {
            enable_lean: false,
            lean_library_path: None,
            verbose: false,
            evaluation_timeout_seconds: 0.01,
            enable_diffusion_rules: false,
            enable_reaction_rules: false,
            enable_thermal_rules: false,
            enable_miscibility_rules: false,
            enable_boundary_rules: false,
            min_evaluation_interval: 1.0,
            max_tiles_per_evaluation: 0,
            tile_activity_threshold: 0.0,
            enable_conservation_checks: false,
            conservation_tolerance: 1e-6,
        }
    }

    /// Create a verbose debugging configuration.
    pub fn debug() -> Self {
        Self {
            verbose: true,
            ..Self::default()
        }
    }

    /// Set whether to enable Lean-backed rules.
    pub fn with_lean(mut self, enable: bool) -> Self {
        self.enable_lean = enable;
        self
    }

    /// Set the Lean library path.
    pub fn with_lean_path(mut self, path: PathBuf) -> Self {
        self.lean_library_path = Some(path);
        self
    }

    /// Set verbose logging.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set the minimum evaluation interval.
    pub fn with_evaluation_interval(mut self, interval: f64) -> Self {
        self.min_evaluation_interval = interval;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        let any_rules_enabled = self.enable_diffusion_rules
            || self.enable_reaction_rules
            || self.enable_thermal_rules
            || self.enable_miscibility_rules
            || self.enable_boundary_rules;

        if self.evaluation_timeout_seconds <= 0.0 {
            return Err("Evaluation timeout must be positive".to_string());
        }
        if self.min_evaluation_interval <= 0.0 {
            return Err("Minimum evaluation interval must be positive".to_string());
        }
        if self.conservation_tolerance <= 0.0 {
            return Err("Conservation tolerance must be positive".to_string());
        }
        if any_rules_enabled && !self.enable_lean {
            return Err(
                "Lean-backed rules are required when semantic rule evaluation is enabled"
                    .to_string(),
            );
        }
        if let (true, Some(path)) = (self.enable_lean, self.lean_library_path.as_ref())
            && !path.exists()
        {
            return Err(format!("Lean library path does not exist: {:?}", path));
        }
        Ok(())
    }
}

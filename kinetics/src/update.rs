//! Semantic update types returned by kinetics evaluation.
//!
//! The semantic update is a compact artifact that `fluidsim` applies
//! to modify the GPU simulation parameters. It contains:
//! - Tilewise transport/diffusion modifiers
//! - Boundary permeability directives
//! - Reaction rate coefficients
//! - Thermal source/sink terms
//!
//! ## Design Principles
//!
//! - Updates are compact and GPU-friendly
//! - Updates are parameterization, NOT replacement state
//! - Updates are time-bounded (valid for a specific window)
//! - Updates can be partially applied or ignored without breaking simulation

use crate::SpeciesId;

/// A scalar value for a specific species.
#[derive(Debug, Clone)]
pub struct SpeciesScalar {
    /// The species this value refers to
    pub species_id: SpeciesId,
    /// The scalar value
    pub value: f64,
}

impl SpeciesScalar {
    pub fn new(species_id: SpeciesId, value: f64) -> Self {
        Self { species_id, value }
    }
}

/// A unique identifier for a reaction type/family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReactionId(pub u32);

impl ReactionId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }
}

/// A unique identifier for a set of related reactions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReactionSetId(pub u32);

impl ReactionSetId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }
}

/// How transport should behave at a boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoundaryTransportMode {
    /// Normal diffusive transport
    Normal,
    /// Impermeable barrier (no transport)
    Impermeable,
    /// Enhanced transport (e.g., active channel)
    Enhanced,
    /// Selective permeability (species-specific)
    Selective,
    /// One-way transport (from A to B only)
    Directional,
}

impl Default for BoundaryTransportMode {
    fn default() -> Self {
        Self::Normal
    }
}

/// Override for miscibility between species or phases.
#[derive(Debug, Clone)]
pub struct MiscibilityOverride {
    /// First species in the pair
    pub species_a: SpeciesId,
    /// Second species in the pair
    pub species_b: SpeciesId,
    /// Miscibility coefficient (0.0 = immiscible, 1.0 = fully miscible)
    pub coefficient: f64,
}

impl MiscibilityOverride {
    pub fn new(species_a: SpeciesId, species_b: SpeciesId, coefficient: f64) -> Self {
        Self {
            species_a,
            species_b,
            coefficient: coefficient.clamp(0.0, 1.0),
        }
    }
}

/// Update directives for a single coarse tile.
#[derive(Debug, Clone)]
pub struct TileUpdate {
    /// The tile ID this update applies to
    pub tile_id: u32,

    // --- Diffusion / transport modifiers ---
    
    /// Multipliers for effective diffusion per species (1.0 = no change)
    pub effective_diffusion_multipliers: Vec<SpeciesScalar>,
    /// Miscibility overrides for species pairs in this tile
    pub miscibility_overrides: Vec<MiscibilityOverride>,

    // --- Thermal behavior ---
    
    /// Heat source/sink rate (W/m³, positive = source, negative = sink)
    pub heat_source_watts_per_m3: Option<f64>,
    /// Effective heat capacity override (J/(kg·K))
    pub effective_heat_capacity: Option<f64>,
    /// Thermal diffusivity override (m²/s)
    pub thermal_diffusivity: Option<f64>,
    /// Target equilibrium temperature (Kelvin, if thermostating)
    pub target_temperature: Option<f64>,

    // --- Equilibrium / semantic targets ---
    
    /// Target concentrations to drive toward (mol/L)
    pub equilibrium_targets: Vec<SpeciesScalar>,

    // --- Reaction control ---
    
    /// Reaction sets enabled in this tile
    pub enabled_reaction_sets: Vec<ReactionSetId>,
    /// Reaction sets explicitly disabled in this tile
    pub disabled_reaction_sets: Vec<ReactionSetId>,
}

impl TileUpdate {
    /// Create a no-op update for a tile.
    pub fn noop(tile_id: u32) -> Self {
        Self {
            tile_id,
            effective_diffusion_multipliers: Vec::new(),
            miscibility_overrides: Vec::new(),
            heat_source_watts_per_m3: None,
            effective_heat_capacity: None,
            thermal_diffusivity: None,
            target_temperature: None,
            equilibrium_targets: Vec::new(),
            enabled_reaction_sets: Vec::new(),
            disabled_reaction_sets: Vec::new(),
        }
    }

    /// Check if this update has any effect.
    pub fn is_noop(&self) -> bool {
        self.effective_diffusion_multipliers.is_empty()
            && self.miscibility_overrides.is_empty()
            && self.heat_source_watts_per_m3.is_none()
            && self.effective_heat_capacity.is_none()
            && self.thermal_diffusivity.is_none()
            && self.target_temperature.is_none()
            && self.equilibrium_targets.is_empty()
            && self.enabled_reaction_sets.is_empty()
            && self.disabled_reaction_sets.is_empty()
    }
}

/// Update directives for a boundary between tiles.
#[derive(Debug, Clone)]
pub struct BoundaryUpdate {
    /// The boundary ID this update applies to
    pub boundary_id: u32,
    /// Overall permeability multiplier (1.0 = no change)
    pub permeability_multiplier: f64,
    /// Transport mode for this boundary
    pub transport_mode: BoundaryTransportMode,
    /// Species-specific permeability multipliers
    pub species_specific_permeability: Vec<SpeciesScalar>,
    /// Thermal conductance multiplier (1.0 = no change)
    pub thermal_conductance_multiplier: f64,
}

impl BoundaryUpdate {
    /// Create a no-op update for a boundary.
    pub fn noop(boundary_id: u32) -> Self {
        Self {
            boundary_id,
            permeability_multiplier: 1.0,
            transport_mode: BoundaryTransportMode::Normal,
            species_specific_permeability: Vec::new(),
            thermal_conductance_multiplier: 1.0,
        }
    }

    /// Check if this update has any effect.
    pub fn is_noop(&self) -> bool {
        (self.permeability_multiplier - 1.0).abs() < 1e-9
            && self.transport_mode == BoundaryTransportMode::Normal
            && self.species_specific_permeability.is_empty()
            && (self.thermal_conductance_multiplier - 1.0).abs() < 1e-9
    }
}

/// A directive describing a reaction to apply.
#[derive(Debug, Clone)]
pub struct ReactionDirective {
    /// The reaction this directive controls
    pub reaction_id: ReactionId,
    /// Human-readable name for the reaction
    pub reaction_name: String,
    /// Reactant A species name (looked up in species registry)
    pub reactant_a: String,
    /// Reactant B species name (looked up in species registry)
    pub reactant_b: String,
    /// Product A species name (empty when the reaction has no first product)
    pub product_a: String,
    /// Product B species name (empty when the reaction has no second product)
    pub product_b: String,
    /// Tiles where this reaction should be applied
    pub applicable_tile_ids: Vec<u32>,
    /// Maximum extent per second (limits reaction rate)
    pub max_extent_per_second: f64,
    /// Physical rate constant (L·mol⁻¹·s⁻¹)
    pub rate_constant: f64,
    /// Effective rate for GPU simulation (pre-scaled for stability)
    pub effective_rate: f64,
    /// Enthalpy change ΔH (J/mol, negative = exothermic → heats up)
    pub enthalpy_delta_j_per_mol: Option<f64>,
    /// Gibbs free energy change ΔG (J/mol, negative = spontaneous)
    pub gibbs_free_energy_j_per_mol: Option<f64>,
    /// Entropy change ΔS (J/(mol·K))
    pub entropy_delta_j_per_mol_k: Option<f64>,
    /// Activation energy (J/mol)
    pub activation_energy_j_per_mol: Option<f64>,
    /// Whether this reaction is reversible
    pub is_reversible: bool,
}

impl ReactionDirective {
    /// Create a new reaction directive.
    pub fn new(reaction_id: ReactionId) -> Self {
        Self {
            reaction_id,
            reaction_name: String::new(),
            reactant_a: String::new(),
            reactant_b: String::new(),
            product_a: String::new(),
            product_b: String::new(),
            applicable_tile_ids: Vec::new(),
            max_extent_per_second: f64::MAX,
            rate_constant: 1.0,
            effective_rate: 1.0,
            enthalpy_delta_j_per_mol: None,
            gibbs_free_energy_j_per_mol: None,
            entropy_delta_j_per_mol_k: None,
            activation_energy_j_per_mol: None,
            is_reversible: false,
        }
    }

    /// Set the tiles where this reaction applies.
    pub fn with_tiles(mut self, tiles: Vec<u32>) -> Self {
        self.applicable_tile_ids = tiles;
        self
    }

    /// Set the rate constant.
    pub fn with_rate_constant(mut self, k: f64) -> Self {
        self.rate_constant = k;
        self
    }
}

/// The complete semantic update returned by kinetics evaluation.
///
/// This is the output of `KineticsEngine::evaluate()`.
/// It contains compact directives for `fluidsim` to apply.
#[derive(Debug, Clone)]
pub struct SemanticUpdate {
    /// Simulation time from which this update is valid (seconds)
    pub valid_from_time_seconds: f64,
    /// Simulation time until which this update is valid (seconds)
    pub valid_until_time_seconds: f64,

    // --- Tilewise updates ---
    
    /// Updates for specific tiles (sparse: only tiles with changes)
    pub tile_updates: Vec<TileUpdate>,

    // --- Boundary updates ---
    
    /// Updates for specific boundaries (sparse: only boundaries with changes)
    pub boundary_updates: Vec<BoundaryUpdate>,

    // --- Reaction directives ---
    
    /// Reaction directives to apply
    pub reaction_directives: Vec<ReactionDirective>,

    // --- Global modifiers ---
    
    /// Global diffusion rate multiplier (1.0 = no change)
    pub global_diffusion_multiplier: f64,
    /// Global temperature adjustment (Kelvin, additive)
    pub global_temperature_adjustment: f64,

    // --- Diagnostics ---
    
    /// Diagnostic messages from the evaluation
    pub diagnostics: Vec<crate::KineticsDiagnostic>,
}

impl SemanticUpdate {
    /// Create a no-op update.
    pub fn noop(valid_from: f64, valid_until: f64) -> Self {
        Self {
            valid_from_time_seconds: valid_from,
            valid_until_time_seconds: valid_until,
            tile_updates: Vec::new(),
            boundary_updates: Vec::new(),
            reaction_directives: Vec::new(),
            global_diffusion_multiplier: 1.0,
            global_temperature_adjustment: 0.0,
            diagnostics: Vec::new(),
        }
    }

    /// Check if this update has any effect.
    pub fn is_noop(&self) -> bool {
        self.tile_updates.iter().all(|u| u.is_noop())
            && self.boundary_updates.iter().all(|u| u.is_noop())
            && self.reaction_directives.is_empty()
            && (self.global_diffusion_multiplier - 1.0).abs() < 1e-9
            && self.global_temperature_adjustment.abs() < 1e-9
    }

    /// Count the number of non-trivial updates.
    pub fn update_count(&self) -> usize {
        let tile_count = self.tile_updates.iter().filter(|u| !u.is_noop()).count();
        let boundary_count = self.boundary_updates.iter().filter(|u| !u.is_noop()).count();
        tile_count + boundary_count + self.reaction_directives.len()
    }

    /// Add a tile update.
    pub fn with_tile_update(mut self, update: TileUpdate) -> Self {
        self.tile_updates.push(update);
        self
    }

    /// Add a boundary update.
    pub fn with_boundary_update(mut self, update: BoundaryUpdate) -> Self {
        self.boundary_updates.push(update);
        self
    }

    /// Add a reaction directive.
    pub fn with_reaction(mut self, directive: ReactionDirective) -> Self {
        self.reaction_directives.push(directive);
        self
    }

    /// Add a diagnostic message.
    pub fn with_diagnostic(mut self, diagnostic: crate::KineticsDiagnostic) -> Self {
        self.diagnostics.push(diagnostic);
        self
    }
}

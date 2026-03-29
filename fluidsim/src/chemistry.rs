//! Chemical evolution rules and reaction stubs.
//!
//! This module provides the interface for per-step chemical evolution.
//! Currently implements a no-op, but the interface allows for future
//! reaction chemistry extensions.

use crate::species::SpeciesRegistry;

/// A rule that evolves chemical concentrations each simulation step.
///
/// Implementations can:
/// - Transform species (reactions)
/// - Apply thermodynamic constraints
/// - Modify concentrations based on local conditions
///
/// The GPU simulation will call this rule's shader or CPU fallback
/// after the diffusion pass each step.
pub trait ChemicalEvolutionRule: Send + Sync {
    /// Get the name of this evolution rule.
    fn name(&self) -> &str;

    /// Called once at simulation start to validate species requirements.
    fn validate(&self, registry: &SpeciesRegistry) -> Result<(), String>;

    /// Get GLSL shader code for GPU evolution, or None if CPU-only.
    /// The shader should implement a function:
    /// ```glsl
    /// void evolve_chemistry(uint cell_index, float dt);
    /// ```
    fn gpu_shader_source(&self) -> Option<&str> {
        None
    }

    /// CPU fallback for evolution. Called only if no GPU shader is provided
    /// or for debugging. Receives mutable access to concentration data.
    ///
    /// `concentrations` layout: [species_index][cell_index]
    fn evolve_cpu(
        &self,
        _concentrations: &mut [Vec<f32>],
        _solid_mask: &[bool],
        _dt: f32,
        _registry: &SpeciesRegistry,
    ) {
        // Default: no-op
    }
}

/// A no-op evolution rule that preserves all concentrations unchanged.
/// This is the default for the demo.
#[derive(Debug, Clone, Default)]
pub struct NoOpEvolution;

impl ChemicalEvolutionRule for NoOpEvolution {
    fn name(&self) -> &str {
        "no-op"
    }

    fn validate(&self, _registry: &SpeciesRegistry) -> Result<(), String> {
        Ok(())
    }

    fn gpu_shader_source(&self) -> Option<&str> {
        // No-op doesn't need a shader - diffusion handles transport
        None
    }
}

// Future: example of what a reaction rule might look like
//
// pub struct AcidBaseNeutralization;
//
// impl ChemicalEvolutionRule for AcidBaseNeutralization {
//     fn name(&self) -> &str { "acid-base" }
//
//     fn validate(&self, registry: &SpeciesRegistry) -> Result<(), String> {
//         // Ensure H+, OH-, and H2O are registered
//         registry.id_of("H+").ok_or("H+ not registered")?;
//         registry.id_of("OH-").ok_or("OH- not registered")?;
//         registry.id_of("H2O").ok_or("H2O not registered")?;
//         Ok(())
//     }
//
//     fn gpu_shader_source(&self) -> Option<&str> {
//         Some(r#"
//             void evolve_chemistry(uint cell_index, float dt) {
//                 float h = concentrations[H_INDEX][cell_index];
//                 float oh = concentrations[OH_INDEX][cell_index];
//                 float reacted = min(h, oh) * reaction_rate * dt;
//                 concentrations[H_INDEX][cell_index] -= reacted;
//                 concentrations[OH_INDEX][cell_index] -= reacted;
//                 concentrations[H2O_INDEX][cell_index] += reacted;
//             }
//         "#)
//     }
// }

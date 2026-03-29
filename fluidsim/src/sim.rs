//! Main simulation state and API.
//!
//! The `Simulation` struct is the primary interface for running the fluid
//! simulation. It orchestrates GPU compute passes, handles inspection
//! requests, and provides access to simulation state.

use crate::chemistry::{ChemicalEvolutionRule, NoOpEvolution};
use crate::gpu::GpuSimulation;
use crate::grid::Grid;
use crate::inspect::{InspectionResult, Inspector};
use crate::scenario::{Scenario, create_demo_scenario};
use crate::solid::MaterialRegistry;
use crate::species::SpeciesRegistry;

use anyhow::Result;


/// Configuration for the simulation.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Grid width in fine cells
    pub width: u32,
    /// Grid height in fine cells
    pub height: u32,
    /// Base diffusion rate multiplier
    pub diffusion_rate: f32,
    /// Number of diffusion sub-steps per frame
    pub diffusion_substeps: u32,
    /// Default mip factor for inspection
    pub inspection_mip: u32,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            diffusion_rate: 0.2,
            diffusion_substeps: 4,
            inspection_mip: 8,
        }
    }
}

/// Render state extracted from the simulation for the renderer.
#[derive(Debug)]
pub struct RenderState {
    /// Grid dimensions
    pub width: u32,
    pub height: u32,
    /// Number of species
    pub species_count: usize,
    /// Concentration data: `[species][cell]`
    pub concentrations: Vec<Vec<f32>>,
    /// Solid mask: 1 = solid, 0 = fluid
    pub solid_mask: Vec<u32>,
    /// Material IDs
    pub material_ids: Vec<u32>,
}

/// The main simulation state.
pub struct Simulation {
    /// Grid dimensions and utilities
    pub grid: Grid,
    /// Species registry
    species_registry: SpeciesRegistry,
    /// Material registry
    material_registry: MaterialRegistry,
    /// GPU simulation state
    gpu: GpuSimulation,
    /// Inspector for mouse hover
    inspector: Inspector,
    /// Chemical evolution rule
    evolution_rule: Box<dyn ChemicalEvolutionRule>,
    /// Configuration
    config: SimulationConfig,
    /// Cached concentration data for inspection
    cached_concentrations: Option<Vec<Vec<f32>>>,
    /// Cached solid mask
    cached_solid_mask: Option<Vec<u32>>,
    /// Cached material IDs
    cached_material_ids: Option<Vec<u32>>,
    /// Whether cached data needs refresh
    cache_dirty: bool,
    /// Simulation time accumulator
    time: f64,
    /// Step counter
    step_count: u64,
    /// Is simulation paused
    paused: bool,
}

impl Simulation {
    /// Create a new simulation from a scenario.
    pub fn from_scenario(scenario: Scenario, config: SimulationConfig) -> Result<Self> {
        let concentrations = scenario.compile_concentrations();
        let solid_mask = scenario.compile_solid_mask();
        let material_ids = scenario.compile_material_ids();
        let diffusion_coeffs = scenario.species_registry.diffusion_coefficients();

        let gpu = GpuSimulation::new(
            scenario.grid.width,
            scenario.grid.height,
            scenario.species_registry.count(),
            &concentrations,
            &solid_mask,
            &material_ids,
            &diffusion_coeffs,
        )?;

        Ok(Self {
            grid: scenario.grid,
            species_registry: scenario.species_registry,
            material_registry: scenario.material_registry,
            gpu,
            inspector: Inspector::new(config.inspection_mip),
            evolution_rule: Box::new(NoOpEvolution),
            config,
            cached_concentrations: None,
            cached_solid_mask: None,
            cached_material_ids: None,
            cache_dirty: true,
            time: 0.0,
            step_count: 0,
            paused: false,
        })
    }

    /// Create the demo simulation.
    pub fn new_demo(config: SimulationConfig) -> Result<Self> {
        let scenario = create_demo_scenario(config.width, config.height);
        Self::from_scenario(scenario, config)
    }

    /// Step the simulation forward by dt seconds.
    pub fn step(&mut self, dt: f32) -> Result<()> {
        if self.paused {
            return Ok(());
        }

        // Run diffusion substeps
        let substep_dt = dt / self.config.diffusion_substeps as f32;
        for _ in 0..self.config.diffusion_substeps {
            self.gpu.step(substep_dt, self.config.diffusion_rate)?;
        }

        // Evolution rule would be applied here
        // Currently no-op, but the hook is in place

        self.time += dt as f64;
        self.step_count += 1;
        self.cache_dirty = true;

        Ok(())
    }

    /// Refresh cached data from GPU.
    fn refresh_cache(&mut self) -> Result<()> {
        if !self.cache_dirty {
            return Ok(());
        }

        self.cached_concentrations = Some(self.gpu.read_concentrations()?);
        
        if self.cached_solid_mask.is_none() {
            self.cached_solid_mask = Some(self.gpu.read_solid_mask()?);
        }
        if self.cached_material_ids.is_none() {
            self.cached_material_ids = Some(self.gpu.read_material_ids()?);
        }

        self.cache_dirty = false;
        Ok(())
    }

    /// Inspect a coarse cell at the given screen position.
    pub fn inspect(&mut self, screen_x: f32, screen_y: f32) -> Result<InspectionResult> {
        self.inspect_with_mip(screen_x, screen_y, self.config.inspection_mip)
    }

    /// Inspect a coarse cell with a specific mip factor.
    pub fn inspect_with_mip(&mut self, screen_x: f32, screen_y: f32, mip: u32) -> Result<InspectionResult> {
        self.refresh_cache()?;

        let coord = self.inspector.screen_to_coarse(screen_x, screen_y, mip);

        let result = self.inspector.inspect(
            coord,
            self.cached_concentrations.as_ref().unwrap(),
            self.cached_solid_mask.as_ref().unwrap(),
            self.cached_material_ids.as_ref().unwrap(),
            self.grid.width,
            self.grid.height,
            &self.species_registry,
            &self.material_registry,
        );

        Ok(result)
    }

    /// Get render state for visualization.
    pub fn render_state(&mut self) -> Result<RenderState> {
        self.refresh_cache()?;

        Ok(RenderState {
            width: self.grid.width,
            height: self.grid.height,
            species_count: self.species_registry.count(),
            concentrations: self.cached_concentrations.clone().unwrap(),
            solid_mask: self.cached_solid_mask.clone().unwrap(),
            material_ids: self.cached_material_ids.clone().unwrap(),
        })
    }

    /// Get the species registry.
    pub fn species_registry(&self) -> &SpeciesRegistry {
        &self.species_registry
    }

    /// Get the material registry.
    pub fn material_registry(&self) -> &MaterialRegistry {
        &self.material_registry
    }

    /// Get the current simulation time.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Get the step count.
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Check if paused.
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Toggle pause state.
    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    /// Set pause state.
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    /// Set the chemical evolution rule.
    pub fn set_evolution_rule(&mut self, rule: Box<dyn ChemicalEvolutionRule>) {
        self.evolution_rule = rule;
    }

    /// Get grid dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.grid.width, self.grid.height)
    }

    /// Get configuration.
    pub fn config(&self) -> &SimulationConfig {
        &self.config
    }

    /// Update diffusion rate.
    pub fn set_diffusion_rate(&mut self, rate: f32) {
        self.config.diffusion_rate = rate;
    }

    /// Update inspection mip factor.
    pub fn set_inspection_mip(&mut self, mip: u32) {
        self.config.inspection_mip = mip;
        self.inspector = Inspector::new(mip);
    }

    /// Resize the simulation grid.
    /// Note: This reinitializes the simulation.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        let scenario = create_demo_scenario(width, height);
        let mut config = self.config.clone();
        config.width = width;
        config.height = height;
        
        *self = Self::from_scenario(scenario, config)?;
        Ok(())
    }
}

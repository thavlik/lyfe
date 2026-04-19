//! Main simulation state and API.
//!
//! The `Simulation` struct is the primary interface for running the fluid
//! simulation. It orchestrates GPU compute passes, handles inspection
//! requests, and provides access to simulation state.

use crate::chemistry::{ChemicalEvolutionRule, NoOpEvolution};
use crate::coarse::CoarseCellData;
use crate::enzyme::{EnzymeEntity, EnzymeField};
use crate::gpu::{GpuRenderBuffers, GpuSimulation, SharedGpuContext};
use crate::grid::Grid;
use crate::inspect::{InspectionResult, Inspector};
use crate::kinetics_integration::{KineticsIntegration, SemanticUpdateApplicator};
use crate::leak::LeakChannel;
use crate::scenario::{
    Scenario, create_acid_base_scenario, create_buffers_scenario, create_catalyst_scenario,
    create_demo_scenario, create_enzyme_scenario, create_leak_scenario,
};
use crate::solid::MaterialRegistry;
use crate::species::SpeciesRegistry;

use anyhow::Result;
use ash::vk;
use std::time::Duration;

/// Configuration for the simulation.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Grid width in fine cells
    pub width: u32,
    /// Grid height in fine cells
    pub height: u32,
    /// Base diffusion rate multiplier
    pub diffusion_rate: f32,
    /// Base thermal diffusion multiplier for the temperature field
    pub thermal_diffusion_rate: f32,
    /// Strength of local electroneutrality projection after transport/reaction.
    /// `1.0` enforces exact per-cell neutrality every pass.
    pub charge_correction_strength: f32,
    /// Minimum diffusion sub-steps per frame (actual count is computed
    /// dynamically from the CFL stability condition and may be higher)
    pub diffusion_substeps: u32,
    /// Default mip factor for inspection
    pub inspection_mip: u32,
    /// Simulation time-scale multiplier.  Simulated dt per frame equals
    /// `wall_dt * time_scale`, so 20.0 means the simulation runs 20×
    /// faster than real-time.  Diffusion substeps are auto-scaled to
    /// maintain numerical stability.
    pub time_scale: f32,
    /// Global multiplier applied to kinetics-provided effective reaction rates
    /// before they are uploaded to the GPU. This lets fluidsim run visually
    /// faster chemistry than lab-scale rates without changing rule definitions.
    pub reaction_rate_scale: f32,
    /// Maximum wall-clock dt to accept per frame (seconds).  Prevents a
    /// spiral-of-death when a frame takes too long.
    pub max_frame_dt: f32,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            diffusion_rate: 5.0,
            thermal_diffusion_rate: 3.0,
            charge_correction_strength: 1.0,
            diffusion_substeps: 4,
            inspection_mip: 8,
            time_scale: 20.0,
            reaction_rate_scale: 8.0,
            max_frame_dt: 1.0 / 15.0,
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
    /// Temperature per cell in Kelvin
    pub temperatures: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct PreparedSimulationStep {
    substeps: u32,
    substep_dt: f32,
    sim_dt: f32,
    diffusion_rate: f32,
    charge_correction_strength: f32,
    reaction_dt: f32,
    thermal_diffusivity: f32,
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
    /// Leak channels embedded in the current scenario.
    leak_channels: Vec<LeakChannel>,
    /// Enzyme entities embedded in the current scenario.
    enzyme_entities: Vec<EnzymeEntity>,
    /// Motion heuristics for the enzyme field.
    enzyme_field: Option<EnzymeField>,
    /// Configuration
    config: SimulationConfig,
    /// Cached concentration data for inspection
    cached_concentrations: Option<Vec<Vec<f32>>>,
    /// Cached solid mask
    cached_solid_mask: Option<Vec<u32>>,
    /// Cached material IDs
    cached_material_ids: Option<Vec<u32>>,
    /// Cached temperatures
    cached_temperatures: Option<Vec<f32>>,
    /// Whether cached data needs refresh
    cache_dirty: bool,
    /// Simulation time accumulator
    time: f64,
    /// Step counter
    step_count: u64,
    /// Is simulation paused
    paused: bool,
    /// Kinetics integration (semantic update engine)
    kinetics: Option<KineticsIntegration>,
    /// Semantic update applicator
    update_applicator: SemanticUpdateApplicator,
    /// Whether the next completed frame should trigger a kinetics evaluation.
    pending_kinetics_evaluation: bool,
    /// Whether CPU readback-driven runtime features are allowed.
    runtime_readbacks_enabled: bool,
    /// Whether low-frequency async coarse inspection readbacks are allowed.
    inspection_readbacks_enabled: bool,
}

fn effective_charge_correction_strength(
    configured_strength: f32,
    leak_channels: &[LeakChannel],
) -> f32 {
    if leak_channels.is_empty() {
        configured_strength
    } else {
        0.0
    }
}

impl Simulation {
    const MIN_KINETICS_INTERVAL_SECONDS: f64 = 1.0;
    const MAX_KINETICS_INTERVAL_SECONDS: f64 = 30.0;
    const MAX_CHARGE_RELAXATION_DIFFUSION_SCALE: f32 = 8.0;

    /// Create a new simulation from a scenario.
    pub fn from_scenario(scenario: Scenario, config: SimulationConfig) -> Result<Self> {
        Self::from_scenario_with_gpu_context(scenario, config, None)
    }

    pub fn from_scenario_with_shared_gpu_context(
        scenario: Scenario,
        config: SimulationConfig,
        context: SharedGpuContext,
    ) -> Result<Self> {
        Self::from_scenario_with_gpu_context(scenario, config, Some(context))
    }

    fn from_scenario_with_gpu_context(
        scenario: Scenario,
        config: SimulationConfig,
        context: Option<SharedGpuContext>,
    ) -> Result<Self> {
        let concentrations = scenario.compile_concentrations();
        let solid_mask = scenario.compile_solid_mask();
        let material_ids = scenario.compile_material_ids();
        let temperatures = scenario.compile_temperatures();
        let diffusion_coeffs = scenario.species_registry.diffusion_coefficients();
        let species_charges = scenario.species_registry.charges();

        let mut gpu = if let Some(context) = context {
            GpuSimulation::new_with_shared_context(
                context,
                scenario.grid.width,
                scenario.grid.height,
                scenario.species_registry.count(),
                &concentrations,
                &solid_mask,
                &material_ids,
                &diffusion_coeffs,
                &species_charges,
            )?
        } else {
            GpuSimulation::new(
                scenario.grid.width,
                scenario.grid.height,
                scenario.species_registry.count(),
                &concentrations,
                &solid_mask,
                &material_ids,
                &diffusion_coeffs,
                &species_charges,
            )?
        };

        gpu.init_reaction_pipeline(&temperatures)?;
        if !scenario.leak_channels.is_empty() {
            gpu.init_leak_pipeline(
                &scenario.leak_channels,
                &scenario.species_registry,
                &solid_mask,
            )?;
        }
        if !scenario.enzyme_entities.is_empty() {
            gpu.init_enzyme_pipeline(&scenario.enzyme_entities, &scenario.species_registry)?;
        }
        let runtime_readbacks_enabled = !gpu.uses_shared_vulkan_context();
        let inspection_readbacks_enabled = true;
        let reaction_rate_scale = config.reaction_rate_scale;

        let mut simulation = Self {
            grid: scenario.grid,
            species_registry: scenario.species_registry,
            material_registry: scenario.material_registry,
            gpu,
            inspector: Inspector::new(config.inspection_mip),
            evolution_rule: Box::new(NoOpEvolution),
            leak_channels: scenario.leak_channels,
            enzyme_entities: scenario.enzyme_entities,
            enzyme_field: scenario.enzyme_field,
            config,
            cached_concentrations: Some(concentrations.clone()),
            cached_solid_mask: Some(solid_mask.clone()),
            cached_material_ids: Some(material_ids.clone()),
            cached_temperatures: Some(temperatures.clone()),
            cache_dirty: false,
            time: 0.0,
            step_count: 0,
            paused: false,
            kinetics: Some(KineticsIntegration::new()?),
            update_applicator: SemanticUpdateApplicator::new(reaction_rate_scale),
            pending_kinetics_evaluation: false,
            runtime_readbacks_enabled,
            inspection_readbacks_enabled,
        };

        simulation.bootstrap_kinetics(
            &concentrations,
            &solid_mask,
            &material_ids,
            &temperatures,
        )?;

        Ok(simulation)
    }

    fn bootstrap_kinetics(
        &mut self,
        concentrations: &[Vec<f32>],
        solid_mask: &[u32],
        material_ids: &[u32],
        temperatures: &[f32],
    ) -> Result<()> {
        let Some(mut kinetics) = self.kinetics.take() else {
            return Ok(());
        };

        kinetics.set_evaluation_interval(Self::MIN_KINETICS_INTERVAL_SECONDS);

        let update_result = kinetics.evaluate(
            self.grid.width,
            self.grid.height,
            self.time,
            concentrations,
            solid_mask,
            material_ids,
            temperatures,
            &self.species_registry,
            &self.material_registry,
        );

        match update_result {
            Ok(update) => {
                let (_applied, gpu_rules) = self.update_applicator.apply(
                    update,
                    &self.species_registry,
                    &mut self.material_registry,
                );

                let mut rules_changed = false;
                if !gpu_rules.is_empty() {
                    match self.gpu.upload_reaction_rules(&gpu_rules) {
                        Ok(changed) => {
                            rules_changed = changed;
                        }
                        Err(error) => {
                            log::error!("Failed to upload bootstrap reaction rules: {}", error);
                            rules_changed = true;
                        }
                    }
                }

                Self::update_kinetics_interval(&mut kinetics, rules_changed)
            }
            Err(error) => {
                log::warn!("Initial kinetics evaluation failed: {}", error);
                kinetics.set_evaluation_interval(Self::MIN_KINETICS_INTERVAL_SECONDS);
            }
        }

        self.kinetics = Some(kinetics);
        Ok(())
    }

    fn update_kinetics_interval(kinetics: &mut KineticsIntegration, rules_changed: bool) {
        if rules_changed {
            kinetics.set_evaluation_interval(Self::MIN_KINETICS_INTERVAL_SECONDS);
            return;
        }

        let next_interval = (kinetics.evaluation_interval() * 2.0).clamp(
            Self::MIN_KINETICS_INTERVAL_SECONDS,
            Self::MAX_KINETICS_INTERVAL_SECONDS,
        );
        kinetics.set_evaluation_interval(next_interval);
    }

    /// Create the demo simulation.
    pub fn new_demo(config: SimulationConfig) -> Result<Self> {
        let scenario = create_demo_scenario(config.width, config.height);
        Self::from_scenario(scenario, config)
    }

    pub fn new_demo_with_shared_gpu_context(
        config: SimulationConfig,
        context: SharedGpuContext,
    ) -> Result<Self> {
        let scenario = create_demo_scenario(config.width, config.height);
        Self::from_scenario_with_shared_gpu_context(scenario, config, context)
    }

    /// Create the acid-base neutralization simulation.
    pub fn new_acid_base(config: SimulationConfig) -> Result<Self> {
        let scenario = create_acid_base_scenario(config.width, config.height);
        Self::from_scenario(scenario, config)
    }

    pub fn new_acid_base_with_shared_gpu_context(
        config: SimulationConfig,
        context: SharedGpuContext,
    ) -> Result<Self> {
        let scenario = create_acid_base_scenario(config.width, config.height);
        Self::from_scenario_with_shared_gpu_context(scenario, config, context)
    }

    /// Create the weak-acid buffer simulation.
    pub fn new_buffers(config: SimulationConfig) -> Result<Self> {
        let scenario = create_buffers_scenario(config.width, config.height);
        Self::from_scenario(scenario, config)
    }

    pub fn new_buffers_with_shared_gpu_context(
        config: SimulationConfig,
        context: SharedGpuContext,
    ) -> Result<Self> {
        let scenario = create_buffers_scenario(config.width, config.height);
        Self::from_scenario_with_shared_gpu_context(scenario, config, context)
    }

    /// Create the dissolved-catalyst phosphorylation simulation.
    pub fn new_catalyst(config: SimulationConfig) -> Result<Self> {
        let scenario = create_catalyst_scenario(config.width, config.height);
        Self::from_scenario(scenario, config)
    }

    pub fn new_catalyst_with_shared_gpu_context(
        config: SimulationConfig,
        context: SharedGpuContext,
    ) -> Result<Self> {
        let scenario = create_catalyst_scenario(config.width, config.height);
        Self::from_scenario_with_shared_gpu_context(scenario, config, context)
    }

    /// Create the enzyme-entity phosphorylation simulation.
    pub fn new_enzyme(config: SimulationConfig) -> Result<Self> {
        let scenario = create_enzyme_scenario(config.width, config.height);
        Self::from_scenario(scenario, config)
    }

    pub fn new_enzyme_with_shared_gpu_context(
        config: SimulationConfig,
        context: SharedGpuContext,
    ) -> Result<Self> {
        let scenario = create_enzyme_scenario(config.width, config.height);
        Self::from_scenario_with_shared_gpu_context(scenario, config, context)
    }

    /// Create the membrane leak-channel simulation.
    pub fn new_leak(config: SimulationConfig) -> Result<Self> {
        let scenario = create_leak_scenario(config.width, config.height);
        Self::from_scenario(scenario, config)
    }

    pub fn new_leak_with_shared_gpu_context(
        config: SimulationConfig,
        context: SharedGpuContext,
    ) -> Result<Self> {
        let scenario = create_leak_scenario(config.width, config.height);
        Self::from_scenario_with_shared_gpu_context(scenario, config, context)
    }

    /// Step the simulation forward by dt seconds of wall-clock time.
    ///
    /// The actual simulated time is `dt * time_scale`.  Diffusion substeps
    /// are computed dynamically so the CFL stability condition holds even
    /// at high diffusion rates.
    pub fn step(&mut self, dt: f32) -> Result<()> {
        let Some(step) = self.prepare_frame_step(dt) else {
            return Ok(());
        };

        self.prepare_dynamic_entities(step.sim_dt)?;

        self.gpu.step_frame(
            step.substeps,
            step.substep_dt,
            step.diffusion_rate,
            step.charge_correction_strength,
            step.reaction_dt,
            step.thermal_diffusivity,
        )?;
        self.finalize_completed_frame()?;
        Ok(())
    }

    pub fn prepare_frame_step(&mut self, dt: f32) -> Option<PreparedSimulationStep> {
        if self.paused {
            log::trace!("Simulation is paused, skipping step");
            return None;
        }

        let wall_dt = dt.min(self.config.max_frame_dt);
        let dt_sim = wall_dt * self.config.time_scale;

        let species_d = self.config.diffusion_rate
            * self.species_registry.max_diffusion_coefficient().max(1.0)
            * Self::MAX_CHARGE_RELAXATION_DIFFUSION_SCALE;
        let max_d = species_d.max(self.config.thermal_diffusion_rate.max(0.0));
        let cfl_substeps = if max_d > 0.0 {
            ((max_d * dt_sim / 0.2).ceil() as u32).max(1)
        } else {
            1
        };
        let substeps = cfl_substeps.max(self.config.diffusion_substeps);
        let substep_dt = dt_sim / substeps as f32;

        log::trace!(
            "time_scale={}, wall_dt={:.4}, dt_sim={:.4}, substeps={}, substep_dt={:.5}, diffusion_rate={}, thermal_diffusion_rate={}",
            self.config.time_scale,
            wall_dt,
            dt_sim,
            substeps,
            substep_dt,
            self.config.diffusion_rate,
            self.config.thermal_diffusion_rate,
        );

        self.time += dt_sim as f64;
        self.step_count += 1;
        self.cache_dirty = true;

        let should_evaluate_kinetics = self
            .kinetics
            .as_mut()
            .filter(|_| self.runtime_readbacks_enabled)
            .map(|k| k.accumulate_time(dt_sim as f64))
            .unwrap_or(false);
        self.pending_kinetics_evaluation |= should_evaluate_kinetics;

        Some(PreparedSimulationStep {
            substeps,
            substep_dt,
            sim_dt: dt_sim,
            diffusion_rate: self.config.diffusion_rate,
            charge_correction_strength: effective_charge_correction_strength(
                self.config.charge_correction_strength,
                &self.leak_channels,
            ),
            reaction_dt: dt_sim,
            thermal_diffusivity: self.config.thermal_diffusion_rate,
        })
    }

    pub fn record_prepared_step(
        &mut self,
        cmd: vk::CommandBuffer,
        step: PreparedSimulationStep,
    ) -> Result<()> {
        self.prepare_dynamic_entities(step.sim_dt)?;
        self.gpu.record_step_frame(
            cmd,
            step.substeps,
            step.substep_dt,
            step.diffusion_rate,
            step.charge_correction_strength,
            step.reaction_dt,
            step.thermal_diffusivity,
        );
        Ok(())
    }

    fn prepare_dynamic_entities(&mut self, sim_dt: f32) -> Result<()> {
        let Some(field) = self.enzyme_field else {
            return Ok(());
        };
        if self.enzyme_entities.is_empty() {
            return Ok(());
        }

        for entity in &mut self.enzyme_entities {
            entity.advance(sim_dt, &field);
        }

        self.sync_enzyme_entities()
    }

    pub fn finalize_completed_frame(&mut self) -> Result<()> {
        if !self.runtime_readbacks_enabled {
            self.pending_kinetics_evaluation = false;
            return Ok(());
        }

        if !self.pending_kinetics_evaluation {
            return Ok(());
        }

        self.pending_kinetics_evaluation = false;
        self.evaluate_kinetics_from_cache()
    }

    pub fn has_pending_kinetics_evaluation(&self) -> bool {
        self.pending_kinetics_evaluation
    }

    fn evaluate_kinetics_from_cache(&mut self) -> Result<()> {
        self.refresh_cache()?;

        let fine_width = self.grid.width;
        let fine_height = self.grid.height;
        let sim_time = self.time;
        let concentrations = self.cached_concentrations.as_ref().unwrap().clone();
        let solid_mask = self.cached_solid_mask.as_ref().unwrap().clone();
        let material_ids = self.cached_material_ids.as_ref().unwrap().clone();
        let temperatures = self
            .cached_temperatures
            .clone()
            .unwrap_or_else(|| vec![293.15; (fine_width * fine_height) as usize]);

        let species_registry = self.species_registry.clone();
        let material_registry = self.material_registry.clone();

        if let Some(ref mut kinetics) = self.kinetics {
            match kinetics.evaluate(
                fine_width,
                fine_height,
                sim_time,
                &concentrations,
                &solid_mask,
                &material_ids,
                &temperatures,
                &species_registry,
                &material_registry,
            ) {
                Ok(update) => {
                    let (_applied, gpu_rules) = self.update_applicator.apply(
                        update,
                        &species_registry,
                        &mut self.material_registry,
                    );

                    if !gpu_rules.is_empty() && self.gpu.reaction.is_none() {
                        let temps = temperatures.clone();
                        if let Err(e) = self.gpu.init_reaction_pipeline(&temps) {
                            log::error!("Failed to initialize reaction pipeline: {}", e);
                        }
                    }

                    let mut rules_changed = false;
                    if !gpu_rules.is_empty() {
                        match self.gpu.upload_reaction_rules(&gpu_rules) {
                            Ok(changed) => {
                                rules_changed = changed;
                            }
                            Err(e) => {
                                log::error!("Failed to upload reaction rules: {}", e);
                                rules_changed = true;
                            }
                        }
                    }

                    Self::update_kinetics_interval(kinetics, rules_changed);
                }
                Err(e) => {
                    log::warn!("Kinetics evaluation failed: {}", e);
                }
            }
        }

        Ok(())
    }

    /// Refresh cached data from GPU.
    fn refresh_cache(&mut self) -> Result<()> {
        if !self.cache_dirty {
            return Ok(());
        }

        self.cached_concentrations = Some(self.gpu.read_concentrations()?);
        self.cached_temperatures = Some(self.gpu.read_temperatures()?);

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
    pub fn inspect_with_mip(
        &mut self,
        screen_x: f32,
        screen_y: f32,
        mip: u32,
    ) -> Result<InspectionResult> {
        self.refresh_cache()?;

        let coord = self.inspector.screen_to_coarse(screen_x, screen_y, mip);

        let result = self.inspector.inspect(
            coord,
            self.cached_concentrations.as_ref().unwrap(),
            self.cached_temperatures.as_ref().unwrap(),
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
    /// Note: Skips expensive GPU readback if data was recently refreshed
    pub fn render_state(&mut self) -> Result<RenderState> {
        // Only refresh every 4 steps to avoid expensive GPU readback every frame
        // This trades some visual latency for much better frame rate
        let should_refresh =
            self.cache_dirty && (self.step_count % 4 == 0 || self.cached_concentrations.is_none());

        if should_refresh {
            self.refresh_cache()?;
        }

        // Use cached data (may be up to 4 frames old)
        let concentrations = self.cached_concentrations.clone().unwrap_or_else(|| {
            vec![
                vec![0.0; (self.grid.width * self.grid.height) as usize];
                self.species_registry.count()
            ]
        });
        let solid_mask = self
            .cached_solid_mask
            .clone()
            .unwrap_or_else(|| vec![0; (self.grid.width * self.grid.height) as usize]);
        let material_ids = self
            .cached_material_ids
            .clone()
            .unwrap_or_else(|| vec![0; (self.grid.width * self.grid.height) as usize]);
        let temperatures = self
            .cached_temperatures
            .clone()
            .unwrap_or_else(|| vec![293.15; (self.grid.width * self.grid.height) as usize]);

        Ok(RenderState {
            width: self.grid.width,
            height: self.grid.height,
            species_count: self.species_registry.count(),
            concentrations,
            solid_mask,
            material_ids,
            temperatures,
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

    pub fn leak_channels(&self) -> &[LeakChannel] {
        &self.leak_channels
    }

    pub fn enzyme_entities(&self) -> &[EnzymeEntity] {
        &self.enzyme_entities
    }

    pub fn enzyme_field(&self) -> Option<&EnzymeField> {
        self.enzyme_field.as_ref()
    }

    pub fn hovered_leak_channel(&self, grid_x: f32, grid_y: f32) -> Option<usize> {
        self.leak_channels
            .iter()
            .position(|channel| channel.contains_grid_point(grid_x, grid_y, 3.2, 1.8))
    }

    pub fn leak_channel_endpoints(&self, index: usize) -> Option<((i32, i32), (i32, i32))> {
        let channel = self.leak_channels.get(index)?;
        let solid_mask = self.cached_solid_mask.as_ref()?;
        channel.resolve_endpoints(self.grid.width, self.grid.height, solid_mask)
    }

    pub fn resolve_leak_channel_endpoints(
        &self,
        channel: &LeakChannel,
    ) -> Option<((i32, i32), (i32, i32))> {
        let solid_mask = self.cached_solid_mask.as_ref()?;
        channel.resolve_endpoints(self.grid.width, self.grid.height, solid_mask)
    }

    pub fn leak_channel_tooltip(&self, index: usize) -> Option<String> {
        let channel = self.leak_channels.get(index)?;
        let species_name = self.species_registry.get(channel.species)?.name.as_ref();
        let ((sink_x, sink_y), (source_x, source_y)) = self.leak_channel_endpoints(index)?;
        Some(format!(
            "Leak Channel\nSpecies: {}\nRate: {:.3}\nFlow: {}\nSink cell: ({}, {})\nSource cell: ({}, {})",
            species_name,
            channel.rate,
            channel.flow_label(),
            sink_x,
            sink_y,
            source_x,
            source_y,
        ))
    }

    pub fn add_leak_channel(&mut self, channel: LeakChannel) -> Result<()> {
        self.leak_channels.push(channel);
        if let Err(error) = self.sync_leak_channels() {
            self.leak_channels.pop();
            return Err(error);
        }
        Ok(())
    }

    pub fn update_leak_channel(&mut self, index: usize, channel: LeakChannel) -> Result<()> {
        let Some(existing) = self.leak_channels.get_mut(index) else {
            return Err(anyhow::anyhow!("Leak channel index {} out of range", index));
        };

        let previous = existing.clone();
        *existing = channel;
        if let Err(error) = self.sync_leak_channels() {
            *self.leak_channels.get_mut(index).unwrap() = previous;
            return Err(error);
        }
        Ok(())
    }

    pub fn remove_leak_channel(&mut self, index: usize) -> Result<LeakChannel> {
        if index >= self.leak_channels.len() {
            return Err(anyhow::anyhow!("Leak channel index {} out of range", index));
        }

        let removed = self.leak_channels.remove(index);
        if let Err(error) = self.sync_leak_channels() {
            self.leak_channels.insert(index, removed.clone());
            return Err(error);
        }

        Ok(removed)
    }

    fn sync_leak_channels(&mut self) -> Result<()> {
        let solid_mask = self
            .cached_solid_mask
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Solid mask cache is unavailable"))?;
        self.gpu
            .upload_leak_channels(&self.leak_channels, &self.species_registry, solid_mask)
    }

    fn sync_enzyme_entities(&mut self) -> Result<()> {
        self.gpu
            .upload_enzyme_entities(&self.enzyme_entities, &self.species_registry)
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

    pub fn gpu_render_buffers(&self) -> GpuRenderBuffers {
        self.gpu.render_buffers()
    }

    pub fn record_render_barriers(&self, cmd: vk::CommandBuffer) {
        self.gpu.record_render_barriers(cmd);
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

    // ============ Async Coarse Grid Inspection API ============
    // These methods provide non-blocking tooltip inspection.
    // The render thread is never blocked waiting for GPU data.

    /// Request an async readback of the coarse cell at the given screen position.
    /// Returns true if the request was accepted, false if rate-limited or no coarse grid.
    pub fn request_async_inspection(&mut self, screen_x: f32, screen_y: f32) -> bool {
        if !self.inspection_readbacks_enabled {
            return false;
        }

        if let Some(ref mut coarse) = self.gpu.coarse_grid {
            let (cx, cy) = coarse.screen_to_coarse(screen_x, screen_y);
            coarse.request_cell_readback(cx, cy)
        } else {
            false
        }
    }

    /// Poll for async inspection completion.
    /// Returns Some(data) if new data is available, None otherwise.
    /// This method never blocks.
    pub fn poll_async_inspection(&mut self) -> Option<CoarseCellData> {
        if !self.inspection_readbacks_enabled {
            return None;
        }

        if let Some(ref mut coarse) = self.gpu.coarse_grid {
            coarse.poll_readback()
        } else {
            None
        }
    }

    /// Get the last cached coarse cell data, even if stale.
    /// The `age` field indicates how old the data is.
    pub fn get_cached_inspection(&self) -> Option<CoarseCellData> {
        if !self.inspection_readbacks_enabled {
            return None;
        }

        if let Some(ref coarse) = self.gpu.coarse_grid {
            coarse.get_cached_cell()
        } else {
            None
        }
    }

    /// Check if an async readback is currently pending.
    pub fn is_inspection_pending(&self) -> bool {
        if let Some(ref coarse) = self.gpu.coarse_grid {
            coarse.is_readback_pending()
        } else {
            false
        }
    }

    pub fn set_async_inspection_interval(&mut self, interval: Duration) {
        if let Some(ref mut coarse) = self.gpu.coarse_grid {
            coarse.set_read_interval(interval);
        }
    }

    /// Get the coarse grid dimensions (if available).
    pub fn coarse_dimensions(&self) -> Option<(u32, u32)> {
        self.gpu
            .coarse_grid
            .as_ref()
            .map(|c| (c.coarse_width, c.coarse_height))
    }

    /// Get the mip factor used for coarse grid.
    pub fn coarse_mip_factor(&self) -> Option<u32> {
        self.gpu.coarse_grid.as_ref().map(|c| c.mip_factor)
    }

    // ============ Kinetics Integration API ============

    /// Check if kinetics integration is enabled.
    pub fn has_kinetics(&self) -> bool {
        self.kinetics.is_some()
    }

    /// Get the kinetics integration (if enabled).
    pub fn kinetics(&self) -> Option<&KineticsIntegration> {
        self.kinetics.as_ref()
    }

    /// Get mutable access to kinetics integration.
    pub fn kinetics_mut(&mut self) -> Option<&mut KineticsIntegration> {
        self.kinetics.as_mut()
    }

    /// Get the last semantic update from kinetics (if any).
    pub fn last_semantic_update(&self) -> Option<&kinetics::SemanticUpdate> {
        self.kinetics.as_ref().and_then(|k| k.last_update())
    }

    /// Get the time since last kinetics evaluation.
    pub fn time_since_kinetics_evaluation(&self) -> f64 {
        self.kinetics
            .as_ref()
            .map(|k| k.time_since_last_evaluation())
            .unwrap_or(0.0)
    }

    /// Set the kinetics evaluation interval.
    pub fn set_kinetics_interval(&mut self, interval_seconds: f64) {
        if let Some(ref mut k) = self.kinetics {
            k.set_evaluation_interval(interval_seconds);
        }
    }

    /// Get kinetics integration statistics.
    pub fn kinetics_stats(&self) -> Option<&crate::kinetics_integration::IntegrationStats> {
        self.kinetics.as_ref().map(|k| k.stats())
    }
}

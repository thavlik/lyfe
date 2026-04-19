use std::time::Instant;

use anyhow::Result;
use fluidsim::{Simulation, SimulationConfig};
use renderer::{EguiRenderer, RenderContext, RenderPipeline};
use winit::window::Window;

use crate::app::{DemoApp, TOOLTIP_REFRESH_INTERVAL};
use crate::colors::compute_species_colors;

impl DemoApp {
    pub(crate) fn initialize(&mut self, window: Window) -> Result<()> {
        let _size = window.inner_size();

        log::info!("Creating render context...");
        let render_ctx = RenderContext::new(&window, self.present_mode)?;
        log::info!("Render context created successfully");

        log::info!("Creating simulation on shared Vulkan device...");
        let simulation = self.create_simulation(&render_ctx)?;
        log::info!("Simulation created successfully");

        log::info!("Creating render pipeline...");
        let species_colors = compute_species_colors(simulation.species_registry());
        let render_pipeline = RenderPipeline::new(
            &render_ctx,
            simulation.dimensions().0,
            simulation.dimensions().1,
            simulation.species_registry().count(),
            &species_colors,
        )?;
        log::info!("Render pipeline created successfully");

        log::info!("Creating egui renderer...");
        let egui_renderer = EguiRenderer::new(&render_ctx, &window)?;
        log::info!("Egui renderer created successfully");

        let species_names = simulation
            .species_registry()
            .iter()
            .map(|species| species.name.to_string())
            .collect();

        let mut simulation = simulation;
        simulation.set_async_inspection_interval(TOOLTIP_REFRESH_INTERVAL);

        self.window = Some(window);
        self.render_ctx = Some(render_ctx);
        self.render_pipeline = Some(render_pipeline);
        self.egui_renderer = Some(egui_renderer);
        self.simulation = Some(simulation);
        self.species_names = species_names;

        Ok(())
    }

    pub(crate) fn simulation_config(&self) -> SimulationConfig {
        let diffusion_rate = match self.scenario {
            crate::cli::ScenarioKind::Leak => 7.5,
            _ => 5.0,
        };
        let time_scale = match self.scenario {
            crate::cli::ScenarioKind::Leak => 24.0,
            _ => 20.0,
        };
        let diffusion_substeps = 4;
        let charge_correction_strength = match self.scenario {
            crate::cli::ScenarioKind::Leak => 0.0,
            _ => 1.0,
        };

        SimulationConfig {
            width: 512,
            height: 512,
            diffusion_rate,
            thermal_diffusion_rate: 3.0,
            charge_correction_strength,
            diffusion_substeps,
            inspection_mip: self.inspection_mip,
            time_scale,
            reaction_rate_scale: 8.0,
            max_frame_dt: 1.0 / 15.0,
        }
    }

    pub(crate) fn create_simulation(&self, render_ctx: &RenderContext) -> Result<Simulation> {
        let config = self.simulation_config();
        let context = render_ctx.shared_gpu_context();
        let mut simulation = match self.scenario {
            crate::cli::ScenarioKind::Basic => {
                Simulation::new_demo_with_shared_gpu_context(config, context)?
            }
            crate::cli::ScenarioKind::AcidBase => {
                Simulation::new_acid_base_with_shared_gpu_context(config, context)?
            }
            crate::cli::ScenarioKind::Buffers => {
                Simulation::new_buffers_with_shared_gpu_context(config, context)?
            }
            crate::cli::ScenarioKind::Catalyst => {
                Simulation::new_catalyst_with_shared_gpu_context(config, context)?
            }
            crate::cli::ScenarioKind::Enzyme => {
                Simulation::new_enzyme_with_shared_gpu_context(config, context)?
            }
            crate::cli::ScenarioKind::Leak => {
                Simulation::new_leak_with_shared_gpu_context(config, context)?
            }
        };
        simulation.set_async_inspection_interval(TOOLTIP_REFRESH_INTERVAL);
        Ok(simulation)
    }

    pub(crate) fn reset_simulation(&mut self) -> Result<()> {
        let render_ctx = self.render_ctx.as_ref().unwrap();
        unsafe {
            render_ctx.device.device_wait_idle()?;
        }

        let simulation = self.create_simulation(render_ctx)?;
        self.species_names = simulation
            .species_registry()
            .iter()
            .map(|species| species.name.to_string())
            .collect();
        self.simulation = Some(simulation);
        self.show_tooltip = false;
        self.tooltip_text.clear();
        self.last_tooltip_coord = None;
        self.hovered_leak_channel = None;
        self.create_menu_open = false;
        self.placement_state = None;
        self.transform_state = None;
        self.selected_entity = None;
        self.inspector_draft = None;
        self.inspector_dirty = false;
        self.inspector_error = None;
        self.detail_probes.clear();
        self.detail_last_refresh = Instant::now() - TOOLTIP_REFRESH_INTERVAL;
        self.last_frame = Instant::now();

        log::info!("Simulation reset ({:?} scenario)", self.scenario);
        Ok(())
    }
}

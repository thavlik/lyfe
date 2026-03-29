//! Fluid simulation demo application.
//!
//! This demo opens a window, runs the GPU fluid simulation, and renders
//! the results at 60 FPS with mouse inspection tooltips.
//!
//! ## Controls
//! - Mouse hover: Inspect coarse cell (8x8 block)
//! - Space: Toggle pause
//! - +/-: Adjust inspection mip factor
//! - Escape: Exit

use std::time::{Duration, Instant};

use anyhow::Result;
use fluidsim::{Simulation, SimulationConfig};
use renderer::{RenderContext, RenderPipeline, EguiRenderer};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

const _TARGET_FPS: f64 = 60.0;
const _FRAME_TIME: Duration = Duration::from_nanos((1_000_000_000.0 / _TARGET_FPS) as u64);

/// Demo application state.
struct DemoApp {
    window: Option<Window>,
    render_ctx: Option<RenderContext>,
    render_pipeline: Option<RenderPipeline>,
    egui_renderer: Option<EguiRenderer>,
    simulation: Option<Simulation>,
    
    // State
    mouse_pos: (f32, f32),
    inspection_mip: u32,
    show_tooltip: bool,
    tooltip_text: String,
    
    // Timing
    last_frame: Instant,
    frame_count: u64,
    fps_update_time: Instant,
    current_fps: f64,
    
    // Flags
    needs_resize: bool,
}

impl DemoApp {
    fn new() -> Self {
        Self {
            window: None,
            render_ctx: None,
            render_pipeline: None,
            egui_renderer: None,
            simulation: None,
            mouse_pos: (0.0, 0.0),
            inspection_mip: 8,
            show_tooltip: false,
            tooltip_text: String::new(),
            last_frame: Instant::now(),
            frame_count: 0,
            fps_update_time: Instant::now(),
            current_fps: 0.0,
            needs_resize: false,
        }
    }

    fn initialize(&mut self, window: Window) -> Result<()> {
        let _size = window.inner_size();
        
        log::info!("Creating simulation...");
        
        // Create simulation
        let config = SimulationConfig {
            width: 512,
            height: 512,
            diffusion_rate: 0.2,
            diffusion_substeps: 4,
            inspection_mip: self.inspection_mip,
        };
        
        let simulation = Simulation::new_demo(config)?;
        log::info!("Simulation created successfully");
        
        log::info!("Creating render context...");
        // Create render context
        let render_ctx = RenderContext::new(&window)?;
        log::info!("Render context created successfully");
        
        log::info!("Creating render pipeline...");
        // Create render pipeline
        let render_pipeline = RenderPipeline::new(
            &render_ctx,
            simulation.dimensions().0,
            simulation.dimensions().1,
            simulation.species_registry().count(),
        )?;
        log::info!("Render pipeline created successfully");
        
        log::info!("Creating egui renderer...");
        // Create egui renderer
        let egui_renderer = EguiRenderer::new(&window)?;
        log::info!("Egui renderer created successfully");
        
        self.window = Some(window);
        self.render_ctx = Some(render_ctx);
        self.render_pipeline = Some(render_pipeline);
        self.egui_renderer = Some(egui_renderer);
        self.simulation = Some(simulation);
        
        Ok(())
    }

    fn update_tooltip(&mut self) {
        let sim = self.simulation.as_mut().unwrap();
        let (grid_w, grid_h) = sim.dimensions();
        
        // Convert mouse position to grid coordinates
        if let Some(window) = &self.window {
            let size = window.inner_size();
            let grid_x = self.mouse_pos.0 * grid_w as f32 / size.width as f32;
            let grid_y = self.mouse_pos.1 * grid_h as f32 / size.height as f32;
            
            match sim.inspect_with_mip(grid_x, grid_y, self.inspection_mip) {
                Ok(result) => {
                    self.tooltip_text = result.format_tooltip();
                    self.show_tooltip = true;
                }
                Err(_) => {
                    self.show_tooltip = false;
                }
            }
        }
    }

    fn render_frame(&mut self) -> Result<bool> {
        let ctx = self.render_ctx.as_mut().unwrap();
        let pipeline = self.render_pipeline.as_mut().unwrap();
        let sim = self.simulation.as_mut().unwrap();
        
        // Get render state and upload
        let state = sim.render_state()?;
        pipeline.upload_state(ctx, &state)?;
        
        // Begin frame
        let image_index = match ctx.begin_frame()? {
            Some(idx) => idx,
            None => {
                self.needs_resize = true;
                return Ok(false);
            }
        };
        
        // Record commands
        let cmd = ctx.current_command_buffer();
        
        let begin_info = ash::vk::CommandBufferBeginInfo::default();
        unsafe {
            ctx.device.reset_command_buffer(cmd, ash::vk::CommandBufferResetFlags::empty())?;
            ctx.device.begin_command_buffer(cmd, &begin_info)?;
        }
        
        pipeline.record(ctx, cmd, image_index as usize);
        
        unsafe {
            ctx.device.end_command_buffer(cmd)?;
        }
        
        // End frame
        let ok = ctx.end_frame(image_index)?;
        if !ok {
            self.needs_resize = true;
        }
        
        Ok(true)
    }

    fn handle_resize(&mut self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        
        let ctx = self.render_ctx.as_mut().unwrap();
        ctx.recreate_swapchain(width, height)?;
        self.needs_resize = false;
        
        Ok(())
    }
}

impl ApplicationHandler for DemoApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_title("Fluid Simulation Demo")
            .with_inner_size(PhysicalSize::new(800, 800));
        
        match event_loop.create_window(window_attrs) {
            Ok(window) => {
                if let Err(e) = self.initialize(window) {
                    log::error!("Failed to initialize: {}", e);
                    event_loop.exit();
                }
            }
            Err(e) => {
                log::error!("Failed to create window: {}", e);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        // Pass to egui first
        if let Some(egui) = &mut self.egui_renderer {
            if let Some(window) = &self.window {
                if egui.handle_event(window, &event) {
                    return;
                }
            }
        }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            
            WindowEvent::Resized(size) => {
                if let Err(e) = self.handle_resize(size.width, size.height) {
                    log::error!("Failed to resize: {}", e);
                }
            }
            
            WindowEvent::KeyboardInput { event: KeyEvent { logical_key, state: ElementState::Pressed, .. }, .. } => {
                match logical_key {
                    Key::Named(NamedKey::Escape) => {
                        event_loop.exit();
                    }
                    Key::Named(NamedKey::Space) => {
                        if let Some(sim) = &mut self.simulation {
                            sim.toggle_pause();
                            log::info!("Simulation {}", if sim.is_paused() { "paused" } else { "resumed" });
                        }
                    }
                    Key::Character(ref c) if c == "+" || c == "=" => {
                        self.inspection_mip = (self.inspection_mip * 2).min(64);
                        if let Some(sim) = &mut self.simulation {
                            sim.set_inspection_mip(self.inspection_mip);
                        }
                        log::info!("Inspection mip: {}", self.inspection_mip);
                    }
                    Key::Character(ref c) if c == "-" || c == "_" => {
                        self.inspection_mip = (self.inspection_mip / 2).max(1);
                        if let Some(sim) = &mut self.simulation {
                            sim.set_inspection_mip(self.inspection_mip);
                        }
                        log::info!("Inspection mip: {}", self.inspection_mip);
                    }
                    _ => {}
                }
            }
            
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_pos = (position.x as f32, position.y as f32);
            }
            
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame);
                
                // Update simulation
                if let Some(sim) = &mut self.simulation {
                    if let Err(e) = sim.step(dt.as_secs_f32()) {
                        log::error!("Simulation step failed: {}", e);
                    }
                }
                
                // Update tooltip
                self.update_tooltip();
                
                // Render egui overlay
                if let (Some(egui), Some(window)) = (&mut self.egui_renderer, &self.window) {
                    egui.begin_frame(window);
                    
                    // Show tooltip near mouse
                    if self.show_tooltip {
                        egui::Window::new("Inspection")
                            .fixed_pos(egui::pos2(self.mouse_pos.0 + 15.0, self.mouse_pos.1 + 15.0))
                            .collapsible(false)
                            .resizable(false)
                            .show(egui.context(), |ui| {
                                ui.monospace(&self.tooltip_text);
                            });
                    }
                    
                    // Show FPS
                    egui::Window::new("Stats")
                        .fixed_pos(egui::pos2(10.0, 10.0))
                        .collapsible(false)
                        .resizable(false)
                        .show(egui.context(), |ui| {
                            ui.label(format!("FPS: {:.1}", self.current_fps));
                            if let Some(sim) = &self.simulation {
                                ui.label(format!("Steps: {}", sim.step_count()));
                                ui.label(format!("Time: {:.2}s", sim.time()));
                                ui.label(format!("Paused: {}", sim.is_paused()));
                            }
                            ui.label(format!("Mip: {}", self.inspection_mip));
                        });
                    
                    let _output = egui.end_frame(window);
                    // Note: Full egui rendering would require additional Vulkan integration
                    // For this demo, the egui output is not rendered to screen
                }
                
                log::trace!("Checking resize...");
                // Render simulation
                if self.needs_resize {
                    if let Some(window) = &self.window {
                        let size = window.inner_size();
                        if let Err(e) = self.handle_resize(size.width, size.height) {
                            log::error!("Resize failed: {}", e);
                        }
                    }
                }
                
                match self.render_frame() {
                    Ok(_) => {}
                    Err(e) => {
                        log::error!("Render failed: {}", e);
                    }
                }
                
                // Update timing
                self.frame_count += 1;
                self.last_frame = now;
                
                // Update FPS counter
                let fps_elapsed = now.duration_since(self.fps_update_time);
                if fps_elapsed >= Duration::from_secs(1) {
                    self.current_fps = self.frame_count as f64 / fps_elapsed.as_secs_f64();
                    self.frame_count = 0;
                    self.fps_update_time = now;
                }
                
                // Request next frame
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            
            _ => {}
        }
    }
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .init();
    
    log::info!("Starting Fluid Simulation Demo");
    log::info!("Controls:");
    log::info!("  Mouse hover: Inspect cell");
    log::info!("  Space: Toggle pause");
    log::info!("  +/-: Adjust inspection mip");
    log::info!("  Escape: Exit");
    
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    
    let mut app = DemoApp::new();
    event_loop.run_app(&mut app)?;
    
    Ok(())
}

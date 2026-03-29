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
    species_names: Vec<String>,  // Cached species names for tooltip display
    
    // Thermal view (momentary display while T is held)
    thermal_view: bool,
    
    // Timing
    last_frame: Instant,
    frame_count: u64,
    fps_update_time: Instant,
    current_fps: f64,
    
    // Flags
    needs_resize: bool,
    smoke_test: bool,
}

impl DemoApp {
    fn new(smoke_test: bool) -> Self {
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
            species_names: Vec::new(),
            thermal_view: false,
            last_frame: Instant::now(),
            frame_count: 0,
            fps_update_time: Instant::now(),
            current_fps: 0.0,
            needs_resize: false,
            smoke_test,
        }
    }

    fn initialize(&mut self, window: Window) -> Result<()> {
        let _size = window.inner_size();
        
        log::info!("Creating simulation...");
        
        // Create simulation
        let config = SimulationConfig {
            width: 512,
            height: 512,
            diffusion_rate: 2.0,
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
        // Compute species colors: explicit overrides, then hash-based fallback
        let species_colors = compute_species_colors(simulation.species_registry());
        // Create render pipeline
        let render_pipeline = RenderPipeline::new(
            &render_ctx,
            simulation.dimensions().0,
            simulation.dimensions().1,
            simulation.species_registry().count(),
            &species_colors,
        )?;
        log::info!("Render pipeline created successfully");
        
        log::info!("Creating egui renderer...");
        // Create egui renderer
        let egui_renderer = EguiRenderer::new(&render_ctx, &window)?;
        log::info!("Egui renderer created successfully");
        
        // Cache species names for tooltip display
        let species_names: Vec<String> = simulation.species_registry()
            .iter()
            .map(|s| s.name.to_string())
            .collect();
        
        self.window = Some(window);
        self.render_ctx = Some(render_ctx);
        self.render_pipeline = Some(render_pipeline);
        self.egui_renderer = Some(egui_renderer);
        self.simulation = Some(simulation);
        self.species_names = species_names;
        
        Ok(())
    }

    fn update_tooltip(&mut self) {
        let sim = self.simulation.as_mut().unwrap();
        let (grid_w, grid_h) = sim.dimensions();
        
        // Convert mouse position (logical coords) to grid coordinates
        if let Some(window) = &self.window {
            let size = window.inner_size();
            let scale = window.scale_factor() as f32;
            // mouse_pos is in logical coords, so use logical size
            let logical_w = size.width as f32 / scale;
            let logical_h = size.height as f32 / scale;
            let grid_x = self.mouse_pos.0 * grid_w as f32 / logical_w;
            let grid_y = self.mouse_pos.1 * grid_h as f32 / logical_h;
            
            // Request async readback of this cell (rate-limited internally)
            let _ = sim.request_async_inspection(grid_x, grid_y);
            
            // Poll for any completed readbacks
            if let Some(data) = sim.poll_async_inspection() {
                // New data available - format it
                let species_lines: String = data.concentrations.iter()
                    .enumerate()
                    .filter(|&(_, c)| *c > 0.001)
                    .map(|(i, c)| {
                        let name = self.species_names.get(i)
                            .map(|s| s.as_str())
                            .unwrap_or("?");
                        format!("  {:<6} {:>8.3} M", name, c)
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                self.tooltip_text = format!(
                    "Coarse Cell ({}, {})\n\
                     Fluid: {} / Solid: {}\n\
                     Age: {:.0}ms\n\
                     Species:\n{}",
                    data.coord.0, data.coord.1,
                    data.fluid_count, data.solid_count,
                    data.age.as_secs_f64() * 1000.0,
                    species_lines,
                );
                self.show_tooltip = true;
            } else if let Some(data) = sim.get_cached_inspection() {
                // Use cached data (may be stale)
                let species_lines: String = data.concentrations.iter()
                    .enumerate()
                    .filter(|&(_, c)| *c > 0.001)
                    .map(|(i, c)| {
                        let name = self.species_names.get(i)
                            .map(|s| s.as_str())
                            .unwrap_or("?");
                        format!("  {:<6} {:>8.3} M", name, c)
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                self.tooltip_text = format!(
                    "Coarse Cell ({}, {})\n\
                     Fluid: {} / Solid: {}\n\
                     Age: {:.0}ms (cached)\n\
                     Species:\n{}",
                    data.coord.0, data.coord.1,
                    data.fluid_count, data.solid_count,
                    data.age.as_secs_f64() * 1000.0,
                    species_lines,
                );
                self.show_tooltip = true;
            } else {
                // No data yet
                self.show_tooltip = false;
            }
        }
    }
    
    fn _update_tooltip_expensive(&mut self) {
        let sim = self.simulation.as_mut().unwrap();
        let (grid_w, grid_h) = sim.dimensions();
        
        // Convert mouse position (logical coords) to grid coordinates
        if let Some(window) = &self.window {
            let size = window.inner_size();
            let scale = window.scale_factor() as f32;
            let logical_w = size.width as f32 / scale;
            let logical_h = size.height as f32 / scale;
            let grid_x = self.mouse_pos.0 * grid_w as f32 / logical_w;
            let grid_y = self.mouse_pos.1 * grid_h as f32 / logical_h;
            
            match sim.inspect_with_mip(grid_x, grid_y, self.inspection_mip) {
                Ok(result) => {
                    let new_tooltip = result.format_tooltip();
                    // Log when tooltip changes significantly
                    if new_tooltip != self.tooltip_text {
                        log::debug!("Inspection at ({:.0}, {:.0}): {}", grid_x, grid_y, 
                            new_tooltip.replace('\n', " | "));
                    }
                    self.tooltip_text = new_tooltip;
                    self.show_tooltip = true;
                }
                Err(e) => {
                    log::trace!("Inspection error: {}", e);
                    self.show_tooltip = false;
                }
            }
        }
    }

    fn render_frame(&mut self) -> Result<bool> {
        let ctx = self.render_ctx.as_mut().unwrap();
        let pipeline = self.render_pipeline.as_mut().unwrap();
        let sim = self.simulation.as_mut().unwrap();
        let egui = self.egui_renderer.as_mut().unwrap();
        
        // Begin frame first - this waits on the in-flight fence,
        // ensuring the previous frame's commands have completed
        // before we touch the command pool or queue.
        let image_index = match ctx.begin_frame()? {
            Some(idx) => idx,
            None => {
                log::debug!("begin_frame returned None, needs resize");
                self.needs_resize = true;
                return Ok(false);
            }
        };

        let t0 = Instant::now();
        
        // Get render state and upload (safe now that previous frame is done)
        let state = sim.render_state()?;
        let t1 = Instant::now();
        
        pipeline.upload_state(ctx, &state)?;
        let t2 = Instant::now();
        
        // Debug: Log frame presentation
        let step = sim.step_count();
        if step > 0 && step % 10 == 0 {
            log::info!("Frame timing: render_state={:.1}ms upload={:.1}ms", 
                (t1-t0).as_secs_f64()*1000.0, 
                (t2-t1).as_secs_f64()*1000.0);
        }

        // Update egui textures before rendering
        egui.set_textures(ctx)?;
        
        log::trace!("Rendering to swapchain image {}", image_index);
        
        // Record commands
        let cmd = ctx.current_command_buffer();
        
        let begin_info = ash::vk::CommandBufferBeginInfo::default();
        unsafe {
            ctx.device.reset_command_buffer(cmd, ash::vk::CommandBufferResetFlags::empty())?;
            ctx.device.begin_command_buffer(cmd, &begin_info)?;
        }
        
        // Record fluid visualization (keeps render pass open)
        pipeline.record(ctx, cmd, image_index as usize, self.thermal_view);
        
        // Record egui draw commands inside the same render pass
        egui.cmd_draw(ctx, cmd, ctx.swapchain_extent)?;
        
        // End render pass
        pipeline.end_render_pass(ctx, cmd);
        
        unsafe {
            ctx.device.end_command_buffer(cmd)?;
        }
        
        // End frame
        let ok = ctx.end_frame(image_index)?;
        if !ok {
            log::debug!("end_frame returned false, needs resize");
            self.needs_resize = true;
        }
        
        // Free egui textures after rendering is complete
        egui.free_textures()?;
        
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
            .with_inner_size(PhysicalSize::new(1600, 1600));
        
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
        // Pass to egui first, but don't skip RedrawRequested - we always need to render
        let egui_wants_event = if let Some(egui) = &mut self.egui_renderer {
            if let Some(window) = &self.window {
                egui.handle_event(window, &event)
            } else {
                false
            }
        } else {
            false
        };

        // For most events, let egui consume them if it wants
        // But always process RedrawRequested, Resized, and CloseRequested
        let is_critical_event = matches!(
            event,
            WindowEvent::RedrawRequested | WindowEvent::Resized(_) | WindowEvent::CloseRequested
        );
        
        if egui_wants_event && !is_critical_event {
            return;
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
                    Key::Character(ref c) if c == "t" || c == "T" => {
                        self.thermal_view = true;
                    }
                    _ => {}
                }
            }
            
            WindowEvent::KeyboardInput { event: KeyEvent { logical_key, state: ElementState::Released, .. }, .. } => {
                match logical_key {
                    Key::Character(ref c) if c == "t" || c == "T" => {
                        self.thermal_view = false;
                    }
                    _ => {}
                }
            }
            
            WindowEvent::CursorMoved { position, .. } => {
                // Store position in logical coordinates for egui
                let scale = self.window.as_ref().map(|w| w.scale_factor()).unwrap_or(1.0) as f32;
                self.mouse_pos = (position.x as f32 / scale, position.y as f32 / scale);
            }
            
            WindowEvent::RedrawRequested => {
                let frame_start = Instant::now();
                let dt = frame_start.duration_since(self.last_frame);
                
                // Update simulation
                let t0 = Instant::now();
                if let Some(sim) = &mut self.simulation {
                    if let Err(e) = sim.step(dt.as_secs_f32()) {
                        log::error!("Simulation step failed: {}", e);
                    }
                }
                let t1 = Instant::now();
                
                // Update tooltip
                self.update_tooltip();
                let t2 = Instant::now();
                
                // Render egui overlay (not actually rendered to screen)
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
                }
                let t3 = Instant::now();
                
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
                let t4 = Instant::now();
                
                // Log step count periodically
                if let Some(sim) = &self.simulation {
                    if sim.step_count() > 0 && sim.step_count() % 10 == 0 {
                        log::info!("Step {}: sim={:.1}ms tooltip={:.1}ms egui={:.1}ms render={:.1}ms total={:.1}ms",
                            sim.step_count(),
                            (t1-t0).as_secs_f64()*1000.0,
                            (t2-t1).as_secs_f64()*1000.0,
                            (t3-t2).as_secs_f64()*1000.0,
                            (t4-t3).as_secs_f64()*1000.0,
                            (t4-frame_start).as_secs_f64()*1000.0);
                    }
                }
                
                // Update timing
                self.frame_count += 1;
                self.last_frame = frame_start;
                
                // Update FPS counter
                let fps_elapsed = frame_start.duration_since(self.fps_update_time);
                if fps_elapsed >= Duration::from_secs(1) {
                    self.current_fps = self.frame_count as f64 / fps_elapsed.as_secs_f64();
                    self.frame_count = 0;
                    self.fps_update_time = frame_start;
                }
                
                // Exit after 5 frames in smoke-test mode
                if self.smoke_test && self.frame_count >= 5 {
                    log::info!("Smoke test: 5 frames rendered, exiting");
                    event_loop.exit();
                    return;
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

impl Drop for DemoApp {
    fn drop(&mut self) {
        log::info!("DemoApp::drop - starting cleanup");
        // Wait for GPU to be idle before cleanup
        if let Some(ctx) = &self.render_ctx {
            unsafe { ctx.device.device_wait_idle().ok(); }
        }
        log::info!("DemoApp::drop - GPU idle");
        
        // Drop simulation first (it has Vulkan resources from its own device)
        drop(self.simulation.take());
        log::info!("DemoApp::drop - simulation dropped");
        
        // Drop egui_renderer before render_ctx (it holds a cloned Device handle)
        drop(self.egui_renderer.take());
        log::info!("DemoApp::drop - egui_renderer dropped");
        
        // Clean up render pipeline before render context is dropped
        // Order matters: pipeline uses ctx's allocator
        if let (Some(pipeline), Some(ctx)) = (self.render_pipeline.as_mut(), self.render_ctx.as_ref()) {
            log::info!("Cleaning up render pipeline...");
            pipeline.destroy(ctx);
        }
        log::info!("DemoApp::drop - pipeline destroyed");
        
        // Now it's safe to drop the render_ctx and other resources
        log::info!("DemoApp::drop - cleanup complete, dropping remaining fields");
    }
}

/// Compute species colors: explicit overrides for known species, hash-based fallback.
fn compute_species_colors(registry: &fluidsim::SpeciesRegistry) -> Vec<[f32; 4]> {
    use std::collections::HashMap;

    // Explicit color assignments for known species (RGB, visually distinct from water blue)
    let overrides: HashMap<&str, [f32; 3]> = HashMap::from([
        ("Na+", [0.95, 0.3, 0.2]),   // Red-orange
        ("K+",  [0.2, 0.85, 0.3]),   // Green
        ("Cl-", [0.85, 0.7, 0.2]),   // Gold/yellow
        ("H+",  [1.0, 0.4, 0.7]),    // Pink
        ("OH-", [0.6, 0.2, 0.9]),    // Purple
        ("Ca2+",[0.9, 0.6, 0.1]),    // Orange
    ]);

    registry.iter().map(|info| {
        if let Some(&rgb) = overrides.get(info.name.as_ref()) {
            [rgb[0], rgb[1], rgb[2], 1.0]
        } else {
            // Hash-based fallback: golden ratio distribution, avoid water-blue hue
            let mut h = (info.index as f32 * 0.618033988749895).fract();
            for _ in 0..8 {
                let dist = (h - 0.54).abs().min(1.0 - (h - 0.54).abs());
                if dist >= 0.12 { break; }
                h = (h + 0.17).fract();
            }
            let rgb = hue_to_rgb(h);
            // Slightly desaturate
            let r = 0.5 + (rgb[0] - 0.5) * 0.85;
            let g = 0.5 + (rgb[1] - 0.5) * 0.85;
            let b = 0.5 + (rgb[2] - 0.5) * 0.85;
            [r, g, b, 1.0]
        }
    }).collect()
}

/// Convert hue (0..1) to RGB.
fn hue_to_rgb(h: f32) -> [f32; 3] {
    let hue = h * 6.0;
    let x = 1.0 - (hue % 2.0 - 1.0).abs();
    if hue < 1.0 { [1.0, x, 0.0] }
    else if hue < 2.0 { [x, 1.0, 0.0] }
    else if hue < 3.0 { [0.0, 1.0, x] }
    else if hue < 4.0 { [0.0, x, 1.0] }
    else if hue < 5.0 { [x, 0.0, 1.0] }
    else { [1.0, 0.0, x] }
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
    
    let smoke_test = std::env::args().any(|a| a == "--smoke-test");
    if smoke_test {
        log::info!("Running in smoke-test mode (5 frames then exit)");
    }

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    
    let mut app = DemoApp::new(smoke_test);
    event_loop.run_app(&mut app)?;
    
    Ok(())
}

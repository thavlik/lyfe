use std::time::{Duration, Instant};

use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

use crate::app::DemoApp;
use crate::app_types::{PerformanceSample, RenderFrameMetrics};

impl ApplicationHandler for DemoApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_title("Fluid Simulation Demo")
            .with_inner_size(PhysicalSize::new(1600, 1600));

        match event_loop.create_window(window_attrs) {
            Ok(window) => {
                if let Err(error) = self.initialize(window) {
                    log::error!("Failed to initialize: {}", error);
                    event_loop.exit();
                }
            }
            Err(error) => {
                log::error!("Failed to create window: {}", error);
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let is_performance_toggle = matches!(
            &event,
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    logical_key: Key::Named(NamedKey::Tab),
                    state: ElementState::Pressed,
                    ..
                },
                ..
            }
        );

        if is_performance_toggle {
            self.performance_overlay = !self.performance_overlay;
            if let Some(window) = &self.window {
                window.request_redraw();
            }
            return;
        }

        let egui_wants_event = if let Some(egui) = &mut self.egui_renderer {
            if let Some(window) = &self.window {
                egui.handle_event(window, &event)
            } else {
                false
            }
        } else {
            false
        };

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
                if let Err(error) = self.handle_resize(size.width, size.height) {
                    log::error!("Failed to resize: {}", error);
                }
            }

            WindowEvent::ModifiersChanged(modifiers) => {
                self.modifiers = modifiers.state();
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match logical_key {
                Key::Named(NamedKey::Escape) => {
                    event_loop.exit();
                }
                Key::Named(NamedKey::Delete) => {
                    self.delete_selected_entity();
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
                Key::Character(ref c)
                    if (c == "r" || c == "R")
                        && !self.modifiers.shift_key()
                        && (self.placement_state.is_some() || self.transform_state.is_some()) =>
                {
                    self.rotate_placement_entity();
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
                Key::Named(NamedKey::Space) => {
                    if let Some(sim) = &mut self.simulation {
                        sim.toggle_pause();
                        log::info!(
                            "Simulation {}",
                            if sim.is_paused() { "paused" } else { "resumed" }
                        );
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
                Key::Character(ref c)
                    if (c == "t" || c == "T") && self.selected_entity.is_some() =>
                {
                    self.begin_transform_selected_entity();
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
                Key::Character(ref c) if c == "t" || c == "T" => {
                    self.thermal_view = true;
                }
                _ => {}
            },

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state: ElementState::Released,
                        ..
                    },
                ..
            } => match logical_key {
                Key::Character(ref c) if (c == "r" || c == "R") && self.modifiers.shift_key() => {
                    if let Err(error) = self.reset_simulation() {
                        log::error!("Failed to reset simulation: {}", error);
                    }
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
                Key::Character(ref c) if c == "t" || c == "T" => {
                    self.thermal_view = false;
                }
                _ => {}
            },

            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state: ElementState::Released,
                ..
            } => {
                self.handle_primary_click();
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let scale = self
                    .window
                    .as_ref()
                    .map(|window| window.scale_factor())
                    .unwrap_or(1.0) as f32;
                self.mouse_pos = (position.x as f32 / scale, position.y as f32 / scale);
            }

            WindowEvent::RedrawRequested => {
                let frame_start = Instant::now();
                let dt = frame_start.duration_since(self.last_frame);

                let t0 = Instant::now();
                if let Some(sim) = &mut self.simulation
                    && let Err(error) = sim.step(dt.as_secs_f32())
                {
                    log::error!("Simulation step failed: {}", error);
                }
                let t1 = Instant::now();

                self.update_tooltip();
                self.update_detail_probes();
                let t2 = Instant::now();

                let performance_overlay = self.performance_overlay;
                let performance_summary = self.performance_summary();
                let performance_samples: Vec<_> =
                    self.performance_samples.iter().copied().collect();
                let hovered_coarse_rect = self.hovered_coarse_rect();
                let mut leak_channel_overlays = self.leak_channel_overlays();
                if let Some(ghost) = self.placement_overlay() {
                    leak_channel_overlays.push(ghost);
                }

                if self.egui_renderer.is_some() && self.window.is_some() {
                    {
                        let window = self.window.as_ref().unwrap();
                        let egui = self.egui_renderer.as_mut().unwrap();
                        egui.begin_frame(window);
                    }

                    let egui_ctx = self.egui_renderer.as_ref().unwrap().context().clone();

                    self.draw_editor_ui(&egui_ctx);
                    self.draw_detail_frame(&egui_ctx);

                    if let Some(rect) = hovered_coarse_rect {
                        Self::draw_hovered_coarse_outline(&egui_ctx, rect);
                    }
                    self.draw_enzymes(&egui_ctx);
                    Self::draw_leak_channels(&egui_ctx, &leak_channel_overlays);
                    self.draw_detail_probes(&egui_ctx);

                    if self.show_tooltip {
                        egui::Window::new("Inspection")
                            .order(egui::Order::Tooltip)
                            .fixed_pos(egui::pos2(self.mouse_pos.0 + 15.0, self.mouse_pos.1 + 15.0))
                            .collapsible(false)
                            .resizable(false)
                            .show(&egui_ctx, |ui| {
                                ui.monospace(&self.tooltip_text);
                            });
                    }

                    Self::draw_performance_overlay(
                        &egui_ctx,
                        performance_overlay,
                        performance_summary,
                        &performance_samples,
                    );

                    let _ = self.apply_selected_entity_changes(false);

                    let window = self.window.as_ref().unwrap();
                    let egui = self.egui_renderer.as_mut().unwrap();
                    let _output = egui.end_frame(window);
                }
                let t3 = Instant::now();

                if self.needs_resize
                    && let Some(window) = &self.window
                {
                    let size = window.inner_size();
                    if let Err(error) = self.handle_resize(size.width, size.height) {
                        log::error!("Resize failed: {}", error);
                    }
                }

                let render_metrics = match self.render_frame() {
                    Ok(metrics) => metrics,
                    Err(error) => {
                        log::error!("Render failed: {}", error);
                        RenderFrameMetrics::default()
                    }
                };
                let t4 = Instant::now();

                self.push_performance_sample(PerformanceSample {
                    timestamp: t4,
                    frame_ms: (t4 - frame_start).as_secs_f64() as f32 * 1000.0,
                    simulation_ms: (t1 - t0).as_secs_f64() as f32 * 1000.0,
                    tooltip_ms: (t2 - t1).as_secs_f64() as f32 * 1000.0,
                    ui_ms: (t3 - t2).as_secs_f64() as f32 * 1000.0,
                    render_ms: render_metrics.render_ms,
                    upload_ms: render_metrics.upload_ms,
                });

                if let Some(sim) = &self.simulation
                    && sim.step_count() > 0
                    && sim.step_count().is_multiple_of(100)
                {
                    log::info!(
                        "Step {}: sim={:.1}ms tooltip={:.1}ms egui={:.1}ms render={:.1}ms total={:.1}ms",
                        sim.step_count(),
                        (t1 - t0).as_secs_f64() * 1000.0,
                        (t2 - t1).as_secs_f64() * 1000.0,
                        (t3 - t2).as_secs_f64() * 1000.0,
                        (t4 - t3).as_secs_f64() * 1000.0,
                        (t4 - frame_start).as_secs_f64() * 1000.0
                    );
                }

                self.frame_count += 1;
                self.last_frame = frame_start;

                let fps_elapsed = frame_start.duration_since(self.fps_update_time);
                if fps_elapsed >= Duration::from_secs(1) {
                    self.current_fps = self.frame_count as f64 / fps_elapsed.as_secs_f64();
                    self.frame_count = 0;
                    self.fps_update_time = frame_start;
                }

                if self.smoke_test && self.frame_count >= 5 {
                    log::info!("Smoke test: 5 frames rendered, exiting");
                    event_loop.exit();
                    return;
                }

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
        let window = self.window.take();
        let mut render_ctx = self.render_ctx.take();
        let mut render_pipeline = self.render_pipeline.take();
        let egui_renderer = self.egui_renderer.take();
        let simulation = self.simulation.take();

        if let Some(ctx) = &render_ctx {
            unsafe {
                ctx.device.device_wait_idle().ok();
            }
        }
        log::info!("DemoApp::drop - GPU idle");

        drop(simulation);
        log::info!("DemoApp::drop - simulation dropped");

        drop(egui_renderer);
        log::info!("DemoApp::drop - egui_renderer dropped");

        if let (Some(pipeline), Some(ctx)) = (render_pipeline.as_mut(), render_ctx.as_ref()) {
            log::info!("Cleaning up render pipeline...");
            pipeline.destroy(ctx);
        }
        log::info!("DemoApp::drop - pipeline destroyed");

        drop(render_pipeline);
        drop(render_ctx.take());
        drop(window);
        log::info!("DemoApp::drop - cleanup complete");
    }
}

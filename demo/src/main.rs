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

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::{Parser, Subcommand};
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
const PERFORMANCE_WINDOW: Duration = Duration::from_secs(30);
const TOOLTIP_REFRESH_INTERVAL: Duration = Duration::from_millis(250);

#[derive(Debug, Clone, Copy)]
struct PerformanceSample {
    timestamp: Instant,
    frame_ms: f32,
    simulation_ms: f32,
    tooltip_ms: f32,
    ui_ms: f32,
    render_ms: f32,
    upload_ms: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct RenderFrameMetrics {
    upload_ms: f32,
    render_ms: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct PerformanceSummary {
    current_frame_ms: f32,
    current_fps: f32,
    average_fps_30s: f32,
    worst_frame_ms_30s: f32,
}

#[derive(Debug, Clone, Copy)]
struct LeakChannelOverlay {
    center: egui::Pos2,
    sink: egui::Pos2,
    source: egui::Pos2,
    color: egui::Color32,
    hovered: bool,
}

/// Which scenario to run.
#[derive(Debug, Clone, Copy, Default)]
enum ScenarioKind {
    #[default]
    Basic,
    AcidBase,
    Buffers,
    Leak,
}

#[derive(Parser)]
#[command(name = "lyfe-demo", about = "Fluid simulation demo")]
struct Cli {
    /// Run in smoke-test mode (render 5 frames then exit)
    #[arg(long)]
    smoke_test: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// The original Na⁺/K⁺/Cl⁻ diffusion demo (default)
    Basic,
    /// Acid-base neutralization: H⁺ + OH⁻ → H₂O
    AcidBase,
    /// Weak-acid buffer vs NaOH with acetic-acid equilibrium
    Buffers,
    /// Buffer scenario with membrane leak channels for K+ and Na+
    Leak,
}

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
    last_tooltip_coord: Option<(u32, u32)>,
    hovered_leak_channel: Option<usize>,
    
    // Thermal view (momentary display while T is held)
    thermal_view: bool,
    performance_overlay: bool,
    
    // Timing
    last_frame: Instant,
    frame_count: u64,
    fps_update_time: Instant,
    current_fps: f64,
    performance_samples: VecDeque<PerformanceSample>,
    
    // Flags
    needs_resize: bool,
    smoke_test: bool,
    scenario: ScenarioKind,
}

impl DemoApp {
    fn new(smoke_test: bool, scenario: ScenarioKind) -> Self {
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
            last_tooltip_coord: None,
            hovered_leak_channel: None,
            thermal_view: false,
            performance_overlay: false,
            last_frame: Instant::now(),
            frame_count: 0,
            fps_update_time: Instant::now(),
            current_fps: 0.0,
            performance_samples: VecDeque::new(),
            needs_resize: false,
            smoke_test,
            scenario,
        }
    }

    fn initialize(&mut self, window: Window) -> Result<()> {
        let _size = window.inner_size();
        
        let diffusion_rate = match self.scenario {
            ScenarioKind::Leak => 7.5,
            _ => 5.0,
        };
        let charge_correction_strength = match self.scenario {
            ScenarioKind::Leak => 0.0,
            _ => 1.0,
        };

        let config = SimulationConfig {
            width: 512,
            height: 512,
            diffusion_rate,
            thermal_diffusion_rate: 3.0,
            charge_correction_strength,
            diffusion_substeps: 4,
            inspection_mip: self.inspection_mip,
            time_scale: 20.0,
            reaction_rate_scale: 8.0,
            max_frame_dt: 1.0 / 15.0,
        };

        log::info!("Creating render context...");
        let render_ctx = RenderContext::new(&window)?;
        log::info!("Render context created successfully");

        log::info!("Creating simulation on shared Vulkan device...");
        let simulation = match self.scenario {
            ScenarioKind::Basic => Simulation::new_demo_with_shared_gpu_context(config, render_ctx.shared_gpu_context())?,
            ScenarioKind::AcidBase => Simulation::new_acid_base_with_shared_gpu_context(config, render_ctx.shared_gpu_context())?,
            ScenarioKind::Buffers => Simulation::new_buffers_with_shared_gpu_context(config, render_ctx.shared_gpu_context())?,
            ScenarioKind::Leak => Simulation::new_leak_with_shared_gpu_context(config, render_ctx.shared_gpu_context())?,
        };
        log::info!("Simulation created successfully");
        
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

            if let Some(channel_index) = sim.hovered_leak_channel(grid_x, grid_y) {
                self.hovered_leak_channel = Some(channel_index);
                self.tooltip_text = sim.leak_channel_tooltip(channel_index)
                    .unwrap_or_else(|| "Leak Channel".to_string());
                self.show_tooltip = true;
                return;
            }
            self.hovered_leak_channel = None;

            let coarse_coord = match (sim.coarse_dimensions(), sim.coarse_mip_factor()) {
                (Some((coarse_w, coarse_h)), Some(mip)) => {
                    let coarse_x = ((grid_x.max(0.0) as u32) / mip).min(coarse_w.saturating_sub(1));
                    let coarse_y = ((grid_y.max(0.0) as u32) / mip).min(coarse_h.saturating_sub(1));
                    Some((coarse_x, coarse_y))
                }
                _ => None,
            };
            
            let cached_data = sim.get_cached_inspection();
            let cached_matches_hover = cached_data.as_ref().map(|data| data.coord) == coarse_coord;
            let cached_is_stale = cached_data.as_ref()
                .map(|data| data.age >= TOOLTIP_REFRESH_INTERVAL)
                .unwrap_or(true);
            let hover_changed = coarse_coord != self.last_tooltip_coord;

            if !sim.is_inspection_pending() && (hover_changed || !cached_matches_hover || cached_is_stale) {
                let _ = sim.request_async_inspection(grid_x, grid_y);
            }
            self.last_tooltip_coord = coarse_coord;
            
            // Poll for any completed readbacks
            if let Some(data) = sim.poll_async_inspection() {
                // New data available - format it
                let mut species_rows: Vec<_> = data.concentrations.iter()
                    .enumerate()
                    .filter(|&(_, c)| *c > 0.001)
                    .map(|(i, c)| {
                        let name = self.species_names.get(i)
                            .map(|s| s.as_str())
                            .unwrap_or("?");
                        (name.to_string(), *c)
                    })
                    .collect();
                species_rows.sort_by(|a, b| b.1.total_cmp(&a.1));
                let species_lines: String = species_rows.iter()
                    .map(|(name, conc)| format!("  {:<6} {:>8.3} M", name, conc))
                    .collect::<Vec<_>>()
                    .join("\n");
                self.tooltip_text = format!(
                    "Coarse Cell ({}, {})\n\
                     Fluid: {} / Solid: {}\n\
                     Temp: {:.2} K ({:.2} C)\n\
                     Species:\n{}",
                    data.coord.0, data.coord.1,
                    data.fluid_count, data.solid_count,
                    data.mean_temperature_kelvin,
                    data.mean_temperature_kelvin - 273.15,
                    species_lines,
                );
                self.show_tooltip = true;
            } else if let Some(data) = cached_data {
                // Use cached data (may be stale)
                let mut species_rows: Vec<_> = data.concentrations.iter()
                    .enumerate()
                    .filter(|&(_, c)| *c > 0.001)
                    .map(|(i, c)| {
                        let name = self.species_names.get(i)
                            .map(|s| s.as_str())
                            .unwrap_or("?");
                        (name.to_string(), *c)
                    })
                    .collect();
                species_rows.sort_by(|a, b| b.1.total_cmp(&a.1));
                let species_lines: String = species_rows.iter()
                    .map(|(name, conc)| format!("  {:<6} {:>8.3} M", name, conc))
                    .collect::<Vec<_>>()
                    .join("\n");
                self.tooltip_text = format!(
                    "Coarse Cell ({}, {})\n\
                     Fluid: {} / Solid: {}\n\
                     Temp: {:.2} K ({:.2} C)\n\
                     Species:\n{}",
                    data.coord.0, data.coord.1,
                    data.fluid_count, data.solid_count,
                    data.mean_temperature_kelvin,
                    data.mean_temperature_kelvin - 273.15,
                    species_lines,
                );
                self.show_tooltip = true;
            } else {
                // No data yet
                self.show_tooltip = false;
            }
        }
    }

    fn hovered_coarse_rect(&self) -> Option<egui::Rect> {
        let sim = self.simulation.as_ref()?;
        let window = self.window.as_ref()?;
        let (coarse_x, coarse_y) = self.last_tooltip_coord?;
        let mip = sim.coarse_mip_factor()?;
        let (grid_w, grid_h) = sim.dimensions();

        let size = window.inner_size();
        let scale = window.scale_factor() as f32;
        let logical_w = size.width as f32 / scale;
        let logical_h = size.height as f32 / scale;

        let cell_w = logical_w * mip as f32 / grid_w as f32;
        let cell_h = logical_h * mip as f32 / grid_h as f32;
        let min_x = coarse_x as f32 * cell_w;
        let min_y = coarse_y as f32 * cell_h;

        Some(egui::Rect::from_min_size(
            egui::pos2(min_x, min_y),
            egui::vec2(cell_w, cell_h),
        ))
    }

    fn draw_hovered_coarse_outline(ctx: &egui::Context, rect: egui::Rect) {
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("coarse_hover_outline"),
        ));

        painter.rect_stroke(
            rect.expand(0.5),
            0.0,
            egui::Stroke::new(3.0, egui::Color32::from_rgba_unmultiplied(8, 12, 18, 220)),
        );
        painter.rect_stroke(
            rect,
            0.0,
            egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 242, 120)),
        );
    }

    fn grid_to_screen(&self, grid_x: f32, grid_y: f32) -> Option<egui::Pos2> {
        let sim = self.simulation.as_ref()?;
        let window = self.window.as_ref()?;
        let (grid_w, grid_h) = sim.dimensions();
        let size = window.inner_size();
        let scale = window.scale_factor() as f32;
        let logical_w = size.width as f32 / scale;
        let logical_h = size.height as f32 / scale;

        Some(egui::pos2(
            grid_x * logical_w / grid_w as f32,
            grid_y * logical_h / grid_h as f32,
        ))
    }

    fn leak_channel_overlays(&self) -> Vec<LeakChannelOverlay> {
        let Some(sim) = self.simulation.as_ref() else { return Vec::new(); };
        let mut overlays = Vec::new();

        for (index, channel) in sim.leak_channels().iter().enumerate() {
            let Some(center) = self.grid_to_screen(channel.x as f32 + 0.5, channel.y as f32 + 0.5) else {
                continue;
            };
            let Some(((sink_x, sink_y), (source_x, source_y))) = sim.leak_channel_endpoints(index) else {
                continue;
            };
            let Some(sink) = self.grid_to_screen(sink_x as f32 + 0.5, sink_y as f32 + 0.5) else {
                continue;
            };
            let Some(source) = self.grid_to_screen(source_x as f32 + 0.5, source_y as f32 + 0.5) else {
                continue;
            };

            let species_name = sim.species_registry()
                .get(channel.species)
                .map(|info| info.name.as_ref())
                .unwrap_or("?");
            let color = match species_name {
                "Na+" => egui::Color32::from_rgb(242, 102, 74),
                "K+" => egui::Color32::from_rgb(92, 214, 110),
                _ => egui::Color32::from_rgb(230, 230, 230),
            };
            overlays.push(LeakChannelOverlay {
                center,
                sink,
                source,
                color,
                hovered: self.hovered_leak_channel == Some(index),
            });
        }

        overlays
    }

    fn draw_leak_channels(ctx: &egui::Context, overlays: &[LeakChannelOverlay]) {
        if overlays.is_empty() {
            return;
        }

        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("leak_channels"),
        ));

        for overlay in overlays {
            let outline = if overlay.hovered {
                egui::Color32::WHITE
            } else {
                egui::Color32::from_rgb(18, 22, 28)
            };

            painter.line_segment(
                [overlay.sink, overlay.source],
                egui::Stroke::new(5.0, egui::Color32::from_rgba_unmultiplied(14, 18, 24, 210)),
            );
            painter.circle_filled(overlay.center, 8.5, egui::Color32::from_rgba_unmultiplied(236, 240, 245, 210));
            painter.circle_stroke(
                overlay.center,
                8.5,
                egui::Stroke::new(if overlay.hovered { 3.0 } else { 2.0 }, outline),
            );
            painter.circle_filled(overlay.center, 4.2, overlay.color);

            let tangent = egui::vec2(overlay.source.x - overlay.sink.x, overlay.source.y - overlay.sink.y).normalized();
            let normal = egui::vec2(-tangent.y, tangent.x);
            let left_top = overlay.center - tangent * 10.0 + normal * 5.0;
            let left_bottom = overlay.center - tangent * 10.0 - normal * 5.0;
            let right_top = overlay.center + tangent * 10.0 + normal * 5.0;
            let right_bottom = overlay.center + tangent * 10.0 - normal * 5.0;
            painter.line_segment([left_top, left_bottom], egui::Stroke::new(2.0, outline));
            painter.line_segment([right_top, right_bottom], egui::Stroke::new(2.0, outline));

            let arrow_base = overlay.center + tangent * 8.0;
            let arrow_tip = overlay.source;
            let arrow_left = arrow_base - tangent * 5.5 + normal * 3.5;
            let arrow_right = arrow_base - tangent * 5.5 - normal * 3.5;
            painter.line_segment([arrow_base, arrow_tip], egui::Stroke::new(2.0, overlay.color));
            painter.line_segment([arrow_tip, arrow_left], egui::Stroke::new(2.0, overlay.color));
            painter.line_segment([arrow_tip, arrow_right], egui::Stroke::new(2.0, overlay.color));
        }
    }

    fn push_performance_sample(&mut self, sample: PerformanceSample) {
        self.performance_samples.push_back(sample);
        let cutoff = sample.timestamp - PERFORMANCE_WINDOW;
        while let Some(front) = self.performance_samples.front() {
            if front.timestamp >= cutoff {
                break;
            }
            self.performance_samples.pop_front();
        }
    }

    fn performance_summary(&self) -> PerformanceSummary {
        let current_frame_ms = self.performance_samples.back().map(|s| s.frame_ms).unwrap_or(0.0);
        let current_fps = if current_frame_ms > 0.0 { 1000.0 / current_frame_ms } else { 0.0 };

        let average_fps_30s = match (self.performance_samples.front(), self.performance_samples.back()) {
            (Some(first), Some(last)) if self.performance_samples.len() >= 2 => {
                let elapsed = (last.timestamp - first.timestamp).as_secs_f32();
                if elapsed > 0.0 {
                    (self.performance_samples.len() as f32 - 1.0) / elapsed
                } else {
                    current_fps
                }
            }
            _ => current_fps,
        };

        let worst_frame_ms_30s = self.performance_samples.iter()
            .map(|s| s.frame_ms)
            .fold(0.0_f32, f32::max);

        PerformanceSummary {
            current_frame_ms,
            current_fps,
            average_fps_30s,
            worst_frame_ms_30s,
        }
    }

    fn draw_performance_overlay(
        ctx: &egui::Context,
        performance_overlay: bool,
        summary: PerformanceSummary,
        samples: &[PerformanceSample],
    ) {
        use egui::{Align2, Color32, Frame, Margin, RichText, Sense, Stroke, Vec2};

        if !performance_overlay {
            return;
        }

        let metrics = [
            ("Frame", Color32::from_rgb(255, 99, 71), samples.iter().map(|s| s.frame_ms).collect::<Vec<_>>()),
            ("Simulation", Color32::from_rgb(86, 204, 242), samples.iter().map(|s| s.simulation_ms).collect::<Vec<_>>()),
            ("Render", Color32::from_rgb(242, 201, 76), samples.iter().map(|s| s.render_ms).collect::<Vec<_>>()),
            ("UI", Color32::from_rgb(155, 81, 224), samples.iter().map(|s| s.ui_ms).collect::<Vec<_>>()),
            ("Upload", Color32::from_rgb(39, 174, 96), samples.iter().map(|s| s.upload_ms).collect::<Vec<_>>()),
            ("Tooltip", Color32::from_rgb(235, 87, 87), samples.iter().map(|s| s.tooltip_ms).collect::<Vec<_>>()),
        ];

        egui::Area::new("performance_stats".into())
            .anchor(Align2::RIGHT_TOP, egui::vec2(-16.0, 16.0))
            .show(ctx, |ui: &mut egui::Ui| {
                Frame::none()
                    .fill(Color32::from_rgba_unmultiplied(8, 12, 18, 220))
                    .stroke(Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 24)))
                    .inner_margin(Margin::same(12.0))
                    .show(ui, |ui: &mut egui::Ui| {
                        ui.set_min_width(240.0);
                        ui.label(RichText::new("Performance").strong().size(18.0).color(Color32::WHITE));
                        ui.add_space(6.0);
                        ui.label(format!("Frame: {:.2} ms", summary.current_frame_ms));
                        ui.label(format!("FPS: {:.1}", summary.current_fps));
                        ui.label(format!("Avg FPS (30s): {:.1}", summary.average_fps_30s));
                        ui.label(format!("Worst Frame (30s): {:.2} ms", summary.worst_frame_ms_30s));
                    });
            });

        egui::Area::new("performance_graph".into())
            .anchor(Align2::LEFT_BOTTOM, egui::vec2(16.0, -16.0))
            .show(ctx, |ui: &mut egui::Ui| {
                let available_width = (ctx.screen_rect().width() - 32.0).max(480.0);
                let graph_size = Vec2::new(available_width, 210.0);

                Frame::none()
                    .fill(Color32::from_rgba_unmultiplied(7, 10, 15, 210))
                    .stroke(Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 24)))
                    .inner_margin(Margin::same(12.0))
                    .show(ui, |ui: &mut egui::Ui| {
                        ui.horizontal_wrapped(|ui: &mut egui::Ui| {
                            for (label, color, _) in &metrics {
                                ui.colored_label(*color, "■");
                                ui.label(*label);
                                ui.add_space(10.0);
                            }
                        });
                        ui.add_space(10.0);

                        let (rect, _) = ui.allocate_exact_size(graph_size, Sense::hover());
                        let painter = ui.painter_at(rect);
                        painter.rect_filled(rect, 10.0, Color32::from_rgba_unmultiplied(15, 21, 30, 200));
                        painter.rect_stroke(
                            rect,
                            10.0,
                            Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 20)),
                        );

                        let left_pad = 42.0;
                        let right_pad = 10.0;
                        let top_pad = 10.0;
                        let bottom_pad = 24.0;
                        let plot = egui::Rect::from_min_max(
                            egui::pos2(rect.left() + left_pad, rect.top() + top_pad),
                            egui::pos2(rect.right() - right_pad, rect.bottom() - bottom_pad),
                        );

                        let max_ms = metrics.iter()
                            .flat_map(|(_, _, values)| values.iter().copied())
                            .fold(16.0_f32, f32::max)
                            .max(1.0);

                        for step in 0..=4 {
                            let t = step as f32 / 4.0;
                            let y = egui::lerp(plot.bottom()..=plot.top(), t);
                            painter.line_segment(
                                [egui::pos2(plot.left(), y), egui::pos2(plot.right(), y)],
                                Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 18)),
                            );
                            let value = max_ms * t;
                            painter.text(
                                egui::pos2(plot.left() - 8.0, y),
                                Align2::RIGHT_CENTER,
                                format!("{value:.0}"),
                                egui::TextStyle::Small.resolve(ui.style()),
                                Color32::from_gray(180),
                            );
                        }

                        if samples.len() >= 2 {
                            for (label, color, values) in &metrics {
                                let _ = label;
                                let points: Vec<_> = values.iter().enumerate().map(|(index, value)| {
                                    let x_t = index as f32 / (values.len().saturating_sub(1) as f32);
                                    let y_t = (*value / max_ms).clamp(0.0, 1.0);
                                    egui::pos2(
                                        egui::lerp(plot.left()..=plot.right(), x_t),
                                        egui::lerp(plot.bottom()..=plot.top(), y_t),
                                    )
                                }).collect();
                                painter.add(egui::Shape::line(points, Stroke::new(2.0, *color)));
                            }
                        }

                        painter.text(
                            egui::pos2(plot.left(), rect.bottom() - 10.0),
                            Align2::LEFT_BOTTOM,
                            "30s ago",
                            egui::TextStyle::Small.resolve(ui.style()),
                            Color32::from_gray(160),
                        );
                        painter.text(
                            egui::pos2(plot.right(), rect.bottom() - 10.0),
                            Align2::RIGHT_BOTTOM,
                            "now",
                            egui::TextStyle::Small.resolve(ui.style()),
                            Color32::from_gray(160),
                        );
                    });
            });
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

    fn render_frame(&mut self) -> Result<RenderFrameMetrics> {
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
                return Ok(RenderFrameMetrics::default());
            }
        };

        let t0 = Instant::now();
        pipeline.bind_simulation_buffers(ctx, sim.gpu_render_buffers());
        let t1 = Instant::now();
        
        // Debug: Log frame presentation
        let step = sim.step_count();
        if step > 0 && step % 10 == 0 {
            log::info!("Frame timing: bind={:.1}ms", 
                (t1-t0).as_secs_f64()*1000.0);
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

        sim.record_render_barriers(cmd);
        
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

        let t3 = Instant::now();
        Ok(RenderFrameMetrics {
            upload_ms: (t1 - t0).as_secs_f64() as f32 * 1000.0,
            render_ms: (t3 - t1).as_secs_f64() as f32 * 1000.0,
        })
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

                let performance_overlay = self.performance_overlay;
                let performance_summary = self.performance_summary();
                let performance_samples: Vec<_> = self.performance_samples.iter().copied().collect();
                let hovered_coarse_rect = self.hovered_coarse_rect();
                let leak_channel_overlays = self.leak_channel_overlays();
                
                // Render egui overlay (not actually rendered to screen)
                if let (Some(egui), Some(window)) = (&mut self.egui_renderer, &self.window) {
                    egui.begin_frame(window);

                    if let Some(rect) = hovered_coarse_rect {
                        Self::draw_hovered_coarse_outline(egui.context(), rect);
                    }
                    Self::draw_leak_channels(egui.context(), &leak_channel_overlays);
                    
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

                    Self::draw_performance_overlay(
                        egui.context(),
                        performance_overlay,
                        performance_summary,
                        &performance_samples,
                    );
                    
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
                
                let render_metrics = match self.render_frame() {
                    Ok(metrics) => metrics,
                    Err(e) => {
                        log::error!("Render failed: {}", e);
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
        ("SO4(2-)", [0.3, 0.7, 0.9]),// Teal
        ("CH3COOH", [0.45, 0.9, 0.55]), // Mint green
        ("CH3COO-", [0.9, 0.55, 0.15]), // Amber
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
    
    let cli = Cli::parse();

    let scenario = match cli.command {
        Some(Commands::AcidBase) => ScenarioKind::AcidBase,
        Some(Commands::Buffers) => ScenarioKind::Buffers,
        Some(Commands::Leak) => ScenarioKind::Leak,
        Some(Commands::Basic) | None => ScenarioKind::Basic,
    };

    log::info!("Starting Fluid Simulation Demo ({:?} scenario)", scenario);
    log::info!("Controls:");
    log::info!("  Mouse hover: Inspect cell");
    log::info!("  Space: Toggle pause");
    log::info!("  +/-: Adjust inspection mip");
    log::info!("  Escape: Exit");
    
    if cli.smoke_test {
        log::info!("Running in smoke-test mode (5 frames then exit)");
    }

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    
    let mut app = DemoApp::new(cli.smoke_test, scenario);
    event_loop.run_app(&mut app)?;
    
    Ok(())
}

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use fluidsim::Simulation;
use renderer::{EguiRenderer, PresentModePreference, RenderContext, RenderPipeline};
use winit::{keyboard::ModifiersState, window::Window};

use crate::app_types::{
    DetailPanelSlot, DetailProbeSnapshot, DetailProbeSpec, LeakChannelDraft, PerformanceSample,
    PerformanceSummary, PlacementState, ScenarioBoxBounds, SelectedEntity, SimulationViewport,
    TransformState,
};
use crate::cli::ScenarioKind;

pub(crate) const PERFORMANCE_WINDOW: Duration = Duration::from_secs(30);
pub(crate) const TOOLTIP_REFRESH_INTERVAL: Duration = Duration::from_millis(250);
pub(crate) const ENTITY_EDIT_DEBOUNCE: Duration = Duration::from_millis(500);
pub(crate) const DEFAULT_EDITOR_LEAK_RATE: f32 = 4.5;
pub(crate) const DETAIL_MARGIN_MIN: f32 = 108.0;
pub(crate) const DETAIL_MARGIN_MAX: f32 = 220.0;
pub(crate) const DETAIL_PANEL_WIDTH: f32 = 250.0;

pub(crate) struct DemoApp {
    pub(crate) window: Option<Window>,
    pub(crate) render_ctx: Option<RenderContext>,
    pub(crate) render_pipeline: Option<RenderPipeline>,
    pub(crate) egui_renderer: Option<EguiRenderer>,
    pub(crate) simulation: Option<Simulation>,
    pub(crate) mouse_pos: (f32, f32),
    pub(crate) inspection_mip: u32,
    pub(crate) show_tooltip: bool,
    pub(crate) tooltip_text: String,
    pub(crate) species_names: Vec<String>,
    pub(crate) last_tooltip_coord: Option<(u32, u32)>,
    pub(crate) hovered_leak_channel: Option<usize>,
    pub(crate) thermal_view: bool,
    pub(crate) performance_overlay: bool,
    pub(crate) modifiers: ModifiersState,
    pub(crate) create_menu_open: bool,
    pub(crate) placement_state: Option<PlacementState>,
    pub(crate) transform_state: Option<TransformState>,
    pub(crate) selected_entity: Option<SelectedEntity>,
    pub(crate) inspector_draft: Option<LeakChannelDraft>,
    pub(crate) inspector_dirty: bool,
    pub(crate) inspector_last_apply: Instant,
    pub(crate) inspector_error: Option<String>,
    pub(crate) last_frame: Instant,
    pub(crate) frame_count: u64,
    pub(crate) fps_update_time: Instant,
    pub(crate) current_fps: f64,
    pub(crate) performance_samples: VecDeque<PerformanceSample>,
    pub(crate) detail_view: bool,
    pub(crate) detail_probes: Vec<DetailProbeSnapshot>,
    pub(crate) detail_last_refresh: Instant,
    pub(crate) present_mode: PresentModePreference,
    pub(crate) needs_resize: bool,
    pub(crate) smoke_test: bool,
    pub(crate) scenario: ScenarioKind,
}

impl DemoApp {
    pub(crate) fn new(
        smoke_test: bool,
        scenario: ScenarioKind,
        detail_view: bool,
        present_mode: PresentModePreference,
    ) -> Self {
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
            modifiers: ModifiersState::empty(),
            create_menu_open: false,
            placement_state: None,
            transform_state: None,
            selected_entity: None,
            inspector_draft: None,
            inspector_dirty: false,
            inspector_last_apply: Instant::now(),
            inspector_error: None,
            last_frame: Instant::now(),
            frame_count: 0,
            fps_update_time: Instant::now(),
            current_fps: 0.0,
            performance_samples: VecDeque::new(),
            detail_view,
            detail_probes: Vec::new(),
            detail_last_refresh: Instant::now() - TOOLTIP_REFRESH_INTERVAL,
            present_mode,
            needs_resize: false,
            smoke_test,
            scenario,
        }
    }

    pub(crate) fn simulation_viewport(&self) -> Option<SimulationViewport> {
        let window = self.window.as_ref()?;
        let size = window.inner_size();
        let scale = window.scale_factor() as f32;
        if scale <= 0.0 {
            return None;
        }

        let logical_w = size.width as f32 / scale;
        let logical_h = size.height as f32 / scale;
        if logical_w <= 0.0 || logical_h <= 0.0 {
            return None;
        }

        let rect = if self.detail_view {
            let margin =
                (logical_w.min(logical_h) * 0.14).clamp(DETAIL_MARGIN_MIN, DETAIL_MARGIN_MAX);
            let side = (logical_w - 2.0 * margin)
                .min(logical_h - 2.0 * margin)
                .max(1.0);
            egui::Rect::from_center_size(
                egui::pos2(logical_w * 0.5, logical_h * 0.5),
                egui::vec2(side, side),
            )
        } else {
            egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(logical_w, logical_h))
        };

        let physical_x = (rect.left() * scale).round().max(0.0) as u32;
        let physical_y = (rect.top() * scale).round().max(0.0) as u32;
        let physical_width = (rect.width() * scale).round().max(1.0) as u32;
        let physical_height = (rect.height() * scale).round().max(1.0) as u32;

        Some(SimulationViewport {
            rect,
            physical_x,
            physical_y,
            physical_width: physical_width.min(size.width.saturating_sub(physical_x).max(1)),
            physical_height: physical_height.min(size.height.saturating_sub(physical_y).max(1)),
        })
    }

    pub(crate) fn scenario_box_bounds(&self) -> Option<ScenarioBoxBounds> {
        let (width, height) = self.simulation.as_ref()?.dimensions();
        Some(ScenarioBoxBounds::from_dimensions(width, height))
    }

    pub(crate) fn mouse_grid_position(&self) -> Option<(f32, f32)> {
        let sim = self.simulation.as_ref()?;
        let viewport = self.simulation_viewport()?;
        let pointer = egui::pos2(self.mouse_pos.0, self.mouse_pos.1);
        if !viewport.rect.contains(pointer) {
            return None;
        }

        let (grid_w, grid_h) = sim.dimensions();
        let grid_x = (pointer.x - viewport.rect.left()) * grid_w as f32 / viewport.rect.width();
        let grid_y = (pointer.y - viewport.rect.top()) * grid_h as f32 / viewport.rect.height();
        Some((
            grid_x.clamp(0.0, grid_w.saturating_sub(1) as f32),
            grid_y.clamp(0.0, grid_h.saturating_sub(1) as f32),
        ))
    }

    pub(crate) fn coarse_rect(&self, coarse_x: u32, coarse_y: u32, mip: u32) -> Option<egui::Rect> {
        let sim = self.simulation.as_ref()?;
        let viewport = self.simulation_viewport()?;
        let (grid_w, grid_h) = sim.dimensions();
        let cell_w = viewport.rect.width() * mip as f32 / grid_w as f32;
        let cell_h = viewport.rect.height() * mip as f32 / grid_h as f32;
        let min_x = viewport.rect.left() + coarse_x as f32 * cell_w;
        let min_y = viewport.rect.top() + coarse_y as f32 * cell_h;
        Some(egui::Rect::from_min_size(
            egui::pos2(min_x, min_y),
            egui::vec2(cell_w, cell_h),
        ))
    }

    pub(crate) fn closest_point_on_rect(rect: egui::Rect, point: egui::Pos2) -> egui::Pos2 {
        egui::pos2(
            point.x.clamp(rect.left(), rect.right()),
            point.y.clamp(rect.top(), rect.bottom()),
        )
    }

    pub(crate) fn clamp_grid_sample(&self, x: f32, y: f32) -> Option<(f32, f32)> {
        let (grid_w, grid_h) = self.simulation.as_ref()?.dimensions();
        Some((
            x.clamp(0.5, grid_w.saturating_sub(1) as f32 - 0.5),
            y.clamp(0.5, grid_h.saturating_sub(1) as f32 - 0.5),
        ))
    }

    pub(crate) fn detail_probe_specs(&self) -> Option<Vec<DetailProbeSpec>> {
        let bounds = self.scenario_box_bounds()?;
        let inset = (self.inspection_mip as f32 * 1.35).max(12.0);
        let outside = (self.inspection_mip as f32 * 1.75).max(14.0);
        let mid_y = (bounds.inner_y0 + bounds.inner_y1) as f32 * 0.5;

        let top_right = self.clamp_grid_sample(
            bounds.inner_x1 as f32 - inset,
            bounds.inner_y0 as f32 + inset,
        )?;
        let bottom_left = self.clamp_grid_sample(
            bounds.inner_x0 as f32 + inset,
            bounds.inner_y1 as f32 - inset,
        )?;
        let center = self.clamp_grid_sample(
            (bounds.inner_x0 + bounds.inner_x1) as f32 * 0.5,
            (bounds.inner_y0 + bounds.inner_y1) as f32 * 0.5,
        )?;
        let outside_left = self.clamp_grid_sample(bounds.outer_x0 as f32 - outside, mid_y)?;
        let outside_right = self.clamp_grid_sample(bounds.outer_x1 as f32 + outside, mid_y)?;

        Some(vec![
            DetailProbeSpec {
                title: "Top Right",
                slot: DetailPanelSlot::TopRight,
                sample_grid: top_right,
            },
            DetailProbeSpec {
                title: "Bottom Left",
                slot: DetailPanelSlot::BottomLeft,
                sample_grid: bottom_left,
            },
            DetailProbeSpec {
                title: "Center",
                slot: DetailPanelSlot::TopCenter,
                sample_grid: center,
            },
            DetailProbeSpec {
                title: "Outside Left",
                slot: DetailPanelSlot::LeftCenter,
                sample_grid: outside_left,
            },
            DetailProbeSpec {
                title: "Outside Right",
                slot: DetailPanelSlot::RightCenter,
                sample_grid: outside_right,
            },
        ])
    }

    pub(crate) fn grid_position_from_mouse(&self) -> Option<(i32, i32)> {
        let (grid_x, grid_y) = self.mouse_grid_position()?;
        Some((grid_x.floor() as i32, grid_y.floor() as i32))
    }

    pub(crate) fn default_leak_species_name(&self) -> String {
        let Some(sim) = self.simulation.as_ref() else {
            return "Na+".to_string();
        };

        if sim.species_registry().id_of("Na+").is_some() {
            "Na+".to_string()
        } else {
            sim.species_registry()
                .iter()
                .next()
                .map(|species| species.name.to_string())
                .unwrap_or_else(|| "Na+".to_string())
        }
    }

    pub(crate) fn default_leak_channel_draft(&self) -> LeakChannelDraft {
        let (x, y) = self.grid_position_from_mouse().unwrap_or((0, 0));
        LeakChannelDraft {
            species_name: self.default_leak_species_name(),
            rate: DEFAULT_EDITOR_LEAK_RATE,
            x,
            y,
            rotation_byte: 0,
        }
    }

    pub(crate) fn grid_to_screen(&self, grid_x: f32, grid_y: f32) -> Option<egui::Pos2> {
        let sim = self.simulation.as_ref()?;
        let viewport = self.simulation_viewport()?;
        let (grid_w, grid_h) = sim.dimensions();

        Some(egui::pos2(
            viewport.rect.left() + grid_x * viewport.rect.width() / grid_w as f32,
            viewport.rect.top() + grid_y * viewport.rect.height() / grid_h as f32,
        ))
    }

    pub(crate) fn grid_to_screen_scale(&self) -> Option<(f32, f32)> {
        let sim = self.simulation.as_ref()?;
        let viewport = self.simulation_viewport()?;
        let (grid_w, grid_h) = sim.dimensions();
        Some((
            viewport.rect.width() / grid_w as f32,
            viewport.rect.height() / grid_h as f32,
        ))
    }

    pub(crate) fn push_performance_sample(&mut self, sample: PerformanceSample) {
        self.performance_samples.push_back(sample);
        let cutoff = sample.timestamp - PERFORMANCE_WINDOW;
        while let Some(front) = self.performance_samples.front() {
            if front.timestamp >= cutoff {
                break;
            }
            self.performance_samples.pop_front();
        }
    }

    pub(crate) fn performance_summary(&self) -> PerformanceSummary {
        let current_frame_ms = self
            .performance_samples
            .back()
            .map(|sample| sample.frame_ms)
            .unwrap_or(0.0);
        let current_fps = if current_frame_ms > 0.0 {
            1000.0 / current_frame_ms
        } else {
            0.0
        };

        let average_fps_30s = match (
            self.performance_samples.front(),
            self.performance_samples.back(),
        ) {
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

        let worst_frame_ms_30s = self
            .performance_samples
            .iter()
            .map(|sample| sample.frame_ms)
            .fold(0.0_f32, f32::max);

        PerformanceSummary {
            current_frame_ms,
            current_fps,
            average_fps_30s,
            worst_frame_ms_30s,
        }
    }
}

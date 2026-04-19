use std::time::Instant;

use anyhow::Result;
use fluidsim::Simulation;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ScenarioBoxBounds {
    pub(crate) outer_x0: u32,
    pub(crate) outer_x1: u32,
    pub(crate) inner_x0: u32,
    pub(crate) inner_y0: u32,
    pub(crate) inner_x1: u32,
    pub(crate) inner_y1: u32,
}

impl ScenarioBoxBounds {
    pub(crate) fn from_dimensions(width: u32, height: u32) -> Self {
        let wall_thickness = 4u32;
        let inner_size = (width.min(height) / 2).max(64);
        let outer_size = inner_size + 2 * wall_thickness;

        let center_x = width / 2;
        let center_y = height / 2;

        let outer_x0 = center_x - outer_size / 2;
        let outer_y0 = center_y - outer_size / 2;
        let outer_x1 = outer_x0 + outer_size;
        let outer_y1 = outer_y0 + outer_size;

        Self {
            outer_x0,
            outer_x1,
            inner_x0: outer_x0 + wall_thickness,
            inner_y0: outer_y0 + wall_thickness,
            inner_x1: outer_x1 - wall_thickness,
            inner_y1: outer_y1 - wall_thickness,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SimulationViewport {
    pub(crate) rect: egui::Rect,
    pub(crate) physical_x: u32,
    pub(crate) physical_y: u32,
    pub(crate) physical_width: u32,
    pub(crate) physical_height: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum DetailPanelSlot {
    TopRight,
    BottomLeft,
    TopCenter,
    LeftCenter,
    RightCenter,
}

impl DetailPanelSlot {
    pub(crate) fn anchor(self) -> egui::Align2 {
        match self {
            Self::TopRight => egui::Align2::RIGHT_TOP,
            Self::BottomLeft => egui::Align2::LEFT_BOTTOM,
            Self::TopCenter => egui::Align2::CENTER_TOP,
            Self::LeftCenter => egui::Align2::LEFT_CENTER,
            Self::RightCenter => egui::Align2::RIGHT_CENTER,
        }
    }

    pub(crate) fn offset(self) -> egui::Vec2 {
        match self {
            Self::TopRight => egui::vec2(-18.0, 18.0),
            Self::BottomLeft => egui::vec2(18.0, -18.0),
            Self::TopCenter => egui::vec2(0.0, 18.0),
            Self::LeftCenter => egui::vec2(18.0, 110.0),
            Self::RightCenter => egui::vec2(-18.0, 0.0),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DetailProbeSnapshot {
    pub(crate) title: &'static str,
    pub(crate) slot: DetailPanelSlot,
    pub(crate) sample_grid: (f32, f32),
    pub(crate) coarse_coord: (u32, u32),
    pub(crate) mip: u32,
    pub(crate) tooltip_text: String,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DetailProbeSpec {
    pub(crate) title: &'static str,
    pub(crate) slot: DetailPanelSlot,
    pub(crate) sample_grid: (f32, f32),
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PerformanceSample {
    pub(crate) timestamp: Instant,
    pub(crate) frame_ms: f32,
    pub(crate) simulation_ms: f32,
    pub(crate) tooltip_ms: f32,
    pub(crate) ui_ms: f32,
    pub(crate) render_ms: f32,
    pub(crate) upload_ms: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RenderFrameMetrics {
    pub(crate) upload_ms: f32,
    pub(crate) render_ms: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PerformanceSummary {
    pub(crate) current_frame_ms: f32,
    pub(crate) current_fps: f32,
    pub(crate) average_fps_30s: f32,
    pub(crate) worst_frame_ms_30s: f32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct LeakChannelOverlay {
    pub(crate) center: egui::Pos2,
    pub(crate) sink: egui::Pos2,
    pub(crate) source: egui::Pos2,
    pub(crate) color: egui::Color32,
    pub(crate) hovered: bool,
    pub(crate) selected: bool,
    pub(crate) ghost: bool,
    pub(crate) valid: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EntityKind {
    LeakChannel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SelectedEntity {
    LeakChannel(usize),
}

#[derive(Debug, Clone)]
pub(crate) struct LeakChannelDraft {
    pub(crate) species_name: String,
    pub(crate) rate: f32,
    pub(crate) x: i32,
    pub(crate) y: i32,
    pub(crate) rotation_byte: u8,
}

impl LeakChannelDraft {
    pub(crate) fn from_channel(channel: &fluidsim::LeakChannel, sim: &Simulation) -> Self {
        let species_name = sim
            .species_registry()
            .get(channel.species)
            .map(|species| species.name.to_string())
            .unwrap_or_default();
        Self {
            species_name,
            rate: channel.rate,
            x: channel.x,
            y: channel.y,
            rotation_byte: channel.rotation_byte(),
        }
    }

    pub(crate) fn to_channel(&self, sim: &Simulation) -> Result<fluidsim::LeakChannel> {
        let species = sim
            .species_registry()
            .id_of(self.species_name.trim())
            .ok_or_else(|| anyhow::anyhow!("Unknown species '{}'", self.species_name.trim()))?;
        Ok(fluidsim::LeakChannel::new(
            self.rate.max(0.0),
            species,
            self.x,
            self.y,
            self.rotation_byte as i8,
        ))
    }

    pub(crate) fn rotate_eighth_turn(&mut self) {
        self.rotation_byte = self
            .rotation_byte
            .wrapping_add(fluidsim::LeakChannel::EIGHTH_TURN);
    }

    pub(crate) fn rotation_degrees(&self) -> u32 {
        let step = (self.rotation_byte as u32 + 16) / 32;
        (step % 8) * 45
    }

    pub(crate) fn set_rotation_degrees(&mut self, degrees: u32) {
        let snapped = ((degrees + 22) / 45) % 8;
        self.rotation_byte = (snapped as u8).saturating_mul(fluidsim::LeakChannel::EIGHTH_TURN);
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PlacementState {
    pub(crate) kind: EntityKind,
    pub(crate) leak_channel: LeakChannelDraft,
}

#[derive(Debug, Clone)]
pub(crate) struct TransformState {
    pub(crate) entity: SelectedEntity,
    pub(crate) leak_channel: LeakChannelDraft,
    pub(crate) mouse_offset: (i32, i32),
}

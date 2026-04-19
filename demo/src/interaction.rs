use anyhow::Result;

use crate::app::{DemoApp, ENTITY_EDIT_DEBOUNCE, TOOLTIP_REFRESH_INTERVAL};
use crate::app_types::{
    DetailProbeSnapshot, EntityKind, LeakChannelOverlay, SelectedEntity, TransformState,
};
use crate::colors::leak_channel_color;
use crate::tooltip::{format_async_inspection_tooltip, format_detail_probe_tooltip};

impl DemoApp {
    pub(crate) fn update_detail_probes(&mut self) {
        if !self.detail_view {
            self.detail_probes.clear();
            return;
        }
        if !self.detail_probes.is_empty()
            && self.detail_last_refresh.elapsed() < TOOLTIP_REFRESH_INTERVAL
        {
            return;
        }

        let Some(specs) = self.detail_probe_specs() else {
            self.detail_probes.clear();
            return;
        };
        let Some(sim) = self.simulation.as_mut() else {
            self.detail_probes.clear();
            return;
        };

        let mut probes = Vec::with_capacity(specs.len());
        for spec in specs {
            match sim.inspect_with_mip(
                spec.sample_grid.0,
                spec.sample_grid.1,
                self.inspection_mip.max(1),
            ) {
                Ok(result) => probes.push(DetailProbeSnapshot {
                    title: spec.title,
                    slot: spec.slot,
                    sample_grid: spec.sample_grid,
                    coarse_coord: (result.coord.x, result.coord.y),
                    mip: result.coord.mip,
                    tooltip_text: format_detail_probe_tooltip(&result),
                }),
                Err(error) => probes.push(DetailProbeSnapshot {
                    title: spec.title,
                    slot: spec.slot,
                    sample_grid: spec.sample_grid,
                    coarse_coord: (0, 0),
                    mip: self.inspection_mip.max(1),
                    tooltip_text: format!("Inspection unavailable\n{}", error),
                }),
            }
        }

        self.detail_probes = probes;
        self.detail_last_refresh = std::time::Instant::now();
    }

    pub(crate) fn begin_placing_entity(&mut self, kind: EntityKind) {
        self.create_menu_open = false;
        self.transform_state = None;
        self.placement_state = Some(match kind {
            EntityKind::LeakChannel => crate::app_types::PlacementState {
                kind,
                leak_channel: self.default_leak_channel_draft(),
            },
        });
    }

    pub(crate) fn preview_leak_channel(&self) -> Option<fluidsim::LeakChannel> {
        let sim = self.simulation.as_ref()?;
        let draft = if let Some(transform) = &self.transform_state {
            if transform.entity != self.selected_entity? {
                return None;
            }
            let mut draft = transform.leak_channel.clone();
            let (mouse_x, mouse_y) = self.grid_position_from_mouse()?;
            draft.x = (mouse_x + transform.mouse_offset.0)
                .clamp(0, sim.dimensions().0.saturating_sub(1) as i32);
            draft.y = (mouse_y + transform.mouse_offset.1)
                .clamp(0, sim.dimensions().1.saturating_sub(1) as i32);
            draft
        } else {
            let placement = self.placement_state.as_ref()?;
            if placement.kind != EntityKind::LeakChannel {
                return None;
            }

            let mut draft = placement.leak_channel.clone();
            let (x, y) = self.grid_position_from_mouse()?;
            draft.x = x;
            draft.y = y;
            draft
        };
        draft.to_channel(sim).ok()
    }

    pub(crate) fn rotate_placement_entity(&mut self) {
        if let Some(transform) = &mut self.transform_state {
            if matches!(transform.entity, SelectedEntity::LeakChannel(_)) {
                transform.leak_channel.rotate_eighth_turn();
            }
        } else if let Some(placement) = &mut self.placement_state
            && placement.kind == EntityKind::LeakChannel
        {
            placement.leak_channel.rotate_eighth_turn();
        }
    }

    pub(crate) fn begin_transform_selected_entity(&mut self) {
        let Some(selection) = self.selected_entity else {
            return;
        };
        let Some((mouse_x, mouse_y)) = self.grid_position_from_mouse() else {
            return;
        };
        let Some(sim) = self.simulation.as_ref() else {
            return;
        };

        let transform_state = match selection {
            SelectedEntity::LeakChannel(index) => {
                let Some(channel) = sim.leak_channels().get(index) else {
                    return;
                };
                let draft = crate::app_types::LeakChannelDraft::from_channel(channel, sim);
                TransformState {
                    entity: selection,
                    mouse_offset: (draft.x - mouse_x, draft.y - mouse_y),
                    leak_channel: draft,
                }
            }
        };

        self.placement_state = None;
        self.transform_state = Some(transform_state);
    }

    pub(crate) fn select_entity(&mut self, selection: Option<SelectedEntity>) {
        let _ = self.apply_selected_entity_changes(true);
        self.transform_state = None;
        self.selected_entity = selection;
        self.inspector_dirty = false;
        self.inspector_error = None;
        self.inspector_last_apply = std::time::Instant::now();
        self.inspector_draft = match selection {
            Some(SelectedEntity::LeakChannel(index)) => self.simulation.as_ref().and_then(|sim| {
                sim.leak_channels()
                    .get(index)
                    .map(|channel| crate::app_types::LeakChannelDraft::from_channel(channel, sim))
            }),
            None => None,
        };
    }

    pub(crate) fn apply_selected_entity_changes(&mut self, force: bool) -> Result<()> {
        if !self.inspector_dirty {
            return Ok(());
        }
        if !force && self.inspector_last_apply.elapsed() < ENTITY_EDIT_DEBOUNCE {
            return Ok(());
        }

        let Some(selection) = self.selected_entity else {
            self.inspector_dirty = false;
            return Ok(());
        };
        let Some(draft) = self.inspector_draft.clone() else {
            self.inspector_dirty = false;
            return Ok(());
        };

        self.inspector_last_apply = std::time::Instant::now();
        let sim = self.simulation.as_mut().unwrap();
        let result = match selection {
            SelectedEntity::LeakChannel(index) => {
                let channel = draft.to_channel(sim)?;
                sim.update_leak_channel(index, channel)
            }
        };

        match result {
            Ok(()) => {
                self.inspector_dirty = false;
                self.inspector_error = None;
                Ok(())
            }
            Err(error) => {
                self.inspector_error = Some(error.to_string());
                Err(error)
            }
        }
    }

    pub(crate) fn handle_primary_click(&mut self) {
        if self.transform_state.is_some() {
            if let Some(selection) = self.selected_entity
                && let Some(channel) = self.preview_leak_channel()
                && let Some(sim) = &mut self.simulation
            {
                match selection {
                    SelectedEntity::LeakChannel(index) => {
                        match sim.update_leak_channel(index, channel) {
                            Ok(()) => {
                                self.transform_state = None;
                                self.select_entity(Some(selection));
                            }
                            Err(error) => {
                                self.inspector_error = Some(error.to_string());
                            }
                        }
                    }
                }
            }
            return;
        }

        if self.placement_state.is_some() {
            if let Some(channel) = self.preview_leak_channel()
                && let Some(sim) = &mut self.simulation
            {
                match sim.add_leak_channel(channel) {
                    Ok(()) => {
                        let index = sim.leak_channels().len().saturating_sub(1);
                        self.placement_state = None;
                        self.select_entity(Some(SelectedEntity::LeakChannel(index)));
                    }
                    Err(error) => {
                        self.inspector_error = Some(error.to_string());
                    }
                }
            }
            return;
        }

        if let Some(index) = self.hovered_leak_channel {
            self.select_entity(Some(SelectedEntity::LeakChannel(index)));
        } else {
            self.select_entity(None);
        }
    }

    pub(crate) fn delete_selected_entity(&mut self) {
        let Some(selection) = self.selected_entity else {
            return;
        };

        if let Some(sim) = &mut self.simulation {
            let result = match selection {
                SelectedEntity::LeakChannel(index) => sim.remove_leak_channel(index).map(|_| ()),
            };

            match result {
                Ok(()) => {
                    self.transform_state = None;
                    self.select_entity(None);
                }
                Err(error) => {
                    self.inspector_error = Some(error.to_string());
                }
            }
        }
    }

    pub(crate) fn placement_overlay(&self) -> Option<LeakChannelOverlay> {
        let sim = self.simulation.as_ref()?;
        let channel = self.preview_leak_channel()?;
        let center = self.grid_to_screen(channel.x as f32 + 0.5, channel.y as f32 + 0.5)?;
        let species_name = sim
            .species_registry()
            .get(channel.species)
            .map(|species| species.name.as_ref())
            .unwrap_or("?");
        let color = leak_channel_color(species_name);

        if let Some(((sink_x, sink_y), (source_x, source_y))) =
            sim.resolve_leak_channel_endpoints(&channel)
        {
            Some(LeakChannelOverlay {
                center,
                sink: self.grid_to_screen(sink_x as f32 + 0.5, sink_y as f32 + 0.5)?,
                source: self.grid_to_screen(source_x as f32 + 0.5, source_y as f32 + 0.5)?,
                color,
                hovered: false,
                selected: false,
                ghost: true,
                valid: true,
            })
        } else {
            Some(LeakChannelOverlay {
                center,
                sink: center,
                source: center,
                color,
                hovered: false,
                selected: false,
                ghost: true,
                valid: false,
            })
        }
    }

    pub(crate) fn update_tooltip(&mut self) {
        let Some((grid_x, grid_y)) = self.mouse_grid_position() else {
            self.hovered_leak_channel = None;
            self.last_tooltip_coord = None;
            self.show_tooltip = false;
            return;
        };

        let sim = self.simulation.as_mut().unwrap();

        if let Some(channel_index) = sim.hovered_leak_channel(grid_x, grid_y) {
            self.hovered_leak_channel = Some(channel_index);
            self.tooltip_text = sim
                .leak_channel_tooltip(channel_index)
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
        let cached_is_stale = cached_data
            .as_ref()
            .map(|data| data.age >= TOOLTIP_REFRESH_INTERVAL)
            .unwrap_or(true);
        let hover_changed = coarse_coord != self.last_tooltip_coord;

        if !sim.is_inspection_pending()
            && (hover_changed || !cached_matches_hover || cached_is_stale)
        {
            let _ = sim.request_async_inspection(grid_x, grid_y);
        }
        self.last_tooltip_coord = coarse_coord;

        if let Some(data) = sim.poll_async_inspection() {
            self.tooltip_text = format_async_inspection_tooltip(
                data.coord,
                data.fluid_count,
                data.solid_count,
                data.mean_temperature_kelvin,
                &data.concentrations,
                &self.species_names,
            );
            self.show_tooltip = true;
        } else if let Some(data) = cached_data {
            self.tooltip_text = format_async_inspection_tooltip(
                data.coord,
                data.fluid_count,
                data.solid_count,
                data.mean_temperature_kelvin,
                &data.concentrations,
                &self.species_names,
            );
            self.show_tooltip = true;
        } else {
            self.show_tooltip = false;
        }
    }

    pub(crate) fn hovered_coarse_rect(&self) -> Option<egui::Rect> {
        let (coarse_x, coarse_y) = self.last_tooltip_coord?;
        let mip = self.simulation.as_ref()?.coarse_mip_factor()?;
        self.coarse_rect(coarse_x, coarse_y, mip)
    }

    pub(crate) fn _update_tooltip_expensive(&mut self) {
        let sim = self.simulation.as_mut().unwrap();
        let (grid_w, grid_h) = sim.dimensions();

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
                    if new_tooltip != self.tooltip_text {
                        log::debug!(
                            "Inspection at ({:.0}, {:.0}): {}",
                            grid_x,
                            grid_y,
                            new_tooltip.replace('\n', " | ")
                        );
                    }
                    self.tooltip_text = new_tooltip;
                    self.show_tooltip = true;
                }
                Err(error) => {
                    log::trace!("Inspection error: {}", error);
                    self.show_tooltip = false;
                }
            }
        }
    }
}

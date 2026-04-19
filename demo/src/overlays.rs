use crate::app::{DETAIL_PANEL_WIDTH, DemoApp};
use crate::app_types::{LeakChannelOverlay, SelectedEntity};
use crate::colors::leak_channel_color;

impl DemoApp {
    pub(crate) fn draw_hovered_coarse_outline(ctx: &egui::Context, rect: egui::Rect) {
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

    pub(crate) fn draw_enzymes(&self, ctx: &egui::Context) {
        let Some(sim) = self.simulation.as_ref() else {
            return;
        };
        if sim.enzyme_entities().is_empty() {
            return;
        }
        let Some((scale_x, scale_y)) = self.grid_to_screen_scale() else {
            return;
        };

        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("enzyme_entities"),
        ));

        for entity in sim.enzyme_entities() {
            let Some(center) = self.grid_to_screen(entity.x, entity.y) else {
                continue;
            };
            let Some(active_site) = self.grid_to_screen(
                entity.active_site_position().x,
                entity.active_site_position().y,
            ) else {
                continue;
            };

            let size_boost = 3.0;
            let half_width = fluidsim::EnzymeEntity::BODY_HALF_WIDTH
                * entity.mobility_scale
                * scale_x
                * size_boost;
            let half_height = fluidsim::EnzymeEntity::BODY_HALF_HEIGHT
                * entity.catalytic_scale
                * scale_y
                * size_boost;
            let (sin_theta, cos_theta) = entity.rotation_radians.sin_cos();
            let rotation = egui::emath::Rot2::from_angle(entity.rotation_radians);

            let mut shell = Vec::with_capacity(36);
            for step in 0..36 {
                let angle = step as f32 / 36.0 * std::f32::consts::TAU;
                let lobe_a = (angle * 2.0).sin() * 0.11;
                let lobe_b = (angle * 3.0 + 0.45).cos() * 0.08;
                let lobe_c = (angle * 5.0 - 0.2).sin() * 0.04;
                let radius = 1.0 + lobe_a - lobe_b + lobe_c;
                let squash = 0.96 + 0.09 * (angle - 0.35).cos();
                let local = egui::vec2(
                    half_width * radius * angle.cos(),
                    half_height * squash * radius * angle.sin(),
                );
                shell.push(center + rotation * local);
            }

            let shadow: Vec<_> = shell
                .iter()
                .map(|point| *point + egui::vec2(8.0, 10.0))
                .collect();
            painter.add(egui::Shape::convex_polygon(
                shadow,
                egui::Color32::from_rgba_unmultiplied(18, 26, 34, 68),
                egui::Stroke::NONE,
            ));

            painter.add(egui::Shape::convex_polygon(
                shell,
                egui::Color32::from_rgb(229, 202, 144),
                egui::Stroke::new(2.4, egui::Color32::from_rgb(102, 63, 28)),
            ));

            for (offset, length_scale, color, width) in [
                (
                    0.38_f32,
                    0.54_f32,
                    egui::Color32::from_rgba_unmultiplied(255, 242, 214, 220),
                    2.8_f32,
                ),
                (
                    0.02_f32,
                    0.62_f32,
                    egui::Color32::from_rgba_unmultiplied(206, 150, 86, 180),
                    2.1_f32,
                ),
                (
                    -0.34_f32,
                    0.48_f32,
                    egui::Color32::from_rgba_unmultiplied(255, 226, 180, 170),
                    1.7_f32,
                ),
            ] {
                let stripe_dir = egui::vec2(
                    cos_theta * half_width * length_scale,
                    sin_theta * half_width * length_scale,
                );
                let stripe_normal = egui::vec2(
                    -sin_theta * half_height * offset,
                    cos_theta * half_height * offset,
                );
                painter.line_segment(
                    [
                        center - stripe_dir + stripe_normal,
                        center + stripe_dir + stripe_normal,
                    ],
                    egui::Stroke::new(width, color),
                );
            }

            let highlight_center =
                center + rotation * egui::vec2(-half_width * 0.18, -half_height * 0.22);
            painter.circle_filled(
                highlight_center,
                half_height * 0.72,
                egui::Color32::from_rgba_unmultiplied(255, 244, 224, 48),
            );

            let pocket_radius = 0.55 * half_height.max(4.0);
            let pocket_shadow = active_site + rotation * egui::vec2(4.0, 3.2);
            painter.circle_filled(
                pocket_shadow,
                pocket_radius * 1.65,
                egui::Color32::from_rgba_unmultiplied(24, 34, 44, 72),
            );
            painter.circle_filled(
                active_site,
                pocket_radius * 1.55,
                egui::Color32::from_rgba_unmultiplied(72, 118, 162, 96),
            );
            painter.circle_filled(
                active_site,
                pocket_radius,
                egui::Color32::from_rgb(102, 170, 218),
            );
            painter.circle_stroke(
                active_site,
                pocket_radius,
                egui::Stroke::new(2.4, egui::Color32::from_rgb(234, 248, 255)),
            );
        }
    }

    pub(crate) fn draw_detail_frame(&self, ctx: &egui::Context) {
        if !self.detail_view {
            return;
        }
        let Some(viewport) = self.simulation_viewport() else {
            return;
        };
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("detail_play_area"),
        ));

        painter.rect_stroke(
            viewport.rect.expand(4.0),
            18.0,
            egui::Stroke::new(3.0, egui::Color32::from_rgba_unmultiplied(7, 12, 18, 180)),
        );
        painter.rect_stroke(
            viewport.rect,
            14.0,
            egui::Stroke::new(
                1.5,
                egui::Color32::from_rgba_unmultiplied(255, 255, 255, 64),
            ),
        );
    }

    pub(crate) fn draw_detail_probes(&self, ctx: &egui::Context) {
        if !self.detail_view || self.detail_probes.is_empty() {
            return;
        }

        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Tooltip,
            egui::Id::new("detail_probe_overlays"),
        ));

        for probe in &self.detail_probes {
            let Some(sample_pos) = self.grid_to_screen(probe.sample_grid.0, probe.sample_grid.1)
            else {
                continue;
            };

            if let Some(rect) =
                self.coarse_rect(probe.coarse_coord.0, probe.coarse_coord.1, probe.mip)
            {
                painter.rect_stroke(
                    rect.expand(0.5),
                    6.0,
                    egui::Stroke::new(
                        1.4,
                        egui::Color32::from_rgba_unmultiplied(255, 231, 140, 180),
                    ),
                );
            }

            let area_response = egui::Area::new(egui::Id::new(("detail_probe", probe.title)))
                .order(egui::Order::Tooltip)
                .anchor(probe.slot.anchor(), probe.slot.offset())
                .show(ctx, |ui| {
                    let response = egui::Frame::none()
                        .fill(egui::Color32::from_rgba_unmultiplied(7, 11, 16, 224))
                        .stroke(egui::Stroke::new(
                            1.0,
                            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 36),
                        ))
                        .inner_margin(egui::Margin::same(10.0))
                        .show(ui, |ui| {
                            ui.set_width(DETAIL_PANEL_WIDTH);
                            ui.label(egui::RichText::new(probe.title).strong().size(15.0));
                            ui.small(format!(
                                "sample ({:.0}, {:.0})",
                                probe.sample_grid.0, probe.sample_grid.1
                            ));
                            ui.add_space(4.0);
                            ui.monospace(&probe.tooltip_text);
                        });
                    response.response.rect
                });

            let panel_rect = area_response.inner;
            let connector_start = Self::closest_point_on_rect(panel_rect, sample_pos);
            let connector_color = egui::Color32::from_rgb(255, 231, 140);
            painter.line_segment(
                [connector_start, sample_pos],
                egui::Stroke::new(1.6, connector_color),
            );
            painter.circle_filled(sample_pos, 4.0, connector_color);
            painter.circle_stroke(
                sample_pos,
                7.5,
                egui::Stroke::new(
                    1.2,
                    egui::Color32::from_rgba_unmultiplied(255, 248, 214, 220),
                ),
            );
        }
    }

    pub(crate) fn leak_channel_overlays(&self) -> Vec<LeakChannelOverlay> {
        let Some(sim) = self.simulation.as_ref() else {
            return Vec::new();
        };
        let mut overlays = Vec::new();

        for (index, channel) in sim.leak_channels().iter().enumerate() {
            let Some(center) = self.grid_to_screen(channel.x as f32 + 0.5, channel.y as f32 + 0.5)
            else {
                continue;
            };
            let Some(((sink_x, sink_y), (source_x, source_y))) = sim.leak_channel_endpoints(index)
            else {
                continue;
            };
            let Some(sink) = self.grid_to_screen(sink_x as f32 + 0.5, sink_y as f32 + 0.5) else {
                continue;
            };
            let Some(source) = self.grid_to_screen(source_x as f32 + 0.5, source_y as f32 + 0.5)
            else {
                continue;
            };

            let species_name = sim
                .species_registry()
                .get(channel.species)
                .map(|info| info.name.as_ref())
                .unwrap_or("?");
            let color = leak_channel_color(species_name);
            overlays.push(LeakChannelOverlay {
                center,
                sink,
                source,
                color,
                hovered: self.hovered_leak_channel == Some(index),
                selected: self.selected_entity == Some(SelectedEntity::LeakChannel(index)),
                ghost: false,
                valid: true,
            });
        }

        overlays
    }

    pub(crate) fn draw_leak_channels(ctx: &egui::Context, overlays: &[LeakChannelOverlay]) {
        if overlays.is_empty() {
            return;
        }

        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("leak_channels"),
        ));

        for overlay in overlays {
            let outline = if overlay.selected {
                egui::Color32::from_rgb(255, 242, 120)
            } else if overlay.hovered {
                egui::Color32::WHITE
            } else {
                egui::Color32::from_rgb(18, 22, 28)
            };
            let flow_color = if overlay.ghost {
                if overlay.valid {
                    egui::Color32::from_rgba_unmultiplied(
                        overlay.color.r(),
                        overlay.color.g(),
                        overlay.color.b(),
                        170,
                    )
                } else {
                    egui::Color32::from_rgba_unmultiplied(255, 120, 120, 170)
                }
            } else {
                overlay.color
            };
            let shell_fill = if overlay.ghost {
                if overlay.valid {
                    egui::Color32::from_rgba_unmultiplied(242, 239, 228, 176)
                } else {
                    egui::Color32::from_rgba_unmultiplied(255, 214, 214, 176)
                }
            } else {
                egui::Color32::from_rgba_unmultiplied(246, 242, 232, 236)
            };
            let shadow_color = if overlay.ghost {
                egui::Color32::from_rgba_unmultiplied(14, 18, 24, 120)
            } else {
                egui::Color32::from_rgba_unmultiplied(14, 18, 24, 210)
            };
            let band_color = if overlay.selected || overlay.hovered {
                outline
            } else if overlay.ghost {
                egui::Color32::from_rgba_unmultiplied(42, 48, 56, 136)
            } else {
                egui::Color32::from_rgba_unmultiplied(34, 40, 48, 220)
            };
            let highlight = if overlay.ghost {
                egui::Color32::from_rgba_unmultiplied(255, 255, 255, 120)
            } else {
                egui::Color32::from_rgba_unmultiplied(255, 255, 255, 185)
            };

            let delta = egui::vec2(
                overlay.source.x - overlay.sink.x,
                overlay.source.y - overlay.sink.y,
            );
            let tangent = if delta.length_sq() > 0.0 {
                delta.normalized()
            } else {
                egui::vec2(1.0, 0.0)
            };
            let normal = egui::vec2(-tangent.y, tangent.x);
            let half_span = (delta.length() * 0.5).max(14.0);
            let body_half_length = (half_span * 0.78).clamp(14.0, 28.0);
            let body_half_width = if overlay.selected {
                9.5
            } else if overlay.hovered {
                8.75
            } else {
                8.0
            };
            let capsule_start = overlay.center - tangent * body_half_length;
            let capsule_end = overlay.center + tangent * body_half_length;
            let stripe_start = capsule_start + tangent * (body_half_length * 0.18);
            let stripe_end = capsule_end - tangent * (body_half_length * 0.34);
            let highlight_offset = normal * (body_half_width * 0.28);
            let highlight_start =
                capsule_start + tangent * (body_half_length * 0.24) + highlight_offset;
            let highlight_end =
                capsule_end - tangent * (body_half_length * 0.58) + highlight_offset;

            painter.line_segment(
                [capsule_start, capsule_end],
                egui::Stroke::new(body_half_width * 2.0 + 6.0, shadow_color),
            );
            painter.line_segment(
                [capsule_start, capsule_end],
                egui::Stroke::new(body_half_width * 2.0 + 2.5, outline),
            );
            painter.line_segment(
                [capsule_start, capsule_end],
                egui::Stroke::new(body_half_width * 2.0, shell_fill),
            );
            painter.line_segment(
                [stripe_start, stripe_end],
                egui::Stroke::new(body_half_width * 1.05, flow_color),
            );
            painter.line_segment(
                [highlight_start, highlight_end],
                egui::Stroke::new(2.0, highlight),
            );

            for band_offset in [-0.42_f32, 0.18_f32] {
                let band_center = overlay.center + tangent * (body_half_length * band_offset);
                let band_half = normal * (body_half_width * 0.82);
                painter.line_segment(
                    [band_center - band_half, band_center + band_half],
                    egui::Stroke::new(2.0, band_color),
                );
            }

            let arrow_tip = capsule_end + tangent * (body_half_width + 7.0);
            let arrow_base = capsule_end - tangent * (body_half_width * 0.15);
            let arrow_half_width = body_half_width * 0.9;
            painter.add(egui::Shape::convex_polygon(
                vec![
                    arrow_tip,
                    arrow_base + normal * arrow_half_width,
                    arrow_base - normal * arrow_half_width,
                ],
                flow_color,
                egui::Stroke::new(
                    if overlay.selected || overlay.hovered {
                        2.5
                    } else {
                        1.8
                    },
                    outline,
                ),
            ));
        }
    }
}

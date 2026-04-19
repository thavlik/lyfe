use crate::app::DemoApp;
use crate::app_types::{EntityKind, PerformanceSample, PerformanceSummary, SelectedEntity};

impl DemoApp {
    pub(crate) fn draw_editor_ui(&mut self, ctx: &egui::Context) {
        use egui::{Align2, Color32, Frame, Margin, RichText, Stroke};

        egui::Area::new("simulation_controls".into())
            .anchor(Align2::LEFT_TOP, egui::vec2(12.0, 12.0))
            .show(ctx, |ui| {
                Frame::none()
                    .fill(Color32::from_rgba_unmultiplied(8, 12, 18, 220))
                    .stroke(Stroke::new(
                        1.0,
                        Color32::from_rgba_unmultiplied(255, 255, 255, 32),
                    ))
                    .inner_margin(Margin::same(8.0))
                    .show(ui, |ui| {
                        let paused = self
                            .simulation
                            .as_ref()
                            .map(|sim| sim.is_paused())
                            .unwrap_or(false);

                        if ui
                            .button(RichText::new(if paused { "PLAY" } else { "PAUSE" }).strong())
                            .clicked()
                            && let Some(sim) = self.simulation.as_mut()
                        {
                            sim.toggle_pause();
                            log::info!(
                                "Simulation {}",
                                if sim.is_paused() { "paused" } else { "resumed" }
                            );
                        }

                        if ui.button(RichText::new("RESTART").strong()).clicked()
                            && let Err(error) = self.reset_simulation()
                        {
                            log::error!("Failed to restart simulation: {}", error);
                        }

                        if ui.button(RichText::new("CREATE").strong()).clicked() {
                            self.create_menu_open = true;
                        }
                    });
            });

        if self.create_menu_open {
            egui::Window::new("Create Entity")
                .anchor(Align2::LEFT_TOP, egui::vec2(12.0, 56.0))
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.strong("Transport");
                    ui.add_space(4.0);
                    if ui.button("Leak Channel").clicked() {
                        self.begin_placing_entity(EntityKind::LeakChannel);
                    }
                });
        }

        if let Some(placement) = &self.placement_state {
            egui::Area::new("entity_placement_status".into())
                .anchor(
                    Align2::LEFT_TOP,
                    egui::vec2(12.0, if self.create_menu_open { 170.0 } else { 56.0 }),
                )
                .show(ctx, |ui| {
                    Frame::none()
                        .fill(Color32::from_rgba_unmultiplied(8, 12, 18, 215))
                        .stroke(Stroke::new(
                            1.0,
                            Color32::from_rgba_unmultiplied(255, 255, 255, 28),
                        ))
                        .inner_margin(Margin::same(10.0))
                        .show(ui, |ui| {
                            ui.label(RichText::new("Placing Leak Channel").strong());
                            ui.label(format!("Species: {}", placement.leak_channel.species_name));
                            ui.label(format!("Rate: {:.2}", placement.leak_channel.rate));
                            ui.label(format!(
                                "Rotation: {} deg",
                                placement.leak_channel.rotation_degrees()
                            ));
                            ui.label("Click in the sim to place. Press r to rotate 45 deg.");
                        });
                });
        }

        if let Some(transform) = &self.transform_state {
            egui::Area::new("entity_transform_status".into())
                .anchor(
                    Align2::LEFT_TOP,
                    egui::vec2(12.0, if self.create_menu_open { 170.0 } else { 56.0 }),
                )
                .show(ctx, |ui| {
                    Frame::none()
                        .fill(Color32::from_rgba_unmultiplied(8, 12, 18, 215))
                        .stroke(Stroke::new(
                            1.0,
                            Color32::from_rgba_unmultiplied(255, 255, 255, 28),
                        ))
                        .inner_margin(Margin::same(10.0))
                        .show(ui, |ui| {
                            ui.label(RichText::new("Transforming Leak Channel").strong());
                            ui.label(format!("Species: {}", transform.leak_channel.species_name));
                            ui.label(format!("Rate: {:.2}", transform.leak_channel.rate));
                            ui.label(format!(
                                "Rotation: {} deg",
                                transform.leak_channel.rotation_degrees()
                            ));
                            ui.label(
                                "Move with the mouse, click to confirm, press r to rotate 45 deg.",
                            );
                        });
                });
        }

        if let Some(SelectedEntity::LeakChannel(index)) = self.selected_entity {
            let (max_x, max_y) = self
                .simulation
                .as_ref()
                .map(|sim| sim.dimensions())
                .unwrap_or((512, 512));
            egui::Area::new("entity_inspector".into())
                .anchor(Align2::LEFT_CENTER, egui::vec2(16.0, 0.0))
                .show(ctx, |ui| {
                    Frame::none()
                        .fill(Color32::from_rgba_unmultiplied(8, 12, 18, 230))
                        .stroke(Stroke::new(
                            1.0,
                            Color32::from_rgba_unmultiplied(255, 255, 255, 32),
                        ))
                        .inner_margin(Margin::same(12.0))
                        .show(ui, |ui| {
                            ui.set_min_width(300.0);
                            ui.label(RichText::new("Leak Channel").strong().size(18.0));
                            ui.small(format!("Entity {}", index + 1));
                            ui.add_space(8.0);

                            if let Some(draft) = &mut self.inspector_draft {
                                ui.label("Species");
                                if ui.text_edit_singleline(&mut draft.species_name).changed() {
                                    self.inspector_dirty = true;
                                }

                                ui.add_space(6.0);
                                if ui
                                    .add(egui::Slider::new(&mut draft.rate, 0.0..=12.0).text("Rate"))
                                    .changed()
                                {
                                    self.inspector_dirty = true;
                                }

                                ui.add_space(6.0);
                                ui.horizontal(|ui| {
                                    ui.label("X");
                                    if ui
                                        .add(
                                            egui::DragValue::new(&mut draft.x)
                                                .range(0..=max_x.saturating_sub(1) as i32),
                                        )
                                        .changed()
                                    {
                                        self.inspector_dirty = true;
                                    }
                                    ui.label("Y");
                                    if ui
                                        .add(
                                            egui::DragValue::new(&mut draft.y)
                                                .range(0..=max_y.saturating_sub(1) as i32),
                                        )
                                        .changed()
                                    {
                                        self.inspector_dirty = true;
                                    }
                                });

                                ui.add_space(6.0);
                                let mut rotation_degrees = draft.rotation_degrees();
                                if ui
                                    .add(
                                        egui::Slider::new(&mut rotation_degrees, 0..=315)
                                            .step_by(45.0)
                                            .text("Rotation"),
                                    )
                                    .changed()
                                {
                                    draft.set_rotation_degrees(rotation_degrees);
                                    self.inspector_dirty = true;
                                }
                                ui.small(format!("Rotation byte: {}", draft.rotation_byte));

                                if let Some(sim) = self.simulation.as_ref() {
                                    match draft.to_channel(sim) {
                                        Ok(channel) => {
                                            ui.small(format!("Flow: {}", channel.flow_label()));
                                            if let Some(((sink_x, sink_y), (source_x, source_y))) =
                                                sim.resolve_leak_channel_endpoints(&channel)
                                            {
                                                ui.small(format!(
                                                    "Sink: ({sink_x}, {sink_y})  Source: ({source_x}, {source_y})"
                                                ));
                                            } else {
                                                ui.colored_label(
                                                    Color32::YELLOW,
                                                    "No valid fluid endpoints for current placement.",
                                                );
                                            }
                                        }
                                        Err(error) => {
                                            ui.colored_label(Color32::YELLOW, error.to_string());
                                        }
                                    }
                                }
                            }

                            ui.add_space(8.0);
                            if self.inspector_dirty {
                                ui.small("Pending live update...");
                            }
                            if let Some(error) = &self.inspector_error {
                                ui.colored_label(Color32::from_rgb(255, 120, 120), error);
                            }
                            if ui.button("Deselect").clicked() {
                                self.select_entity(None);
                            }
                        });
                });
        }
    }

    pub(crate) fn draw_performance_overlay(
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
            (
                "Frame",
                Color32::from_rgb(255, 99, 71),
                samples
                    .iter()
                    .map(|sample| sample.frame_ms)
                    .collect::<Vec<_>>(),
            ),
            (
                "Simulation",
                Color32::from_rgb(86, 204, 242),
                samples
                    .iter()
                    .map(|sample| sample.simulation_ms)
                    .collect::<Vec<_>>(),
            ),
            (
                "Render",
                Color32::from_rgb(242, 201, 76),
                samples
                    .iter()
                    .map(|sample| sample.render_ms)
                    .collect::<Vec<_>>(),
            ),
            (
                "UI",
                Color32::from_rgb(155, 81, 224),
                samples
                    .iter()
                    .map(|sample| sample.ui_ms)
                    .collect::<Vec<_>>(),
            ),
            (
                "Upload",
                Color32::from_rgb(39, 174, 96),
                samples
                    .iter()
                    .map(|sample| sample.upload_ms)
                    .collect::<Vec<_>>(),
            ),
            (
                "Tooltip",
                Color32::from_rgb(235, 87, 87),
                samples
                    .iter()
                    .map(|sample| sample.tooltip_ms)
                    .collect::<Vec<_>>(),
            ),
        ];

        egui::Area::new("performance_stats".into())
            .anchor(Align2::RIGHT_TOP, egui::vec2(-16.0, 16.0))
            .show(ctx, |ui: &mut egui::Ui| {
                Frame::none()
                    .fill(Color32::from_rgba_unmultiplied(8, 12, 18, 220))
                    .stroke(Stroke::new(
                        1.0,
                        Color32::from_rgba_unmultiplied(255, 255, 255, 24),
                    ))
                    .inner_margin(Margin::same(12.0))
                    .show(ui, |ui: &mut egui::Ui| {
                        ui.set_min_width(240.0);
                        ui.label(
                            RichText::new("Performance")
                                .strong()
                                .size(18.0)
                                .color(Color32::WHITE),
                        );
                        ui.add_space(6.0);
                        ui.label(format!("Frame: {:.2} ms", summary.current_frame_ms));
                        ui.label(format!("FPS: {:.1}", summary.current_fps));
                        ui.label(format!("Avg FPS (30s): {:.1}", summary.average_fps_30s));
                        ui.label(format!(
                            "Worst Frame (30s): {:.2} ms",
                            summary.worst_frame_ms_30s
                        ));
                    });
            });

        egui::Area::new("performance_graph".into())
            .anchor(Align2::LEFT_BOTTOM, egui::vec2(16.0, -16.0))
            .show(ctx, |ui: &mut egui::Ui| {
                let available_width = (ctx.screen_rect().width() - 32.0).max(480.0);
                let graph_size = Vec2::new(available_width, 210.0);

                Frame::none()
                    .fill(Color32::from_rgba_unmultiplied(7, 10, 15, 210))
                    .stroke(Stroke::new(
                        1.0,
                        Color32::from_rgba_unmultiplied(255, 255, 255, 24),
                    ))
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
                        painter.rect_filled(
                            rect,
                            10.0,
                            Color32::from_rgba_unmultiplied(15, 21, 30, 200),
                        );
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

                        let max_ms = metrics
                            .iter()
                            .flat_map(|(_, _, values)| values.iter().copied())
                            .fold(16.0_f32, f32::max)
                            .max(1.0);

                        for step in 0..=4 {
                            let t = step as f32 / 4.0;
                            let y = egui::lerp(plot.bottom()..=plot.top(), t);
                            painter.line_segment(
                                [egui::pos2(plot.left(), y), egui::pos2(plot.right(), y)],
                                Stroke::new(
                                    1.0,
                                    Color32::from_rgba_unmultiplied(255, 255, 255, 18),
                                ),
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
                            for (_, color, values) in &metrics {
                                let points: Vec<_> = values
                                    .iter()
                                    .enumerate()
                                    .map(|(index, value)| {
                                        let x_t =
                                            index as f32 / (values.len().saturating_sub(1) as f32);
                                        let y_t = (*value / max_ms).clamp(0.0, 1.0);
                                        egui::pos2(
                                            egui::lerp(plot.left()..=plot.right(), x_t),
                                            egui::lerp(plot.bottom()..=plot.top(), y_t),
                                        )
                                    })
                                    .collect();
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
}

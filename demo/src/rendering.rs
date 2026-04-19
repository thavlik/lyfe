use std::time::Instant;

use anyhow::Result;
use renderer::RenderViewport;

use crate::app::DemoApp;
use crate::app_types::RenderFrameMetrics;

impl DemoApp {
    pub(crate) fn render_frame(&mut self) -> Result<RenderFrameMetrics> {
        let render_viewport = self.simulation_viewport().map(|viewport| RenderViewport {
            x: viewport.physical_x,
            y: viewport.physical_y,
            width: viewport.physical_width,
            height: viewport.physical_height,
        });
        let ctx = self.render_ctx.as_mut().unwrap();
        let pipeline = self.render_pipeline.as_mut().unwrap();
        let sim = self.simulation.as_mut().unwrap();
        let egui = self.egui_renderer.as_mut().unwrap();

        let image_index = match ctx.begin_frame()? {
            Some(index) => index,
            None => {
                log::debug!("begin_frame returned None, needs resize");
                self.needs_resize = true;
                return Ok(RenderFrameMetrics::default());
            }
        };

        let t0 = Instant::now();
        pipeline.bind_simulation_buffers(ctx, sim.gpu_render_buffers());
        let t1 = Instant::now();

        let step = sim.step_count();
        if step > 0 && step.is_multiple_of(100) {
            log::info!(
                "Frame timing: bind={:.1}ms",
                (t1 - t0).as_secs_f64() * 1000.0
            );
        }

        egui.set_textures(ctx)?;

        log::trace!("Rendering to swapchain image {}", image_index);

        let cmd = ctx.current_command_buffer();
        let begin_info = ash::vk::CommandBufferBeginInfo::default();
        unsafe {
            ctx.device
                .reset_command_buffer(cmd, ash::vk::CommandBufferResetFlags::empty())?;
            ctx.device.begin_command_buffer(cmd, &begin_info)?;
        }

        sim.record_render_barriers(cmd);

        let render_viewport =
            render_viewport.unwrap_or_else(|| RenderViewport::fullscreen(ctx.swapchain_extent));
        pipeline.record(
            ctx,
            cmd,
            image_index as usize,
            self.thermal_view,
            render_viewport,
        );

        egui.cmd_draw(ctx, cmd, ctx.swapchain_extent)?;
        pipeline.end_render_pass(ctx, cmd);

        unsafe {
            ctx.device.end_command_buffer(cmd)?;
        }

        let ok = ctx.end_frame(image_index)?;
        if !ok {
            log::debug!("end_frame returned false, needs resize");
            self.needs_resize = true;
        }

        egui.free_textures()?;

        let t3 = Instant::now();
        Ok(RenderFrameMetrics {
            upload_ms: (t1 - t0).as_secs_f64() as f32 * 1000.0,
            render_ms: (t3 - t1).as_secs_f64() as f32 * 1000.0,
        })
    }

    pub(crate) fn handle_resize(&mut self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return Ok(());
        }

        let ctx = self.render_ctx.as_mut().unwrap();
        ctx.recreate_swapchain(width, height)?;
        self.needs_resize = false;

        Ok(())
    }
}

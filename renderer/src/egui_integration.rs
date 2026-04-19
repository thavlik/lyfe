//! egui integration for Vulkan rendering.
//!
//! Provides tooltip and UI overlay rendering using egui-ash-renderer.

use anyhow::Result;
use ash::vk;
use egui::{ClippedPrimitive, Context, FullOutput, TexturesDelta};
use egui_ash_renderer::{Options, Renderer};
use egui_winit::State;
use winit::window::Window;

use crate::context::RenderContext;

/// egui renderer for Vulkan using egui-ash-renderer.
pub struct EguiRenderer {
    pub ctx: Context,
    pub state: State,
    pub renderer: Renderer,
    /// Cached output from end_frame for rendering
    cached_output: Option<CachedEguiOutput>,
}

struct CachedEguiOutput {
    primitives: Vec<ClippedPrimitive>,
    textures_delta: TexturesDelta,
}

impl EguiRenderer {
    pub fn new(render_ctx: &RenderContext, window: &Window) -> Result<Self> {
        let ctx = Context::default();
        let viewport_id = ctx.viewport_id();
        let state = State::new(ctx.clone(), viewport_id, window, None, None, None);

        // Create egui-ash-renderer with default allocator
        // (uses std::sync::Mutex internally, our RenderContext uses parking_lot::Mutex)
        let renderer = Renderer::with_default_allocator(
            &render_ctx.instance,
            render_ctx.physical_device,
            render_ctx.device.clone(),
            render_ctx.render_pass,
            Options {
                srgb_framebuffer: false, // Our swapchain is not sRGB
                ..Default::default()
            },
        )?;

        Ok(Self {
            ctx,
            state,
            renderer,
            cached_output: None,
        })
    }

    pub fn handle_event(&mut self, window: &Window, event: &winit::event::WindowEvent) -> bool {
        self.state.on_window_event(window, event).consumed
    }

    pub fn begin_frame(&mut self, window: &Window) {
        let raw_input = self.state.take_egui_input(window);
        self.ctx.begin_pass(raw_input);
    }

    pub fn end_frame(&mut self, window: &Window) -> FullOutput {
        let output = self.ctx.end_pass();
        self.state
            .handle_platform_output(window, output.platform_output.clone());

        // Tessellate and cache for rendering
        let primitives = self
            .ctx
            .tessellate(output.shapes.clone(), output.pixels_per_point);
        self.cached_output = Some(CachedEguiOutput {
            primitives,
            textures_delta: output.textures_delta.clone(),
        });

        output
    }

    pub fn context(&self) -> &Context {
        &self.ctx
    }

    /// Update textures before rendering (must be called before cmd_draw)
    pub fn set_textures(&mut self, ctx: &RenderContext) -> Result<()> {
        if let Some(output) = &self.cached_output
            && !output.textures_delta.set.is_empty()
        {
            self.renderer.set_textures(
                ctx.graphics_queue,
                ctx.command_pool,
                &output.textures_delta.set,
            )?;
        }
        Ok(())
    }

    /// Record egui draw commands to the command buffer.
    /// Must be called inside an active render pass.
    pub fn cmd_draw(
        &mut self,
        _ctx: &RenderContext,
        cmd: vk::CommandBuffer,
        extent: vk::Extent2D,
    ) -> Result<()> {
        if let Some(output) = &self.cached_output {
            self.renderer
                .cmd_draw(cmd, extent, self.ctx.pixels_per_point(), &output.primitives)?;
        }
        Ok(())
    }

    /// Free textures after rendering is complete.
    pub fn free_textures(&mut self) -> Result<()> {
        if let Some(output) = self.cached_output.take()
            && !output.textures_delta.free.is_empty()
        {
            self.renderer.free_textures(&output.textures_delta.free)?;
        }
        Ok(())
    }
}

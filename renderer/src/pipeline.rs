//! Rendering pipeline for fluid visualization.

use anyhow::{Context as _, Result};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use fluidsim::RenderState;
use gpu_allocator::vulkan::{Allocator, AllocationCreateDesc, AllocationScheme, Allocation};
use gpu_allocator::MemoryLocation;


use crate::context::RenderContext;

/// Push constants for the visualization shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct VisualizationPushConstants {
    pub width: u32,
    pub height: u32,
    pub species_count: u32,
    pub frame_counter: u32,  // For debug visualization
}

/// A GPU buffer for the render pipeline.
pub struct RenderBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
    pub size: u64,
}

impl RenderBuffer {
    pub fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &str,
    ) -> Result<Self> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation = allocator.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;

        unsafe {
            device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
        }

        Ok(Self { buffer, allocation, size })
    }

    pub fn write<T: Pod>(&mut self, data: &[T]) -> Result<()> {
        let bytes = bytemuck::cast_slice(data);
        let mapped = self.allocation.mapped_slice_mut()
            .context("Buffer not mapped")?;
        mapped[..bytes.len()].copy_from_slice(bytes);
        Ok(())
    }
}

/// Rendering pipeline for visualizing fluid simulation.
pub struct RenderPipeline {
    // Graphics pipeline
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,

    // Data buffers
    pub concentration_buffer: RenderBuffer,
    pub solid_mask_buffer: RenderBuffer,
    pub material_id_buffer: RenderBuffer,

    // Staging buffer
    pub staging_buffer: RenderBuffer,

    // Cached sizes
    pub grid_width: u32,
    pub grid_height: u32,
    pub species_count: usize,
    pub cell_count: usize,
    
    // Frame counter for debug visualization
    pub frame_counter: u32,
}

impl RenderPipeline {
    pub fn new(ctx: &RenderContext, grid_width: u32, grid_height: u32, species_count: usize) -> Result<Self> {
        let cell_count = (grid_width * grid_height) as usize;
        
        // Buffer sizes
        let conc_size = (species_count * cell_count * std::mem::size_of::<f32>()) as u64;
        let mask_size = (cell_count * std::mem::size_of::<u32>()) as u64;

        // Create buffers
        let mut alloc = ctx.allocator.lock();

        let concentration_buffer = RenderBuffer::new(
            &ctx.device,
            &mut alloc,
            conc_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            "render_concentrations",
        )?;

        let solid_mask_buffer = RenderBuffer::new(
            &ctx.device,
            &mut alloc,
            mask_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            "render_solid_mask",
        )?;

        let material_id_buffer = RenderBuffer::new(
            &ctx.device,
            &mut alloc,
            mask_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            "render_material_ids",
        )?;

        let staging_buffer = RenderBuffer::new(
            &ctx.device,
            &mut alloc,
            conc_size.max(mask_size),
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "render_staging",
        )?;

        drop(alloc);

        // Create descriptor set layout
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings);
        let descriptor_set_layout = unsafe { ctx.device.create_descriptor_set_layout(&layout_info, None)? };

        // Create pipeline layout
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<VisualizationPushConstants>() as u32);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        let pipeline_layout = unsafe { ctx.device.create_pipeline_layout(&pipeline_layout_info, None)? };

        // Compile shaders
        let compiler = shaderc::Compiler::new().context("Failed to create shader compiler")?;
        let mut options = shaderc::CompileOptions::new().context("Failed to create compile options")?;
        options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_2 as u32);

        let vert_source = include_str!("../shaders/fullscreen.vert");
        let frag_source = include_str!("../shaders/visualization.frag");

        let vert_spirv = compiler.compile_into_spirv(
            vert_source, shaderc::ShaderKind::Vertex, "fullscreen.vert", "main", Some(&options)
        ).context("Failed to compile vertex shader")?;

        let frag_spirv = compiler.compile_into_spirv(
            frag_source, shaderc::ShaderKind::Fragment, "visualization.frag", "main", Some(&options)
        ).context("Failed to compile fragment shader")?;

        let vert_module_info = vk::ShaderModuleCreateInfo::default().code(vert_spirv.as_binary());
        let frag_module_info = vk::ShaderModuleCreateInfo::default().code(frag_spirv.as_binary());

        let vert_module = unsafe { ctx.device.create_shader_module(&vert_module_info, None)? };
        let frag_module = unsafe { ctx.device.create_shader_module(&frag_module_info, None)? };

        let entry_name = c"main";
        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(entry_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(entry_name),
        ];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(ctx.swapchain_extent.width as f32)
            .height(ctx.swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        let scissor = vk::Rect2D::default()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(ctx.swapchain_extent);

        let viewports = [viewport];
        let scissors = [scissor];
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&dynamic_states);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(ctx.render_pass)
            .subpass(0);

        let pipeline = unsafe {
            ctx.device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|e| anyhow::anyhow!("Failed to create graphics pipeline: {:?}", e.1))?[0]
        };

        unsafe {
            ctx.device.destroy_shader_module(vert_module, None);
            ctx.device.destroy_shader_module(frag_module, None);
        }

        // Create descriptor pool and set
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(3),
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe { ctx.device.create_descriptor_pool(&pool_info, None)? };

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(std::slice::from_ref(&descriptor_set_layout));
        let descriptor_set = unsafe { ctx.device.allocate_descriptor_sets(&alloc_info)? }[0];

        // Update descriptor set
        let conc_info = [vk::DescriptorBufferInfo::default()
            .buffer(concentration_buffer.buffer)
            .offset(0)
            .range(conc_size)];
        let mask_info = [vk::DescriptorBufferInfo::default()
            .buffer(solid_mask_buffer.buffer)
            .offset(0)
            .range(mask_size)];
        let mat_info = [vk::DescriptorBufferInfo::default()
            .buffer(material_id_buffer.buffer)
            .offset(0)
            .range(mask_size)];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&conc_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&mask_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&mat_info),
        ];

        unsafe { ctx.device.update_descriptor_sets(&writes, &[]) };

        Ok(Self {
            pipeline_layout,
            pipeline,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            concentration_buffer,
            solid_mask_buffer,
            material_id_buffer,
            staging_buffer,
            grid_width,
            grid_height,
            species_count,
            cell_count,
            frame_counter: 0,
        })
    }

    /// Upload render state data to GPU using synchronous copies.
    pub fn upload_state(&mut self, ctx: &RenderContext, state: &RenderState) -> Result<()> {
        // Flatten concentrations
        let mut flat_conc: Vec<f32> = Vec::with_capacity(self.species_count * self.cell_count);
        for species_conc in &state.concentrations {
            flat_conc.extend_from_slice(species_conc);
        }

        // Debug: Log a sample concentration to verify data is changing
        let center_idx = self.cell_count / 2 + self.grid_width as usize / 2;
        if self.frame_counter % 30 == 0 {
            log::debug!("Upload frame {}: center concentrations = [{:.4}, {:.4}, {:.4}]",
                self.frame_counter,
                flat_conc.get(center_idx).unwrap_or(&0.0),
                flat_conc.get(self.cell_count + center_idx).unwrap_or(&0.0),
                flat_conc.get(2 * self.cell_count + center_idx).unwrap_or(&0.0));
        }

        let conc_size = (self.species_count * self.cell_count * std::mem::size_of::<f32>()) as u64;
        let mask_size = (self.cell_count * std::mem::size_of::<u32>()) as u64;

        // Upload concentrations with barrier
        self.staging_buffer.write(&flat_conc)?;
        copy_buffer_with_barrier(ctx, self.staging_buffer.buffer, self.concentration_buffer.buffer, conc_size)?;

        // Upload solid mask with barrier
        self.staging_buffer.write(&state.solid_mask)?;
        copy_buffer_with_barrier(ctx, self.staging_buffer.buffer, self.solid_mask_buffer.buffer, mask_size)?;

        // Upload material IDs with barrier
        self.staging_buffer.write(&state.material_ids)?;
        copy_buffer_with_barrier(ctx, self.staging_buffer.buffer, self.material_id_buffer.buffer, mask_size)?;

        Ok(())
    }

    /// Record rendering commands.
    pub fn record(&mut self, ctx: &RenderContext, cmd: vk::CommandBuffer, image_index: usize) {
        // Increment frame counter
        self.frame_counter = self.frame_counter.wrapping_add(1);
        
        // Note: Memory barriers for buffer transfers are now in copy_buffer_with_barrier()
        
        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.1, 1.0], // Dark blue background
            },
        }];

        let render_pass_info = vk::RenderPassBeginInfo::default()
            .render_pass(ctx.render_pass)
            .framebuffer(ctx.framebuffers[image_index])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: ctx.swapchain_extent,
            })
            .clear_values(&clear_values);

        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(ctx.swapchain_extent.width as f32)
            .height(ctx.swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        let scissor = vk::Rect2D::default()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(ctx.swapchain_extent);

        let push_constants = VisualizationPushConstants {
            width: self.grid_width,
            height: self.grid_height,
            species_count: self.species_count as u32,
            frame_counter: self.frame_counter,
        };

        unsafe {
            ctx.device.cmd_begin_render_pass(cmd, &render_pass_info, vk::SubpassContents::INLINE);
            ctx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            ctx.device.cmd_set_viewport(cmd, 0, &[viewport]);
            ctx.device.cmd_set_scissor(cmd, 0, &[scissor]);
            ctx.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            ctx.device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push_constants),
            );
            // Draw fullscreen triangle
            ctx.device.cmd_draw(cmd, 3, 1, 0, 0);
            // Note: Render pass is NOT ended here - caller must end it after egui rendering
        }
    }

    /// End the render pass. Call this after egui rendering.
    pub fn end_render_pass(&self, ctx: &RenderContext, cmd: vk::CommandBuffer) {
        unsafe {
            ctx.device.cmd_end_render_pass(cmd);
        }
    }

    pub fn destroy(&mut self, ctx: &RenderContext) {
        unsafe {
            ctx.device.destroy_pipeline(self.pipeline, None);
            ctx.device.destroy_pipeline_layout(self.pipeline_layout, None);
            ctx.device.destroy_descriptor_pool(self.descriptor_pool, None);
            ctx.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            let mut alloc = ctx.allocator.lock();

            ctx.device.destroy_buffer(self.concentration_buffer.buffer, None);
            alloc.free(std::mem::take(&mut self.concentration_buffer.allocation)).ok();

            ctx.device.destroy_buffer(self.solid_mask_buffer.buffer, None);
            alloc.free(std::mem::take(&mut self.solid_mask_buffer.allocation)).ok();

            ctx.device.destroy_buffer(self.material_id_buffer.buffer, None);
            alloc.free(std::mem::take(&mut self.material_id_buffer.allocation)).ok();

            ctx.device.destroy_buffer(self.staging_buffer.buffer, None);
            alloc.free(std::mem::take(&mut self.staging_buffer.allocation)).ok();
        }
    }
}

fn copy_buffer_with_barrier(
    ctx: &RenderContext,
    src: vk::Buffer,
    dst: vk::Buffer,
    size: u64,
) -> Result<()> {
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(ctx.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmd = unsafe { ctx.device.allocate_command_buffers(&alloc_info)? }[0];

    let begin_info = vk::CommandBufferBeginInfo::default()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { ctx.device.begin_command_buffer(cmd, &begin_info)? };

    let copy_region = vk::BufferCopy::default().size(size);
    unsafe { ctx.device.cmd_copy_buffer(cmd, src, dst, &[copy_region]) };

    // Add memory barrier to make the transfer visible to shader reads
    let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .buffer(dst)
        .offset(0)
        .size(vk::WHOLE_SIZE);

    unsafe {
        ctx.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[barrier],
            &[],
        );
    }

    unsafe { ctx.device.end_command_buffer(cmd)? };

    let fence_info = vk::FenceCreateInfo::default();
    let fence = unsafe { ctx.device.create_fence(&fence_info, None)? };

    let submit_info = vk::SubmitInfo::default()
        .command_buffers(std::slice::from_ref(&cmd));

    unsafe {
        ctx.device.queue_submit(ctx.graphics_queue, &[submit_info], fence)?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        ctx.device.destroy_fence(fence, None);
        ctx.device.free_command_buffers(ctx.command_pool, &[cmd]);
    }

    Ok(())
}

//! GPU compute pipeline for fluid simulation.
//!
//! This module handles all Vulkan resources for running the diffusion
//! simulation on the GPU. It manages:
//! - Storage buffers for concentration data (ping-pong)
//! - Solid mask and material ID buffers
//! - Compute pipeline for diffusion passes
//! - Synchronization and buffer transfers
//! - Coarse grid computation for async tooltip readback
//!
//! ## Buffer Layout
//!
//! Concentration data uses `[species][cell]` layout:
//! - Total size: `species_count * cell_count * sizeof(f32)`
//! - Species `i` at cell `j`: `buffer[i * cell_count + j]`
//!
//! This layout allows efficient per-species diffusion passes where
//! each compute invocation processes one species across the grid.

use anyhow::{Context, Result, bail};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::vulkan::{Allocator, AllocationCreateDesc, AllocationScheme, Allocation};
use gpu_allocator::MemoryLocation;
use parking_lot::Mutex;

use std::sync::Arc;

use crate::coarse::CoarseGrid;

/// Push constants for the diffusion compute shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DiffusionPushConstants {
    /// Grid width in cells
    pub width: u32,
    /// Grid height in cells
    pub height: u32,
    /// Number of species channels
    pub species_count: u32,
    /// Index of species being processed this pass (or 0xFFFFFFFF for all)
    pub species_index: u32,
    /// Diffusion rate multiplier
    pub diffusion_rate: f32,
    /// Time step
    pub dt: f32,
    /// Padding for alignment
    pub _pad: [u32; 2],
}

/// A GPU buffer with its allocation.
pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
    pub size: u64,
}

impl GpuBuffer {
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

    /// Write data to a CPU-visible buffer.
    pub fn write<T: Pod>(&mut self, data: &[T]) -> Result<()> {
        let bytes = bytemuck::cast_slice(data);
        let mapped = self.allocation.mapped_slice_mut()
            .context("Buffer not mapped for CPU access")?;
        mapped[..bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    /// Read data from a CPU-visible buffer.
    pub fn read<T: Pod + Clone>(&self, count: usize) -> Result<Vec<T>> {
        let mapped = self.allocation.mapped_slice()
            .context("Buffer not mapped for CPU access")?;
        let bytes = &mapped[..count * std::mem::size_of::<T>()];
        Ok(bytemuck::cast_slice(bytes).to_vec())
    }
}

/// GPU resources for the fluid simulation.
pub struct GpuSimulation {
    // Vulkan handles
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub compute_queue: vk::Queue,
    pub compute_queue_family: u32,

    // Memory allocator
    pub allocator: Arc<Mutex<Allocator>>,

    // Grid dimensions
    pub width: u32,
    pub height: u32,
    pub cell_count: usize,
    pub species_count: usize,

    // Concentration buffers (ping-pong)
    pub conc_buffer_a: GpuBuffer,
    pub conc_buffer_b: GpuBuffer,
    pub current_buffer: usize, // 0 = A is current, 1 = B is current

    // Solid mask and material buffers
    pub solid_mask: GpuBuffer,
    pub material_ids: GpuBuffer,

    // Diffusion coefficients per species
    pub diffusion_coeffs: GpuBuffer,

    // Staging buffer for uploads (CpuToGpu)
    pub staging_buffer: GpuBuffer,
    
    // Readback buffer for downloads (GpuToCpu)
    pub readback_buffer: GpuBuffer,

    // Compute pipeline
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub diffusion_pipeline: vk::Pipeline,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>, // [0] = A->B, [1] = B->A

    // Command resources
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub fence: vk::Fence,
    
    // Coarse grid for async tooltip readback
    pub coarse_grid: Option<CoarseGrid>,
}

impl GpuSimulation {
    /// Create a new GPU simulation context.
    /// 
    /// This creates a Vulkan instance and device suitable for compute-only
    /// workloads.
    pub fn new(
        width: u32,
        height: u32,
        species_count: usize,
        initial_concentrations: &[Vec<f32>],
        solid_mask_data: &[u32],
        material_ids_data: &[u32],
        diffusion_coeffs_data: &[f32],
    ) -> Result<Self> {
        let cell_count = (width * height) as usize;

        // Validate input sizes
        if initial_concentrations.len() != species_count {
            bail!("Expected {} species, got {}", species_count, initial_concentrations.len());
        }
        for (i, conc) in initial_concentrations.iter().enumerate() {
            if conc.len() != cell_count {
                bail!("Species {} has {} cells, expected {}", i, conc.len(), cell_count);
            }
        }
        if solid_mask_data.len() != cell_count {
            bail!("Solid mask has {} cells, expected {}", solid_mask_data.len(), cell_count);
        }
        if material_ids_data.len() != cell_count {
            bail!("Material IDs has {} cells, expected {}", material_ids_data.len(), cell_count);
        }
        if diffusion_coeffs_data.len() != species_count {
            bail!("Diffusion coeffs has {} entries, expected {}", diffusion_coeffs_data.len(), species_count);
        }

        // Create Vulkan instance
        let entry = unsafe { ash::Entry::load()? };
        
        let app_name = c"FluidSim";
        let engine_name = c"FluidSim Engine";
        
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_2);

        let extensions: Vec<*const i8> = vec![];

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions);

        let instance = unsafe { entry.create_instance(&instance_info, None)? };

        // Select physical device (prefer discrete GPU)
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        if physical_devices.is_empty() {
            bail!("No Vulkan-capable GPU found");
        }

        let physical_device = physical_devices.into_iter()
            .max_by_key(|&pd| {
                let props = unsafe { instance.get_physical_device_properties(pd) };
                match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
                    _ => 0,
                }
            })
            .unwrap();

        // Find compute queue family
        let queue_families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let compute_queue_family = queue_families.iter()
            .position(|qf| qf.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .context("No compute queue family found")? as u32;

        // Create logical device
        let queue_priority = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(compute_queue_family)
            .queue_priorities(&queue_priority);

        let device_extensions: Vec<*const i8> = vec![];

        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extensions);

        let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
        let compute_queue = unsafe { device.get_device_queue(compute_queue_family, 0) };

        // Create memory allocator
        let allocator = Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })?;
        let allocator = Arc::new(Mutex::new(allocator));

        // Calculate buffer sizes
        let conc_buffer_size = (species_count * cell_count * std::mem::size_of::<f32>()) as u64;
        let mask_buffer_size = (cell_count * std::mem::size_of::<u32>()) as u64;
        let coeffs_buffer_size = (species_count * std::mem::size_of::<f32>()) as u64;

        // Create buffers
        let mut alloc = allocator.lock();
        
        let conc_buffer_a = GpuBuffer::new(
            &device,
            &mut alloc,
            conc_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            "conc_buffer_a",
        )?;

        let conc_buffer_b = GpuBuffer::new(
            &device,
            &mut alloc,
            conc_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            "conc_buffer_b",
        )?;

        let solid_mask = GpuBuffer::new(
            &device,
            &mut alloc,
            mask_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            "solid_mask",
        )?;

        let material_ids = GpuBuffer::new(
            &device,
            &mut alloc,
            mask_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            "material_ids",
        )?;

        let diffusion_coeffs = GpuBuffer::new(
            &device,
            &mut alloc,
            coeffs_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            "diffusion_coeffs",
        )?;

        // Staging buffer for uploads and readback
        let staging_size = conc_buffer_size.max(mask_buffer_size);
        let mut staging_buffer = GpuBuffer::new(
            &device,
            &mut alloc,
            staging_size,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::CpuToGpu,
            "staging",
        )?;
        
        // Readback buffer for GPU->CPU transfers
        // Use CpuToGpu since some drivers have issues with GpuToCpu
        let readback_buffer = GpuBuffer::new(
            &device,
            &mut alloc,
            staging_size,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::CpuToGpu,  // CpuToGpu also works for reads on most drivers
            "readback",
        )?;

        drop(alloc);

        // Create command pool and buffer for uploads
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(compute_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_info, None)? };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { device.allocate_command_buffers(&alloc_info)? }[0];

        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { device.create_fence(&fence_info, None)? };

        // Upload initial data
        // Flatten concentration data
        let mut flat_conc: Vec<f32> = Vec::with_capacity(species_count * cell_count);
        for species_conc in initial_concentrations {
            flat_conc.extend_from_slice(species_conc);
        }

        // Upload concentrations
        staging_buffer.write(&flat_conc)?;
        Self::copy_buffer_sync(&device, command_pool, compute_queue, fence, staging_buffer.buffer, conc_buffer_a.buffer, conc_buffer_size, Some(command_buffer))?;

        // Upload solid mask
        staging_buffer.write(solid_mask_data)?;
        Self::copy_buffer_sync(&device, command_pool, compute_queue, fence, staging_buffer.buffer, solid_mask.buffer, mask_buffer_size, Some(command_buffer))?;

        // Upload material IDs
        staging_buffer.write(material_ids_data)?;
        Self::copy_buffer_sync(&device, command_pool, compute_queue, fence, staging_buffer.buffer, material_ids.buffer, mask_buffer_size, Some(command_buffer))?;

        // Upload diffusion coefficients
        staging_buffer.write(diffusion_coeffs_data)?;
        Self::copy_buffer_sync(&device, command_pool, compute_queue, fence, staging_buffer.buffer, diffusion_coeffs.buffer, coeffs_buffer_size, Some(command_buffer))?;

        // Create descriptor set layout
        let bindings = [
            // Binding 0: Source concentrations (read)
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Binding 1: Destination concentrations (write)
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Binding 2: Solid mask
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Binding 3: Diffusion coefficients
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings);
        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None)? };

        // Create pipeline layout with push constants
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<DiffusionPushConstants>() as u32);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

        // Compile diffusion shader
        let shader_source = include_str!("../shaders/diffusion.comp");
        let compiler = shaderc::Compiler::new().context("Failed to create shader compiler")?;
        let mut options = shaderc::CompileOptions::new().context("Failed to create compile options")?;
        options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_2 as u32);
        options.set_optimization_level(shaderc::OptimizationLevel::Performance);

        let spirv = compiler.compile_into_spirv(
            shader_source,
            shaderc::ShaderKind::Compute,
            "diffusion.comp",
            "main",
            Some(&options),
        ).context("Failed to compile diffusion shader")?;

        let shader_module_info = vk::ShaderModuleCreateInfo::default()
            .code(spirv.as_binary());
        let shader_module = unsafe { device.create_shader_module(&shader_module_info, None)? };

        let entry_name = c"main";
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(entry_name);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(pipeline_layout);

        let diffusion_pipeline = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|e| anyhow::anyhow!("Failed to create compute pipeline: {:?}", e.1))?[0]
        };

        unsafe { device.destroy_shader_module(shader_module, None) };

        // Create descriptor pool and sets
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(8), // 4 bindings * 2 sets
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(2)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        let layouts = [descriptor_set_layout, descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);
        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };

        // Update descriptor sets
        // Set 0: A -> B
        Self::update_descriptor_set(
            &device,
            descriptor_sets[0],
            conc_buffer_a.buffer,
            conc_buffer_b.buffer,
            solid_mask.buffer,
            diffusion_coeffs.buffer,
            conc_buffer_size,
            mask_buffer_size,
            coeffs_buffer_size,
        );

        // Set 1: B -> A
        Self::update_descriptor_set(
            &device,
            descriptor_sets[1],
            conc_buffer_b.buffer,
            conc_buffer_a.buffer,
            solid_mask.buffer,
            diffusion_coeffs.buffer,
            conc_buffer_size,
            mask_buffer_size,
            coeffs_buffer_size,
        );

        // Create coarse grid for async tooltip readback (mip 8)
        let coarse_grid = CoarseGrid::new(
            device.clone(),
            compute_queue,
            compute_queue_family,
            allocator.clone(),
            width,
            height,
            species_count,
            8, // mip factor
            conc_buffer_a.buffer,
            conc_buffer_b.buffer,
            solid_mask.buffer,
            conc_buffer_size,
            mask_buffer_size,
        ).ok(); // Don't fail if coarse grid creation fails

        Ok(Self {
            instance,
            physical_device,
            device,
            compute_queue,
            compute_queue_family,
            allocator,
            width,
            height,
            cell_count,
            species_count,
            conc_buffer_a,
            conc_buffer_b,
            current_buffer: 0,
            solid_mask,
            material_ids,
            diffusion_coeffs,
            staging_buffer,
            readback_buffer,
            descriptor_set_layout,
            pipeline_layout,
            diffusion_pipeline,
            descriptor_pool,
            descriptor_sets,
            command_pool,
            command_buffer,
            fence,
            coarse_grid,
        })
    }

    fn copy_buffer_sync(
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        fence: vk::Fence,
        src: vk::Buffer,
        dst: vk::Buffer,
        size: u64,
        reuse_cmd: Option<vk::CommandBuffer>,
    ) -> Result<()> {
        let (cmd, allocated) = if let Some(cmd) = reuse_cmd {
            (cmd, false)
        } else {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = unsafe { device.allocate_command_buffers(&alloc_info)? }[0];
            (cmd, true)
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(cmd, &begin_info)? };

        let copy_region = vk::BufferCopy::default()
            .size(size);
        unsafe { device.cmd_copy_buffer(cmd, src, dst, &[copy_region]) };

        unsafe { device.end_command_buffer(cmd)? };

        let submit_info = vk::SubmitInfo::default()
            .command_buffers(std::slice::from_ref(&cmd));
        
        unsafe {
            device.reset_fences(&[fence])?;
            device.queue_submit(queue, &[submit_info], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            if allocated {
                device.free_command_buffers(command_pool, &[cmd]);
            }
        }

        Ok(())
    }

    fn update_descriptor_set(
        device: &ash::Device,
        set: vk::DescriptorSet,
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        mask_buffer: vk::Buffer,
        coeffs_buffer: vk::Buffer,
        conc_size: u64,
        mask_size: u64,
        coeffs_size: u64,
    ) {
        let src_info = [vk::DescriptorBufferInfo::default()
            .buffer(src_buffer)
            .offset(0)
            .range(conc_size)];
        let dst_info = [vk::DescriptorBufferInfo::default()
            .buffer(dst_buffer)
            .offset(0)
            .range(conc_size)];
        let mask_info = [vk::DescriptorBufferInfo::default()
            .buffer(mask_buffer)
            .offset(0)
            .range(mask_size)];
        let coeffs_info = [vk::DescriptorBufferInfo::default()
            .buffer(coeffs_buffer)
            .offset(0)
            .range(coeffs_size)];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&src_info),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&dst_info),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&mask_info),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&coeffs_info),
        ];

        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    /// Run one diffusion step on the GPU.
    pub fn step(&mut self, dt: f32, diffusion_rate: f32) -> Result<()> {
        let push_constants = DiffusionPushConstants {
            width: self.width,
            height: self.height,
            species_count: self.species_count as u32,
            species_index: 0xFFFFFFFF, // Process all species
            diffusion_rate,
            dt,
            _pad: [0, 0],
        };

        // Select descriptor set based on current buffer
        let descriptor_set = self.descriptor_sets[self.current_buffer];

        unsafe {
            self.device.reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(self.command_buffer, &begin_info)?;

            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.diffusion_pipeline,
            );

            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            self.device.cmd_push_constants(
                self.command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push_constants),
            );

            // Dispatch: one thread per cell, process all species
            let workgroup_size = 256u32;
            let num_groups = (self.cell_count as u32 + workgroup_size - 1) / workgroup_size;
            self.device.cmd_dispatch(self.command_buffer, num_groups, self.species_count as u32, 1);

            // Memory barrier before coarse grid computation
            // Ensures diffusion writes complete before coarse reads
            let buffer_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .buffer(if self.current_buffer == 0 { self.conc_buffer_b.buffer } else { self.conc_buffer_a.buffer })
                .offset(0)
                .size(vk::WHOLE_SIZE);

            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[buffer_barrier],
                &[],
            );

            // Run coarse grid computation (reads from destination buffer which will become current after swap)
            // After diffusion: if current_buffer == 0, we wrote to B and will swap to 1 (B becomes current)
            // So coarse grid should read from the buffer that will be current after swap
            if let Some(ref coarse) = self.coarse_grid {
                // After this step completes, current_buffer will have been swapped
                // So coarse should read from 1 - current_buffer (the destination of diffusion)
                let next_buffer = 1 - self.current_buffer;
                coarse.record_compute(self.command_buffer, next_buffer);
            }

            self.device.end_command_buffer(self.command_buffer)?;

            let submit_info = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.command_buffer));

            self.device.reset_fences(&[self.fence])?;
            self.device.queue_submit(self.compute_queue, &[submit_info], self.fence)?;
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
        }

        // Swap buffers
        self.current_buffer = 1 - self.current_buffer;

        Ok(())
    }

    /// Read back concentration data from GPU.
    /// Returns `[species][cell]` layout.
    pub fn read_concentrations(&mut self) -> Result<Vec<Vec<f32>>> {
        let current_buffer = if self.current_buffer == 0 {
            &self.conc_buffer_a
        } else {
            &self.conc_buffer_b
        };

        let total_size = (self.species_count * self.cell_count * std::mem::size_of::<f32>()) as u64;

        Self::copy_buffer_sync(
            &self.device,
            self.command_pool,
            self.compute_queue,
            self.fence,
            current_buffer.buffer,
            self.readback_buffer.buffer,
            total_size,
            Some(self.command_buffer),
        )?;

        let flat_data: Vec<f32> = self.readback_buffer.read(self.species_count * self.cell_count)?;

        // Reshape to [species][cell]
        let mut concentrations = Vec::with_capacity(self.species_count);
        for s in 0..self.species_count {
            let start = s * self.cell_count;
            let end = start + self.cell_count;
            concentrations.push(flat_data[start..end].to_vec());
        }

        Ok(concentrations)
    }

    /// Read back solid mask data.
    pub fn read_solid_mask(&mut self) -> Result<Vec<u32>> {
        let size = (self.cell_count * std::mem::size_of::<u32>()) as u64;

        Self::copy_buffer_sync(
            &self.device,
            self.command_pool,
            self.compute_queue,
            self.fence,
            self.solid_mask.buffer,
            self.readback_buffer.buffer,
            size,
            Some(self.command_buffer),
        )?;

        self.readback_buffer.read(self.cell_count)
    }

    /// Read back material ID data.
    pub fn read_material_ids(&mut self) -> Result<Vec<u32>> {
        let size = (self.cell_count * std::mem::size_of::<u32>()) as u64;

        Self::copy_buffer_sync(
            &self.device,
            self.command_pool,
            self.compute_queue,
            self.fence,
            self.material_ids.buffer,
            self.readback_buffer.buffer,
            size,
            Some(self.command_buffer),
        )?;

        self.readback_buffer.read(self.cell_count)
    }

    /// Get the current concentration buffer for rendering.
    pub fn current_concentration_buffer(&self) -> vk::Buffer {
        if self.current_buffer == 0 {
            self.conc_buffer_a.buffer
        } else {
            self.conc_buffer_b.buffer
        }
    }
}

impl Drop for GpuSimulation {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            
            // Drop coarse_grid FIRST - it holds cloned device handle and references our buffers
            drop(self.coarse_grid.take());

            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_pipeline(self.diffusion_pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            // Buffers and allocations
            let mut alloc = self.allocator.lock();
            
            self.device.destroy_buffer(self.conc_buffer_a.buffer, None);
            alloc.free(std::mem::take(&mut self.conc_buffer_a.allocation)).ok();
            
            self.device.destroy_buffer(self.conc_buffer_b.buffer, None);
            alloc.free(std::mem::take(&mut self.conc_buffer_b.allocation)).ok();
            
            self.device.destroy_buffer(self.solid_mask.buffer, None);
            alloc.free(std::mem::take(&mut self.solid_mask.allocation)).ok();
            
            self.device.destroy_buffer(self.material_ids.buffer, None);
            alloc.free(std::mem::take(&mut self.material_ids.allocation)).ok();
            
            self.device.destroy_buffer(self.diffusion_coeffs.buffer, None);
            alloc.free(std::mem::take(&mut self.diffusion_coeffs.allocation)).ok();
            
            self.device.destroy_buffer(self.staging_buffer.buffer, None);
            alloc.free(std::mem::take(&mut self.staging_buffer.allocation)).ok();
            
            self.device.destroy_buffer(self.readback_buffer.buffer, None);
            alloc.free(std::mem::take(&mut self.readback_buffer.allocation)).ok();
            
            drop(alloc);

            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

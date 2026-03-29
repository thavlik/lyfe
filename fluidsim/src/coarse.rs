//! Coarse grid computation and async readback for tooltip inspection.
//!
//! This module provides:
//! - GPU-side coarse grid averaging (runs every frame after diffusion)
//! - Async non-blocking readback of single coarse cells
//! - Rate-limited reads (at most once per second)
//!
//! The design ensures the render thread is never blocked waiting for
//! tooltip data - all readbacks are asynchronous.

use anyhow::{Context, Result};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::vulkan::{Allocator, AllocationCreateDesc, AllocationScheme, Allocation};
use gpu_allocator::MemoryLocation;
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Push constants for the coarse grid compute shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CoarsePushConstants {
    pub full_width: u32,
    pub full_height: u32,
    pub coarse_width: u32,
    pub coarse_height: u32,
    pub species_count: u32,
    pub mip_factor: u32,
    pub _pad: [u32; 2],
}

/// A small buffer for reading back a single coarse cell's data.
struct ReadbackCell {
    /// Species concentrations for this cell
    concentrations: Vec<f32>,
    /// Number of fluid cells in this coarse cell
    fluid_count: u32,
    /// Number of solid cells
    solid_count: u32,
    /// Coarse cell coordinates
    coord: (u32, u32),
    /// When this data was read
    timestamp: Instant,
}

/// State of an async readback operation.
enum ReadbackState {
    /// No readback in flight
    Idle,
    /// Readback command submitted, waiting for fence
    Pending {
        fence: vk::Fence,
        target_coord: (u32, u32),
        submit_time: Instant,
    },
    /// Readback complete, data available
    Ready(ReadbackCell),
}

/// Coarse grid computation and async readback system.
pub struct CoarseGrid {
    // Vulkan handles (borrowed from GpuSimulation)
    device: ash::Device,
    queue: vk::Queue,
    
    // Coarse grid dimensions
    pub mip_factor: u32,
    pub coarse_width: u32,
    pub coarse_height: u32,
    pub species_count: usize,
    pub full_width: u32,
    pub full_height: u32,
    
    // GPU buffers for coarse grid
    coarse_conc_buffer: CoarseBuffer,
    coarse_fluid_count_buffer: CoarseBuffer,
    
    // Small readback buffer (just one coarse cell worth of data)
    cell_readback_buffer: CoarseBuffer,
    
    // Compute pipeline for coarse averaging
    coarse_pipeline: vk::Pipeline,
    coarse_pipeline_layout: vk::PipelineLayout,
    coarse_descriptor_set_layout: vk::DescriptorSetLayout,
    coarse_descriptor_pool: vk::DescriptorPool,
    coarse_descriptor_sets: Vec<vk::DescriptorSet>, // [0] for buffer A, [1] for buffer B
    
    // Command resources for async readback
    readback_command_pool: vk::CommandPool,
    readback_command_buffer: vk::CommandBuffer,
    readback_fence: vk::Fence,
    
    // Async readback state
    readback_state: ReadbackState,
    last_read_time: Instant,
    min_read_interval: Duration,
    
    // Latest cached result (may be stale)
    cached_cell: Option<ReadbackCell>,
    
    // Allocator reference
    allocator: Arc<Mutex<Allocator>>,
}

/// GPU buffer wrapper for coarse grid.
struct CoarseBuffer {
    buffer: vk::Buffer,
    allocation: Allocation,
    size: u64,
}

impl CoarseBuffer {
    fn new(
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

    fn read<T: Pod + Clone>(&self, count: usize) -> Result<Vec<T>> {
        let mapped = self.allocation.mapped_slice()
            .context("Buffer not mapped for CPU access")?;
        let bytes = &mapped[..count * std::mem::size_of::<T>()];
        Ok(bytemuck::cast_slice(bytes).to_vec())
    }
}

/// Result of inspecting a coarse cell (async).
#[derive(Debug, Clone)]
pub struct CoarseCellData {
    /// Average concentrations per species
    pub concentrations: Vec<f32>,
    /// Number of fluid cells in this coarse cell
    pub fluid_count: u32,
    /// Number of solid cells
    pub solid_count: u32,
    /// Coarse cell coordinates
    pub coord: (u32, u32),
    /// Age of this data
    pub age: Duration,
}

impl CoarseGrid {
    /// Create a new coarse grid system.
    pub fn new(
        device: ash::Device,
        queue: vk::Queue,
        queue_family: u32,
        allocator: Arc<Mutex<Allocator>>,
        full_width: u32,
        full_height: u32,
        species_count: usize,
        mip_factor: u32,
        src_buffer_a: vk::Buffer,
        src_buffer_b: vk::Buffer,
        solid_mask_buffer: vk::Buffer,
        conc_buffer_size: u64,
        mask_buffer_size: u64,
    ) -> Result<Self> {
        let coarse_width = (full_width + mip_factor - 1) / mip_factor;
        let coarse_height = (full_height + mip_factor - 1) / mip_factor;
        let coarse_cell_count = (coarse_width * coarse_height) as usize;
        
        log::info!("Creating coarse grid: {}x{} (mip {})", coarse_width, coarse_height, mip_factor);
        
        // Buffer sizes
        let coarse_conc_size = (species_count * coarse_cell_count * std::mem::size_of::<f32>()) as u64;
        let coarse_count_size = (coarse_cell_count * std::mem::size_of::<u32>()) as u64;
        // Readback buffer: concentrations for one cell + fluid count
        let cell_readback_size = ((species_count + 1) * std::mem::size_of::<f32>()) as u64;
        
        let mut alloc = allocator.lock();
        
        // Create coarse grid buffers (GPU only, updated every frame)
        let coarse_conc_buffer = CoarseBuffer::new(
            &device,
            &mut alloc,
            coarse_conc_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::GpuOnly,
            "coarse_concentrations",
        )?;
        
        let coarse_fluid_count_buffer = CoarseBuffer::new(
            &device,
            &mut alloc,
            coarse_count_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::GpuOnly,
            "coarse_fluid_counts",
        )?;
        
        // Small readback buffer (CPU visible)
        let cell_readback_buffer = CoarseBuffer::new(
            &device,
            &mut alloc,
            cell_readback_size.max(256), // Minimum size for alignment
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::CpuToGpu,
            "cell_readback",
        )?;
        
        drop(alloc);
        
        // Compile coarse grid shader
        let shader_source = include_str!("../shaders/coarse_grid.comp");
        let compiler = shaderc::Compiler::new().context("Failed to create shader compiler")?;
        let mut options = shaderc::CompileOptions::new().context("Failed to create compile options")?;
        options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_2 as u32);
        options.set_optimization_level(shaderc::OptimizationLevel::Performance);

        let spirv = compiler.compile_into_spirv(
            shader_source,
            shaderc::ShaderKind::Compute,
            "coarse_grid.comp",
            "main",
            Some(&options),
        ).context("Failed to compile coarse grid shader")?;

        let shader_module_info = vk::ShaderModuleCreateInfo::default()
            .code(spirv.as_binary());
        let shader_module = unsafe { device.create_shader_module(&shader_module_info, None)? };
        
        // Create descriptor set layout for coarse shader
        let bindings = [
            // Binding 0: Full concentrations (read)
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Binding 1: Solid mask (read)
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Binding 2: Coarse concentrations (write)
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Binding 3: Coarse fluid counts (write)
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings);
        let coarse_descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None)? };
        
        // Create pipeline layout
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<CoarsePushConstants>() as u32);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&coarse_descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        let coarse_pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };
        
        // Create compute pipeline
        let entry_name = c"main";
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(entry_name);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(coarse_pipeline_layout);

        let coarse_pipeline = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|e| anyhow::anyhow!("Failed to create coarse pipeline: {:?}", e.1))?[0]
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
        let coarse_descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        let layouts = [coarse_descriptor_set_layout, coarse_descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(coarse_descriptor_pool)
            .set_layouts(&layouts);
        let coarse_descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };
        
        // Update descriptor sets for both ping-pong buffers
        Self::update_coarse_descriptor_set(
            &device,
            coarse_descriptor_sets[0],
            src_buffer_a,
            solid_mask_buffer,
            coarse_conc_buffer.buffer,
            coarse_fluid_count_buffer.buffer,
            conc_buffer_size,
            mask_buffer_size,
            coarse_conc_size,
            coarse_count_size,
        );
        
        Self::update_coarse_descriptor_set(
            &device,
            coarse_descriptor_sets[1],
            src_buffer_b,
            solid_mask_buffer,
            coarse_conc_buffer.buffer,
            coarse_fluid_count_buffer.buffer,
            conc_buffer_size,
            mask_buffer_size,
            coarse_conc_size,
            coarse_count_size,
        );
        
        // Create command pool and buffer for async readback
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let readback_command_pool = unsafe { device.create_command_pool(&pool_info, None)? };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(readback_command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let readback_command_buffer = unsafe { device.allocate_command_buffers(&alloc_info)? }[0];

        let fence_info = vk::FenceCreateInfo::default();
        let readback_fence = unsafe { device.create_fence(&fence_info, None)? };
        
        Ok(Self {
            device,
            queue,
            mip_factor,
            coarse_width,
            coarse_height,
            species_count,
            full_width,
            full_height,
            coarse_conc_buffer,
            coarse_fluid_count_buffer,
            cell_readback_buffer,
            coarse_pipeline,
            coarse_pipeline_layout,
            coarse_descriptor_set_layout,
            coarse_descriptor_pool,
            coarse_descriptor_sets,
            readback_command_pool,
            readback_command_buffer,
            readback_fence,
            readback_state: ReadbackState::Idle,
            last_read_time: Instant::now() - Duration::from_secs(10), // Allow immediate first read
            min_read_interval: Duration::from_millis(200),
            cached_cell: None,
            allocator,
        })
    }
    
    fn update_coarse_descriptor_set(
        device: &ash::Device,
        set: vk::DescriptorSet,
        full_conc_buffer: vk::Buffer,
        solid_mask_buffer: vk::Buffer,
        coarse_conc_buffer: vk::Buffer,
        coarse_count_buffer: vk::Buffer,
        full_conc_size: u64,
        mask_size: u64,
        coarse_conc_size: u64,
        coarse_count_size: u64,
    ) {
        let full_conc_info = [vk::DescriptorBufferInfo::default()
            .buffer(full_conc_buffer)
            .offset(0)
            .range(full_conc_size)];
        let mask_info = [vk::DescriptorBufferInfo::default()
            .buffer(solid_mask_buffer)
            .offset(0)
            .range(mask_size)];
        let coarse_conc_info = [vk::DescriptorBufferInfo::default()
            .buffer(coarse_conc_buffer)
            .offset(0)
            .range(coarse_conc_size)];
        let coarse_count_info = [vk::DescriptorBufferInfo::default()
            .buffer(coarse_count_buffer)
            .offset(0)
            .range(coarse_count_size)];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&full_conc_info),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&mask_info),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&coarse_conc_info),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&coarse_count_info),
        ];

        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }
    
    /// Record coarse grid computation commands into an existing command buffer.
    /// Should be called every frame after diffusion, before submitting the command buffer.
    pub fn record_compute(&self, cmd: vk::CommandBuffer, current_buffer: usize) {
        let push_constants = CoarsePushConstants {
            full_width: self.full_width,
            full_height: self.full_height,
            coarse_width: self.coarse_width,
            coarse_height: self.coarse_height,
            species_count: self.species_count as u32,
            mip_factor: self.mip_factor,
            _pad: [0, 0],
        };

        let descriptor_set = self.coarse_descriptor_sets[current_buffer];

        unsafe {
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.coarse_pipeline,
            );

            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.coarse_pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            self.device.cmd_push_constants(
                cmd,
                self.coarse_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push_constants),
            );

            // Dispatch: one thread per coarse cell per species
            // Using 8x8 workgroups
            let groups_x = (self.coarse_width + 7) / 8;
            let groups_y = (self.coarse_height + 7) / 8;
            let groups_z = self.species_count as u32;
            self.device.cmd_dispatch(cmd, groups_x, groups_y, groups_z);
        }
    }
    
    /// Convert screen coordinates to coarse cell coordinates.
    pub fn screen_to_coarse(&self, screen_x: f32, screen_y: f32) -> (u32, u32) {
        let grid_x = (screen_x.max(0.0) as u32).min(self.full_width - 1);
        let grid_y = (screen_y.max(0.0) as u32).min(self.full_height - 1);
        let coarse_x = (grid_x / self.mip_factor).min(self.coarse_width - 1);
        let coarse_y = (grid_y / self.mip_factor).min(self.coarse_height - 1);
        (coarse_x, coarse_y)
    }
    
    /// Request async readback of a coarse cell.
    /// Returns immediately - check poll_readback() later for results.
    /// Rate-limited to at most once per second.
    pub fn request_cell_readback(&mut self, coarse_x: u32, coarse_y: u32) -> bool {
        // Check rate limit
        let now = Instant::now();
        if now.duration_since(self.last_read_time) < self.min_read_interval {
            return false;
        }
        
        // Check if a readback is already in flight
        if matches!(self.readback_state, ReadbackState::Pending { .. }) {
            return false;
        }
        
        // Bounds check
        if coarse_x >= self.coarse_width || coarse_y >= self.coarse_height {
            return false;
        }
        
        let coarse_idx = coarse_y * self.coarse_width + coarse_x;
        let coarse_cell_count = (self.coarse_width * self.coarse_height) as usize;
        
        // Record copy commands for this cell's data
        unsafe {
            self.device.reset_command_buffer(self.readback_command_buffer, vk::CommandBufferResetFlags::empty()).ok();
            
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            if self.device.begin_command_buffer(self.readback_command_buffer, &begin_info).is_err() {
                return false;
            }

            let barriers = [
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .buffer(self.coarse_conc_buffer.buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE),
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .buffer(self.coarse_fluid_count_buffer.buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE),
            ];

            self.device.cmd_pipeline_barrier(
                self.readback_command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &barriers,
                &[],
            );
            
            // Copy concentrations for each species from this coarse cell
            let mut copy_regions = Vec::with_capacity(self.species_count + 1);
            for s in 0..self.species_count {
                let src_offset = ((s * coarse_cell_count) + coarse_idx as usize) * std::mem::size_of::<f32>();
                let dst_offset = s * std::mem::size_of::<f32>();
                copy_regions.push(vk::BufferCopy::default()
                    .src_offset(src_offset as u64)
                    .dst_offset(dst_offset as u64)
                    .size(std::mem::size_of::<f32>() as u64));
            }
            
            self.device.cmd_copy_buffer(
                self.readback_command_buffer,
                self.coarse_conc_buffer.buffer,
                self.cell_readback_buffer.buffer,
                &copy_regions,
            );
            
            // Copy fluid count
            let fluid_count_offset = coarse_idx as usize * std::mem::size_of::<u32>();
            let dst_offset = self.species_count * std::mem::size_of::<f32>();
            let count_copy = vk::BufferCopy::default()
                .src_offset(fluid_count_offset as u64)
                .dst_offset(dst_offset as u64)
                .size(std::mem::size_of::<u32>() as u64);
            
            self.device.cmd_copy_buffer(
                self.readback_command_buffer,
                self.coarse_fluid_count_buffer.buffer,
                self.cell_readback_buffer.buffer,
                &[count_copy],
            );
            
            if self.device.end_command_buffer(self.readback_command_buffer).is_err() {
                return false;
            }
            
            // Reset and submit with fence
            self.device.reset_fences(&[self.readback_fence]).ok();
            
            let submit_info = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.readback_command_buffer));
            
            if self.device.queue_submit(self.queue, &[submit_info], self.readback_fence).is_err() {
                return false;
            }
        }
        
        self.last_read_time = now;
        self.readback_state = ReadbackState::Pending {
            fence: self.readback_fence,
            target_coord: (coarse_x, coarse_y),
            submit_time: now,
        };
        
        true
    }
    
    /// Poll for async readback completion.
    /// Returns Some if new data is available, None otherwise.
    /// This is non-blocking.
    pub fn poll_readback(&mut self) -> Option<CoarseCellData> {
        match &self.readback_state {
            ReadbackState::Idle => None,
            ReadbackState::Pending { fence, target_coord, submit_time } => {
                // Check if fence is signaled (non-blocking)
                let status = unsafe {
                    self.device.get_fence_status(*fence)
                };
                
                if status == Ok(true) {
                    // Readback complete - read the data
                    let coord = *target_coord;
                    let _submit_time = *submit_time;
                    
                    // Read concentrations
                    let concs: Vec<f32> = match self.cell_readback_buffer.read(self.species_count) {
                        Ok(v) => v,
                        Err(_) => {
                            self.readback_state = ReadbackState::Idle;
                            return None;
                        }
                    };
                    
                    // Read fluid count (stored after concentrations)
                    let mapped = self.cell_readback_buffer.allocation.mapped_slice();
                    let fluid_count = if let Some(slice) = mapped {
                        let offset = self.species_count * std::mem::size_of::<f32>();
                        let bytes = &slice[offset..offset + 4];
                        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                    } else {
                        0
                    };
                    
                    let mip_squared = self.mip_factor * self.mip_factor;
                    let solid_count = mip_squared.saturating_sub(fluid_count);
                    
                    let cell = ReadbackCell {
                        concentrations: concs.clone(),
                        fluid_count,
                        solid_count,
                        coord,
                        timestamp: Instant::now(),
                    };
                    
                    self.cached_cell = Some(cell);
                    self.readback_state = ReadbackState::Idle;
                    
                    Some(CoarseCellData {
                        concentrations: concs,
                        fluid_count,
                        solid_count,
                        coord,
                        age: Duration::ZERO,
                    })
                } else {
                    None
                }
            }
            ReadbackState::Ready(cell) => {
                let data = CoarseCellData {
                    concentrations: cell.concentrations.clone(),
                    fluid_count: cell.fluid_count,
                    solid_count: cell.solid_count,
                    coord: cell.coord,
                    age: cell.timestamp.elapsed(),
                };
                self.readback_state = ReadbackState::Idle;
                Some(data)
            }
        }
    }
    
    /// Get the most recently read cell data (may be stale).
    pub fn get_cached_cell(&self) -> Option<CoarseCellData> {
        self.cached_cell.as_ref().map(|cell| CoarseCellData {
            concentrations: cell.concentrations.clone(),
            fluid_count: cell.fluid_count,
            solid_count: cell.solid_count,
            coord: cell.coord,
            age: cell.timestamp.elapsed(),
        })
    }
    
    /// Check if a readback is in flight.
    pub fn is_readback_pending(&self) -> bool {
        matches!(self.readback_state, ReadbackState::Pending { .. })
    }
    
    /// Set the minimum interval between readbacks.
    pub fn set_read_interval(&mut self, interval: Duration) {
        self.min_read_interval = interval;
    }
}

impl Drop for CoarseGrid {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            
            self.device.destroy_fence(self.readback_fence, None);
            self.device.destroy_command_pool(self.readback_command_pool, None);
            
            self.device.destroy_pipeline(self.coarse_pipeline, None);
            self.device.destroy_pipeline_layout(self.coarse_pipeline_layout, None);
            self.device.destroy_descriptor_pool(self.coarse_descriptor_pool, None);
            self.device.destroy_descriptor_set_layout(self.coarse_descriptor_set_layout, None);
            
            let mut alloc = self.allocator.lock();
            
            self.device.destroy_buffer(self.coarse_conc_buffer.buffer, None);
            alloc.free(std::mem::take(&mut self.coarse_conc_buffer.allocation)).ok();
            
            self.device.destroy_buffer(self.coarse_fluid_count_buffer.buffer, None);
            alloc.free(std::mem::take(&mut self.coarse_fluid_count_buffer.allocation)).ok();
            
            self.device.destroy_buffer(self.cell_readback_buffer.buffer, None);
            alloc.free(std::mem::take(&mut self.cell_readback_buffer.allocation)).ok();
        }
    }
}

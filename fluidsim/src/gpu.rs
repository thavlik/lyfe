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
use crate::leak::LeakChannel;
use crate::species::SpeciesRegistry;

#[derive(Clone)]
pub struct SharedGpuContext {
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub allocator: Arc<Mutex<Allocator>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuRenderBuffers {
    pub concentration: vk::Buffer,
    pub solid_mask: vk::Buffer,
    pub material_ids: vk::Buffer,
    pub temperature: vk::Buffer,
}

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

/// Push constants for the reaction compute shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ReactionPushConstants {
    pub width: u32,
    pub height: u32,
    pub species_count: u32,
    pub num_reactions: u32,
    pub dt: f32,
    pub _pad: [u32; 3],
}

/// Push constants for the leak-channel compute shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LeakPushConstants {
    pub width: u32,
    pub height: u32,
    pub species_count: u32,
    pub num_channels: u32,
    pub dt: f32,
    pub _pad: [u32; 3],
}

/// Push constants for the thermal diffusion compute shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ThermalPushConstants {
    pub width: u32,
    pub height: u32,
    pub thermal_diffusivity: f32,
    pub dt: f32,
    pub _pad: [u32; 2],
}

/// Push constants for the charge-neutrality projection shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ChargeProjectionPushConstants {
    pub width: u32,
    pub height: u32,
    pub species_count: u32,
    pub correction_strength: f32,
    pub _pad: [u32; 3],
}

/// A single GPU-packed reaction rule.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuReactionRule {
    pub reactant_a_index: u32,
    pub reactant_b_index: u32,
    pub product_a_index: u32,
    pub product_b_index: u32,
    pub effective_rate_bits: u32, // f32 bit-cast
    pub enthalpy_delta_bits: u32, // f32 bit-cast
    pub entropy_delta_bits: u32, // f32 bit-cast
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuLeakChannel {
    pub species_index: u32,
    pub sink_x: i32,
    pub sink_y: i32,
    pub source_x: i32,
    pub source_y: i32,
    pub rate: f32,
    pub rotation_byte: u32,
    pub _pad: u32,
}

impl GpuReactionRule {
    pub const NONE: u32 = u32::MAX;

    pub fn new(
        a: u32,
        b: Option<u32>,
        product_a: Option<u32>,
        product_b: Option<u32>,
        rate: f32,
        enthalpy: f32,
        entropy: f32,
    ) -> Self {
        Self {
            reactant_a_index: a,
            reactant_b_index: b.unwrap_or(Self::NONE),
            product_a_index: product_a.unwrap_or(Self::NONE),
            product_b_index: product_b.unwrap_or(Self::NONE),
            effective_rate_bits: rate.to_bits(),
            enthalpy_delta_bits: enthalpy.to_bits(),
            entropy_delta_bits: entropy.to_bits(),
        }
    }
}

/// Maximum number of reaction rules the GPU buffer can hold.
const MAX_REACTION_RULES: usize = 16;
const MAX_LEAK_CHANNELS: usize = 32;

/// Optional reaction pipeline state.
/// Created lazily when reaction rules are first uploaded.
pub struct ReactionPipelineState {
    pub reaction_descriptor_set_layout: vk::DescriptorSetLayout,
    pub reaction_pipeline_layout: vk::PipelineLayout,
    pub reaction_pipeline: vk::Pipeline,
    pub reaction_descriptor_pool: vk::DescriptorPool,
    /// Four descriptor sets: concentration current buffer × temperature current buffer.
    pub reaction_descriptor_sets: Vec<vk::DescriptorSet>,
    /// Temperature buffers (GPU-only, per-cell f32 Kelvin)
    pub temperature_buffer_a: GpuBuffer,
    pub temperature_buffer_b: GpuBuffer,
    pub temperature_current_buffer: usize,
    pub thermal_descriptor_set_layout: vk::DescriptorSetLayout,
    pub thermal_pipeline_layout: vk::PipelineLayout,
    pub thermal_pipeline: vk::Pipeline,
    pub thermal_descriptor_pool: vk::DescriptorPool,
    /// Two descriptor sets: [0] = A->B, [1] = B->A.
    pub thermal_descriptor_sets: Vec<vk::DescriptorSet>,
    /// Reaction rules buffer (CPU-visible, small)
    pub rules_buffer: GpuBuffer,
    /// Number of currently active reaction rules
    pub active_rule_count: u32,
    /// Hash of the current rule set (for recompilation tracking)
    pub rule_set_hash: u64,
}

pub struct LeakPipelineState {
    pub leak_descriptor_set_layout: vk::DescriptorSetLayout,
    pub leak_pipeline_layout: vk::PipelineLayout,
    pub leak_pipeline: vk::Pipeline,
    pub leak_descriptor_pool: vk::DescriptorPool,
    pub leak_descriptor_sets: Vec<vk::DescriptorSet>,
    pub channels_buffer: GpuBuffer,
    pub active_channel_count: u32,
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
    pub entry: Option<ash::Entry>,
    // Vulkan handles
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub compute_queue: vk::Queue,
    pub compute_queue_family: u32,

    // Memory allocator
    pub allocator: Option<Arc<Mutex<Allocator>>>,
    pub owns_vulkan_context: bool,

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
    // Integer ionic charges / valences per species
    pub species_charges: GpuBuffer,

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
    pub charge_descriptor_set_layout: vk::DescriptorSetLayout,
    pub charge_pipeline_layout: vk::PipelineLayout,
    pub charge_projection_pipeline: vk::Pipeline,
    pub charge_descriptor_pool: vk::DescriptorPool,
    pub charge_descriptor_sets: Vec<vk::DescriptorSet>, // [0] = A current, [1] = B current

    // Command resources
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub fence: vk::Fence,
    
    // Coarse grid for async tooltip readback
    pub coarse_grid: Option<CoarseGrid>,

    // Reaction pipeline (created lazily when rules are first uploaded)
    pub reaction: Option<ReactionPipelineState>,
    pub leak: Option<LeakPipelineState>,
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
        species_charges_data: &[i32],
    ) -> Result<Self> {
        let (entry, context) = Self::create_owned_context()?;
        Self::new_from_context(
            Some(entry),
            context,
            true,
            width,
            height,
            species_count,
            initial_concentrations,
            solid_mask_data,
            material_ids_data,
            diffusion_coeffs_data,
            species_charges_data,
        )
    }

    pub fn new_with_shared_context(
        context: SharedGpuContext,
        width: u32,
        height: u32,
        species_count: usize,
        initial_concentrations: &[Vec<f32>],
        solid_mask_data: &[u32],
        material_ids_data: &[u32],
        diffusion_coeffs_data: &[f32],
        species_charges_data: &[i32],
    ) -> Result<Self> {
        Self::new_from_context(
            None,
            context,
            false,
            width,
            height,
            species_count,
            initial_concentrations,
            solid_mask_data,
            material_ids_data,
            diffusion_coeffs_data,
            species_charges_data,
        )
    }

    fn create_owned_context() -> Result<(ash::Entry, SharedGpuContext)> {
        let entry = unsafe { ash::Entry::load()? };

        let app_name = c"FluidSim";
        let engine_name = c"FluidSim Engine";

        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_2);

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info);

        let instance = unsafe { entry.create_instance(&instance_info, None)? };

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

        let queue_families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let compute_queue_family = queue_families.iter()
            .enumerate()
            .find_map(|(index, qf)| {
                (qf.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && qf.queue_flags.contains(vk::QueueFlags::COMPUTE))
                    .then_some(index as u32)
            })
            .or_else(|| {
                queue_families.iter().enumerate().find_map(|(index, qf)| {
                    qf.queue_flags.contains(vk::QueueFlags::COMPUTE)
                        .then_some(index as u32)
                })
            })
            .context("No compute-capable queue family found")?;

        let queue_priority = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(compute_queue_family)
            .queue_priorities(&queue_priority);

        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info));

        let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
        let queue = unsafe { device.get_device_queue(compute_queue_family, 0) };

        let allocator = Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })?;

        Ok((
            entry,
            SharedGpuContext {
                instance,
                physical_device,
                device,
                queue,
                queue_family: compute_queue_family,
                allocator: Arc::new(Mutex::new(allocator)),
            },
        ))
    }

    fn new_from_context(
        entry: Option<ash::Entry>,
        context: SharedGpuContext,
        owns_vulkan_context: bool,
        width: u32,
        height: u32,
        species_count: usize,
        initial_concentrations: &[Vec<f32>],
        solid_mask_data: &[u32],
        material_ids_data: &[u32],
        diffusion_coeffs_data: &[f32],
        species_charges_data: &[i32],
    ) -> Result<Self> {
        let cell_count = (width * height) as usize;

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
        if species_charges_data.len() != species_count {
            bail!("Species charges has {} entries, expected {}", species_charges_data.len(), species_count);
        }

        let SharedGpuContext {
            instance,
            physical_device,
            device,
            queue: compute_queue,
            queue_family: compute_queue_family,
            allocator,
        } = context;

        let conc_buffer_size = (species_count * cell_count * std::mem::size_of::<f32>()) as u64;
        let mask_buffer_size = (cell_count * std::mem::size_of::<u32>()) as u64;
        let coeffs_buffer_size = (species_count * std::mem::size_of::<f32>()) as u64;
        let charges_buffer_size = (species_count * std::mem::size_of::<i32>()) as u64;

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

        let species_charges = GpuBuffer::new(
            &device,
            &mut alloc,
            charges_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            "species_charges",
        )?;

        let staging_size = conc_buffer_size.max(mask_buffer_size).max(charges_buffer_size);
        let mut staging_buffer = GpuBuffer::new(
            &device,
            &mut alloc,
            staging_size,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::CpuToGpu,
            "staging",
        )?;

        let readback_buffer = GpuBuffer::new(
            &device,
            &mut alloc,
            staging_size,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::CpuToGpu,
            "readback",
        )?;

        drop(alloc);

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

        let mut flat_conc: Vec<f32> = Vec::with_capacity(species_count * cell_count);
        for species_conc in initial_concentrations {
            flat_conc.extend_from_slice(species_conc);
        }

        staging_buffer.write(&flat_conc)?;
        Self::copy_buffer_sync(&device, command_pool, compute_queue, fence, staging_buffer.buffer, conc_buffer_a.buffer, conc_buffer_size, Some(command_buffer))?;

        staging_buffer.write(solid_mask_data)?;
        Self::copy_buffer_sync(&device, command_pool, compute_queue, fence, staging_buffer.buffer, solid_mask.buffer, mask_buffer_size, Some(command_buffer))?;

        staging_buffer.write(material_ids_data)?;
        Self::copy_buffer_sync(&device, command_pool, compute_queue, fence, staging_buffer.buffer, material_ids.buffer, mask_buffer_size, Some(command_buffer))?;

        staging_buffer.write(diffusion_coeffs_data)?;
        Self::copy_buffer_sync(&device, command_pool, compute_queue, fence, staging_buffer.buffer, diffusion_coeffs.buffer, coeffs_buffer_size, Some(command_buffer))?;

        staging_buffer.write(species_charges_data)?;
        Self::copy_buffer_sync(&device, command_pool, compute_queue, fence, staging_buffer.buffer, species_charges.buffer, charges_buffer_size, Some(command_buffer))?;

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings);
        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None)? };

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<DiffusionPushConstants>() as u32);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

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

        let charge_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let charge_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&charge_bindings);
        let charge_descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(&charge_layout_info, None)?
        };

        let charge_push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<ChargeProjectionPushConstants>() as u32);

        let charge_pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&charge_descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&charge_push_constant_range));
        let charge_pipeline_layout = unsafe {
            device.create_pipeline_layout(&charge_pipeline_layout_info, None)?
        };

        let charge_spirv = compiler.compile_into_spirv(
            include_str!("../shaders/charge_projection.comp"),
            shaderc::ShaderKind::Compute,
            "charge_projection.comp",
            "main",
            Some(&options),
        ).context("Failed to compile charge projection shader")?;

        let charge_shader_module_info = vk::ShaderModuleCreateInfo::default()
            .code(charge_spirv.as_binary());
        let charge_shader_module = unsafe { device.create_shader_module(&charge_shader_module_info, None)? };

        let charge_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(charge_shader_module)
            .name(entry_name);

        let charge_pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(charge_stage_info)
            .layout(charge_pipeline_layout);

        let charge_projection_pipeline = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &[charge_pipeline_info], None)
                .map_err(|e| anyhow::anyhow!("Failed to create charge projection pipeline: {:?}", e.1))?[0]
        };

        unsafe { device.destroy_shader_module(charge_shader_module, None) };

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(10),
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

        let charge_pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(6),
        ];
        let charge_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(2)
            .pool_sizes(&charge_pool_sizes);
        let charge_descriptor_pool = unsafe { device.create_descriptor_pool(&charge_pool_info, None)? };

        let charge_layouts = [charge_descriptor_set_layout, charge_descriptor_set_layout];
        let charge_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(charge_descriptor_pool)
            .set_layouts(&charge_layouts);
        let charge_descriptor_sets = unsafe { device.allocate_descriptor_sets(&charge_alloc_info)? };

        Self::update_descriptor_set(
            &device,
            descriptor_sets[0],
            conc_buffer_a.buffer,
            conc_buffer_b.buffer,
            solid_mask.buffer,
            diffusion_coeffs.buffer,
            species_charges.buffer,
            conc_buffer_size,
            mask_buffer_size,
            coeffs_buffer_size,
            charges_buffer_size,
        );

        Self::update_descriptor_set(
            &device,
            descriptor_sets[1],
            conc_buffer_b.buffer,
            conc_buffer_a.buffer,
            solid_mask.buffer,
            diffusion_coeffs.buffer,
            species_charges.buffer,
            conc_buffer_size,
            mask_buffer_size,
            coeffs_buffer_size,
            charges_buffer_size,
        );

        Self::update_charge_descriptor_set(
            &device,
            charge_descriptor_sets[0],
            conc_buffer_a.buffer,
            solid_mask.buffer,
            species_charges.buffer,
            conc_buffer_size,
            mask_buffer_size,
            charges_buffer_size,
        );

        Self::update_charge_descriptor_set(
            &device,
            charge_descriptor_sets[1],
            conc_buffer_b.buffer,
            solid_mask.buffer,
            species_charges.buffer,
            conc_buffer_size,
            mask_buffer_size,
            charges_buffer_size,
        );

        let coarse_grid = None;

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            compute_queue,
            compute_queue_family,
            allocator: Some(allocator),
            owns_vulkan_context,
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
            species_charges,
            staging_buffer,
            readback_buffer,
            descriptor_set_layout,
            pipeline_layout,
            diffusion_pipeline,
            descriptor_pool,
            descriptor_sets,
            charge_descriptor_set_layout,
            charge_pipeline_layout,
            charge_projection_pipeline,
            charge_descriptor_pool,
            charge_descriptor_sets,
            command_pool,
            command_buffer,
            fence,
            coarse_grid,
            reaction: None,
            leak: None,
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
            unsafe {
                device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            }
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

    fn readback_buffer_sync(
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
            unsafe {
                device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            }
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

        let barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .buffer(src)
            .offset(0)
            .size(vk::WHOLE_SIZE);

        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );
        }

        let copy_region = vk::BufferCopy::default().size(size);
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
        charges_buffer: vk::Buffer,
        conc_size: u64,
        mask_size: u64,
        coeffs_size: u64,
        charges_size: u64,
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
        let charges_info = [vk::DescriptorBufferInfo::default()
            .buffer(charges_buffer)
            .offset(0)
            .range(charges_size)];

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
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&charges_info),
        ];

        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    fn update_charge_descriptor_set(
        device: &ash::Device,
        set: vk::DescriptorSet,
        conc_buffer: vk::Buffer,
        mask_buffer: vk::Buffer,
        charges_buffer: vk::Buffer,
        conc_size: u64,
        mask_size: u64,
        charges_size: u64,
    ) {
        let conc_info = [vk::DescriptorBufferInfo::default()
            .buffer(conc_buffer)
            .offset(0)
            .range(conc_size)];
        let mask_info = [vk::DescriptorBufferInfo::default()
            .buffer(mask_buffer)
            .offset(0)
            .range(mask_size)];
        let charges_info = [vk::DescriptorBufferInfo::default()
            .buffer(charges_buffer)
            .offset(0)
            .range(charges_size)];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&conc_info),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&mask_info),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&charges_info),
        ];

        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    fn reaction_descriptor_index(conc_current_buffer: usize, temperature_current_buffer: usize) -> usize {
        conc_current_buffer * 2 + temperature_current_buffer
    }

    fn update_reaction_descriptor_set(
        device: &ash::Device,
        set: vk::DescriptorSet,
        conc_buffer: vk::Buffer,
        mask_buffer: vk::Buffer,
        temperature_buffer: vk::Buffer,
        rules_buffer: vk::Buffer,
        conc_size: u64,
        mask_size: u64,
        temperature_size: u64,
        rules_size: u64,
    ) {
        let conc_info = [vk::DescriptorBufferInfo::default()
            .buffer(conc_buffer)
            .offset(0)
            .range(conc_size)];
        let mask_info = [vk::DescriptorBufferInfo::default()
            .buffer(mask_buffer)
            .offset(0)
            .range(mask_size)];
        let temp_info = [vk::DescriptorBufferInfo::default()
            .buffer(temperature_buffer)
            .offset(0)
            .range(temperature_size)];
        let rules_info = [vk::DescriptorBufferInfo::default()
            .buffer(rules_buffer)
            .offset(0)
            .range(rules_size)];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&conc_info),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&mask_info),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&temp_info),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&rules_info),
        ];

        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    fn update_thermal_descriptor_set(
        device: &ash::Device,
        set: vk::DescriptorSet,
        src_temperature_buffer: vk::Buffer,
        dst_temperature_buffer: vk::Buffer,
        mask_buffer: vk::Buffer,
        temperature_size: u64,
        mask_size: u64,
    ) {
        let src_info = [vk::DescriptorBufferInfo::default()
            .buffer(src_temperature_buffer)
            .offset(0)
            .range(temperature_size)];
        let dst_info = [vk::DescriptorBufferInfo::default()
            .buffer(dst_temperature_buffer)
            .offset(0)
            .range(temperature_size)];
        let mask_info = [vk::DescriptorBufferInfo::default()
            .buffer(mask_buffer)
            .offset(0)
            .range(mask_size)];

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
        ];

        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    fn update_leak_descriptor_set(
        device: &ash::Device,
        set: vk::DescriptorSet,
        conc_buffer: vk::Buffer,
        channels_buffer: vk::Buffer,
        conc_size: u64,
        channels_size: u64,
    ) {
        let conc_info = [vk::DescriptorBufferInfo::default()
            .buffer(conc_buffer)
            .offset(0)
            .range(conc_size)];
        let channel_info = [vk::DescriptorBufferInfo::default()
            .buffer(channels_buffer)
            .offset(0)
            .range(channels_size)];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&conc_info),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&channel_info),
        ];

        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    fn current_concentration_buffer_handle(&self) -> vk::Buffer {
        if self.current_buffer == 0 {
            self.conc_buffer_a.buffer
        } else {
            self.conc_buffer_b.buffer
        }
    }

    fn temperature_buffer_handle(rxn: &ReactionPipelineState, current_buffer: usize) -> vk::Buffer {
        if current_buffer == 0 {
            rxn.temperature_buffer_a.buffer
        } else {
            rxn.temperature_buffer_b.buffer
        }
    }

    fn record_compute_buffer_barrier(&self, cmd: vk::CommandBuffer, buffer: vk::Buffer) {
        let barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .buffer(buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE);

        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );
        }
    }

    fn record_diffusion_dispatch(
        &self,
        cmd: vk::CommandBuffer,
        descriptor_set: vk::DescriptorSet,
        push_constants: &DiffusionPushConstants,
    ) {
        unsafe {
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.diffusion_pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            self.device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(push_constants),
            );

            let workgroup_size = 256u32;
            let num_groups = (self.cell_count as u32 + workgroup_size - 1) / workgroup_size;
            self.device.cmd_dispatch(cmd, num_groups, self.species_count as u32, 1);
        }
    }

    fn record_leak_dispatch(
        &self,
        cmd: vk::CommandBuffer,
        leak: &LeakPipelineState,
        descriptor_set: vk::DescriptorSet,
        push_constants: &LeakPushConstants,
    ) {
        unsafe {
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, leak.leak_pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                leak.leak_pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            self.device.cmd_push_constants(
                cmd,
                leak.leak_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(push_constants),
            );

            let workgroup_size = 64u32;
            let num_groups = (push_constants.num_channels + workgroup_size - 1) / workgroup_size;
            self.device.cmd_dispatch(cmd, num_groups.max(1), 1, 1);
        }
    }

    fn record_charge_dispatch(
        &self,
        cmd: vk::CommandBuffer,
        descriptor_set: vk::DescriptorSet,
        push_constants: &ChargeProjectionPushConstants,
    ) {
        unsafe {
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.charge_projection_pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.charge_pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            self.device.cmd_push_constants(
                cmd,
                self.charge_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(push_constants),
            );

            let workgroup_size = 256u32;
            let num_groups = (self.cell_count as u32 + workgroup_size - 1) / workgroup_size;
            self.device.cmd_dispatch(cmd, num_groups, 1, 1);
        }
    }

    fn record_reaction_dispatch(
        &self,
        cmd: vk::CommandBuffer,
        rxn: &ReactionPipelineState,
        descriptor_set: vk::DescriptorSet,
        push_constants: &ReactionPushConstants,
    ) {
        unsafe {
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, rxn.reaction_pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                rxn.reaction_pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            self.device.cmd_push_constants(
                cmd,
                rxn.reaction_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(push_constants),
            );

            let workgroup_size = 256u32;
            let num_groups = (self.cell_count as u32 + workgroup_size - 1) / workgroup_size;
            self.device.cmd_dispatch(cmd, num_groups, 1, 1);
        }
    }

    fn record_temperature_dispatch(
        &self,
        cmd: vk::CommandBuffer,
        rxn: &ReactionPipelineState,
        descriptor_set: vk::DescriptorSet,
        push_constants: &ThermalPushConstants,
    ) {
        unsafe {
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, rxn.thermal_pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                rxn.thermal_pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            self.device.cmd_push_constants(
                cmd,
                rxn.thermal_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(push_constants),
            );

            let workgroup_size = 256u32;
            let num_groups = (self.cell_count as u32 + workgroup_size - 1) / workgroup_size;
            self.device.cmd_dispatch(cmd, num_groups, 1, 1);
        }
    }

    pub fn record_step_frame(
        &mut self,
        cmd: vk::CommandBuffer,
        substeps: u32,
        substep_dt: f32,
        diffusion_rate: f32,
        charge_correction_strength: f32,
        reaction_dt: f32,
        thermal_diffusivity: f32,
    ) {
        if substeps == 0 {
            return;
        }

        let charge_push = ChargeProjectionPushConstants {
            width: self.width,
            height: self.height,
            species_count: self.species_count as u32,
            correction_strength: charge_correction_strength,
            _pad: [0; 3],
        };
        let reaction_substep_dt = if substeps > 0 {
            reaction_dt / substeps as f32
        } else {
            reaction_dt
        };

        let mut temperature_current_buffer = self.reaction.as_ref()
            .map(|rxn| rxn.temperature_current_buffer)
            .unwrap_or(0);

        let mut input_barriers = vec![
            vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .buffer(self.current_concentration_buffer_handle())
                .offset(0)
                .size(vk::WHOLE_SIZE),
            vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .buffer(self.solid_mask.buffer)
                .offset(0)
                .size(vk::WHOLE_SIZE),
            vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .buffer(self.diffusion_coeffs.buffer)
                .offset(0)
                .size(vk::WHOLE_SIZE),
            vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .buffer(self.species_charges.buffer)
                .offset(0)
                .size(vk::WHOLE_SIZE),
        ];

        if let Some(rxn) = self.reaction.as_ref() {
            input_barriers.push(
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .buffer(Self::temperature_buffer_handle(rxn, temperature_current_buffer))
                    .offset(0)
                    .size(vk::WHOLE_SIZE),
            );
            input_barriers.push(
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .buffer(rxn.rules_buffer.buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE),
            );
        }

        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &input_barriers,
                &[],
            );
        }

        for _ in 0..substeps {
            let diffusion_push = DiffusionPushConstants {
                width: self.width,
                height: self.height,
                species_count: self.species_count as u32,
                species_index: 0xFFFFFFFF,
                diffusion_rate,
                dt: substep_dt,
                _pad: [0, 0],
            };

            let diffusion_descriptor_set = self.descriptor_sets[self.current_buffer];
            self.record_diffusion_dispatch(cmd, diffusion_descriptor_set, &diffusion_push);

            let next_buffer = 1 - self.current_buffer;
            let next_concentration_buffer = if next_buffer == 0 {
                self.conc_buffer_a.buffer
            } else {
                self.conc_buffer_b.buffer
            };
            self.record_compute_buffer_barrier(cmd, next_concentration_buffer);

            self.current_buffer = next_buffer;

            let charge_descriptor_set = self.charge_descriptor_sets[self.current_buffer];
            self.record_charge_dispatch(cmd, charge_descriptor_set, &charge_push);
            self.record_compute_buffer_barrier(cmd, self.current_concentration_buffer_handle());

            if let Some(rxn) = self.reaction.as_ref() {
                let reaction_push = ReactionPushConstants {
                    width: self.width,
                    height: self.height,
                    species_count: self.species_count as u32,
                    num_reactions: rxn.active_rule_count,
                    dt: reaction_substep_dt,
                    _pad: [0; 3],
                };
                let reaction_descriptor_set = rxn.reaction_descriptor_sets[
                    Self::reaction_descriptor_index(self.current_buffer, temperature_current_buffer)
                ];
                self.record_reaction_dispatch(cmd, rxn, reaction_descriptor_set, &reaction_push);
                self.record_compute_buffer_barrier(cmd, self.current_concentration_buffer_handle());
                self.record_compute_buffer_barrier(
                    cmd,
                    Self::temperature_buffer_handle(rxn, temperature_current_buffer),
                );

                let thermal_push = ThermalPushConstants {
                    width: self.width,
                    height: self.height,
                    thermal_diffusivity,
                    dt: substep_dt,
                    _pad: [0; 2],
                };
                let thermal_descriptor_set = rxn.thermal_descriptor_sets[temperature_current_buffer];
                self.record_temperature_dispatch(cmd, rxn, thermal_descriptor_set, &thermal_push);
                temperature_current_buffer = 1 - temperature_current_buffer;
                self.record_compute_buffer_barrier(
                    cmd,
                    Self::temperature_buffer_handle(rxn, temperature_current_buffer),
                );
            }
        }

        if let Some(leak) = self.leak.as_ref() {
            if leak.active_channel_count > 0 {
                let leak_push = LeakPushConstants {
                    width: self.width,
                    height: self.height,
                    species_count: self.species_count as u32,
                    num_channels: leak.active_channel_count,
                    dt: reaction_dt,
                    _pad: [0; 3],
                };
                let leak_descriptor_set = leak.leak_descriptor_sets[self.current_buffer];
                self.record_leak_dispatch(cmd, leak, leak_descriptor_set, &leak_push);
                self.record_compute_buffer_barrier(cmd, self.current_concentration_buffer_handle());
            }
        }

        if let Some(ref coarse) = self.coarse_grid {
            coarse.record_compute(cmd, self.current_buffer, temperature_current_buffer);
        }

        if let Some(rxn) = self.reaction.as_mut() {
            rxn.temperature_current_buffer = temperature_current_buffer;
        }

        self.record_render_barriers(cmd);
    }

    /// Execute a full simulation step as a single GPU submission.
    pub fn step_frame(
        &mut self,
        substeps: u32,
        substep_dt: f32,
        diffusion_rate: f32,
        charge_correction_strength: f32,
        reaction_dt: f32,
        thermal_diffusivity: f32,
    ) -> Result<()> {
        unsafe {
            self.device.reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(self.command_buffer, &begin_info)?;
        }

        self.record_step_frame(
            self.command_buffer,
            substeps,
            substep_dt,
            diffusion_rate,
            charge_correction_strength,
            reaction_dt,
            thermal_diffusivity,
        );

        unsafe {
            self.device.end_command_buffer(self.command_buffer)?;

            let submit_info = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.command_buffer));

            self.device.reset_fences(&[self.fence])?;
            self.device.queue_submit(self.compute_queue, &[submit_info], self.fence)?;
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
        }

        Ok(())
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
                let next_buffer = 1 - self.current_buffer;
                let temperature_current_buffer = self.reaction.as_ref()
                    .map(|rxn| rxn.temperature_current_buffer)
                    .unwrap_or(0);
                coarse.record_compute(self.command_buffer, next_buffer, temperature_current_buffer);
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

    /// Project the current concentration buffer onto a locally electroneutral state.
    pub fn step_charge_projection(&mut self, correction_strength: f32) -> Result<()> {
        let push_constants = ChargeProjectionPushConstants {
            width: self.width,
            height: self.height,
            species_count: self.species_count as u32,
            correction_strength,
            _pad: [0; 3],
        };

        let descriptor_set = self.charge_descriptor_sets[self.current_buffer];

        unsafe {
            self.device.reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(self.command_buffer, &begin_info)?;

            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.charge_projection_pipeline,
            );

            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.charge_pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            self.device.cmd_push_constants(
                self.command_buffer,
                self.charge_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push_constants),
            );

            let workgroup_size = 256u32;
            let num_groups = (self.cell_count as u32 + workgroup_size - 1) / workgroup_size;
            self.device.cmd_dispatch(self.command_buffer, num_groups, 1, 1);

            self.device.end_command_buffer(self.command_buffer)?;

            let submit_info = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.command_buffer));

            self.device.reset_fences(&[self.fence])?;
            self.device.queue_submit(self.compute_queue, &[submit_info], self.fence)?;
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
        }

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

        Self::readback_buffer_sync(
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

    /// Read back per-cell temperatures in Kelvin.
    pub fn read_temperatures(&mut self) -> Result<Vec<f32>> {
        let rxn = match &self.reaction {
            Some(r) => r,
            None => return Ok(vec![293.15; self.cell_count]),
        };

        let temperature_buffer = if rxn.temperature_current_buffer == 0 {
            rxn.temperature_buffer_a.buffer
        } else {
            rxn.temperature_buffer_b.buffer
        };

        let size = (self.cell_count * std::mem::size_of::<f32>()) as u64;

        Self::readback_buffer_sync(
            &self.device,
            self.command_pool,
            self.compute_queue,
            self.fence,
            temperature_buffer,
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

    pub fn current_temperature_buffer(&self) -> vk::Buffer {
        self.reaction.as_ref()
            .map(|rxn| Self::temperature_buffer_handle(rxn, rxn.temperature_current_buffer))
            .unwrap_or(vk::Buffer::null())
    }

    pub fn render_buffers(&self) -> GpuRenderBuffers {
        GpuRenderBuffers {
            concentration: self.current_concentration_buffer(),
            solid_mask: self.solid_mask.buffer,
            material_ids: self.material_ids.buffer,
            temperature: self.current_temperature_buffer(),
        }
    }

    pub fn uses_shared_vulkan_context(&self) -> bool {
        !self.owns_vulkan_context
    }

    pub fn record_render_barriers(&self, cmd: vk::CommandBuffer) {
        let mut barriers = vec![
            vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .buffer(self.current_concentration_buffer())
                .offset(0)
                .size(vk::WHOLE_SIZE),
            vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .buffer(self.solid_mask.buffer)
                .offset(0)
                .size(vk::WHOLE_SIZE),
            vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .buffer(self.material_ids.buffer)
                .offset(0)
                .size(vk::WHOLE_SIZE),
        ];

        let temperature = self.current_temperature_buffer();
        if temperature != vk::Buffer::null() {
            barriers.push(
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .buffer(temperature)
                    .offset(0)
                    .size(vk::WHOLE_SIZE),
            );
        }

        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &barriers,
                &[],
            );
        }
    }

    /// Initialize the reaction pipeline (lazy, called when first rules arrive).
    ///
    /// Creates: temperature buffer, rules buffer, descriptor sets, pipeline.
    pub fn init_reaction_pipeline(&mut self, initial_temperatures: &[f32]) -> Result<()> {
        if self.reaction.is_some() {
            return Ok(()); // Already initialized
        }

        let temp_buffer_size = (self.cell_count * std::mem::size_of::<f32>()) as u64;
        let rules_buffer_size = (MAX_REACTION_RULES * std::mem::size_of::<GpuReactionRule>()) as u64;
        let conc_buffer_size = (self.species_count * self.cell_count * std::mem::size_of::<f32>()) as u64;
        let mask_buffer_size = (self.cell_count * std::mem::size_of::<u32>()) as u64;

        let mut alloc = self.allocator.as_ref().unwrap().lock();

        let temperature_buffer_a = GpuBuffer::new(
            &self.device,
            &mut alloc,
            temp_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::GpuOnly,
            "temperature_a",
        )?;

        let temperature_buffer_b = GpuBuffer::new(
            &self.device,
            &mut alloc,
            temp_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::GpuOnly,
            "temperature_b",
        )?;

        // CPU-visible reaction rules buffer
        let rules_buffer = GpuBuffer::new(
            &self.device,
            &mut alloc,
            rules_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            "reaction_rules",
        )?;

        drop(alloc);

        // Upload initial temperatures via staging
        self.staging_buffer.write(initial_temperatures)?;
        Self::copy_buffer_sync(
            &self.device,
            self.command_pool,
            self.compute_queue,
            self.fence,
            self.staging_buffer.buffer,
            temperature_buffer_a.buffer,
            temp_buffer_size,
            Some(self.command_buffer),
        )?;
        Self::copy_buffer_sync(
            &self.device,
            self.command_pool,
            self.compute_queue,
            self.fence,
            self.staging_buffer.buffer,
            temperature_buffer_b.buffer,
            temp_buffer_size,
            Some(self.command_buffer),
        )?;

        let compiler = shaderc::Compiler::new().context("Failed to create shader compiler")?;
        let mut options = shaderc::CompileOptions::new().context("Failed to create compile options")?;
        options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_2 as u32);
        options.set_optimization_level(shaderc::OptimizationLevel::Performance);

        let reaction_spirv = compiler.compile_into_spirv(
            include_str!("../shaders/reaction.comp"),
            shaderc::ShaderKind::Compute,
            "reaction.comp",
            "main",
            Some(&options),
        ).context("Failed to compile reaction shader")?;

        let thermal_spirv = compiler.compile_into_spirv(
            include_str!("../shaders/thermal_diffusion.comp"),
            shaderc::ShaderKind::Compute,
            "thermal_diffusion.comp",
            "main",
            Some(&options),
        ).context("Failed to compile thermal diffusion shader")?;

        let reaction_shader_module_info = vk::ShaderModuleCreateInfo::default()
            .code(reaction_spirv.as_binary());
        let reaction_shader_module = unsafe {
            self.device.create_shader_module(&reaction_shader_module_info, None)?
        };

        let thermal_shader_module_info = vk::ShaderModuleCreateInfo::default()
            .code(thermal_spirv.as_binary());
        let thermal_shader_module = unsafe {
            self.device.create_shader_module(&thermal_shader_module_info, None)?
        };

        let reaction_bindings = [
            // 0: Concentrations (read-write)
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // 1: Solid mask (read)
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // 2: Temperatures (read-write)
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // 3: Reaction rules (read)
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let reaction_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&reaction_bindings);
        let reaction_descriptor_set_layout = unsafe {
            self.device.create_descriptor_set_layout(&reaction_layout_info, None)?
        };

        let reaction_push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<ReactionPushConstants>() as u32);

        let reaction_pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&reaction_descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&reaction_push_constant_range));
        let reaction_pipeline_layout = unsafe {
            self.device.create_pipeline_layout(&reaction_pipeline_layout_info, None)?
        };

        let entry_name = c"main";
        let reaction_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(reaction_shader_module)
            .name(entry_name);

        let reaction_pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(reaction_stage_info)
            .layout(reaction_pipeline_layout);

        let reaction_pipeline = unsafe {
            self.device.create_compute_pipelines(vk::PipelineCache::null(), &[reaction_pipeline_info], None)
                .map_err(|e| anyhow::anyhow!("Failed to create reaction pipeline: {:?}", e.1))?[0]
        };

        let thermal_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let thermal_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&thermal_bindings);
        let thermal_descriptor_set_layout = unsafe {
            self.device.create_descriptor_set_layout(&thermal_layout_info, None)?
        };

        let thermal_push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<ThermalPushConstants>() as u32);

        let thermal_pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&thermal_descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&thermal_push_constant_range));
        let thermal_pipeline_layout = unsafe {
            self.device.create_pipeline_layout(&thermal_pipeline_layout_info, None)?
        };

        let thermal_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(thermal_shader_module)
            .name(entry_name);

        let thermal_pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(thermal_stage_info)
            .layout(thermal_pipeline_layout);

        let thermal_pipeline = unsafe {
            self.device.create_compute_pipelines(vk::PipelineCache::null(), &[thermal_pipeline_info], None)
                .map_err(|e| anyhow::anyhow!("Failed to create thermal pipeline: {:?}", e.1))?[0]
        };

        unsafe {
            self.device.destroy_shader_module(reaction_shader_module, None);
            self.device.destroy_shader_module(thermal_shader_module, None);
        };

        let reaction_pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(16),
        ];
        let reaction_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(4)
            .pool_sizes(&reaction_pool_sizes);
        let reaction_descriptor_pool = unsafe {
            self.device.create_descriptor_pool(&reaction_pool_info, None)?
        };

        let reaction_layouts = [
            reaction_descriptor_set_layout,
            reaction_descriptor_set_layout,
            reaction_descriptor_set_layout,
            reaction_descriptor_set_layout,
        ];
        let reaction_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(reaction_descriptor_pool)
            .set_layouts(&reaction_layouts);
        let reaction_descriptor_sets = unsafe {
            self.device.allocate_descriptor_sets(&reaction_alloc_info)?
        };

        let conc_buffers = [self.conc_buffer_a.buffer, self.conc_buffer_b.buffer];
        let temp_buffers = [temperature_buffer_a.buffer, temperature_buffer_b.buffer];
        for (conc_index, &conc_buf) in conc_buffers.iter().enumerate() {
            for (temp_index, &temp_buf) in temp_buffers.iter().enumerate() {
                let set_index = Self::reaction_descriptor_index(conc_index, temp_index);
                Self::update_reaction_descriptor_set(
                    &self.device,
                    reaction_descriptor_sets[set_index],
                    conc_buf,
                    self.solid_mask.buffer,
                    temp_buf,
                    rules_buffer.buffer,
                    conc_buffer_size,
                    mask_buffer_size,
                    temp_buffer_size,
                    rules_buffer_size,
                );
            }
        }
        let thermal_pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(6),
        ];
        let thermal_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(2)
            .pool_sizes(&thermal_pool_sizes);
        let thermal_descriptor_pool = unsafe {
            self.device.create_descriptor_pool(&thermal_pool_info, None)?
        };

        let thermal_layouts = [thermal_descriptor_set_layout, thermal_descriptor_set_layout];
        let thermal_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(thermal_descriptor_pool)
            .set_layouts(&thermal_layouts);
        let thermal_descriptor_sets = unsafe {
            self.device.allocate_descriptor_sets(&thermal_alloc_info)?
        };

        Self::update_thermal_descriptor_set(
            &self.device,
            thermal_descriptor_sets[0],
            temperature_buffer_a.buffer,
            temperature_buffer_b.buffer,
            self.solid_mask.buffer,
            temp_buffer_size,
            mask_buffer_size,
        );
        Self::update_thermal_descriptor_set(
            &self.device,
            thermal_descriptor_sets[1],
            temperature_buffer_b.buffer,
            temperature_buffer_a.buffer,
            self.solid_mask.buffer,
            temp_buffer_size,
            mask_buffer_size,
        );

        self.reaction = Some(ReactionPipelineState {
            reaction_descriptor_set_layout,
            reaction_pipeline_layout,
            reaction_pipeline,
            reaction_descriptor_pool,
            reaction_descriptor_sets,
            temperature_buffer_a,
            temperature_buffer_b,
            temperature_current_buffer: 0,
            thermal_descriptor_set_layout,
            thermal_pipeline_layout,
            thermal_pipeline,
            thermal_descriptor_pool,
            thermal_descriptor_sets,
            rules_buffer,
            active_rule_count: 0,
            rule_set_hash: 0,
        });

        self.coarse_grid = CoarseGrid::new(
            self.device.clone(),
            self.compute_queue,
            self.compute_queue_family,
            self.allocator.as_ref().expect("allocator should exist").clone(),
            self.width,
            self.height,
            self.species_count,
            8,
            self.conc_buffer_a.buffer,
            self.conc_buffer_b.buffer,
            self.reaction.as_ref().expect("reaction initialized").temperature_buffer_a.buffer,
            self.reaction.as_ref().expect("reaction initialized").temperature_buffer_b.buffer,
            self.solid_mask.buffer,
            (self.species_count * self.cell_count * std::mem::size_of::<f32>()) as u64,
            (self.cell_count * std::mem::size_of::<f32>()) as u64,
            (self.cell_count * std::mem::size_of::<u32>()) as u64,
        ).ok();

        log::info!("Reaction pipeline initialized");
        Ok(())
    }

    /// Upload reaction rules to the GPU.
    ///
    /// Returns true if the rules changed (hash differs) and were uploaded.
    /// Returns false if the rules are the same as before (no upload needed).
    pub fn upload_reaction_rules(&mut self, rules: &[GpuReactionRule]) -> Result<bool> {
        if rules.len() > MAX_REACTION_RULES {
            bail!("Too many reaction rules: {} > {}", rules.len(), MAX_REACTION_RULES);
        }

        let rxn = self.reaction.as_mut()
            .context("Reaction pipeline not initialized")?;

        // Compute hash of the rule set to avoid redundant uploads
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        rules.len().hash(&mut hasher);
        for r in rules {
            r.reactant_a_index.hash(&mut hasher);
            r.reactant_b_index.hash(&mut hasher);
            r.product_a_index.hash(&mut hasher);
            r.product_b_index.hash(&mut hasher);
            r.effective_rate_bits.hash(&mut hasher);
            r.enthalpy_delta_bits.hash(&mut hasher);
            r.entropy_delta_bits.hash(&mut hasher);
        }
        let new_hash = hasher.finish();

        if new_hash == rxn.rule_set_hash && rxn.active_rule_count == rules.len() as u32 {
            return Ok(false); // Same rules, skip upload
        }

        let rules_size = (rules.len() * std::mem::size_of::<GpuReactionRule>()) as u64;
        if rules_size > 0 {
            self.staging_buffer.write(rules)?;
            Self::copy_buffer_sync(
                &self.device,
                self.command_pool,
                self.compute_queue,
                self.fence,
                self.staging_buffer.buffer,
                rxn.rules_buffer.buffer,
                rules_size,
                Some(self.command_buffer),
            )?;
        }

        rxn.active_rule_count = rules.len() as u32;
        rxn.rule_set_hash = new_hash;

        log::debug!("Uploaded {} reaction rules (hash={:#x})", rules.len(), new_hash);
        Ok(true)
    }

    /// Run the reaction compute pass on the current buffer (in-place).
    ///
    /// This should be called after diffusion but before the next frame.
    /// The reaction shader reads and writes the current concentration buffer
    /// in place (not ping-pong).
    pub fn step_reactions(&mut self, dt: f32) -> Result<()> {
        let rxn = match &self.reaction {
            Some(r) => r,
            None => return Ok(()),
        };

        let push = ReactionPushConstants {
            width: self.width,
            height: self.height,
            species_count: self.species_count as u32,
            num_reactions: rxn.active_rule_count,
            dt,
            _pad: [0; 3],
        };

        // The reaction shader reads/writes the CURRENT buffer in-place
        let descriptor_set = rxn.reaction_descriptor_sets[
            Self::reaction_descriptor_index(self.current_buffer, rxn.temperature_current_buffer)
        ];

        unsafe {
            self.device.reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(self.command_buffer, &begin_info)?;

            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                rxn.reaction_pipeline,
            );

            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                rxn.reaction_pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            self.device.cmd_push_constants(
                self.command_buffer,
                rxn.reaction_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push),
            );

            let workgroup_size = 256u32;
            let num_groups = (self.cell_count as u32 + workgroup_size - 1) / workgroup_size;
            self.device.cmd_dispatch(self.command_buffer, num_groups, 1, 1);

            self.device.end_command_buffer(self.command_buffer)?;

            let submit_info = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.command_buffer));

            self.device.reset_fences(&[self.fence])?;
            self.device.queue_submit(self.compute_queue, &[submit_info], self.fence)?;
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
        }

        Ok(())
    }

    pub fn init_leak_pipeline(
        &mut self,
        channels: &[LeakChannel],
        species_registry: &SpeciesRegistry,
        solid_mask: &[u32],
    ) -> Result<()> {
        if channels.is_empty() {
            return Ok(());
        }
        if channels.len() > MAX_LEAK_CHANNELS {
            bail!("Too many leak channels: {} > {}", channels.len(), MAX_LEAK_CHANNELS);
        }
        if self.leak.is_some() {
            return Ok(());
        }

        let packed_channels: Vec<GpuLeakChannel> = channels.iter().filter_map(|channel| {
            let species_index = species_registry.index_of(channel.species)? as u32;
            let ((sink_x, sink_y), (source_x, source_y)) =
                channel.resolve_endpoints(self.width, self.height, solid_mask)?;
            Some(GpuLeakChannel {
                species_index,
                sink_x,
                sink_y,
                source_x,
                source_y,
                rate: channel.rate,
                rotation_byte: channel.rotation as u8 as u32,
                _pad: 0,
            })
        }).collect();

        let channels_buffer_size = (MAX_LEAK_CHANNELS * std::mem::size_of::<GpuLeakChannel>()) as u64;
        let conc_buffer_size = (self.species_count * self.cell_count * std::mem::size_of::<f32>()) as u64;

        let mut alloc = self.allocator.as_ref().unwrap().lock();
        let channels_buffer = GpuBuffer::new(
            &self.device,
            &mut alloc,
            channels_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            "leak_channels",
        )?;
        drop(alloc);

        self.staging_buffer.write(&packed_channels)?;
        Self::copy_buffer_sync(
            &self.device,
            self.command_pool,
            self.compute_queue,
            self.fence,
            self.staging_buffer.buffer,
            channels_buffer.buffer,
            (packed_channels.len() * std::mem::size_of::<GpuLeakChannel>()) as u64,
            Some(self.command_buffer),
        )?;

        let compiler = shaderc::Compiler::new().context("Failed to create shader compiler")?;
        let mut options = shaderc::CompileOptions::new().context("Failed to create compile options")?;
        options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_2 as u32);
        options.set_optimization_level(shaderc::OptimizationLevel::Performance);

        let leak_spirv = compiler.compile_into_spirv(
            include_str!("../shaders/leak.comp"),
            shaderc::ShaderKind::Compute,
            "leak.comp",
            "main",
            Some(&options),
        ).context("Failed to compile leak shader")?;

        let leak_shader_module_info = vk::ShaderModuleCreateInfo::default()
            .code(leak_spirv.as_binary());
        let leak_shader_module = unsafe {
            self.device.create_shader_module(&leak_shader_module_info, None)?
        };

        let leak_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let leak_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&leak_bindings);
        let leak_descriptor_set_layout = unsafe {
            self.device.create_descriptor_set_layout(&leak_layout_info, None)?
        };

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<LeakPushConstants>() as u32);
        let leak_pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&leak_descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        let leak_pipeline_layout = unsafe {
            self.device.create_pipeline_layout(&leak_pipeline_layout_info, None)?
        };

        let entry_name = c"main";
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(leak_shader_module)
            .name(entry_name);
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(leak_pipeline_layout);
        let leak_pipeline = unsafe {
            self.device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|e| anyhow::anyhow!("Failed to create leak pipeline: {:?}", e.1))?[0]
        };

        unsafe {
            self.device.destroy_shader_module(leak_shader_module, None);
        }

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(4)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(2)
            .pool_sizes(&pool_sizes);
        let leak_descriptor_pool = unsafe {
            self.device.create_descriptor_pool(&pool_info, None)?
        };

        let layouts = [leak_descriptor_set_layout, leak_descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(leak_descriptor_pool)
            .set_layouts(&layouts);
        let leak_descriptor_sets = unsafe {
            self.device.allocate_descriptor_sets(&alloc_info)?
        };

        Self::update_leak_descriptor_set(
            &self.device,
            leak_descriptor_sets[0],
            self.conc_buffer_a.buffer,
            channels_buffer.buffer,
            conc_buffer_size,
            channels_buffer_size,
        );
        Self::update_leak_descriptor_set(
            &self.device,
            leak_descriptor_sets[1],
            self.conc_buffer_b.buffer,
            channels_buffer.buffer,
            conc_buffer_size,
            channels_buffer_size,
        );

        self.leak = Some(LeakPipelineState {
            leak_descriptor_set_layout,
            leak_pipeline_layout,
            leak_pipeline,
            leak_descriptor_pool,
            leak_descriptor_sets,
            channels_buffer,
            active_channel_count: packed_channels.len() as u32,
        });

        Ok(())
    }

    /// Run one thermal diffusion step on the current temperature buffer.
    pub fn step_temperature(&mut self, dt: f32, thermal_diffusivity: f32) -> Result<()> {
        let rxn = match &mut self.reaction {
            Some(r) => r,
            None => return Ok(()),
        };

        let push_constants = ThermalPushConstants {
            width: self.width,
            height: self.height,
            thermal_diffusivity,
            dt,
            _pad: [0; 2],
        };
        let descriptor_set = rxn.thermal_descriptor_sets[rxn.temperature_current_buffer];
        let pipeline = rxn.thermal_pipeline;
        let pipeline_layout = rxn.thermal_pipeline_layout;

        unsafe {
            self.device.reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(self.command_buffer, &begin_info)?;

            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );

            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            self.device.cmd_push_constants(
                self.command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push_constants),
            );

            let workgroup_size = 256u32;
            let num_groups = (self.cell_count as u32 + workgroup_size - 1) / workgroup_size;
            self.device.cmd_dispatch(self.command_buffer, num_groups, 1, 1);

            self.device.end_command_buffer(self.command_buffer)?;

            let submit_info = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.command_buffer));

            self.device.reset_fences(&[self.fence])?;
            self.device.queue_submit(self.compute_queue, &[submit_info], self.fence)?;
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
        }

        rxn.temperature_current_buffer = 1 - rxn.temperature_current_buffer;
        Ok(())
    }
}

impl Drop for GpuSimulation {
    fn drop(&mut self) {
        log::info!("GpuSimulation::drop - starting");
        unsafe {
            self.device.device_wait_idle().ok();
            log::info!("GpuSimulation::drop - device idle");
            
            // Drop coarse_grid FIRST - it holds cloned device handle and references our buffers
            drop(self.coarse_grid.take());
            log::info!("GpuSimulation::drop - coarse_grid dropped");

            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);

            // Clean up reaction pipeline if present
            if let Some(mut rxn) = self.reaction.take() {
                self.device.destroy_descriptor_pool(rxn.reaction_descriptor_pool, None);
                self.device.destroy_pipeline(rxn.reaction_pipeline, None);
                self.device.destroy_pipeline_layout(rxn.reaction_pipeline_layout, None);
                self.device.destroy_descriptor_set_layout(rxn.reaction_descriptor_set_layout, None);
                self.device.destroy_descriptor_pool(rxn.thermal_descriptor_pool, None);
                self.device.destroy_pipeline(rxn.thermal_pipeline, None);
                self.device.destroy_pipeline_layout(rxn.thermal_pipeline_layout, None);
                self.device.destroy_descriptor_set_layout(rxn.thermal_descriptor_set_layout, None);

                let mut alloc = self.allocator.as_ref().unwrap().lock();
                self.device.destroy_buffer(rxn.temperature_buffer_a.buffer, None);
                alloc.free(std::mem::take(&mut rxn.temperature_buffer_a.allocation)).ok();
                self.device.destroy_buffer(rxn.temperature_buffer_b.buffer, None);
                alloc.free(std::mem::take(&mut rxn.temperature_buffer_b.allocation)).ok();
                self.device.destroy_buffer(rxn.rules_buffer.buffer, None);
                alloc.free(std::mem::take(&mut rxn.rules_buffer.allocation)).ok();
                drop(alloc);
            }
            log::info!("GpuSimulation::drop - reaction pipeline dropped");

            if let Some(mut leak) = self.leak.take() {
                self.device.destroy_descriptor_pool(leak.leak_descriptor_pool, None);
                self.device.destroy_pipeline(leak.leak_pipeline, None);
                self.device.destroy_pipeline_layout(leak.leak_pipeline_layout, None);
                self.device.destroy_descriptor_set_layout(leak.leak_descriptor_set_layout, None);

                let mut alloc = self.allocator.as_ref().unwrap().lock();
                self.device.destroy_buffer(leak.channels_buffer.buffer, None);
                alloc.free(std::mem::take(&mut leak.channels_buffer.allocation)).ok();
                drop(alloc);
            }
            log::info!("GpuSimulation::drop - leak pipeline dropped");

            self.device.destroy_descriptor_pool(self.charge_descriptor_pool, None);
            self.device.destroy_pipeline(self.charge_projection_pipeline, None);
            self.device.destroy_pipeline_layout(self.charge_pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.charge_descriptor_set_layout, None);

            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_pipeline(self.diffusion_pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            // Buffers and allocations
            let mut alloc = self.allocator.as_ref().unwrap().lock();
            
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

            self.device.destroy_buffer(self.species_charges.buffer, None);
            alloc.free(std::mem::take(&mut self.species_charges.allocation)).ok();
            
            self.device.destroy_buffer(self.staging_buffer.buffer, None);
            alloc.free(std::mem::take(&mut self.staging_buffer.allocation)).ok();
            
            self.device.destroy_buffer(self.readback_buffer.buffer, None);
            alloc.free(std::mem::take(&mut self.readback_buffer.allocation)).ok();
            
            drop(alloc);
            log::info!("GpuSimulation::drop - buffers freed");

            // Drop the allocator BEFORE destroying the device.
            // gpu_allocator::Allocator::drop calls vkFreeMemory, which requires a live device.
            drop(self.allocator.take());
            log::info!("GpuSimulation::drop - allocator dropped, device handle: {:?}", self.device.handle());

            if self.owns_vulkan_context {
                self.device.destroy_device(None);
                self.instance.destroy_instance(None);
                log::info!("GpuSimulation::drop - owned Vulkan context destroyed");
            } else {
                log::info!("GpuSimulation::drop - shared Vulkan context left alive");
            }
            log::info!("GpuSimulation::drop - done");
        }
    }
}

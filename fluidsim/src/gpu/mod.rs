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
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use parking_lot::Mutex;
use std::sync::Arc;

use crate::coarse::{CoarseGrid, CoarseGridCreateInfo, CoarseGridSourceBuffers};
use crate::enzyme::EnzymeEntity;
use crate::leak::LeakChannel;
use crate::species::SpeciesRegistry;

mod buffer;
mod dispatch;
mod drop_impl;
mod pipelines;
mod setup;

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

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DiffusionPushConstants {
    pub width: u32,
    pub height: u32,
    pub species_count: u32,
    pub species_index: u32,
    pub diffusion_rate: f32,
    pub dt: f32,
    pub _pad: [u32; 2],
}

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

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct EnzymePushConstants {
    pub width: u32,
    pub height: u32,
    pub species_count: u32,
    pub num_enzymes: u32,
    pub dt: f32,
    pub base_turnover_rate: f32,
    pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ThermalPushConstants {
    pub width: u32,
    pub height: u32,
    pub thermal_diffusivity: f32,
    pub dt: f32,
    pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ChargeProjectionPushConstants {
    pub width: u32,
    pub height: u32,
    pub species_count: u32,
    pub correction_strength: f32,
    pub _pad: [u32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuReactionRule {
    pub reactant_a_index: u32,
    pub reactant_b_index: u32,
    pub product_a_index: u32,
    pub product_b_index: u32,
    pub catalyst_index: u32,
    pub kinetic_model: u32,
    pub effective_rate_bits: u32,
    pub km_reactant_a_bits: u32,
    pub km_reactant_b_bits: u32,
    pub enthalpy_delta_bits: u32,
    pub entropy_delta_bits: u32,
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

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuEnzymeEntity {
    pub active_site_x: i32,
    pub active_site_y: i32,
    pub glucose_index: u32,
    pub atp_index: u32,
    pub g6p_index: u32,
    pub adp_index: u32,
    pub catalytic_scale: f32,
    pub thermal_bias: f32,
    pub km_glucose: f32,
    pub km_atp: f32,
}

pub struct GpuReactionRuleConfig {
    pub reactant_a_index: u32,
    pub reactant_b_index: Option<u32>,
    pub product_a_index: Option<u32>,
    pub product_b_index: Option<u32>,
    pub catalyst_index: Option<u32>,
    pub kinetic_model: u32,
    pub rate: f32,
    pub km_reactant_a: f32,
    pub km_reactant_b: f32,
    pub enthalpy: f32,
    pub entropy: f32,
}

pub struct GpuSimulationCreateInfo<'a> {
    pub width: u32,
    pub height: u32,
    pub species_count: usize,
    pub initial_concentrations: &'a [Vec<f32>],
    pub solid_mask_data: &'a [u32],
    pub material_ids_data: &'a [u32],
    pub diffusion_coeffs_data: &'a [f32],
    pub species_charges_data: &'a [i32],
}

#[derive(Debug, Clone, Copy)]
pub struct StepFrameParams {
    pub substeps: u32,
    pub substep_dt: f32,
    pub diffusion_rate: f32,
    pub charge_correction_strength: f32,
    pub reaction_dt: f32,
    pub thermal_diffusivity: f32,
}

const MAX_REACTION_RULES: usize = 16;
const MAX_LEAK_CHANNELS: usize = 32;
const MAX_ENZYME_ENTITIES: usize = 32;

pub struct ReactionPipelineState {
    pub reaction_descriptor_set_layout: vk::DescriptorSetLayout,
    pub reaction_pipeline_layout: vk::PipelineLayout,
    pub reaction_pipeline: vk::Pipeline,
    pub reaction_descriptor_pool: vk::DescriptorPool,
    pub reaction_descriptor_sets: Vec<vk::DescriptorSet>,
    pub temperature_buffer_a: GpuBuffer,
    pub temperature_buffer_b: GpuBuffer,
    pub temperature_current_buffer: usize,
    pub thermal_descriptor_set_layout: vk::DescriptorSetLayout,
    pub thermal_pipeline_layout: vk::PipelineLayout,
    pub thermal_pipeline: vk::Pipeline,
    pub thermal_descriptor_pool: vk::DescriptorPool,
    pub thermal_descriptor_sets: Vec<vk::DescriptorSet>,
    pub rules_buffer: GpuBuffer,
    pub active_rule_count: u32,
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

pub struct EnzymePipelineState {
    pub enzyme_descriptor_set_layout: vk::DescriptorSetLayout,
    pub enzyme_pipeline_layout: vk::PipelineLayout,
    pub enzyme_pipeline: vk::Pipeline,
    pub enzyme_descriptor_pool: vk::DescriptorPool,
    pub enzyme_descriptor_sets: Vec<vk::DescriptorSet>,
    pub enzymes_buffer: GpuBuffer,
    pub active_enzyme_count: u32,
}

pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
    pub size: u64,
}

pub struct GpuSimulation {
    pub entry: Option<ash::Entry>,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub compute_queue: vk::Queue,
    pub compute_queue_family: u32,
    pub allocator: Option<Arc<Mutex<Allocator>>>,
    pub owns_vulkan_context: bool,
    pub width: u32,
    pub height: u32,
    pub cell_count: usize,
    pub species_count: usize,
    pub conc_buffer_a: GpuBuffer,
    pub conc_buffer_b: GpuBuffer,
    pub current_buffer: usize,
    pub solid_mask: GpuBuffer,
    pub material_ids: GpuBuffer,
    pub diffusion_coeffs: GpuBuffer,
    pub species_charges: GpuBuffer,
    pub staging_buffer: GpuBuffer,
    pub readback_buffer: GpuBuffer,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub diffusion_pipeline: vk::Pipeline,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub charge_descriptor_set_layout: vk::DescriptorSetLayout,
    pub charge_pipeline_layout: vk::PipelineLayout,
    pub charge_projection_pipeline: vk::Pipeline,
    pub charge_descriptor_pool: vk::DescriptorPool,
    pub charge_descriptor_sets: Vec<vk::DescriptorSet>,
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub fence: vk::Fence,
    pub coarse_grid: Option<CoarseGrid>,
    pub reaction: Option<ReactionPipelineState>,
    pub leak: Option<LeakPipelineState>,
    pub enzyme: Option<EnzymePipelineState>,
}

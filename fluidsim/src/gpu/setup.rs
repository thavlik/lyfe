use super::*;

impl GpuReactionRule {
    pub const NONE: u32 = u32::MAX;

    pub fn new(config: GpuReactionRuleConfig) -> Self {
        Self {
            reactant_a_index: config.reactant_a_index,
            reactant_b_index: config.reactant_b_index.unwrap_or(Self::NONE),
            product_a_index: config.product_a_index.unwrap_or(Self::NONE),
            product_b_index: config.product_b_index.unwrap_or(Self::NONE),
            catalyst_index: config.catalyst_index.unwrap_or(Self::NONE),
            kinetic_model: config.kinetic_model,
            effective_rate_bits: config.rate.to_bits(),
            km_reactant_a_bits: config.km_reactant_a.to_bits(),
            km_reactant_b_bits: config.km_reactant_b.to_bits(),
            enthalpy_delta_bits: config.enthalpy.to_bits(),
            entropy_delta_bits: config.entropy.to_bits(),
        }
    }
}

impl GpuSimulation {
    pub fn new(config: GpuSimulationCreateInfo<'_>) -> Result<Self> {
        let (entry, context) = Self::create_owned_context()?;
        Self::new_from_context(Some(entry), context, true, config)
    }

    pub fn new_with_shared_context(
        context: SharedGpuContext,
        config: GpuSimulationCreateInfo<'_>,
    ) -> Result<Self> {
        Self::new_from_context(None, context, false, config)
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

        let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&instance_info, None)? };

        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        if physical_devices.is_empty() {
            bail!("No Vulkan-capable GPU found");
        }

        let physical_device = physical_devices
            .into_iter()
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

        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let compute_queue_family = queue_families
            .iter()
            .enumerate()
            .find_map(|(index, qf)| {
                (qf.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && qf.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .then_some(index as u32)
            })
            .or_else(|| {
                queue_families.iter().enumerate().find_map(|(index, qf)| {
                    qf.queue_flags
                        .contains(vk::QueueFlags::COMPUTE)
                        .then_some(index as u32)
                })
            })
            .context("No compute-capable queue family found")?;

        let queue_priority = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(compute_queue_family)
            .queue_priorities(&queue_priority);

        let device_info =
            vk::DeviceCreateInfo::default().queue_create_infos(std::slice::from_ref(&queue_info));

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
        config: GpuSimulationCreateInfo<'_>,
    ) -> Result<Self> {
        let GpuSimulationCreateInfo {
            width,
            height,
            species_count,
            initial_concentrations,
            solid_mask_data,
            material_ids_data,
            diffusion_coeffs_data,
            species_charges_data,
        } = config;
        let cell_count = (width * height) as usize;

        if initial_concentrations.len() != species_count {
            bail!(
                "Expected {} species, got {}",
                species_count,
                initial_concentrations.len()
            );
        }
        for (index, conc) in initial_concentrations.iter().enumerate() {
            if conc.len() != cell_count {
                bail!(
                    "Species {} has {} cells, expected {}",
                    index,
                    conc.len(),
                    cell_count
                );
            }
        }
        if solid_mask_data.len() != cell_count {
            bail!(
                "Solid mask has {} cells, expected {}",
                solid_mask_data.len(),
                cell_count
            );
        }
        if material_ids_data.len() != cell_count {
            bail!(
                "Material IDs has {} cells, expected {}",
                material_ids_data.len(),
                cell_count
            );
        }
        if diffusion_coeffs_data.len() != species_count {
            bail!(
                "Diffusion coeffs has {} entries, expected {}",
                diffusion_coeffs_data.len(),
                species_count
            );
        }
        if species_charges_data.len() != species_count {
            bail!(
                "Species charges has {} entries, expected {}",
                species_charges_data.len(),
                species_count
            );
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
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            "conc_buffer_a",
        )?;
        let conc_buffer_b = GpuBuffer::new(
            &device,
            &mut alloc,
            conc_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
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

        let staging_size = conc_buffer_size
            .max(mask_buffer_size)
            .max(charges_buffer_size);
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

        let mut flat_conc = Vec::with_capacity(species_count * cell_count);
        for species_conc in initial_concentrations {
            flat_conc.extend_from_slice(species_conc);
        }

        staging_buffer.write(&flat_conc)?;
        Self::copy_buffer_sync(
            &device,
            command_pool,
            compute_queue,
            fence,
            staging_buffer.buffer,
            conc_buffer_a.buffer,
            conc_buffer_size,
            Some(command_buffer),
        )?;

        staging_buffer.write(solid_mask_data)?;
        Self::copy_buffer_sync(
            &device,
            command_pool,
            compute_queue,
            fence,
            staging_buffer.buffer,
            solid_mask.buffer,
            mask_buffer_size,
            Some(command_buffer),
        )?;

        staging_buffer.write(material_ids_data)?;
        Self::copy_buffer_sync(
            &device,
            command_pool,
            compute_queue,
            fence,
            staging_buffer.buffer,
            material_ids.buffer,
            mask_buffer_size,
            Some(command_buffer),
        )?;

        staging_buffer.write(diffusion_coeffs_data)?;
        Self::copy_buffer_sync(
            &device,
            command_pool,
            compute_queue,
            fence,
            staging_buffer.buffer,
            diffusion_coeffs.buffer,
            coeffs_buffer_size,
            Some(command_buffer),
        )?;

        staging_buffer.write(species_charges_data)?;
        Self::copy_buffer_sync(
            &device,
            command_pool,
            compute_queue,
            fence,
            staging_buffer.buffer,
            species_charges.buffer,
            charges_buffer_size,
            Some(command_buffer),
        )?;

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

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&layout_info, None)? };

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<DiffusionPushConstants>() as u32);
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

        let shader_source = include_str!("../../shaders/diffusion.comp");
        let compiler = shaderc::Compiler::new().context("Failed to create shader compiler")?;
        let mut options =
            shaderc::CompileOptions::new().context("Failed to create compile options")?;
        options.set_target_env(
            shaderc::TargetEnv::Vulkan,
            shaderc::EnvVersion::Vulkan1_2 as u32,
        );
        options.set_optimization_level(shaderc::OptimizationLevel::Performance);

        let spirv = compiler
            .compile_into_spirv(
                shader_source,
                shaderc::ShaderKind::Compute,
                "diffusion.comp",
                "main",
                Some(&options),
            )
            .context("Failed to compile diffusion shader")?;

        let shader_module_info = vk::ShaderModuleCreateInfo::default().code(spirv.as_binary());
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
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|error| {
                    anyhow::anyhow!("Failed to create compute pipeline: {:?}", error.1)
                })?[0]
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

        let charge_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&charge_bindings);
        let charge_descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&charge_layout_info, None)? };

        let charge_push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<ChargeProjectionPushConstants>() as u32);
        let charge_pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&charge_descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&charge_push_constant_range));
        let charge_pipeline_layout =
            unsafe { device.create_pipeline_layout(&charge_pipeline_layout_info, None)? };

        let charge_spirv = compiler
            .compile_into_spirv(
                include_str!("../../shaders/charge_projection.comp"),
                shaderc::ShaderKind::Compute,
                "charge_projection.comp",
                "main",
                Some(&options),
            )
            .context("Failed to compile charge projection shader")?;

        let charge_shader_module_info =
            vk::ShaderModuleCreateInfo::default().code(charge_spirv.as_binary());
        let charge_shader_module =
            unsafe { device.create_shader_module(&charge_shader_module_info, None)? };

        let charge_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(charge_shader_module)
            .name(entry_name);
        let charge_pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(charge_stage_info)
            .layout(charge_pipeline_layout);
        let charge_projection_pipeline = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[charge_pipeline_info], None)
                .map_err(|error| {
                    anyhow::anyhow!("Failed to create charge projection pipeline: {:?}", error.1)
                })?[0]
        };
        unsafe { device.destroy_shader_module(charge_shader_module, None) };

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(10)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(2)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        let layouts = [descriptor_set_layout, descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);
        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };

        let charge_pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(6)];
        let charge_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(2)
            .pool_sizes(&charge_pool_sizes);
        let charge_descriptor_pool =
            unsafe { device.create_descriptor_pool(&charge_pool_info, None)? };

        let charge_layouts = [charge_descriptor_set_layout, charge_descriptor_set_layout];
        let charge_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(charge_descriptor_pool)
            .set_layouts(&charge_layouts);
        let charge_descriptor_sets =
            unsafe { device.allocate_descriptor_sets(&charge_alloc_info)? };

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
            coarse_grid: None,
            reaction: None,
            leak: None,
            enzyme: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn copy_buffer_sync(
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

        let copy_region = vk::BufferCopy::default().size(size);
        unsafe { device.cmd_copy_buffer(cmd, src, dst, &[copy_region]) };
        unsafe { device.end_command_buffer(cmd)? };

        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
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

    #[allow(clippy::too_many_arguments)]
    pub(super) fn readback_buffer_sync(
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

        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
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

    #[allow(clippy::too_many_arguments)]
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

    #[allow(clippy::too_many_arguments)]
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
}

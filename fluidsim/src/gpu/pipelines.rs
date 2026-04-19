use super::*;

impl GpuSimulation {
    fn pack_leak_channels(
        &self,
        channels: &[LeakChannel],
        species_registry: &SpeciesRegistry,
        solid_mask: &[u32],
    ) -> Result<Vec<GpuLeakChannel>> {
        if channels.len() > MAX_LEAK_CHANNELS {
            bail!(
                "Too many leak channels: {} > {}",
                channels.len(),
                MAX_LEAK_CHANNELS
            );
        }

        let mut packed_channels = Vec::with_capacity(channels.len());
        for channel in channels {
            let species_index = species_registry
                .index_of(channel.species)
                .context("Leak channel species is not registered in this scenario")?
                as u32;
            let ((sink_x, sink_y), (source_x, source_y)) = channel
                .resolve_endpoints(self.width, self.height, solid_mask)
                .context("Leak channel does not resolve to valid fluid endpoints")?;

            packed_channels.push(GpuLeakChannel {
                species_index,
                sink_x,
                sink_y,
                source_x,
                source_y,
                rate: channel.rate,
                rotation_byte: channel.rotation as u8 as u32,
                _pad: 0,
            });
        }

        Ok(packed_channels)
    }

    fn pack_enzyme_entities(
        &self,
        entities: &[EnzymeEntity],
        species_registry: &SpeciesRegistry,
    ) -> Result<Vec<GpuEnzymeEntity>> {
        if entities.len() > MAX_ENZYME_ENTITIES {
            bail!(
                "Too many enzyme entities: {} > {}",
                entities.len(),
                MAX_ENZYME_ENTITIES
            );
        }

        let glucose_index = species_registry
            .index_of_name("Glucose")
            .context("Enzyme scenario is missing Glucose")? as u32;
        let atp_index = species_registry
            .index_of_name("ATP")
            .context("Enzyme scenario is missing ATP")? as u32;
        let g6p_index = species_registry
            .index_of_name("G6P")
            .context("Enzyme scenario is missing G6P")? as u32;
        let adp_index = species_registry
            .index_of_name("ADP")
            .context("Enzyme scenario is missing ADP")? as u32;

        Ok(entities
            .iter()
            .map(|entity| {
                let (active_site_x, active_site_y) = entity.active_site_cell();
                GpuEnzymeEntity {
                    active_site_x,
                    active_site_y,
                    glucose_index,
                    atp_index,
                    g6p_index,
                    adp_index,
                    catalytic_scale: entity.catalytic_scale,
                    thermal_bias: entity.thermal_bias,
                    km_glucose: 0.12,
                    km_atp: 0.08,
                }
            })
            .collect())
    }

    pub(super) fn reaction_descriptor_index(
        conc_current_buffer: usize,
        temperature_current_buffer: usize,
    ) -> usize {
        conc_current_buffer * 2 + temperature_current_buffer
    }

    #[allow(clippy::too_many_arguments)]
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

    #[allow(clippy::too_many_arguments)]
    fn update_leak_descriptor_set(
        device: &ash::Device,
        set: vk::DescriptorSet,
        conc_buffer: vk::Buffer,
        channels_buffer: vk::Buffer,
        charges_buffer: vk::Buffer,
        conc_size: u64,
        channels_size: u64,
        charges_size: u64,
    ) {
        let conc_info = [vk::DescriptorBufferInfo::default()
            .buffer(conc_buffer)
            .offset(0)
            .range(conc_size)];
        let channel_info = [vk::DescriptorBufferInfo::default()
            .buffer(channels_buffer)
            .offset(0)
            .range(channels_size)];
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
                .buffer_info(&channel_info),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&charges_info),
        ];

        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    #[allow(clippy::too_many_arguments)]
    fn update_enzyme_descriptor_set(
        device: &ash::Device,
        set: vk::DescriptorSet,
        conc_buffer: vk::Buffer,
        temperature_buffer: vk::Buffer,
        enzymes_buffer: vk::Buffer,
        conc_size: u64,
        temperature_size: u64,
        enzymes_size: u64,
    ) {
        let conc_info = [vk::DescriptorBufferInfo::default()
            .buffer(conc_buffer)
            .offset(0)
            .range(conc_size)];
        let temp_info = [vk::DescriptorBufferInfo::default()
            .buffer(temperature_buffer)
            .offset(0)
            .range(temperature_size)];
        let enzyme_info = [vk::DescriptorBufferInfo::default()
            .buffer(enzymes_buffer)
            .offset(0)
            .range(enzymes_size)];

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
                .buffer_info(&temp_info),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&enzyme_info),
        ];

        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    pub fn init_reaction_pipeline(&mut self, initial_temperatures: &[f32]) -> Result<()> {
        if self.reaction.is_some() {
            return Ok(());
        }

        let temp_buffer_size = (self.cell_count * std::mem::size_of::<f32>()) as u64;
        let rules_buffer_size =
            (MAX_REACTION_RULES * std::mem::size_of::<GpuReactionRule>()) as u64;
        let conc_buffer_size =
            (self.species_count * self.cell_count * std::mem::size_of::<f32>()) as u64;
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
        let rules_buffer = GpuBuffer::new(
            &self.device,
            &mut alloc,
            rules_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            "reaction_rules",
        )?;
        drop(alloc);

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
        let mut options =
            shaderc::CompileOptions::new().context("Failed to create compile options")?;
        options.set_target_env(
            shaderc::TargetEnv::Vulkan,
            shaderc::EnvVersion::Vulkan1_2 as u32,
        );
        options.set_optimization_level(shaderc::OptimizationLevel::Performance);

        let reaction_spirv = compiler
            .compile_into_spirv(
                include_str!("../../shaders/reaction.comp"),
                shaderc::ShaderKind::Compute,
                "reaction.comp",
                "main",
                Some(&options),
            )
            .context("Failed to compile reaction shader")?;
        let thermal_spirv = compiler
            .compile_into_spirv(
                include_str!("../../shaders/thermal_diffusion.comp"),
                shaderc::ShaderKind::Compute,
                "thermal_diffusion.comp",
                "main",
                Some(&options),
            )
            .context("Failed to compile thermal diffusion shader")?;

        let reaction_shader_module_info =
            vk::ShaderModuleCreateInfo::default().code(reaction_spirv.as_binary());
        let reaction_shader_module = unsafe {
            self.device
                .create_shader_module(&reaction_shader_module_info, None)?
        };

        let thermal_shader_module_info =
            vk::ShaderModuleCreateInfo::default().code(thermal_spirv.as_binary());
        let thermal_shader_module = unsafe {
            self.device
                .create_shader_module(&thermal_shader_module_info, None)?
        };

        let reaction_bindings = [
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
        ];

        let reaction_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&reaction_bindings);
        let reaction_descriptor_set_layout = unsafe {
            self.device
                .create_descriptor_set_layout(&reaction_layout_info, None)?
        };

        let reaction_push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<ReactionPushConstants>() as u32);
        let reaction_pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&reaction_descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&reaction_push_constant_range));
        let reaction_pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&reaction_pipeline_layout_info, None)?
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
            self.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[reaction_pipeline_info],
                    None,
                )
                .map_err(|error| {
                    anyhow::anyhow!("Failed to create reaction pipeline: {:?}", error.1)
                })?[0]
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

        let thermal_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&thermal_bindings);
        let thermal_descriptor_set_layout = unsafe {
            self.device
                .create_descriptor_set_layout(&thermal_layout_info, None)?
        };

        let thermal_push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<ThermalPushConstants>() as u32);
        let thermal_pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&thermal_descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&thermal_push_constant_range));
        let thermal_pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&thermal_pipeline_layout_info, None)?
        };

        let thermal_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(thermal_shader_module)
            .name(entry_name);
        let thermal_pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(thermal_stage_info)
            .layout(thermal_pipeline_layout);
        let thermal_pipeline = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[thermal_pipeline_info], None)
                .map_err(|error| {
                    anyhow::anyhow!("Failed to create thermal pipeline: {:?}", error.1)
                })?[0]
        };

        unsafe {
            self.device
                .destroy_shader_module(reaction_shader_module, None);
            self.device
                .destroy_shader_module(thermal_shader_module, None);
        }

        let reaction_pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(16)];
        let reaction_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(4)
            .pool_sizes(&reaction_pool_sizes);
        let reaction_descriptor_pool = unsafe {
            self.device
                .create_descriptor_pool(&reaction_pool_info, None)?
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
        let reaction_descriptor_sets =
            unsafe { self.device.allocate_descriptor_sets(&reaction_alloc_info)? };

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

        let thermal_pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(6)];
        let thermal_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(2)
            .pool_sizes(&thermal_pool_sizes);
        let thermal_descriptor_pool = unsafe {
            self.device
                .create_descriptor_pool(&thermal_pool_info, None)?
        };

        let thermal_layouts = [thermal_descriptor_set_layout, thermal_descriptor_set_layout];
        let thermal_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(thermal_descriptor_pool)
            .set_layouts(&thermal_layouts);
        let thermal_descriptor_sets =
            unsafe { self.device.allocate_descriptor_sets(&thermal_alloc_info)? };

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

        self.coarse_grid = CoarseGrid::new(CoarseGridCreateInfo {
            device: self.device.clone(),
            queue: self.compute_queue,
            queue_family: self.compute_queue_family,
            allocator: self
                .allocator
                .as_ref()
                .expect("allocator should exist")
                .clone(),
            full_width: self.width,
            full_height: self.height,
            species_count: self.species_count,
            mip_factor: 8,
            source_buffers: CoarseGridSourceBuffers {
                src_buffer_a: self.conc_buffer_a.buffer,
                src_buffer_b: self.conc_buffer_b.buffer,
                temperature_buffer_a: self
                    .reaction
                    .as_ref()
                    .expect("reaction initialized")
                    .temperature_buffer_a
                    .buffer,
                temperature_buffer_b: self
                    .reaction
                    .as_ref()
                    .expect("reaction initialized")
                    .temperature_buffer_b
                    .buffer,
                solid_mask_buffer: self.solid_mask.buffer,
                conc_buffer_size: (self.species_count
                    * self.cell_count
                    * std::mem::size_of::<f32>()) as u64,
                temperature_buffer_size: (self.cell_count * std::mem::size_of::<f32>()) as u64,
                mask_buffer_size: (self.cell_count * std::mem::size_of::<u32>()) as u64,
            },
        })
        .ok();

        log::info!("Reaction pipeline initialized");
        Ok(())
    }

    pub fn upload_reaction_rules(&mut self, rules: &[GpuReactionRule]) -> Result<bool> {
        if rules.len() > MAX_REACTION_RULES {
            bail!(
                "Too many reaction rules: {} > {}",
                rules.len(),
                MAX_REACTION_RULES
            );
        }

        let rxn = self
            .reaction
            .as_mut()
            .context("Reaction pipeline not initialized")?;

        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        rules.len().hash(&mut hasher);
        for rule in rules {
            rule.reactant_a_index.hash(&mut hasher);
            rule.reactant_b_index.hash(&mut hasher);
            rule.product_a_index.hash(&mut hasher);
            rule.product_b_index.hash(&mut hasher);
            rule.catalyst_index.hash(&mut hasher);
            rule.kinetic_model.hash(&mut hasher);
            rule.effective_rate_bits.hash(&mut hasher);
            rule.km_reactant_a_bits.hash(&mut hasher);
            rule.km_reactant_b_bits.hash(&mut hasher);
            rule.enthalpy_delta_bits.hash(&mut hasher);
            rule.entropy_delta_bits.hash(&mut hasher);
        }
        let new_hash = hasher.finish();

        if new_hash == rxn.rule_set_hash && rxn.active_rule_count == rules.len() as u32 {
            return Ok(false);
        }

        let rules_size = std::mem::size_of_val(rules) as u64;
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

        log::debug!(
            "Uploaded {} reaction rules (hash={:#x})",
            rules.len(),
            new_hash
        );
        Ok(true)
    }

    pub fn step_reactions(&mut self, dt: f32) -> Result<()> {
        let rxn = match &self.reaction {
            Some(rxn) => rxn,
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

        let descriptor_set = rxn.reaction_descriptor_sets
            [Self::reaction_descriptor_index(self.current_buffer, rxn.temperature_current_buffer)];

        unsafe {
            self.device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(self.command_buffer, &begin_info)?;

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
            let num_groups = (self.cell_count as u32).div_ceil(workgroup_size);
            self.device
                .cmd_dispatch(self.command_buffer, num_groups, 1, 1);

            self.device.end_command_buffer(self.command_buffer)?;

            let submit_info = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.command_buffer));

            self.device.reset_fences(&[self.fence])?;
            self.device
                .queue_submit(self.compute_queue, &[submit_info], self.fence)?;
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
        if self.leak.is_some() {
            return Ok(());
        }

        let packed_channels = self.pack_leak_channels(channels, species_registry, solid_mask)?;

        let channels_buffer_size =
            (MAX_LEAK_CHANNELS * std::mem::size_of::<GpuLeakChannel>()) as u64;
        let conc_buffer_size =
            (self.species_count * self.cell_count * std::mem::size_of::<f32>()) as u64;
        let charges_buffer_size = (self.species_count * std::mem::size_of::<i32>()) as u64;

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
        let mut options =
            shaderc::CompileOptions::new().context("Failed to create compile options")?;
        options.set_target_env(
            shaderc::TargetEnv::Vulkan,
            shaderc::EnvVersion::Vulkan1_2 as u32,
        );
        options.set_optimization_level(shaderc::OptimizationLevel::Performance);

        let leak_spirv = compiler
            .compile_into_spirv(
                include_str!("../../shaders/leak.comp"),
                shaderc::ShaderKind::Compute,
                "leak.comp",
                "main",
                Some(&options),
            )
            .context("Failed to compile leak shader")?;

        let leak_shader_module_info =
            vk::ShaderModuleCreateInfo::default().code(leak_spirv.as_binary());
        let leak_shader_module = unsafe {
            self.device
                .create_shader_module(&leak_shader_module_info, None)?
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
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let leak_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&leak_bindings);
        let leak_descriptor_set_layout = unsafe {
            self.device
                .create_descriptor_set_layout(&leak_layout_info, None)?
        };

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<LeakPushConstants>() as u32);
        let leak_pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&leak_descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        let leak_pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&leak_pipeline_layout_info, None)?
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
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|error| anyhow::anyhow!("Failed to create leak pipeline: {:?}", error.1))?
                [0]
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
        let leak_descriptor_pool = unsafe { self.device.create_descriptor_pool(&pool_info, None)? };

        let layouts = [leak_descriptor_set_layout, leak_descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(leak_descriptor_pool)
            .set_layouts(&layouts);
        let leak_descriptor_sets = unsafe { self.device.allocate_descriptor_sets(&alloc_info)? };

        Self::update_leak_descriptor_set(
            &self.device,
            leak_descriptor_sets[0],
            self.conc_buffer_a.buffer,
            channels_buffer.buffer,
            self.species_charges.buffer,
            conc_buffer_size,
            channels_buffer_size,
            charges_buffer_size,
        );
        Self::update_leak_descriptor_set(
            &self.device,
            leak_descriptor_sets[1],
            self.conc_buffer_b.buffer,
            channels_buffer.buffer,
            self.species_charges.buffer,
            conc_buffer_size,
            channels_buffer_size,
            charges_buffer_size,
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

    pub fn upload_leak_channels(
        &mut self,
        channels: &[LeakChannel],
        species_registry: &SpeciesRegistry,
        solid_mask: &[u32],
    ) -> Result<()> {
        if self.leak.is_none() {
            return self.init_leak_pipeline(channels, species_registry, solid_mask);
        }

        let packed_channels = self.pack_leak_channels(channels, species_registry, solid_mask)?;
        let leak = self.leak.as_mut().unwrap();
        leak.active_channel_count = packed_channels.len() as u32;

        if packed_channels.is_empty() {
            return Ok(());
        }

        self.staging_buffer.write(&packed_channels)?;
        Self::copy_buffer_sync(
            &self.device,
            self.command_pool,
            self.compute_queue,
            self.fence,
            self.staging_buffer.buffer,
            leak.channels_buffer.buffer,
            (packed_channels.len() * std::mem::size_of::<GpuLeakChannel>()) as u64,
            Some(self.command_buffer),
        )?;

        Ok(())
    }

    pub fn init_enzyme_pipeline(
        &mut self,
        entities: &[EnzymeEntity],
        species_registry: &SpeciesRegistry,
    ) -> Result<()> {
        if entities.is_empty() {
            return Ok(());
        }
        if self.enzyme.is_some() {
            return Ok(());
        }

        let (temperature_buffer_a, temperature_buffer_b) = {
            let rxn = self
                .reaction
                .as_ref()
                .context("Reaction pipeline must be initialized before enzyme pipeline")?;
            (
                rxn.temperature_buffer_a.buffer,
                rxn.temperature_buffer_b.buffer,
            )
        };
        let packed_enzymes = self.pack_enzyme_entities(entities, species_registry)?;

        let enzymes_buffer_size =
            (MAX_ENZYME_ENTITIES * std::mem::size_of::<GpuEnzymeEntity>()) as u64;
        let conc_buffer_size =
            (self.species_count * self.cell_count * std::mem::size_of::<f32>()) as u64;
        let temp_buffer_size = (self.cell_count * std::mem::size_of::<f32>()) as u64;

        let mut alloc = self.allocator.as_ref().unwrap().lock();
        let enzymes_buffer = GpuBuffer::new(
            &self.device,
            &mut alloc,
            enzymes_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            "enzyme_entities",
        )?;
        drop(alloc);

        self.staging_buffer.write(&packed_enzymes)?;
        Self::copy_buffer_sync(
            &self.device,
            self.command_pool,
            self.compute_queue,
            self.fence,
            self.staging_buffer.buffer,
            enzymes_buffer.buffer,
            (packed_enzymes.len() * std::mem::size_of::<GpuEnzymeEntity>()) as u64,
            Some(self.command_buffer),
        )?;

        let compiler = shaderc::Compiler::new().context("Failed to create shader compiler")?;
        let mut options =
            shaderc::CompileOptions::new().context("Failed to create compile options")?;
        options.set_target_env(
            shaderc::TargetEnv::Vulkan,
            shaderc::EnvVersion::Vulkan1_2 as u32,
        );
        options.set_optimization_level(shaderc::OptimizationLevel::Performance);

        let enzyme_spirv = compiler
            .compile_into_spirv(
                include_str!("../../shaders/enzyme.comp"),
                shaderc::ShaderKind::Compute,
                "enzyme.comp",
                "main",
                Some(&options),
            )
            .context("Failed to compile enzyme shader")?;

        let enzyme_shader_module_info =
            vk::ShaderModuleCreateInfo::default().code(enzyme_spirv.as_binary());
        let enzyme_shader_module = unsafe {
            self.device
                .create_shader_module(&enzyme_shader_module_info, None)?
        };

        let enzyme_bindings = [
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
        let enzyme_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&enzyme_bindings);
        let enzyme_descriptor_set_layout = unsafe {
            self.device
                .create_descriptor_set_layout(&enzyme_layout_info, None)?
        };

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<EnzymePushConstants>() as u32);
        let enzyme_pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&enzyme_descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        let enzyme_pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&enzyme_pipeline_layout_info, None)?
        };

        let entry_name = c"main";
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(enzyme_shader_module)
            .name(entry_name);
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(enzyme_pipeline_layout);
        let enzyme_pipeline = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|error| {
                    anyhow::anyhow!("Failed to create enzyme pipeline: {:?}", error.1)
                })?[0]
        };

        unsafe {
            self.device
                .destroy_shader_module(enzyme_shader_module, None);
        }

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(12)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(4)
            .pool_sizes(&pool_sizes);
        let enzyme_descriptor_pool =
            unsafe { self.device.create_descriptor_pool(&pool_info, None)? };

        let layouts = [
            enzyme_descriptor_set_layout,
            enzyme_descriptor_set_layout,
            enzyme_descriptor_set_layout,
            enzyme_descriptor_set_layout,
        ];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(enzyme_descriptor_pool)
            .set_layouts(&layouts);
        let enzyme_descriptor_sets = unsafe { self.device.allocate_descriptor_sets(&alloc_info)? };

        for conc_current_buffer in 0..2 {
            for temp_current_buffer in 0..2 {
                let descriptor_index =
                    Self::reaction_descriptor_index(conc_current_buffer, temp_current_buffer);
                let conc_buffer = if conc_current_buffer == 0 {
                    self.conc_buffer_a.buffer
                } else {
                    self.conc_buffer_b.buffer
                };
                let temp_buffer = if temp_current_buffer == 0 {
                    temperature_buffer_a
                } else {
                    temperature_buffer_b
                };
                Self::update_enzyme_descriptor_set(
                    &self.device,
                    enzyme_descriptor_sets[descriptor_index],
                    conc_buffer,
                    temp_buffer,
                    enzymes_buffer.buffer,
                    conc_buffer_size,
                    temp_buffer_size,
                    enzymes_buffer_size,
                );
            }
        }

        self.enzyme = Some(EnzymePipelineState {
            enzyme_descriptor_set_layout,
            enzyme_pipeline_layout,
            enzyme_pipeline,
            enzyme_descriptor_pool,
            enzyme_descriptor_sets,
            enzymes_buffer,
            active_enzyme_count: packed_enzymes.len() as u32,
        });

        Ok(())
    }

    pub fn upload_enzyme_entities(
        &mut self,
        entities: &[EnzymeEntity],
        species_registry: &SpeciesRegistry,
    ) -> Result<()> {
        if self.enzyme.is_none() {
            return self.init_enzyme_pipeline(entities, species_registry);
        }

        let packed_enzymes = self.pack_enzyme_entities(entities, species_registry)?;
        let enzyme = self.enzyme.as_mut().unwrap();
        enzyme.active_enzyme_count = packed_enzymes.len() as u32;

        if packed_enzymes.is_empty() {
            return Ok(());
        }

        self.staging_buffer.write(&packed_enzymes)?;
        Self::copy_buffer_sync(
            &self.device,
            self.command_pool,
            self.compute_queue,
            self.fence,
            self.staging_buffer.buffer,
            enzyme.enzymes_buffer.buffer,
            (packed_enzymes.len() * std::mem::size_of::<GpuEnzymeEntity>()) as u64,
            Some(self.command_buffer),
        )?;

        Ok(())
    }

    pub fn step_temperature(&mut self, dt: f32, thermal_diffusivity: f32) -> Result<()> {
        let rxn = match &mut self.reaction {
            Some(rxn) => rxn,
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
            self.device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(self.command_buffer, &begin_info)?;

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
            let num_groups = (self.cell_count as u32).div_ceil(workgroup_size);
            self.device
                .cmd_dispatch(self.command_buffer, num_groups, 1, 1);

            self.device.end_command_buffer(self.command_buffer)?;

            let submit_info = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.command_buffer));

            self.device.reset_fences(&[self.fence])?;
            self.device
                .queue_submit(self.compute_queue, &[submit_info], self.fence)?;
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
        }

        rxn.temperature_current_buffer = 1 - rxn.temperature_current_buffer;
        Ok(())
    }
}

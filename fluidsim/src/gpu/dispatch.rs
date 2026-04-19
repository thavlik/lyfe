use super::*;

impl GpuSimulation {
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
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.diffusion_pipeline,
            );
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
            let num_groups = (self.cell_count as u32).div_ceil(workgroup_size);
            self.device
                .cmd_dispatch(cmd, num_groups, self.species_count as u32, 1);
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
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, leak.leak_pipeline);
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
            let num_groups = push_constants.num_channels.div_ceil(workgroup_size);
            self.device.cmd_dispatch(cmd, num_groups.max(1), 1, 1);
        }
    }

    fn record_enzyme_dispatch(
        &self,
        cmd: vk::CommandBuffer,
        enzyme: &EnzymePipelineState,
        descriptor_set: vk::DescriptorSet,
        push_constants: &EnzymePushConstants,
    ) {
        unsafe {
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                enzyme.enzyme_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                enzyme.enzyme_pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            self.device.cmd_push_constants(
                cmd,
                enzyme.enzyme_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(push_constants),
            );

            let workgroup_size = 64u32;
            let num_groups = push_constants.num_enzymes.div_ceil(workgroup_size);
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
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.charge_projection_pipeline,
            );
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
            let num_groups = (self.cell_count as u32).div_ceil(workgroup_size);
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
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                rxn.reaction_pipeline,
            );
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
            let num_groups = (self.cell_count as u32).div_ceil(workgroup_size);
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
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                rxn.thermal_pipeline,
            );
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
            let num_groups = (self.cell_count as u32).div_ceil(workgroup_size);
            self.device.cmd_dispatch(cmd, num_groups, 1, 1);
        }
    }

    pub fn record_step_frame(&mut self, cmd: vk::CommandBuffer, params: StepFrameParams) {
        let StepFrameParams {
            substeps,
            substep_dt,
            diffusion_rate,
            charge_correction_strength,
            reaction_dt,
            thermal_diffusivity,
        } = params;
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

        let mut temperature_current_buffer = self
            .reaction
            .as_ref()
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
                    .src_access_mask(
                        vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE,
                    )
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .buffer(Self::temperature_buffer_handle(
                        rxn,
                        temperature_current_buffer,
                    ))
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
        if let Some(enzyme) = self.enzyme.as_ref() {
            input_barriers.push(
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .buffer(enzyme.enzymes_buffer.buffer)
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
                let reaction_descriptor_set = rxn.reaction_descriptor_sets
                    [Self::reaction_descriptor_index(
                        self.current_buffer,
                        temperature_current_buffer,
                    )];
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
                let thermal_descriptor_set =
                    rxn.thermal_descriptor_sets[temperature_current_buffer];
                self.record_temperature_dispatch(cmd, rxn, thermal_descriptor_set, &thermal_push);
                temperature_current_buffer = 1 - temperature_current_buffer;
                self.record_compute_buffer_barrier(
                    cmd,
                    Self::temperature_buffer_handle(rxn, temperature_current_buffer),
                );
            }

            if let Some(leak) = self.leak.as_ref()
                && leak.active_channel_count > 0
            {
                let leak_push = LeakPushConstants {
                    width: self.width,
                    height: self.height,
                    species_count: self.species_count as u32,
                    num_channels: leak.active_channel_count,
                    dt: substep_dt,
                    _pad: [0; 3],
                };
                let leak_descriptor_set = leak.leak_descriptor_sets[self.current_buffer];
                self.record_leak_dispatch(cmd, leak, leak_descriptor_set, &leak_push);
                self.record_compute_buffer_barrier(cmd, self.current_concentration_buffer_handle());
            }

            if let Some(enzyme) = self.enzyme.as_ref()
                && enzyme.active_enzyme_count > 0
            {
                let enzyme_push = EnzymePushConstants {
                    width: self.width,
                    height: self.height,
                    species_count: self.species_count as u32,
                    num_enzymes: enzyme.active_enzyme_count,
                    dt: reaction_substep_dt,
                    base_turnover_rate: 0.22,
                    _pad: [0; 2],
                };
                let enzyme_descriptor_set = enzyme.enzyme_descriptor_sets
                    [Self::reaction_descriptor_index(
                        self.current_buffer,
                        temperature_current_buffer,
                    )];
                self.record_enzyme_dispatch(cmd, enzyme, enzyme_descriptor_set, &enzyme_push);
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
            self.device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(self.command_buffer, &begin_info)?;
        }

        self.record_step_frame(
            self.command_buffer,
            StepFrameParams {
                substeps,
                substep_dt,
                diffusion_rate,
                charge_correction_strength,
                reaction_dt,
                thermal_diffusivity,
            },
        );

        unsafe {
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

    pub fn step(&mut self, dt: f32, diffusion_rate: f32) -> Result<()> {
        let push_constants = DiffusionPushConstants {
            width: self.width,
            height: self.height,
            species_count: self.species_count as u32,
            species_index: 0xFFFFFFFF,
            diffusion_rate,
            dt,
            _pad: [0, 0],
        };

        let descriptor_set = self.descriptor_sets[self.current_buffer];

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

            let workgroup_size = 256u32;
            let num_groups = (self.cell_count as u32).div_ceil(workgroup_size);
            self.device.cmd_dispatch(
                self.command_buffer,
                num_groups,
                self.species_count as u32,
                1,
            );

            let buffer_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .buffer(if self.current_buffer == 0 {
                    self.conc_buffer_b.buffer
                } else {
                    self.conc_buffer_a.buffer
                })
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

            if let Some(ref coarse) = self.coarse_grid {
                let next_buffer = 1 - self.current_buffer;
                let temperature_current_buffer = self
                    .reaction
                    .as_ref()
                    .map(|rxn| rxn.temperature_current_buffer)
                    .unwrap_or(0);
                coarse.record_compute(self.command_buffer, next_buffer, temperature_current_buffer);
            }

            self.device.end_command_buffer(self.command_buffer)?;

            let submit_info = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.command_buffer));

            self.device.reset_fences(&[self.fence])?;
            self.device
                .queue_submit(self.compute_queue, &[submit_info], self.fence)?;
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
        }

        self.current_buffer = 1 - self.current_buffer;
        Ok(())
    }

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
            self.device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(self.command_buffer, &begin_info)?;

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

        let flat_data: Vec<f32> = self
            .readback_buffer
            .read(self.species_count * self.cell_count)?;

        let mut concentrations = Vec::with_capacity(self.species_count);
        for species_index in 0..self.species_count {
            let start = species_index * self.cell_count;
            let end = start + self.cell_count;
            concentrations.push(flat_data[start..end].to_vec());
        }

        Ok(concentrations)
    }

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

    pub fn read_temperatures(&mut self) -> Result<Vec<f32>> {
        let rxn = match &self.reaction {
            Some(rxn) => rxn,
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

    pub fn current_concentration_buffer(&self) -> vk::Buffer {
        if self.current_buffer == 0 {
            self.conc_buffer_a.buffer
        } else {
            self.conc_buffer_b.buffer
        }
    }

    pub fn current_temperature_buffer(&self) -> vk::Buffer {
        self.reaction
            .as_ref()
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
                    .src_access_mask(
                        vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE,
                    )
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
}

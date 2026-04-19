use super::*;

impl Drop for GpuSimulation {
    fn drop(&mut self) {
        log::info!("GpuSimulation::drop - starting");
        unsafe {
            self.device.device_wait_idle().ok();
            log::info!("GpuSimulation::drop - device idle");

            drop(self.coarse_grid.take());
            log::info!("GpuSimulation::drop - coarse_grid dropped");

            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);

            if let Some(mut rxn) = self.reaction.take() {
                self.device
                    .destroy_descriptor_pool(rxn.reaction_descriptor_pool, None);
                self.device.destroy_pipeline(rxn.reaction_pipeline, None);
                self.device
                    .destroy_pipeline_layout(rxn.reaction_pipeline_layout, None);
                self.device
                    .destroy_descriptor_set_layout(rxn.reaction_descriptor_set_layout, None);
                self.device
                    .destroy_descriptor_pool(rxn.thermal_descriptor_pool, None);
                self.device.destroy_pipeline(rxn.thermal_pipeline, None);
                self.device
                    .destroy_pipeline_layout(rxn.thermal_pipeline_layout, None);
                self.device
                    .destroy_descriptor_set_layout(rxn.thermal_descriptor_set_layout, None);

                let mut alloc = self.allocator.as_ref().unwrap().lock();
                self.device
                    .destroy_buffer(rxn.temperature_buffer_a.buffer, None);
                alloc
                    .free(std::mem::take(&mut rxn.temperature_buffer_a.allocation))
                    .ok();
                self.device
                    .destroy_buffer(rxn.temperature_buffer_b.buffer, None);
                alloc
                    .free(std::mem::take(&mut rxn.temperature_buffer_b.allocation))
                    .ok();
                self.device.destroy_buffer(rxn.rules_buffer.buffer, None);
                alloc
                    .free(std::mem::take(&mut rxn.rules_buffer.allocation))
                    .ok();
                drop(alloc);
            }
            log::info!("GpuSimulation::drop - reaction pipeline dropped");

            if let Some(mut leak) = self.leak.take() {
                self.device
                    .destroy_descriptor_pool(leak.leak_descriptor_pool, None);
                self.device.destroy_pipeline(leak.leak_pipeline, None);
                self.device
                    .destroy_pipeline_layout(leak.leak_pipeline_layout, None);
                self.device
                    .destroy_descriptor_set_layout(leak.leak_descriptor_set_layout, None);

                let mut alloc = self.allocator.as_ref().unwrap().lock();
                self.device
                    .destroy_buffer(leak.channels_buffer.buffer, None);
                alloc
                    .free(std::mem::take(&mut leak.channels_buffer.allocation))
                    .ok();
                drop(alloc);
            }
            log::info!("GpuSimulation::drop - leak pipeline dropped");

            if let Some(mut enzyme) = self.enzyme.take() {
                self.device
                    .destroy_descriptor_pool(enzyme.enzyme_descriptor_pool, None);
                self.device.destroy_pipeline(enzyme.enzyme_pipeline, None);
                self.device
                    .destroy_pipeline_layout(enzyme.enzyme_pipeline_layout, None);
                self.device
                    .destroy_descriptor_set_layout(enzyme.enzyme_descriptor_set_layout, None);

                let mut alloc = self.allocator.as_ref().unwrap().lock();
                self.device
                    .destroy_buffer(enzyme.enzymes_buffer.buffer, None);
                alloc
                    .free(std::mem::take(&mut enzyme.enzymes_buffer.allocation))
                    .ok();
                drop(alloc);
            }
            log::info!("GpuSimulation::drop - enzyme pipeline dropped");

            self.device
                .destroy_descriptor_pool(self.charge_descriptor_pool, None);
            self.device
                .destroy_pipeline(self.charge_projection_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.charge_pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.charge_descriptor_set_layout, None);

            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_pipeline(self.diffusion_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            let mut alloc = self.allocator.as_ref().unwrap().lock();

            self.device.destroy_buffer(self.conc_buffer_a.buffer, None);
            alloc
                .free(std::mem::take(&mut self.conc_buffer_a.allocation))
                .ok();

            self.device.destroy_buffer(self.conc_buffer_b.buffer, None);
            alloc
                .free(std::mem::take(&mut self.conc_buffer_b.allocation))
                .ok();

            self.device.destroy_buffer(self.solid_mask.buffer, None);
            alloc
                .free(std::mem::take(&mut self.solid_mask.allocation))
                .ok();

            self.device.destroy_buffer(self.material_ids.buffer, None);
            alloc
                .free(std::mem::take(&mut self.material_ids.allocation))
                .ok();

            self.device
                .destroy_buffer(self.diffusion_coeffs.buffer, None);
            alloc
                .free(std::mem::take(&mut self.diffusion_coeffs.allocation))
                .ok();

            self.device
                .destroy_buffer(self.species_charges.buffer, None);
            alloc
                .free(std::mem::take(&mut self.species_charges.allocation))
                .ok();

            self.device.destroy_buffer(self.staging_buffer.buffer, None);
            alloc
                .free(std::mem::take(&mut self.staging_buffer.allocation))
                .ok();

            self.device
                .destroy_buffer(self.readback_buffer.buffer, None);
            alloc
                .free(std::mem::take(&mut self.readback_buffer.allocation))
                .ok();

            drop(alloc);
            log::info!("GpuSimulation::drop - buffers freed");

            drop(self.allocator.take());
            log::info!(
                "GpuSimulation::drop - allocator dropped, device handle: {:?}",
                self.device.handle()
            );

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

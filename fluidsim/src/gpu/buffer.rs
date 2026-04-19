use super::*;

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

        Ok(Self {
            buffer,
            allocation,
            size,
        })
    }

    pub fn write<T: Pod>(&mut self, data: &[T]) -> Result<()> {
        let bytes = bytemuck::cast_slice(data);
        let mapped = self
            .allocation
            .mapped_slice_mut()
            .context("Buffer not mapped for CPU access")?;
        mapped[..bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    pub fn read<T: Pod + Clone>(&self, count: usize) -> Result<Vec<T>> {
        let mapped = self
            .allocation
            .mapped_slice()
            .context("Buffer not mapped for CPU access")?;
        let bytes = &mapped[..count * std::mem::size_of::<T>()];
        Ok(bytemuck::cast_slice(bytes).to_vec())
    }
}

use ash::vk;

use crate::{util::find_memorytype_index, vulkan_context::VulkanContext};

// Small controls should not each cost a megabyte. Growth rounds up geometrically,
// and completed frames retain their high-water capacity for subsequent frames.
const MIN_BUFFER_SIZE: usize = 64 * 1024;

/// An exclusively owned, persistently mapped host-coherent buffer.
pub(crate) struct Buffer<T> {
    pub handle: vk::Buffer,
    memory: vk::DeviceMemory,
    usage: vk::BufferUsageFlags,
    capacity: usize,
    ptr: std::ptr::NonNull<T>,
}

impl<T: Copy> Buffer<T> {
    pub fn with_capacity(
        context: &VulkanContext,
        usage: vk::BufferUsageFlags,
        elements: usize,
    ) -> Self {
        let element_size = std::mem::size_of::<T>();
        assert_ne!(element_size, 0);
        let bytes = elements
            .checked_mul(element_size)
            .expect("buffer size overflow");
        let size = bytes
            .max(MIN_BUFFER_SIZE)
            .checked_next_power_of_two()
            .expect("buffer size overflow") as vk::DeviceSize;
        let device = context.device;
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let handle = unsafe { device.create_buffer(&info, None).unwrap() };
        let requirements = unsafe { device.get_buffer_memory_requirements(handle) };
        let memory_index = find_memorytype_index(
            &requirements,
            &context.memory_properties,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .expect("Unable to find host-coherent buffer memory");
        let allocation = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_index);
        let memory = unsafe { device.allocate_memory(&allocation, None).unwrap() };
        let ptr = unsafe {
            device.bind_buffer_memory(handle, memory, 0).unwrap();
            let ptr = device
                .map_memory(memory, 0, requirements.size, vk::MemoryMapFlags::empty())
                .unwrap();
            std::ptr::NonNull::new_unchecked(ptr.cast())
        };
        Self {
            handle,
            memory,
            usage,
            // Allocation padding is not usable buffer space. Only the size passed
            // to create_buffer determines what a recorded GPU command may read.
            capacity: size as usize / element_size,
            ptr,
        }
    }

    /// Grow an unused buffer. No recorded or executing command may reference it.
    pub unsafe fn reserve(&mut self, context: &VulkanContext, elements: usize) {
        if elements > self.capacity {
            let replacement = Self::with_capacity(context, self.usage, elements);
            self.cleanup(context.device);
            *self = replacement;
        }
    }

    /// The caller owns this buffer exclusively until the GPU finishes reading it.
    pub unsafe fn write(&mut self, offset: usize, data: &[T]) {
        assert!(
            offset
                .checked_add(data.len())
                .is_some_and(|end| end <= self.capacity),
            "write exceeds buffer capacity"
        );
        self.ptr
            .as_ptr()
            .add(offset)
            .copy_from_nonoverlapping(data.as_ptr(), data.len());
    }

    /// No recorded or executing command may reference this buffer after cleanup.
    pub unsafe fn cleanup(&self, device: &ash::Device) {
        device.unmap_memory(self.memory);
        device.destroy_buffer(self.handle, None);
        device.free_memory(self.memory, None);
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

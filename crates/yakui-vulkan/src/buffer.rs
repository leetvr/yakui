use ash::vk;

use crate::{
    allocator::{Allocation, Allocator},
    vulkan_context::VulkanContext,
};

// Small controls should not each cost a megabyte. Growth rounds up geometrically,
// and completed frames retain their high-water capacity for subsequent frames.
const MIN_BUFFER_SIZE: usize = 64 * 1024;

/// An exclusively owned, persistently mapped host-coherent buffer.
pub(crate) struct Buffer<T> {
    pub handle: vk::Buffer,
    memory: Allocation,
    usage: vk::BufferUsageFlags,
    capacity: usize,
    ptr: std::ptr::NonNull<T>,
}

impl<T: Copy> Buffer<T> {
    pub fn with_capacity(
        context: &VulkanContext,
        allocator: &mut Allocator,
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
        let memory = unsafe { allocator.buffer(context, handle) };
        let ptr = memory.ptr.expect("buffer memory must be mapped").cast();
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
    pub unsafe fn reserve(
        &mut self,
        context: &VulkanContext,
        allocator: &mut Allocator,
        elements: usize,
    ) {
        if elements > self.capacity {
            let replacement = Self::with_capacity(context, allocator, self.usage, elements);
            let previous = std::mem::replace(self, replacement);
            previous.cleanup(context.device, allocator);
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
    pub unsafe fn cleanup(self, device: &ash::Device, allocator: &mut Allocator) {
        device.destroy_buffer(self.handle, None);
        allocator.free(device, self.memory);
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

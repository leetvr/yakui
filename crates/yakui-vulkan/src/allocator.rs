//! Adapted from lazy_vulkan's offset_allocator + aligned bind-offset mechanism.
//! Source: leetvr/lazy_vulkan, 33aec6209289d3dd8ee684276b9f15b6a904fc21.
//! Copyright (c) 2023 Let Eyes Equals Two, Pty Ltd; see LAZY-VULKAN-LICENSE-MIT.
use std::ptr::NonNull;

use ash::vk;

use crate::{util::find_memorytype_index, VulkanContext};

const BLOCK_SIZE: u64 = 16 * 1024 * 1024;

/// Memory owned by this renderer, excluding host-imported images.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryStats {
    /// Backing Vulkan device-memory allocations (shared blocks and dedicated allocations).
    pub device_allocations: usize,
    /// Bytes reserved from Vulkan, including reusable free space in blocks.
    pub reserved_bytes: u64,
    /// Live buffer/image allocations, including resources awaiting frame completion.
    pub live_allocations: usize,
}

// Buffer and optimal-image pools stay separate, so bufferImageGranularity cannot
// couple their bindings. Each pool also uses one compatible Vulkan memory type.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Buffer,
    Image,
}
struct Block {
    memory: vk::DeviceMemory,
    ptr: Option<NonNull<u8>>,
    memory_type: u32,
    kind: Kind,
    size: u64,
    offsets: offset_allocator::Allocator,
    live: usize,
}

enum Storage {
    Pooled {
        block: usize,
        range: offset_allocator::Allocation,
    },
    Dedicated {
        size: u64,
    },
}
pub(crate) struct Allocation {
    pub memory: vk::DeviceMemory,
    pub offset: u64,
    pub ptr: Option<NonNull<u8>>,
    storage: Storage,
}

#[derive(Default)]
pub(crate) struct Allocator {
    blocks: Vec<Block>,
    dedicated: usize,
    dedicated_bytes: u64,
}
impl Allocator {
    pub unsafe fn buffer(&mut self, context: &VulkanContext, buffer: vk::Buffer) -> Allocation {
        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
        context.device.get_buffer_memory_requirements2(
            &vk::BufferMemoryRequirementsInfo2::default().buffer(buffer),
            &mut requirements,
        );
        let requirements = requirements.memory_requirements;
        let allocation = self.allocate(
            context,
            requirements,
            Kind::Buffer,
            dedicated.requires_dedicated_allocation == vk::TRUE,
            vk::MemoryDedicatedAllocateInfo::default().buffer(buffer),
        );
        context
            .device
            .bind_buffer_memory(buffer, allocation.memory, allocation.offset)
            .unwrap();
        allocation
    }

    pub unsafe fn image(&mut self, context: &VulkanContext, image: vk::Image) -> Allocation {
        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
        context.device.get_image_memory_requirements2(
            &vk::ImageMemoryRequirementsInfo2::default().image(image),
            &mut requirements,
        );
        let requirements = requirements.memory_requirements;
        let allocation = self.allocate(
            context,
            requirements,
            Kind::Image,
            dedicated.requires_dedicated_allocation == vk::TRUE,
            vk::MemoryDedicatedAllocateInfo::default().image(image),
        );
        context
            .device
            .bind_image_memory(image, allocation.memory, allocation.offset)
            .unwrap();
        allocation
    }

    unsafe fn allocate(
        &mut self,
        context: &VulkanContext,
        requirements: vk::MemoryRequirements,
        kind: Kind,
        dedicated: bool,
        mut resource: vk::MemoryDedicatedAllocateInfo<'_>,
    ) -> Allocation {
        let flags = if kind == Kind::Buffer {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        } else {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        };
        let memory_type = find_memorytype_index(&requirements, &context.memory_properties, flags)
            .expect("No compatible Vulkan memory type");
        let block_size = requirements
            .size
            .max(BLOCK_SIZE)
            .checked_next_power_of_two();
        // offset_allocator addresses blocks with u32 offsets. Mandatory dedicated
        // allocations and resources too large for a block use their exact requirements.
        if dedicated || block_size.is_none_or(|size| size > u32::MAX as u64) {
            let memory = context
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(requirements.size)
                        .memory_type_index(memory_type)
                        .push_next(&mut resource),
                    None,
                )
                .unwrap();
            let ptr = Self::map(context.device, memory, kind);
            self.dedicated += 1;
            self.dedicated_bytes += requirements.size;
            return Allocation {
                memory,
                offset: 0,
                ptr,
                storage: Storage::Dedicated {
                    size: requirements.size,
                },
            };
        }
        for (index, block) in self.blocks.iter_mut().enumerate() {
            if block.kind == kind && block.memory_type == memory_type {
                if let Some((range, offset)) = allocate_offset(
                    &mut block.offsets,
                    requirements.size,
                    requirements.alignment,
                ) {
                    return Self::from_block(index, block, range, offset);
                }
            }
        }
        let size = block_size.unwrap();
        let memory = context
            .device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(size)
                    .memory_type_index(memory_type),
                None,
            )
            .unwrap();
        let ptr = Self::map(context.device, memory, kind);
        let mut block = Block {
            memory,
            ptr,
            memory_type,
            kind,
            size,
            offsets: offset_allocator::Allocator::new(size as u32),
            live: 0,
        };
        let (range, offset) = allocate_offset(
            &mut block.offsets,
            requirements.size,
            requirements.alignment,
        )
        .unwrap();
        let allocation = Self::from_block(self.blocks.len(), &mut block, range, offset);
        self.blocks.push(block);
        allocation
    }

    unsafe fn map(
        device: &ash::Device,
        memory: vk::DeviceMemory,
        kind: Kind,
    ) -> Option<NonNull<u8>> {
        if kind == Kind::Buffer {
            Some(
                NonNull::new(
                    device
                        .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                        .unwrap()
                        .cast(),
                )
                .unwrap(),
            )
        } else {
            None
        }
    }
    fn from_block(
        index: usize,
        block: &mut Block,
        range: offset_allocator::Allocation,
        offset: u64,
    ) -> Allocation {
        block.live += 1;
        Allocation {
            memory: block.memory,
            offset,
            ptr: block
                .ptr
                .map(|ptr| unsafe { NonNull::new_unchecked(ptr.as_ptr().add(offset as usize)) }),
            storage: Storage::Pooled {
                block: index,
                range,
            },
        }
    }

    /// Destroy the bound resource first, and only after all GPU references end.
    pub unsafe fn free(&mut self, device: &ash::Device, allocation: Allocation) {
        match allocation.storage {
            Storage::Pooled { block, range } => {
                let block = &mut self.blocks[block];
                block.offsets.free(range);
                block.live -= 1;
            }
            Storage::Dedicated { size } => {
                if allocation.ptr.is_some() {
                    device.unmap_memory(allocation.memory);
                }
                device.free_memory(allocation.memory, None);
                self.dedicated -= 1;
                self.dedicated_bytes -= size;
            }
        }
    }
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            device_allocations: self.blocks.len() + self.dedicated,
            reserved_bytes: self.blocks.iter().map(|block| block.size).sum::<u64>()
                + self.dedicated_bytes,
            live_allocations: self.blocks.iter().map(|block| block.live).sum::<usize>()
                + self.dedicated,
        }
    }
    /// The renderer has waited for idle and destroyed every buffer/image binding.
    pub unsafe fn cleanup(&mut self, device: &ash::Device) {
        assert_eq!(
            self.stats().live_allocations,
            0,
            "live allocator bindings at cleanup"
        );
        for block in self.blocks.drain(..) {
            if block.ptr.is_some() {
                device.unmap_memory(block.memory);
            }
            device.free_memory(block.memory, None);
        }
    }
}

fn allocate_offset(
    offsets: &mut offset_allocator::Allocator,
    size: u64,
    alignment: u64,
) -> Option<(offset_allocator::Allocation, u64)> {
    // lazy_vulkan first tries the requested size: an already aligned range needs
    // no padding, including a resource that occupies an entire fresh block.
    let range = offsets.allocate(u32::try_from(size).ok()?)?;
    if range.offset as u64 % alignment == 0 {
        return Some((range, range.offset as u64));
    }
    offsets.free(range);
    // A second allocation may choose a different hole. Reserve worst-case
    // padding and recompute alignment from the offset actually returned.
    let reserved = u32::try_from(size.checked_add(alignment - 1)?).ok()?;
    let range = offsets.allocate(reserved)?;
    let offset = (range.offset as u64 + alignment - 1) & !(alignment - 1);
    Some((range, offset))
}

#[cfg(test)]
mod tests {
    use super::allocate_offset;
    #[test]
    fn fragmented_ranges_align_and_reclaim_their_padding() {
        let mut offsets = offset_allocator::Allocator::new(1024 * 1024);
        let mut ranges = Vec::new();
        for index in 0..400 {
            let alignment = 1 << (index % 9);
            let size = 13 + index % 71;
            let (range, offset) = allocate_offset(&mut offsets, size, alignment).unwrap();
            assert_eq!(offset % alignment, 0);
            assert!(offset >= range.offset as u64);
            assert!(offset + size <= range.offset as u64 + size + alignment - 1);
            ranges.push(range);
        }
        for range in ranges.into_iter().rev() {
            offsets.free(range);
        }
        assert_eq!(offsets.storage_report().total_free_space, 1024 * 1024);
        assert!(allocate_offset(&mut offsets, 1024 * 1024, 256).is_some());
    }
}

use ash::vk;

use crate::allocator::{Allocation, Allocator};

#[derive(Clone)]
/// A wrapper around handles into your Vulkan renderer to call various methods on [`crate::YakuiVulkan`]
///
/// ## Safety
/// It is **very** important that you pass the correct handles to this struct, or you will have a terrible time.
/// Once you create a [`crate::YakuiVulkan`] instance, you **must** use the same handles each time you call a
/// method on that instance.
///
/// See the documentation on each member for specific safety requirements.
pub struct VulkanContext<'a> {
    /// A valid Vulkan device (see the [crate level documentation](`crate`) for specifics)
    pub device: &'a ash::Device,
    /// A queue that can call render and transfer commands
    pub queue: vk::Queue,
    /// Memory properties used for [`crate::YakuiVulkan`]'s allocation commands
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// Device properties used for setting paint limits
    pub properties: vk::PhysicalDeviceProperties,
}

impl<'a> VulkanContext<'a> {
    /// Construct a new [`VulkanContext`] wrapper.
    pub fn new(
        device: &'a ash::Device,
        queue: vk::Queue,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
        properties: vk::PhysicalDeviceProperties,
    ) -> Self {
        Self {
            device,
            queue,
            memory_properties,
            properties,
        }
    }

    pub(crate) unsafe fn create_image(
        &self,
        allocator: &mut Allocator,
        extent: vk::Extent2D,
        format: vk::Format,
    ) -> (vk::Image, Allocation) {
        let device = self.device;

        let image = device
            .create_image(
                &vk::ImageCreateInfo {
                    image_type: vk::ImageType::TYPE_2D,
                    format,
                    extent: extent.into(),
                    mip_levels: 1,
                    array_layers: 1,
                    samples: vk::SampleCountFlags::TYPE_1,
                    tiling: vk::ImageTiling::OPTIMAL,
                    usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    ..Default::default()
                },
                None,
            )
            .unwrap();

        let memory = allocator.image(self, image);
        (image, memory)
    }

    pub(crate) unsafe fn create_image_view(
        &self,
        image: vk::Image,
        format: vk::Format,
    ) -> vk::ImageView {
        self.device
            .create_image_view(
                &vk::ImageViewCreateInfo {
                    image,
                    format,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        level_count: 1,
                        layer_count: 1,
                        ..Default::default()
                    },
                    view_type: vk::ImageViewType::TYPE_2D,
                    ..Default::default()
                },
                None,
            )
            .unwrap()
    }
}

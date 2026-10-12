use ash::vk;

use crate::vulkan_context::VulkanContext;

/// A thin wrapper around descriptor related functionality
pub struct Descriptors {
    pub(crate) pool: vk::DescriptorPool,
    pub(crate) set: vk::DescriptorSet,
    pub(crate) layout: vk::DescriptorSetLayout,
    texture_count: u32,
    free_slots: Vec<u32>,
}

impl Descriptors {
    pub(crate) fn new(vulkan_context: &VulkanContext) -> Descriptors {
        let device = vulkan_context.device;

        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                    .pool_sizes(&[vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: 1000,
                    }]),
                None,
            )
        }
        .unwrap();

        let flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND
            | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND];
        let mut binding_flags =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);

        let layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(&[vk::DescriptorSetLayoutBinding {
                        binding: 0,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        stage_flags: vk::ShaderStageFlags::FRAGMENT,
                        descriptor_count: 1000,
                        ..Default::default()
                    }])
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                    .push_next(&mut binding_flags),
                None,
            )
        }
        .unwrap();

        let set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(std::slice::from_ref(&layout)),
            )
        }
        .unwrap()[0];

        Descriptors {
            pool,
            set,
            layout,
            texture_count: 0,
            free_slots: Vec::new(),
        }
    }

    pub(crate) unsafe fn update_texture_descriptor_set(
        &mut self,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
        vulkan_context: &VulkanContext,
    ) -> u32 {
        // A descriptor remains immutable while any frame can sample it. Slots
        // reach this list only when the owning texture's retirement frame ends.
        let texture_id = self.free_slots.pop().unwrap_or_else(|| {
            assert!(self.texture_count < 1000, "yakui-vulkan texture descriptor capacity exhausted (1000 live or pending textures)");
            let id = self.texture_count;
            self.texture_count += 1;
            id
        });
        vulkan_context.device.update_descriptor_sets(
            std::slice::from_ref(
                &vk::WriteDescriptorSet::default()
                    .image_info(std::slice::from_ref(
                        &vk::DescriptorImageInfo::default()
                            .sampler(sampler)
                            .image_view(image_view)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    ))
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .dst_array_element(texture_id as _)
                    .dst_set(self.set),
            ),
            &[],
        );

        texture_id
    }

    pub(crate) fn release_texture(&mut self, id: u32) {
        self.free_slots.push(id);
    }

    /// ## Safety
    /// Descriptors will be unusable after this function has been called.
    pub unsafe fn cleanup(&self, device: &ash::Device) {
        device.destroy_descriptor_pool(self.pool, None);
        device.destroy_descriptor_set_layout(self.layout, None);
    }
}

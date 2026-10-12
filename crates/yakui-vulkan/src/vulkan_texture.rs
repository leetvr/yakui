use std::collections::VecDeque;

use ash::vk;
use yakui_core as yakui;

use crate::{buffer::Buffer, descriptors::Descriptors, vulkan_context::VulkanContext, Vertex};

pub(crate) const NO_TEXTURE_ID: u32 = u32::MAX;

/// A container around a Vulkan created texture
pub struct VulkanTexture {
    image: vk::Image,
    memory: vk::DeviceMemory,
    pub(crate) sampler: vk::Sampler,
    pub(crate) view: vk::ImageView,
    pub(crate) id: u32,
}

/// A container for information about a texture.
/// Populate this struct and pass it to [`crate::YakuiVulkan::add_user_texture()`] to create user managed
/// textures that you can then use in [`yakui`] code.
pub struct VulkanTextureCreateInfo<T> {
    image_data: T,
    format: vk::Format,
    resolution: vk::Extent2D,
    min_filter: vk::Filter,
    mag_filter: vk::Filter,
    address_mode: vk::SamplerAddressMode,
}

impl<T: AsRef<[u8]>> VulkanTextureCreateInfo<T> {
    /// Construct a new [`VulkanTextureCreateInfo`] wrapper. Ensure `image_data` refers to an image that matches
    /// the rest of the parameters.
    pub fn new(
        image_data: T,
        format: vk::Format,
        resolution: vk::Extent2D,
        min_filter: vk::Filter,
        mag_filter: vk::Filter,
        address_mode: vk::SamplerAddressMode,
    ) -> Self {
        Self {
            image_data,
            format,
            resolution,
            min_filter,
            mag_filter,
            address_mode,
        }
    }
}

impl VulkanTexture {
    /// Create a [`VulkanTexture`] from a pre-existing [`vk::Image`]. Most users will instead want to call
    /// [`super::YakuiVulkan::create_user_texture()`].
    ///
    /// ## Safety
    /// Takes ownership of the image, its memory and its view. They are destroyed
    /// when the texture is removed and its final frame completes, or at cleanup.
    ///
    /// - All Vulkan handles must have been created on the same `vulkan_context`
    pub fn from_image(
        vulkan_context: &VulkanContext,
        descriptors: &mut Descriptors,
        image: vk::Image,
        memory: vk::DeviceMemory,
        view: vk::ImageView,
    ) -> Self {
        let address_mode = vk::SamplerAddressMode::CLAMP_TO_EDGE;
        let filter = vk::Filter::LINEAR;
        let sampler = unsafe {
            vulkan_context
                .device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .address_mode_u(address_mode)
                        .address_mode_v(address_mode)
                        .address_mode_w(address_mode)
                        .mag_filter(filter)
                        .min_filter(filter),
                    None,
                )
                .unwrap()
        };
        let id =
            unsafe { descriptors.update_texture_descriptor_set(view, sampler, vulkan_context) };
        VulkanTexture {
            image,
            memory,
            sampler,
            view,
            id,
        }
    }

    pub(crate) fn new<T: AsRef<[u8]>>(
        vulkan_context: &VulkanContext,
        descriptors: &mut Descriptors,
        create_info: VulkanTextureCreateInfo<T>,
        queue: &mut UploadQueue,
    ) -> Self {
        let VulkanTextureCreateInfo {
            image_data,
            format,
            resolution,
            min_filter,
            mag_filter,
            address_mode,
        } = create_info;

        let (image, memory) = unsafe { vulkan_context.create_image(resolution, format) };
        unsafe {
            queue.push(vulkan_context, image, resolution, image_data.as_ref());
        }
        let view = unsafe { vulkan_context.create_image_view(image, format) };

        let sampler = unsafe {
            vulkan_context
                .device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .address_mode_u(address_mode)
                        .address_mode_v(address_mode)
                        .address_mode_w(address_mode)
                        .mag_filter(mag_filter)
                        .min_filter(min_filter),
                    None,
                )
                .unwrap()
        };

        let id =
            unsafe { descriptors.update_texture_descriptor_set(view, sampler, vulkan_context) };

        VulkanTexture {
            image,
            memory,
            view,
            sampler,
            id,
        }
    }

    pub(crate) fn from_yakui_texture(
        vulkan_context: &VulkanContext,
        descriptors: &mut Descriptors,
        texture: &yakui::paint::Texture,
        queue: &mut UploadQueue,
    ) -> Self {
        let resolution = vk::Extent2D {
            width: texture.size().x,
            height: texture.size().y,
        };

        let format = get_format(texture.format());
        let image_data = texture.data();

        let mag_filter = get_filter(texture.mag_filter);
        let min_filter = get_filter(texture.min_filter);
        let address_mode = get_address_mode(texture.address_mode);
        VulkanTexture::new(
            vulkan_context,
            descriptors,
            VulkanTextureCreateInfo::new(
                image_data,
                format,
                resolution,
                min_filter,
                mag_filter,
                address_mode,
            ),
            queue,
        )
    }

    pub(crate) unsafe fn cleanup(&self, device: &ash::Device) {
        device.destroy_sampler(self.sampler, None);
        device.destroy_image_view(self.view, None);
        device.destroy_image(self.image, None);
        device.free_memory(self.memory, None);
    }
}

fn get_format(yakui_format: yakui::paint::TextureFormat) -> vk::Format {
    match yakui_format {
        yakui::paint::TextureFormat::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
        yakui::paint::TextureFormat::Rgba8SrgbPremultiplied => vk::Format::R8G8B8A8_SRGB,
        yakui::paint::TextureFormat::R8 => vk::Format::R8_UNORM,
    }
}

fn get_filter(yakui_filter: yakui::paint::TextureFilter) -> vk::Filter {
    match yakui_filter {
        yakui::paint::TextureFilter::Linear => vk::Filter::LINEAR,
        yakui::paint::TextureFilter::Nearest => vk::Filter::NEAREST,
    }
}

fn get_address_mode(yakui_address_mode: yakui::paint::AddressMode) -> vk::SamplerAddressMode {
    match yakui_address_mode {
        yakui::paint::AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        yakui::paint::AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
    }
}

#[derive(Default)]
pub(crate) struct UploadQueue {
    phase: FrameResources,
    in_flight: VecDeque<FrameResources>,
    available: Vec<FrameResources>,
    textures: Vec<(vk::Image, vk::Extent2D, vk::Buffer, usize)>,
    pre_barriers: Vec<vk::ImageMemoryBarrier<'static>>,
    post_barriers: Vec<vk::ImageMemoryBarrier<'static>>,
}

impl UploadQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Seal all resources recorded for this frame. Further writes use another slot.
    pub fn phase_submitted(&mut self) {
        assert!(
            self.textures.is_empty(),
            "record pending texture uploads before submission"
        );
        let next = self.available.pop().unwrap_or_default();
        let phase = std::mem::replace(&mut self.phase, next);
        self.in_flight.push_back(phase);
    }

    /// The host has waited for all transfer and drawing work in the oldest frame.
    pub unsafe fn phase_executed(
        &mut self,
        context: &VulkanContext,
        descriptors: &mut Descriptors,
    ) {
        let mut finished = self.in_flight.pop_front().expect("no frames in flight");
        for texture in finished.graveyard.drain(..) {
            texture.cleanup(context.device);
            descriptors.release_texture(texture.id);
        }
        finished.geometry_used = 0;
        finished.staging_used = 0;
        for (_, fill) in &mut finished.buffers {
            *fill = 0;
        }
        self.available.push(finished);
    }

    /// Retire after this frame, including any already-recorded uploads/draws.
    pub unsafe fn dispose(&mut self, texture: VulkanTexture) {
        // Frames are submitted and completed in order on the host's graphics
        // queue. Waiting for this frame also covers earlier uses of the texture.
        self.phase.graveyard.push(texture);
    }

    pub unsafe fn geometry(
        &mut self,
        context: &VulkanContext,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> (vk::Buffer, vk::Buffer) {
        let index = self.phase.geometry_used;
        if index == self.phase.geometry.len() {
            self.phase.geometry.push(GeometryBuffers {
                vertices: Buffer::with_capacity(
                    context,
                    vk::BufferUsageFlags::VERTEX_BUFFER,
                    vertices.len(),
                ),
                indices: Buffer::with_capacity(
                    context,
                    vk::BufferUsageFlags::INDEX_BUFFER,
                    indices.len(),
                ),
            });
        }
        let pair = &mut self.phase.geometry[index];
        // This pair has not been used in this frame. Earlier frames cannot own
        // it because only GPU-completed slots enter the available pool. A second
        // paint call receives another pair so it cannot overwrite recorded data.
        pair.vertices.reserve(context, vertices.len());
        pair.indices.reserve(context, indices.len());
        pair.vertices.write(0, vertices);
        pair.indices.write(0, indices);
        self.phase.geometry_used += 1;
        (pair.vertices.handle, pair.indices.handle)
    }

    unsafe fn push(
        &mut self,
        vulkan_context: &VulkanContext,
        image: vk::Image,
        extent: vk::Extent2D,
        data: &[u8],
    ) {
        self.pre_barriers.push(vk::ImageMemoryBarrier {
            dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            image,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            },
            ..Default::default()
        });
        let (buffer, offset) = self.phase.push(vulkan_context, data);
        self.textures.push((image, extent, buffer, offset));
        self.post_barriers.push(vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            image,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            },
            ..Default::default()
        });
    }

    pub unsafe fn record(&mut self, vulkan_context: &VulkanContext, cmd: vk::CommandBuffer) {
        let device = vulkan_context.device;
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &self.pre_barriers,
        );
        self.pre_barriers.clear();

        for (image, extent, buffer, offset) in self.textures.drain(..) {
            device.cmd_copy_buffer_to_image(
                cmd,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy {
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        layer_count: 1,
                        ..Default::default()
                    },
                    image_extent: extent.into(),
                    buffer_offset: offset as vk::DeviceSize,
                    ..Default::default()
                }],
            );
        }

        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &self.post_barriers,
        );
        self.post_barriers.clear();
    }

    pub unsafe fn cleanup(&mut self, device: &ash::Device) {
        self.phase.cleanup(device);
        for mut phase in self.in_flight.drain(..).chain(self.available.drain(..)) {
            phase.cleanup(device);
        }
    }
}

#[derive(Default)]
struct FrameResources {
    buffers: Vec<(Buffer<u8>, usize)>,
    staging_used: usize,
    geometry: Vec<GeometryBuffers>,
    geometry_used: usize,
    graveyard: Vec<VulkanTexture>,
}

struct GeometryBuffers {
    vertices: Buffer<Vertex>,
    indices: Buffer<u32>,
}

impl FrameResources {
    unsafe fn cleanup(&mut self, device: &ash::Device) {
        for (buffer, _) in self.buffers.drain(..) {
            buffer.cleanup(device);
        }
        for pair in self.geometry.drain(..) {
            pair.vertices.cleanup(device);
            pair.indices.cleanup(device);
        }
        for texture in self.graveyard.drain(..) {
            texture.cleanup(device);
        }
    }

    unsafe fn push(&mut self, context: &VulkanContext, data: &[u8]) -> (vk::Buffer, usize) {
        loop {
            let index = self.staging_used;
            if index == self.buffers.len() {
                self.buffers.push((
                    Buffer::with_capacity(context, vk::BufferUsageFlags::TRANSFER_SRC, data.len()),
                    0,
                ));
            }
            let (buffer, fill) = &mut self.buffers[index];
            // Image copy offsets must be four-byte aligned, even for odd-sized
            // R8 images. Never relocate a chunk referenced by an earlier copy.
            let start = (*fill + 3) & !3;
            if *fill == 0 {
                buffer.reserve(context, data.len());
            }
            if start + data.len() <= buffer.capacity() {
                *fill = start + data.len();
                buffer.write(start, data);
                return (buffer.handle, start);
            }
            self.staging_used += 1;
        }
    }
}

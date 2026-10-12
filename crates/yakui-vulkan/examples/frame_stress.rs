//! Real-device regression check: three in-flight frames, two paints per frame,
//! growing geometry/staging buffers, and more texture allocations than slots.
//! Run with `cargo run -p yakui-vulkan --example frame_stress`.
use ash::vk;
use std::{
    ffi::CStr,
    sync::atomic::{AtomicUsize, Ordering},
};
use yakui_core::{
    geometry::{Constraints, Rect, Vec2},
    paint::{PaintMesh, Pipeline, Vertex},
    widget::{LayoutContext, PaintContext, Widget},
    TextureId, Yakui,
};
use yakui_vulkan::{Options, VulkanContext, VulkanTextureCreateInfo, YakuiVulkan};

const EXTENT: vk::Extent2D = vk::Extent2D {
    width: 64,
    height: 64,
};
static VALIDATION_ERRORS: AtomicUsize = AtomicUsize::new(0);

unsafe extern "system" fn validation(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _kind: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        VALIDATION_ERRORS.fetch_add(1, Ordering::Relaxed);
        eprintln!("{}", CStr::from_ptr((*data).p_message).to_string_lossy());
    }
    vk::FALSE
}

#[derive(Debug, Default)]
struct Quad {
    color: [f32; 4],
    right: bool,
    large: bool,
    texture: Option<TextureId>,
}

impl Widget for Quad {
    type Props<'a> = Self;
    type Response = ();
    fn new() -> Self {
        Self::default()
    }
    fn update(&mut self, props: Self) {
        *self = props;
    }
    fn layout(&self, _ctx: LayoutContext<'_>, constraints: Constraints) -> Vec2 {
        constraints.constrain(Vec2::splat(64.0))
    }
    fn paint(&self, ctx: PaintContext<'_>) {
        let x = if self.right { 32.0 } else { 0.0 };
        let vertices = [
            Vertex::new([x, 0.0], [0.0, 0.0], self.color),
            Vertex::new([x + 32.0, 0.0], [1.0, 0.0], self.color),
            Vertex::new([x + 32.0, 64.0], [1.0, 1.0], self.color),
            Vertex::new([x, 64.0], [0.0, 1.0], self.color),
        ];
        // Extra vertices stress allocations without spending millions of pixels
        // on overdraw. The visible quad still exercises the renderer's output.
        for index in 0..if self.large { 20 } else { 1 } {
            let mut data = vertices.to_vec();
            if self.large {
                data.resize(20_000, vertices[0]);
            }
            let mut mesh = PaintMesh::new(data, [0, 1, 2, 0, 2, 3]);
            mesh.texture = self.texture.map(|id| (id, Rect::ONE));
            if index % 2 == 1 {
                mesh.pipeline = Pipeline::Text;
            }
            ctx.paint.add_mesh(mesh);
        }
    }
}

struct Frame {
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    image: vk::Image,
    image_memory: vk::DeviceMemory,
    view: vk::ImageView,
    framebuffer: vk::Framebuffer,
    readback: vk::Buffer,
    readback_memory: vk::DeviceMemory,
    pixels: *const u8,
}

fn memory_type(
    context: &VulkanContext,
    requirements: vk::MemoryRequirements,
    flags: vk::MemoryPropertyFlags,
) -> u32 {
    (0..context.memory_properties.memory_type_count)
        .find(|&i| {
            requirements.memory_type_bits & (1 << i) != 0
                && context.memory_properties.memory_types[i as usize]
                    .property_flags
                    .contains(flags)
        })
        .expect("required memory type")
}

unsafe fn frame(context: &VulkanContext, pool: vk::CommandPool, pass: vk::RenderPass) -> Frame {
    let device = context.device;
    let cmd = device
        .allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
        .unwrap()[0];
    let fence = device
        .create_fence(&vk::FenceCreateInfo::default(), None)
        .unwrap();
    let image = device
        .create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .extent(EXTENT.into())
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC),
            None,
        )
        .unwrap();
    let requirements = device.get_image_memory_requirements(image);
    let image_memory = device
        .allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type(
                    context,
                    requirements,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )),
            None,
        )
        .unwrap();
    device.bind_image_memory(image, image_memory, 0).unwrap();
    let range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1);
    let view = device
        .create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .subresource_range(range),
            None,
        )
        .unwrap();
    let framebuffer = device
        .create_framebuffer(
            &vk::FramebufferCreateInfo::default()
                .render_pass(pass)
                .attachments(&[view])
                .width(EXTENT.width)
                .height(EXTENT.height)
                .layers(1),
            None,
        )
        .unwrap();
    let readback = device
        .create_buffer(
            &vk::BufferCreateInfo::default()
                .size(64 * 64 * 4)
                .usage(vk::BufferUsageFlags::TRANSFER_DST),
            None,
        )
        .unwrap();
    let requirements = device.get_buffer_memory_requirements(readback);
    let readback_memory = device
        .allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type(
                    context,
                    requirements,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )),
            None,
        )
        .unwrap();
    device
        .bind_buffer_memory(readback, readback_memory, 0)
        .unwrap();
    let pixels = device
        .map_memory(
            readback_memory,
            0,
            requirements.size,
            vk::MemoryMapFlags::empty(),
        )
        .unwrap()
        .cast();
    Frame {
        cmd,
        fence,
        image,
        image_memory,
        view,
        framebuffer,
        readback,
        readback_memory,
        pixels,
    }
}

fn color(index: usize) -> [u8; 4] {
    [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]][index % 3]
}

fn main() {
    unsafe {
        run();
    }
}

unsafe fn run() {
    let entry = ash::Entry::load().expect("load Vulkan");
    let layers = [c"VK_LAYER_KHRONOS_validation".as_ptr()];
    let mut extensions = vec![ash::ext::debug_utils::NAME.as_ptr()];
    if cfg!(target_os = "macos") {
        extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
    }
    let app = vk::ApplicationInfo::default()
        .application_name(c"Yakui frame stress")
        .api_version(vk::API_VERSION_1_2);
    let flags = if cfg!(target_os = "macos") {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::empty()
    };
    let validation_features = [vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION];
    let mut validation_info =
        vk::ValidationFeaturesEXT::default().enabled_validation_features(&validation_features);
    let instance = entry
        .create_instance(
            &vk::InstanceCreateInfo::default()
                .application_info(&app)
                .flags(flags)
                .enabled_layer_names(&layers)
                .enabled_extension_names(&extensions)
                .push_next(&mut validation_info),
            None,
        )
        .unwrap();
    let debug = ash::ext::debug_utils::Instance::new(&entry, &instance);
    let messenger = debug
        .create_debug_utils_messenger(
            &vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(validation)),
            None,
        )
        .unwrap();
    let physical = instance.enumerate_physical_devices().unwrap()[0];
    let family = instance
        .get_physical_device_queue_family_properties(physical)
        .iter()
        .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .unwrap() as u32;
    let properties = instance.get_physical_device_properties(physical);
    println!(
        "Device: {}",
        CStr::from_ptr(properties.device_name.as_ptr()).to_string_lossy()
    );
    let priorities = [1.0];
    let queues = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(family)
        .queue_priorities(&priorities)];
    let mut indexing = vk::PhysicalDeviceDescriptorIndexingFeatures::default()
        .descriptor_binding_partially_bound(true)
        .descriptor_binding_sampled_image_update_after_bind(true);
    let mut timeline =
        vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);
    let mut device_extensions = Vec::new();
    if cfg!(target_os = "macos") {
        device_extensions.push(ash::khr::portability_subset::NAME.as_ptr());
    }
    let device = instance
        .create_device(
            physical,
            &vk::DeviceCreateInfo::default()
                .queue_create_infos(&queues)
                .enabled_extension_names(&device_extensions)
                .push_next(&mut indexing)
                .push_next(&mut timeline),
            None,
        )
        .unwrap();
    let queue = device.get_device_queue(family, 0);
    let context = VulkanContext::new(
        &device,
        queue,
        instance.get_physical_device_memory_properties(physical),
        properties,
    );
    let attachments = [vk::AttachmentDescription::default()
        .format(vk::Format::R8G8B8A8_UNORM)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)];
    let references = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let subpasses = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&references)];
    let dependencies = [vk::SubpassDependency::default()
        .src_subpass(0)
        .dst_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
    let pass = device
        .create_render_pass(
            &vk::RenderPassCreateInfo::default()
                .attachments(&attachments)
                .subpasses(&subpasses)
                .dependencies(&dependencies),
            None,
        )
        .unwrap();
    let pool = device
        .create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )
        .unwrap();
    let frames: Vec<_> = (0..3).map(|_| frame(&context, pool, pass)).collect();
    let mut gate_type =
        vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
    let gate = device
        .create_semaphore(
            &vk::SemaphoreCreateInfo::default().push_next(&mut gate_type),
            None,
        )
        .unwrap();
    let mut renderer = YakuiVulkan::new(
        &context,
        Options {
            render_pass: pass,
            ..Default::default()
        },
    );
    let mut yak = Yakui::new();
    yak.set_surface_size(Vec2::splat(64.0));
    yak.set_unscaled_viewport(Rect::from_pos_size(Vec2::ZERO, Vec2::splat(64.0)));
    renderer.set_paint_limits(&context, &mut yak);
    for batch in 0..112 {
        for (slot, frame) in frames.iter().enumerate() {
            let number = batch * 3 + slot;
            device
                .reset_command_buffer(frame.cmd, vk::CommandBufferResetFlags::empty())
                .unwrap();
            device
                .begin_command_buffer(frame.cmd, &vk::CommandBufferBeginInfo::default())
                .unwrap();
            // Reusing the offscreen target/readback needs a GPU dependency too;
            // waiting on its fence makes reuse possible but does not specify the
            // next render-pass layout transition or transfer memory accesses.
            device.cmd_pipeline_barrier(
                frame.cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(
                        vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
                    )
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)],
                &[],
                &[],
            );
            let right = color(number + 1);
            let texture = renderer.create_user_texture(
                &context,
                VulkanTextureCreateInfo::new(
                    right.to_vec(),
                    vk::Format::R8G8B8A8_UNORM,
                    vk::Extent2D {
                        width: 1,
                        height: 1,
                    },
                    vk::Filter::NEAREST,
                    vk::Filter::NEAREST,
                    vk::SamplerAddressMode::CLAMP_TO_EDGE,
                ),
            );
            let mut retired = vec![texture];
            for i in 0..4 {
                // Three odd R8 uploads followed by a larger one exercise aligned
                // offsets and staging growth on reused frame slots.
                let (width, height) = if batch == 1 && i == 3 {
                    (513, 513)
                } else {
                    (3, 1)
                };
                retired.push(renderer.create_user_texture(
                    &context,
                    VulkanTextureCreateInfo::new(
                        vec![255; width * height],
                        vk::Format::R8_UNORM,
                        vk::Extent2D {
                            width: width as u32,
                            height: height as u32,
                        },
                        vk::Filter::NEAREST,
                        vk::Filter::NEAREST,
                        vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    ),
                ));
            }
            let left = color(number);
            yak.start();
            yak.dom().do_widget::<Quad>(Quad {
                color: left.map(|c| c as f32 / 255.0),
                large: batch == 1,
                ..Default::default()
            });
            yak.finish();
            let first = yak.paint();
            renderer.transfer(first, &context, frame.cmd);
            let clear = [vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.0; 4] },
            }];
            device.cmd_begin_render_pass(
                frame.cmd,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(pass)
                    .framebuffer(frame.framebuffer)
                    .render_area(EXTENT.into())
                    .clear_values(&clear),
                vk::SubpassContents::INLINE,
            );
            device.cmd_set_viewport(
                frame.cmd,
                0,
                &[vk::Viewport {
                    width: 64.0,
                    height: 64.0,
                    max_depth: 1.0,
                    ..Default::default()
                }],
            );
            device.cmd_set_scissor(frame.cmd, 0, &[EXTENT.into()]);
            renderer.paint(first, &context, frame.cmd, EXTENT);
            yak.start();
            yak.dom().do_widget::<Quad>(Quad {
                color: [1.0; 4],
                right: true,
                texture: Some(texture),
                ..Default::default()
            });
            yak.finish();
            renderer.paint(yak.paint(), &context, frame.cmd, EXTENT);
            device.cmd_end_render_pass(frame.cmd);
            device.cmd_copy_image_to_buffer(
                frame.cmd,
                frame.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                frame.readback,
                &[vk::BufferImageCopy::default()
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_extent(EXTENT.into())],
            );
            device.cmd_pipeline_barrier(
                frame.cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)],
                &[],
                &[],
            );
            device.end_command_buffer(frame.cmd).unwrap();
            for texture in retired {
                assert!(renderer.remove_user_texture(texture));
            }
            // Hold GPU execution until all three frames have been recorded. A
            // shared geometry buffer would deterministically corrupt this batch.
            let waits = [batch as u64 + 1];
            let mut values =
                vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&waits);
            let submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&frame.cmd))
                .wait_semaphores(std::slice::from_ref(&gate))
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::TOP_OF_PIPE])
                .push_next(&mut values);
            device.queue_submit(queue, &[submit], frame.fence).unwrap();
            renderer.transfers_submitted();
        }
        device
            .signal_semaphore(
                &vk::SemaphoreSignalInfo::default()
                    .semaphore(gate)
                    .value(batch as u64 + 1),
            )
            .unwrap();
        let fences: Vec<_> = frames.iter().map(|frame| frame.fence).collect();
        device.wait_for_fences(&fences, true, u64::MAX).unwrap();
        for (slot, frame) in frames.iter().enumerate() {
            let pixels = std::slice::from_raw_parts(frame.pixels, 64 * 64 * 4);
            let number = batch * 3 + slot;
            assert_eq!(
                &pixels[(32 * 64 + 16) * 4..][..4],
                &color(number),
                "left, frame {number}"
            );
            assert_eq!(
                &pixels[(32 * 64 + 48) * 4..][..4],
                &color(number + 1),
                "right, frame {number}"
            );
            renderer.transfers_finished(&context);
        }
        device.reset_fences(&fences).unwrap();
    }
    renderer.cleanup(&device);
    for frame in frames {
        device.destroy_fence(frame.fence, None);
        device.destroy_framebuffer(frame.framebuffer, None);
        device.destroy_image_view(frame.view, None);
        device.destroy_image(frame.image, None);
        device.free_memory(frame.image_memory, None);
        device.unmap_memory(frame.readback_memory);
        device.destroy_buffer(frame.readback, None);
        device.free_memory(frame.readback_memory, None);
    }
    device.destroy_semaphore(gate, None);
    device.destroy_command_pool(pool, None);
    device.destroy_render_pass(pass, None);
    device.destroy_device(None);
    debug.destroy_debug_utils_messenger(messenger, None);
    instance.destroy_instance(None);
    assert_eq!(
        VALIDATION_ERRORS.load(Ordering::Relaxed),
        0,
        "Vulkan validation errors"
    );
    println!("Passed: 336 frames, 3 in flight, 672 paints, 1680 textures, >10 MiB geometry growth, zero validation errors.");
}

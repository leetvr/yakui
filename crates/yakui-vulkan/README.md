A Vulkan renderer for [yakui](https://github.com/SecondHalfGames/yakui), using `ash`.
The application owns the Vulkan device, queue, command buffers, submission and
completion fences. `YakuiVulkan` owns its graphics pipeline, descriptors, textures
and reusable frame resources.

Pass an existing render pass/subpass in `Options`, or enable dynamic rendering and
set `Options::dynamic_rendering_format`. The application begins rendering, sets its
viewport/scissor and controls attachment layouts. There is no requirement to draw
to a swapchain image or use the presentation layout.

The renderer requires Vulkan 1.2 and these enabled descriptor indexing features:
`descriptorBindingPartiallyBound` and
`descriptorBindingSampledImageUpdateAfterBind`. The descriptor array holds up to
1,000 live or pending-retirement textures. Removed textures reclaim their slots
only after GPU completion. Buffers currently use persistently mapped,
host-visible, host-coherent memory.

Memory uses lazy_vulkan's offset-allocator/aligned binding approach, adapted from
leetvr/lazy_vulkan `33aec6209289d3dd8ee684276b9f15b6a904fc21` (MIT notice retained
in LAZY-VULKAN-LICENSE-MIT). Geometry, staging buffers and textures suballocate
shared blocks per Vulkan memory type. Buffer and optimal-image pools are separate.
Blocks start at 16 MiB and grow to fit larger resources; mandatory dedicated
allocations are honored. There is no lazy_vulkan renderer/context dependency or
fixed multi-gigabyte slab. The host continues to own submission and completion.

Retired images release their ranges after frame completion. Growing unused
buffers return their old ranges after destruction; completed frame buffers remain
allocated for reuse. Free block space is reused and backing blocks stay allocated
until cleanup. `memory_stats()` reports backing allocations/reserved bytes and
live bindings (including resources still awaiting completion), excluding imported
host image memory.

## Frame ownership

Record uploads outside rendering, then one or more paints inside rendering:

```rust,ignore
unsafe {
    renderer.transfer(yak.paint(), &context, command_buffer);
    // Begin a render pass or dynamic rendering and set viewport/scissor.
    renderer.paint(yak.paint(), &context, command_buffer, resolution);
    // End rendering and finish/submit the command buffer on context.queue.
}
renderer.transfers_submitted();
```

Each submission owns its geometry and staging buffers exclusively. Multiple
`paint()` calls within the same frame receive separate geometry buffers. Buffers
grow as necessary; completed resource slots retain their capacity for reuse.

After the host waits on the oldest frame's fence or timeline value:

```rust,ignore
unsafe { renderer.transfers_finished(&context); }
```

Despite the historical method names, this completion notification must cover
**all transfers and draws** recorded for that frame. Submit and retire frames in
order on the context's queue. Do not resubmit command buffers referencing retired
resources. No empty submissions are needed to initialize the renderer.

Texture removals/replacements retire their Vulkan handles and descriptor slots
with the current frame, conservatively covering earlier submitted uses and any
pending uploads. `remove_user_texture()` follows the same rule. `from_image()`
takes ownership of the supplied image, allocation and view. `borrowed_view()`
registers a host-owned sampled image without taking those handles. Keep the image
and view alive until all frames using the registration, including its removal
frame, have completed. Yakui still owns and retires its sampler/descriptor slot.

Call `cleanup()` before destroying the device. It waits for device idle and
releases all owned resources. Do not use the renderer afterwards.

See `examples/demo.rs` for a windowed application. Run the real-device regression
check with Khronos validation installed:

```sh
cargo run -p yakui-vulkan --example frame_stress
```

It checks rendered pixels with three gated frames in flight, multiple paints per
frame, geometry growth beyond 10 MiB, clipped/offscreen draws, straight and
premultiplied alpha, borrowed image ownership, aligned odd-sized R8 uploads, staging growth,
and more texture allocations than the descriptor pool's capacity. It also checks
backing allocations stabilize after warmup and cleanup releases all memory. It enables
synchronization validation and fails on any validation error.

Paint opacity is carried independently of color alpha and is applied to the final
premultiplied fragment color. The renderer blends with `ONE, ONE_MINUS_SRC_ALPHA`
for both color and alpha. Yakui-managed textures preserve their declared alpha
convention; mark user data with `VulkanTextureCreateInfo::premultiplied(true)`
when RGB already includes alpha.

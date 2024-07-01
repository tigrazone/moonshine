// abstraction for GPU commands

const std = @import("std");
const vk = @import("vulkan");

const core = @import("./core.zig");
const VulkanContext = core.VulkanContext;
const VkAllocator = core.Allocator;
const vk_helpers = core.vk_helpers;

pool: vk.CommandPool,
buffer: VulkanContext.CommandBuffer,

const Self = @This();

pub fn create(vc: *const VulkanContext, name: [*:0]const u8) !Self {
    const pool = try vc.device.createCommandPool(&.{
        .queue_family_index = vc.physical_device.queue_family_index,
        .flags = .{
            .transient_bit = true,
        },
    }, null);
    errdefer vc.device.destroyCommandPool(pool, null);

    var buffer: vk.CommandBuffer = undefined;
    try vc.device.allocateCommandBuffers(&.{
        .level = vk.CommandBufferLevel.primary,
        .command_pool = pool,
        .command_buffer_count = 1,
    }, @ptrCast(&buffer));

    try vk_helpers.setDebugName(vc, buffer, name);

    return Self {
        .pool = pool,
        .buffer = VulkanContext.CommandBuffer.init(buffer, vc.device_dispatch),
    };
}

pub fn destroy(self: *Self, vc: *const VulkanContext) void {
    vc.device.destroyCommandPool(self.pool, null);
}

// start recording work
pub fn begin(self: *Self) !void {
    try self.buffer.beginCommandBuffer(&.{
        .flags = .{
            .one_time_submit_bit = true,
        },
    });
}

// submit recorded work
pub fn submit(self: *Self, vc: *const VulkanContext) !void {
    try self.buffer.endCommandBuffer();

    const submit_info = vk.SubmitInfo2 {
        .command_buffer_info_count = 1,
        .p_command_buffer_infos = @ptrCast(&vk.CommandBufferSubmitInfo {
            .command_buffer = self.buffer.handle,
            .device_mask = 0,
        }),
        .wait_semaphore_info_count = 0,
        .p_wait_semaphore_infos = undefined,
        .signal_semaphore_info_count = 0,
        .p_signal_semaphore_infos = undefined,
    };

    try vc.queue.submit2(1, @ptrCast(&submit_info), .null_handle);
}

pub fn submitAndIdleUntilDone(self: *Self, vc: *const VulkanContext) !void {
    try self.submit(vc);
    try self.idleUntilDone(vc);
}

// must be called at some point if you want a guarantee your work is actually done
pub fn idleUntilDone(self: *Self, vc: *const VulkanContext) !void {
    try vc.queue.waitIdle();
    try vc.device.resetCommandPool(self.pool, .{});
}

pub fn uploadDataToImage(self: *Self, dst_image: vk.Image, src_data: VkAllocator.HostBuffer(u8), extent: vk.Extent2D, dst_layout: vk.ImageLayout) void {
    self.buffer.pipelineBarrier2(&vk.DependencyInfo {
        .image_memory_barrier_count = 1,
        .p_image_memory_barriers = @ptrCast(&vk.ImageMemoryBarrier2 {
            .dst_stage_mask = .{ .copy_bit = true },
            .dst_access_mask = .{ .transfer_write_bit = true },
            .old_layout = .undefined,
            .new_layout = .transfer_dst_optimal,
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .image = dst_image,
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = vk.REMAINING_ARRAY_LAYERS,
            },
        }),
    });
    self.copyBufferToImage(src_data.handle, dst_image, .transfer_dst_optimal, extent);
    self.buffer.pipelineBarrier2(&vk.DependencyInfo {
        .image_memory_barrier_count = 1,
        .p_image_memory_barriers = @ptrCast(&vk.ImageMemoryBarrier2 {
            .src_stage_mask = .{ .copy_bit = true },
            .src_access_mask = .{ .transfer_write_bit = true },
            .old_layout = .transfer_dst_optimal,
            .new_layout = dst_layout,
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .image = dst_image,
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = vk.REMAINING_ARRAY_LAYERS,
            },
        }),
    });
}

// buffers must have appropriate flags
pub fn copyBuffer(self: *Self, vc: *const VulkanContext, dst: vk.Buffer, src: vk.Buffer, regions: []const vk.BufferCopy) void {
    vc.device.cmdCopyBuffer(self.buffer, src, dst, @intCast(regions.len), regions.ptr);
}

// buffers must have appropriate flags
// uploads whole host buffer to gpu buffer
pub fn uploadBuffer(self: *Self, comptime T: type, dst: VkAllocator.DeviceBuffer(T), src: VkAllocator.HostBuffer(T)) void {
    const region = vk.BufferCopy {
        .src_offset = 0,
        .dst_offset = 0,
        .size = src.sizeInBytes(),
    };

    self.buffer.copyBuffer(src.handle, dst.handle, 1, @ptrCast(&region));
}

pub fn updateBuffer(self: *Self, comptime T: type, dst: VkAllocator.DeviceBuffer(T), src: []const T, offset: vk.DeviceSize) void {
    const bytes = std.mem.sliceAsBytes(src);
    self.buffer.updateBuffer(dst.handle, offset * @sizeOf(T), bytes.len, src.ptr);
}

// size in number of data to duplicate, not bytes
pub fn fillBuffer(self: *Self, dst: vk.Buffer, size: vk.DeviceSize, data: anytype) void {
    if (@sizeOf(@TypeOf(data)) != 4) @compileError("fill buffer requires data to be 4 bytes");
    self.buffer.fillBuffer(dst, 0, size * 4, @bitCast(data));
}

pub fn clearColorImage(self: *Self, dst: vk.Image, layout: vk.ImageLayout, color: vk.ClearColorValue) void {
    self.buffer.clearColorImage(dst, layout, &color, 1, &[1]vk.ImageSubresourceRange{
        .{
            .aspect_mask = .{ .color_bit = true },
            .base_mip_level = 0,
            .level_count = vk.REMAINING_MIP_LEVELS,
            .base_array_layer = 0,
            .layer_count = vk.REMAINING_ARRAY_LAYERS,
        }
    });
}

pub fn buildAccelerationStructures(self: *Self, infos: []const vk.AccelerationStructureBuildGeometryInfoKHR, build_range_infos: []const [*]const vk.AccelerationStructureBuildRangeInfoKHR) void {
    std.debug.assert(infos.len == build_range_infos.len);
    self.buffer.buildAccelerationStructuresKHR(@intCast(infos.len), infos.ptr, build_range_infos.ptr);
}

pub fn copyImageToBuffer(self: *Self, src: vk.Image, layout: vk.ImageLayout, extent: vk.Extent2D, dst: vk.Buffer) void {
    const copy = vk.BufferImageCopy {
        .buffer_offset = 0,
        .buffer_row_length = 0,
        .buffer_image_height = 0,
        .image_subresource = .{
            .aspect_mask = .{ .color_bit = true },
            .mip_level = 0,
            .base_array_layer = 0,
            .layer_count = 1,
        },
        .image_offset = .{
            .x = 0,
            .y = 0,
            .z = 0,
        },
        .image_extent = .{
            .width = extent.width,
            .height = extent.height,
            .depth = 1,
        },
    };
    self.buffer.copyImageToBuffer(src, layout, dst, 1, @ptrCast(&copy));
}

pub fn copyBufferToImage(self: *Self, src: vk.Buffer, dst: vk.Image, layout: vk.ImageLayout, extent: vk.Extent2D) void {
    const copy = vk.BufferImageCopy {
        .buffer_offset = 0,
        .buffer_row_length = 0,
        .buffer_image_height = 0,
        .image_subresource = .{
            .aspect_mask = .{ .color_bit = true },
            .mip_level = 0,
            .base_array_layer = 0,
            .layer_count = 1,
        },
        .image_offset = .{
            .x = 0,
            .y = 0,
            .z = 0,
        },
        .image_extent = .{
            .width = extent.width,
            .height = extent.height,
            .depth = 1,
        },
    };
    self.buffer.copyBufferToImage(src, dst, layout, 1, @ptrCast(&copy));
}
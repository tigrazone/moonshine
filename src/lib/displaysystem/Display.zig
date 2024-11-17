const std = @import("std");
const vk = @import("vulkan");

const engine = @import("../engine.zig");
const core = engine.core;
const VulkanContext = core.VulkanContext;
const DestructionQueue = core.DestructionQueue;
const VkAllocator = core.Allocator;
const Encoder = core.Encoder;
const vk_helpers = core.vk_helpers;

const Swapchain = engine.displaysystem.Swapchain;

const metrics = @import("build_options").vk_metrics;

// DOUBLE BUFFER STRATEGY
// This uses the strategy that I think is most decent while being the simplest to implement.
//
// We have two "frames in flight", frame A and frame B, which cycle.
// While frame A is being processed by the GPU and displayed, we are recording frame B.
// Essentially this means that while frame A is doing GPU work, frame B is doing CPU work.
//
// This means that for e.g., vertex animation, we don't need to keep two separate GPU vertex buffers,
// as just one GPU task is being done at a time. We just queue the update in command buffer B via a
// buffer copy or update command, which doesn't affect the work of command buffer A.
//
// The catch here is that we must keep two copies of non-static GPU-accesible host data, as when we are updating
// the data for command buffer B, command buffer A could be using that data in another operation.
// In many cases, this can be avoided by using vkCmdUpdateBuffer rather than a transfer operation.
pub const frames_in_flight = 2;

const Self = @This();

frames: [frames_in_flight]Frame,
frame_index: u8,

swapchain: Swapchain,

timestamp_period: if (metrics) f32 else void,
last_frame_time_ns: if (metrics) f64 else void,

// uses initial_extent as the render extent -- that is, the buffer that is actually being rendered into, irrespective of window size
// then during rendering the render buffer is blitted into the swapchain images
pub fn create(vc: *const VulkanContext, initial_extent: vk.Extent2D, surface: vk.SurfaceKHR) !Self {
    var swapchain = try Swapchain.create(vc, initial_extent, surface);
    errdefer swapchain.destroy(vc);

    var frames: [frames_in_flight]Frame = undefined;
    inline for (&frames, 0..) |*frame, i| {
        frame.* = try Frame.create(vc, std.fmt.comptimePrint("frame {}", .{i}));
    }
    try vc.device.resetFences(1, @ptrCast(&frames[0].fence));

    const timestamp_period = if (metrics) blk: {
        var properties = vk.PhysicalDeviceProperties2 {
            .properties = undefined,
        };

        vc.instance.getPhysicalDeviceProperties2(vc.physical_device.handle, &properties);

        break :blk properties.properties.limits.timestamp_period;
    } else {};

    return Self {
        .swapchain = swapchain,
        .frames = frames,
        .frame_index = 0,

        .timestamp_period = timestamp_period,
        .last_frame_time_ns = if (metrics) 0.0 else {},
    };
}

pub fn destroy(self: *Self, vc: *const VulkanContext) void {
    self.swapchain.destroy(vc);
    inline for (&self.frames) |*frame| {
        frame.destroy(vc);
    }
}

pub fn startFrame(self: *Self, vc: *const VulkanContext) !*Encoder {
    const frame = &self.frames[self.frame_index];

    _ = try self.swapchain.acquireNextImage(vc, frame.image_acquired);

    try frame.encoder.begin();

    if (metrics) frame.encoder.buffer.writeTimestamp2(.{ .top_of_pipe_bit = true }, frame.query_pool, 0);

    return &frame.encoder;
}

pub fn recreate(self: *Self, vc: *const VulkanContext, new_extent: vk.Extent2D) !vk.SwapchainKHR {
    return try self.swapchain.recreate(vc, new_extent);
}

pub fn endFrame(self: *Self, vc: *const VulkanContext) !vk.Result {
    const result = blk: {
        const frame = self.frames[self.frame_index];

        if (metrics) frame.encoder.buffer.writeTimestamp2(.{ .bottom_of_pipe_bit = true }, frame.query_pool, 1);

        try frame.encoder.submit(vc.queue, .{
            .wait_semaphore_infos = &[_]vk.SemaphoreSubmitInfoKHR {
                .{
                    .semaphore = frame.image_acquired,
                    .value = 0,
                    .stage_mask = .{ .color_attachment_output_bit = true },
                    .device_index = 0,
                }
            },
            .signal_semaphore_infos = &[_]vk.SemaphoreSubmitInfoKHR {
                .{
                    .semaphore = frame.command_completed,
                    .value = 0,
                    .stage_mask =  .{ .color_attachment_output_bit = true },
                    .device_index = 0,
                }
            },
            .fence = frame.fence,
        });

        break :blk self.swapchain.present(vc, frame.command_completed);
    };

    self.frame_index = (self.frame_index + 1) % frames_in_flight;

    // wait for next frame to ensure CPU is not too far ahead of GPU
    var next_frame = &self.frames[self.frame_index];
    _ = try vc.device.waitForFences(1, @ptrCast(&next_frame.fence), vk.TRUE, std.math.maxInt(u64));

    // collect metrics if enabled
    if (metrics) {
        var timestamps: [2]u64 = undefined;
        const query_result = try vc.device.getQueryPoolResults(next_frame.query_pool, 0, 2, 2 * @sizeOf(u64), &timestamps, @sizeOf(u64), .{.@"64_bit" = true });
        self.last_frame_time_ns = if (query_result == .success) @as(f64, @floatFromInt(timestamps[1] - timestamps[0])) * self.timestamp_period else std.math.nan(f64);
    }

    // reset resources associated with next frame so it is ready for use on next startFrame
    try next_frame.reset(vc);

    return result;
}

const Frame = struct {
    image_acquired: vk.Semaphore,
    command_completed: vk.Semaphore,
    fence: vk.Fence,

    encoder: Encoder,

    query_pool: if (metrics) vk.QueryPool else void,

    fn create(vc: *const VulkanContext, name: [*:0]const u8) !Frame {
        const image_acquired = try vc.device.createSemaphore(&.{}, null);
        errdefer vc.device.destroySemaphore(image_acquired, null);

        const command_completed = try vc.device.createSemaphore(&.{}, null);
        errdefer vc.device.destroySemaphore(command_completed, null);

        const fence = try vc.device.createFence(&.{
            .flags = .{ .signaled_bit = true },
        }, null);

        var encoder = try Encoder.create(vc, name);
        errdefer encoder.destroy(vc);

        const query_pool = if (metrics) try vc.device.createQueryPool(&.{
            .query_type = .timestamp,
            .query_count = 2,
        }, null) else undefined;
        errdefer if (metrics) vc.device.destroyQueryPool(query_pool, null);
        if (metrics) vc.device.resetQueryPool(query_pool, 0, 2);

        return Frame {
            .image_acquired = image_acquired,
            .command_completed = command_completed,
            .fence = fence,

            .encoder = encoder,

            .query_pool = query_pool,
        };
    }

    // frame must not be in use
    fn reset(self: *Frame, vc: *const VulkanContext) !void {
        try vc.device.resetFences(1, @ptrCast(&self.fence));

        if (metrics) {
            vc.device.resetQueryPool(self.query_pool, 0, 2);
        }
        try vc.device.resetCommandPool(self.encoder.pool, .{});
        self.encoder.clearResources(vc);
    }

    fn destroy(self: *Frame, vc: *const VulkanContext) void {
        vc.device.destroySemaphore(self.image_acquired, null);
        vc.device.destroySemaphore(self.command_completed, null);
        vc.device.destroyFence(self.fence, null);
        self.encoder.destroy(vc);
        if (metrics) vc.device.destroyQueryPool(self.query_pool, null);
    }
};

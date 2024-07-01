const std = @import("std");
const vk = @import("vulkan");
const shaders = @import("shaders");

const engine = @import("../engine.zig");
const core = engine.core;
const VulkanContext = core.VulkanContext;
const VkAllocator = core.Allocator;
const Encoder = core.Encoder;

const hrtsystem = engine.hrtsystem;
const Pipeline = hrtsystem.pipeline.ObjectPickPipeline;
const Sensor = core.Sensor;
const Camera = hrtsystem.Camera;

const F32x2 = @import("../vector.zig").Vec2(f32);

const Self = @This();

const ClickDataShader = extern struct {
    instance_index: i32, // -1 if clicked background
    geometry_index: u32,
    primitive_index: u32,
    barycentrics: F32x2,

    pub fn toClickedObject(self: ClickDataShader) ?ClickedObject {
        if (self.instance_index == -1) {
            return null;
        } else {
            return ClickedObject {
                .instance_index = @intCast(self.instance_index),
                .geometry_index = self.geometry_index,
                .primitive_index = self.primitive_index,
                .barycentrics = self.barycentrics,
            };
        }
    }
};

pub const ClickedObject = struct {
    instance_index: u32,
    geometry_index: u32,
    primitive_index: u32,
    barycentrics: F32x2,
};

buffer: VkAllocator.HostBuffer(ClickDataShader),
pipeline: Pipeline,

encoder: Encoder,
ready_fence: vk.Fence,

pub fn create(vc: *const VulkanContext, vk_allocator: *VkAllocator, allocator: std.mem.Allocator, transfer_encoder: Encoder) !Self {
    const buffer = try vk_allocator.createHostBuffer(vc, ClickDataShader, 1, .{ .storage_buffer_bit = true });
    errdefer buffer.destroy(vc);

    var pipeline = try Pipeline.create(vc, vk_allocator, allocator, transfer_encoder, .{}, .{}, .{});
    errdefer pipeline.destroy(vc);

    const encoder = try Encoder.create(vc, "object picker");
    errdefer encoder.destroy(vc);

    const ready_fence = try vc.device.createFence(&.{
        .flags = .{},
    }, null);
    errdefer vc.device.destroyFence(ready_fence, null);

    return Self {
        .buffer = buffer,
        .pipeline = pipeline,

        .encoder = encoder,
        .ready_fence = ready_fence,
    };
}

pub fn getClickedObject(self: *Self, vc: *const VulkanContext, normalized_coords: F32x2, camera: Camera, accel: vk.AccelerationStructureKHR, sensor: Sensor) !?ClickedObject {
    // begin
    try self.encoder.begin();

    // bind pipeline + sets
    self.pipeline.recordBindPipeline(self.encoder.buffer);
    self.pipeline.recordPushDescriptors(self.encoder.buffer, Pipeline.PushSetBindings {
        .tlas = accel,
        .output_image = .{ .view = sensor.image.view },
        .click_data = self.buffer.handle,
    });

    self.pipeline.recordPushConstants(self.encoder.buffer, .{ .lens = camera.lenses.items[0], .click_position = normalized_coords });

    // trace rays
    self.pipeline.recordTraceRays(self.encoder.buffer, vk.Extent2D { .width = 1, .height = 1 });

    // end
    try self.encoder.submit(vc.queue, .{ .fence = self.ready_fence });

    _ = try vc.device.waitForFences(1, @ptrCast(&self.ready_fence), vk.TRUE, std.math.maxInt(u64));
    try vc.device.resetFences(1, @ptrCast(&self.ready_fence));
    try vc.device.resetCommandPool(self.encoder.pool, .{});

    return self.buffer.data[0].toClickedObject();
}

pub fn destroy(self: *Self, vc: *const VulkanContext) void {
    self.buffer.destroy(vc);
    self.pipeline.destroy(vc);
    self.encoder.destroy(vc);
    vc.device.destroyFence(self.ready_fence, null);
}

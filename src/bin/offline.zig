const std = @import("std");
const vk = @import("vulkan");

const engine = @import("engine");

const core = engine.core;
const VulkanContext = core.VulkanContext;
const Encoder = core.Encoder;
const Pipeline = engine.hrtsystem.pipeline.PathTracing;
const Scene = engine.hrtsystem.Scene;

const vk_helpers = core.vk_helpers;
const exr = engine.fileformats.exr;

const vector = engine.vector;
const F32x3 = vector.Vec3(f32);
const Mat3x4 = vector.Mat3x4(f32);

const Config = struct {
    in_filepath: []const u8, // must be gltf/glb
    out_filepath: []const u8, // must be exr
    skybox_filepath: []const u8, // must be exr
    spp: u32,
    extent: vk.Extent2D,

    fn fromCli(allocator: std.mem.Allocator) !Config {
        const args = try std.process.argsAlloc(allocator);
        defer std.process.argsFree(allocator, args);
        if (args.len < 4) return error.BadArgs;

        const in_filepath = args[1];
        if (!std.mem.eql(u8, std.fs.path.extension(in_filepath), ".glb") and !std.mem.eql(u8, std.fs.path.extension(in_filepath), ".gltf")) return error.OnlySupportsGltfInput;

        const skybox_filepath = args[2];
        if (!std.mem.eql(u8, std.fs.path.extension(skybox_filepath), ".exr")) return error.OnlySupportsExrSkybox;

        const out_filepath = args[3];
        if (!std.mem.eql(u8, std.fs.path.extension(out_filepath), ".exr")) return error.OnlySupportsExrOutput;

        const spp = if (args.len > 4) try std.fmt.parseInt(u32, args[4], 10) else 16;

        return Config {
            .in_filepath = try allocator.dupe(u8, in_filepath),
            .out_filepath = try allocator.dupe(u8, out_filepath),
            .skybox_filepath = try allocator.dupe(u8, skybox_filepath),
            .spp = spp,
            .extent = vk.Extent2D { .width = 1280, .height = 720 }, // TODO: cli
        };
    }

    fn destroy(self: Config, allocator: std.mem.Allocator) void {
        allocator.free(self.in_filepath);
        allocator.free(self.out_filepath);
        allocator.free(self.skybox_filepath);
    }
};

const IntervalLogger = struct {
    last_time: std.time.Instant,

    fn start() !IntervalLogger {
        return IntervalLogger {
            .last_time = try std.time.Instant.now(),
        };
    }

    fn log(self: *IntervalLogger, state: []const u8) !void {
        const new_time = try std.time.Instant.now();
        const elapsed = new_time.since(self.last_time);
        const ms = elapsed / std.time.ns_per_ms;
        const s = ms / std.time.ms_per_s;
        try std.io.getStdOut().writer().print("{}.{:0>3} seconds to {s}\n", .{ s, ms, state });
        self.last_time = new_time;
    }
};

pub const required_vulkan_functions = engine.hrtsystem.required_vulkan_functions;

pub fn main() !void {
    var logger = try IntervalLogger.start();

    var gpa = std.heap.GeneralPurposeAllocator(.{}) {};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = try Config.fromCli(allocator);
    defer config.destroy(allocator);

    const context = try VulkanContext.create(allocator, "offline", &.{}, &engine.hrtsystem.required_device_extensions, &engine.hrtsystem.required_device_features, null);
    defer context.destroy(allocator);

    var encoder = try Encoder.create(&context, "main");
    defer encoder.destroy(&context);

    try logger.log("set up initial state");

    try encoder.begin();
    var scene = try Scene.fromGltfExr(&context, allocator, &encoder, config.in_filepath, config.skybox_filepath, config.extent);
    defer scene.destroy(&context, allocator);
    try encoder.submitAndIdleUntilDone(&context);

    try logger.log("load world");

    try encoder.begin();
    var pipeline = try Pipeline.create(&context, allocator, &encoder, .{ scene.world.materials.textures.descriptor_layout.handle, scene.world.constant_specta.descriptor_layout.handle }, .{
        .max_bounces = 1024,
        .env_samples_per_bounce = 1,
        .mesh_samples_per_bounce = 1,
    }, .{ scene.background.sampler });
    defer pipeline.destroy(&context);
    try encoder.submitAndIdleUntilDone(&context);

    try logger.log("create pipeline");

    const output_buffer = try core.mem.DownloadBuffer([4]f32).create(&context, scene.camera.sensors.items[0].extent.width * scene.camera.sensors.items[0].extent.height, "output");
    defer output_buffer.destroy(&context);

    // actual ray tracing
    {
        try encoder.begin();

        // prepare our stuff
        scene.camera.sensors.items[0].recordPrepareForCapture(encoder.buffer, .{ .ray_tracing_shader_bit_khr = true }, .{});

        // bind our stuff
        pipeline.recordBindPipeline(encoder.buffer);
        pipeline.recordBindAdditionalDescriptorSets(encoder.buffer, .{ scene.world.materials.textures.descriptor_set, scene.world.constant_specta.descriptor_set });
        pipeline.recordPushDescriptors(encoder.buffer, scene.pushDescriptors(0, 0));

        for (0..config.spp) |sample_count| {
            // push our stuff
            pipeline.recordPushConstants(encoder.buffer, .{ .lens = scene.camera.lenses.items[0], .sample_count = scene.camera.sensors.items[0].sample_count });

            // trace our stuff
            pipeline.recordTraceRays(encoder.buffer, scene.camera.sensors.items[0].extent);

            // if not last invocation, need barrier cuz we write to images
            if (sample_count != config.spp) {
                encoder.buffer.pipelineBarrier2(&vk.DependencyInfo {
                    .image_memory_barrier_count = 1,
                    .p_image_memory_barriers = &[_]vk.ImageMemoryBarrier2 {
                        .{
                            .src_stage_mask = .{ .ray_tracing_shader_bit_khr = true },
                            .src_access_mask = if (sample_count == 0) .{ .shader_storage_write_bit = true } else .{ .shader_storage_write_bit = true, .shader_storage_read_bit = true },
                            .dst_stage_mask = .{ .ray_tracing_shader_bit_khr = true },
                            .dst_access_mask = .{ .shader_storage_write_bit = true, .shader_storage_read_bit = true },
                            .old_layout = .general,
                            .new_layout = .general,
                            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
                            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
                            .image = scene.camera.sensors.items[0].image.handle,
                            .subresource_range = .{
                                .aspect_mask = .{ .color_bit = true },
                                .base_mip_level = 0,
                                .level_count = 1,
                                .base_array_layer = 0,
                                .layer_count = vk.REMAINING_ARRAY_LAYERS,
                            },
                        }
                    },
                });
            }
            scene.camera.sensors.items[0].sample_count += 1;
        }

        // copy our stuff
        scene.camera.sensors.items[0].recordPrepareForCopy(encoder.buffer, .{ .ray_tracing_shader_bit_khr = true }, .{ .copy_bit = true });

        // copy rendered image to host-visible staging buffer
        encoder.copyImageToBuffer(scene.camera.sensors.items[0].image.handle, .transfer_src_optimal, scene.camera.sensors.items[0].extent, output_buffer.handle);

        try encoder.submitAndIdleUntilDone(&context);
    }

    try logger.log("render");

    // now done with GPU stuff/all rendering; can write from output buffer to exr
    try exr.helpers.Rgba2D.save(exr.helpers.Rgba2D { .ptr = output_buffer.slice.ptr, .extent = scene.camera.sensors.items[0].extent }, allocator, config.out_filepath);

    try logger.log("write exr");
}

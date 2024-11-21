const std = @import("std");
const vk = @import("vulkan");

const engine = @import("engine");

const core = engine.core;
const VulkanContext = core.VulkanContext;
const Encoder = core.Encoder;
const Pipeline = engine.hrtsystem.pipeline.PathTracing;
const Scene = engine.hrtsystem.Scene;
const World = engine.hrtsystem.World;
const MeshManager = engine.hrtsystem.MeshManager;
const MaterialManager = engine.hrtsystem.MaterialManager;
const TextureManager = MaterialManager.TextureManager;
const Accel = engine.hrtsystem.Accel;
const Camera = engine.hrtsystem.Camera;
const Background = engine.hrtsystem.BackgroundManager;

const exr = engine.fileformats.exr;
const Rgba2D = exr.helpers.Rgba2D;

const vector = engine.vector;
const F32x4 = vector.Vec4(f32);
const F32x3 = vector.Vec3(f32);
const F32x2 = vector.Vec2(f32);
const U32x3 = vector.Vec3(u32);
const Mat3x4 = vector.Mat3x4(f32);

const vk_helpers = engine.core.vk_helpers;

const TestingContext = struct {
    vc: VulkanContext,
    encoder: Encoder,
    output_buffer: core.mem.DownloadBuffer([4]f32),

    fn create(allocator: std.mem.Allocator, extent: vk.Extent2D) !TestingContext {
        const vc = try VulkanContext.create(allocator, "engine-tests", &.{}, &engine.hrtsystem.required_device_extensions, &engine.hrtsystem.required_device_features, null);
        errdefer vc.destroy(allocator);

        var encoder = try Encoder.create(&vc, "main");
        errdefer encoder.destroy(&vc);

        const output_buffer = try core.mem.DownloadBuffer([4]f32).create(&vc, extent.width * extent.height, "output");
        errdefer output_buffer.destroy(&vc);

        return TestingContext {
            .vc = vc,
            .encoder = encoder,
            .output_buffer = output_buffer,
        };
    }

    fn renderToOutput(self: *TestingContext, pipeline: *const Pipeline, scene: *const Scene, spp: usize) !void {
        try self.encoder.begin();

        // prepare our stuff
        scene.camera.sensors.items[0].recordPrepareForCapture(self.encoder.buffer, .{ .ray_tracing_shader_bit_khr = true }, .{});

        // bind our stuff
        pipeline.recordBindPipeline(self.encoder.buffer);
        pipeline.recordBindAdditionalDescriptorSets(self.encoder.buffer, .{ scene.world.materials.textures.descriptor_set, scene.world.constant_specta.descriptor_set });
        pipeline.recordPushDescriptors(self.encoder.buffer, scene.pushDescriptors(0, 0));

        for (0..spp) |sample_count| {
            // push our stuff
            pipeline.recordPushConstants(self.encoder.buffer, .{ .lens = scene.camera.lenses.items[0], .sample_count = scene.camera.sensors.items[0].sample_count });

            // trace our stuff
            pipeline.recordTraceRays(self.encoder.buffer, scene.camera.sensors.items[0].extent);

            // if not last invocation, need barrier cuz we write to images
            if (sample_count != spp) {
                self.encoder.buffer.pipelineBarrier2(&vk.DependencyInfo {
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
        scene.camera.sensors.items[0].recordPrepareForCopy(self.encoder.buffer, .{ .ray_tracing_shader_bit_khr = true }, .{ .copy_bit = true });

        // copy output image to host-visible staging buffer
        self.encoder.copyImageToBuffer(scene.camera.sensors.items[0].image.handle, .transfer_src_optimal, scene.camera.sensors.items[0].extent, self.output_buffer.handle);

        try self.encoder.submitAndIdleUntilDone(&self.vc);
    }

    fn destroy(self: *TestingContext, allocator: std.mem.Allocator) void {
        self.output_buffer.destroy(&self.vc);
        self.encoder.destroy(&self.vc);
        self.vc.destroy(allocator);
    }
};

// creates a unit icosphere placed at the origin with the specified subdivision level
//
// http://blog.andreaskahler.com/2009/06/creating-icosphere-mesh-in-code.html
// https://observablehq.com/@mourner/fast-icosphere-mesh
fn icosphere(order: usize, allocator: std.mem.Allocator, encoder: *Encoder, reverse_winding_order: bool) !MeshManager.Mesh {
    const Subdivider = struct {
        const MidpointCache = std.AutoArrayHashMapUnmanaged(u64, u32);

        const Self = @This();

        cache: MidpointCache,
        positions: std.ArrayListUnmanaged(F32x3),
        triangles: std.ArrayListUnmanaged(U32x3),

        allocator: std.mem.Allocator,

        fn init(gpa: std.mem.Allocator, initial_positions: []const F32x3, initial_triangles: []const U32x3) std.mem.Allocator.Error!Self {
            return Self {
              .cache = MidpointCache {},
              .positions = std.ArrayListUnmanaged(F32x3).fromOwnedSlice(try gpa.dupe(F32x3, initial_positions)),
              .triangles = std.ArrayListUnmanaged(U32x3).fromOwnedSlice(try gpa.dupe(U32x3, initial_triangles)),
              .allocator = gpa,
            };
        }

        fn deinit(self: *Self) void {
            self.cache.deinit(self.allocator);
            self.positions.deinit(self.allocator);
            self.triangles.deinit(self.allocator);
        }

        fn subdivide(self: *Self) std.mem.Allocator.Error!void {
            var next_triangles = std.ArrayListUnmanaged(U32x3) {};
            for (self.triangles.items) |triangle| {
                const a = try self.get_midpoint(triangle.x, triangle.y);
                const b = try self.get_midpoint(triangle.y, triangle.z);
                const c = try self.get_midpoint(triangle.z, triangle.x);

                try next_triangles.append(self.allocator, U32x3.new(triangle.x, a, c));
                try next_triangles.append(self.allocator, U32x3.new(triangle.y, b, a));
                try next_triangles.append(self.allocator, U32x3.new(triangle.z, c, b));
                try next_triangles.append(self.allocator, U32x3.new(a, b, c));
            }
            self.triangles.deinit(self.allocator);
            self.triangles = next_triangles;
        }

        fn get_midpoint(self: *Self, index1: u32, index2: u32) std.mem.Allocator.Error!u32 {
            const smaller = if (index1 < index2) index1 else index2;
            const greater = if (index1 < index2) index2 else index1;
            const key = (@as(u64, @intCast(smaller)) << 32) + greater;

            if (self.cache.get(key)) |x| {
                return x;
            }  else {
                const point1 = self.positions.items[index1];
                const point2 = self.positions.items[index2];
                const midpoint = point1.add(point2).div_scalar(2.0);
                try self.positions.append(self.allocator, midpoint);

                const new_index: u32 = @intCast(self.positions.items.len - 1);
                try self.cache.put(self.allocator, key, new_index);

                return new_index;
            }
        }
    };

    // create icosahedron
    const t = (1.0 + std.math.sqrt(5.0)) / 2.0;

    const initial_positions = [12]F32x3 {
        F32x3.new(-1,  t,  0),
        F32x3.new( 1,  t,  0),
        F32x3.new(-1, -t,  0),
        F32x3.new( 1, -t,  0),
        F32x3.new( 0, -1,  t),
        F32x3.new( 0,  1,  t),
        F32x3.new( 0, -1, -t),
        F32x3.new( 0,  1, -t),
        F32x3.new( t,  0, -1),
        F32x3.new( t,  0,  1),
        F32x3.new(-t,  0, -1),
        F32x3.new(-t,  0,  1),
    };

    const initial_triangles = [20]U32x3 {
        U32x3.new(0, 11, 5),
        U32x3.new(0, 5, 1),
        U32x3.new(0, 1, 7),
        U32x3.new(0, 7, 10),
        U32x3.new(0, 10, 11),
        U32x3.new(1, 5, 9),
        U32x3.new(5, 11, 4),
        U32x3.new(11, 10, 2),
        U32x3.new(10, 7, 6),
        U32x3.new(7, 1, 8),
        U32x3.new(3, 9, 4),
        U32x3.new(3, 4, 2),
        U32x3.new(3, 2, 6),
        U32x3.new(3, 6, 8),
        U32x3.new(3, 8, 9),
        U32x3.new(4, 9, 5),
        U32x3.new(2, 4, 11),
        U32x3.new(6, 2, 10),
        U32x3.new(8, 6, 7),
        U32x3.new(9, 8, 1),
    };

    var subdivider = try Subdivider.init(allocator, &initial_positions, &initial_triangles);
    defer subdivider.deinit();

    for (0..order) |_| {
        try subdivider.subdivide();
    }

    const positions = try encoder.uploadAllocator().dupe(F32x3, subdivider.positions.items);

    for (positions) |*position| {
        position.* = position.unit();
    }

    const indices = try encoder.uploadAllocator().dupe(U32x3, subdivider.triangles.items);
    if (reverse_winding_order) {
        for (indices) |*index| {
            index.* = U32x3.new(index.z, index.y, index.x);
        }
    }

    const mesh = MeshManager.Mesh {
        .name = "icosphere",
        .positions = encoder.upload_allocator.getBufferSlice(positions),
        .normals = null, // TODO: add normals here when normals are robust enough
        .texcoords = null,
        .indices = encoder.upload_allocator.getBufferSlice(indices),
    };
    return mesh;
}

// TODO: use actual statistical tests

// some furnace tests
//
// theoretically any shape works for the furnace test
// the reason to use a sphere (rather than e.g., a box or pyramid with less geometric complexity)
// is that a sphere is the simplest shape that will test the BRDF with all incoming directions
// this is technically an argument for supporting primitives other than triangles,
// if the goal is just to test the BRDF in the most comprehensive way

// used so that we can care less for less perceptable error
fn luminance(color: [3]f32) f32 {
    return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2];
}

// spectral causes a decent amount of color variance so our per-pixel error bounds
// are fairly high.
// however, we additionally make sure that the total image average is 1
fn assertWhiteFurnaceImage(image: []const [4]f32) !void {
    var average: f64 = 0.0;
    for (image) |pixel| {
        const val = luminance(pixel[0..3].*);
        average += val / @as(f64, @floatFromInt(image.len));
        if (!std.math.approxEqAbs(f32, val, 1.0, 0.3)) return error.NonWhitePixel;
    }
    if (!std.math.approxEqAbs(f64, average, 1.0, 0.001)) return error.NonWhiteAverage;
}

test "white sphere on white background is white" {
    const allocator = std.testing.allocator;
    const extent = vk.Extent2D { .width = 32, .height = 32 };
    var tc = try TestingContext.create(allocator, extent);
    defer tc.destroy(allocator);

    try tc.encoder.begin();
    var world = try World.createEmpty(&tc.vc, allocator, &tc.encoder);

    // add sphere to world
    {
        const mesh_handle = try world.meshes.upload(&tc.vc, allocator, &tc.encoder, try icosphere(5, allocator, &tc.encoder, false));

        const normal: *F32x2 = @ptrCast(try tc.encoder.uploadAllocator().alignedAlloc(u8, vk_helpers.texelBlockSize(vk_helpers.typeToFormat(F32x2)), @sizeOf(F32x2)));
        normal.* = MaterialManager.Material.default_normal;
        const normal_texture = try world.materials.textures.upload(&tc.vc, F32x2, allocator, &tc.encoder, tc.encoder.upload_allocator.getBufferSlice(normal), vk.Extent2D { .width = 1, .height = 1 }, "");

        const albedo: *F32x4 = @ptrCast(try tc.encoder.uploadAllocator().alignedAlloc(u8, vk_helpers.texelBlockSize(vk_helpers.typeToFormat(F32x4)), @sizeOf(F32x4)));
        albedo.* = F32x4.new(1, 1, 1, std.math.nan(f32));
        const albedo_texture = try world.materials.textures.upload(&tc.vc, F32x4, allocator, &tc.encoder, tc.encoder.upload_allocator.getBufferSlice(albedo), vk.Extent2D { .width = 1, .height = 1 }, "");

        const emissive: *F32x4 = @ptrCast(try tc.encoder.uploadAllocator().alignedAlloc(u8, vk_helpers.texelBlockSize(vk_helpers.typeToFormat(F32x4)), @sizeOf(F32x4)));
        emissive.* = F32x4.new(0, 0, 0, std.math.nan(f32));
        const emissive_texture = try world.materials.textures.upload(&tc.vc, F32x4, allocator, &tc.encoder, tc.encoder.upload_allocator.getBufferSlice(emissive), vk.Extent2D { .width = 1, .height = 1 }, "");

        const material_handle = try world.materials.upload(&tc.vc, allocator, &tc.encoder, MaterialManager.Material {
            .normal = normal_texture,
            .emissive = emissive_texture,
            .bsdf = MaterialManager.PolymorphicBSDF {
                .lambert = MaterialManager.Lambert {
                    .color = albedo_texture,
                }
            }
        }, "white");

        _ = try world.accel.uploadInstance(&tc.vc, allocator, &tc.encoder, world.meshes, world.materials, Accel.Instance {
            .visible = true,
            .transform = Mat3x4.identity,
            .geometries = &[1]Accel.Geometry {
                .{
                    .material = material_handle,
                    .mesh = mesh_handle,
                }
            },
        });
    }

    var camera = Camera {};
    _ = try camera.appendLens(allocator, Camera.Lens {
        .origin = F32x3.new(-3, 0, 0),
        .forward = F32x3.new(1, 0, 0),
        .up = F32x3.new(0, 0, 1),
        .vfov = std.math.pi / 4.0,
        .aperture = 0,
        .focus_distance = 1,
    });
    _ = try camera.appendSensor(&tc.vc, allocator, extent);

    var background = try Background.create(&tc.vc, allocator);
    var white = [4]f32 {1, 1, 1, 1};
    const image = Rgba2D {
        .ptr = @ptrCast(&white),
        .extent = .{
            .width = 1,
            .height = 1,
        }
    };
    try background.addBackground(&tc.vc, allocator, &tc.encoder, image, "white");

    var scene = Scene {
        .world = world,
        .camera = camera,
        .background = background,
    };
    defer scene.destroy(&tc.vc, allocator);

    var pipeline = try Pipeline.create(&tc.vc, allocator, &tc.encoder, .{ scene.world.materials.textures.descriptor_layout.handle, scene.world.constant_specta.descriptor_layout.handle }, .{
        .max_bounces = 1024,
        .env_samples_per_bounce = 0,
        .mesh_samples_per_bounce = 0,
    }, .{ scene.background.sampler });
    defer pipeline.destroy(&tc.vc);
    try tc.encoder.submitAndIdleUntilDone(&tc.vc);

    try tc.renderToOutput(&pipeline, &scene, 512);
    try assertWhiteFurnaceImage(tc.output_buffer.slice);

    // do that again but with env sampling
    try tc.encoder.begin();
    const other_pipeline = try pipeline.recreate(&tc.vc, allocator, &tc.encoder, .{
        .max_bounces = 1024,
        .env_samples_per_bounce = 1,
        .mesh_samples_per_bounce = 0,
    });
    defer tc.vc.device.destroyPipeline(other_pipeline, null);
    try tc.encoder.submitAndIdleUntilDone(&tc.vc);

    try tc.renderToOutput(&pipeline, &scene, 512);
    try assertWhiteFurnaceImage(tc.output_buffer.slice);
}

test "inside illuminating sphere is white" {
    const allocator = std.testing.allocator;

    const extent = vk.Extent2D { .width = 32, .height = 32 };
    var tc = try TestingContext.create(allocator, extent);
    defer tc.destroy(allocator);

    try tc.encoder.begin();
    var world = try World.createEmpty(&tc.vc, allocator, &tc.encoder);

    // add sphere to world
    {
        const mesh_handle = try world.meshes.upload(&tc.vc, allocator, &tc.encoder, try icosphere(5, allocator, &tc.encoder, true));

        const normal: *F32x2 = @ptrCast(try tc.encoder.uploadAllocator().alignedAlloc(u8, vk_helpers.texelBlockSize(vk_helpers.typeToFormat(F32x2)), @sizeOf(F32x2)));
        normal.* = MaterialManager.Material.default_normal;
        const normal_texture = try world.materials.textures.upload(&tc.vc, F32x2, allocator, &tc.encoder, tc.encoder.upload_allocator.getBufferSlice(normal), vk.Extent2D { .width = 1, .height = 1 }, "");

        const albedo: *F32x4 = @ptrCast(try tc.encoder.uploadAllocator().alignedAlloc(u8, vk_helpers.texelBlockSize(vk_helpers.typeToFormat(F32x4)), @sizeOf(F32x4)));
        albedo.* = F32x4.new(0.5, 0.5, 0.5, std.math.nan(f32));
        const albedo_texture = try world.materials.textures.upload(&tc.vc, F32x4, allocator, &tc.encoder, tc.encoder.upload_allocator.getBufferSlice(albedo), vk.Extent2D { .width = 1, .height = 1 }, "");

        const emissive: *F32x4 = @ptrCast(try tc.encoder.uploadAllocator().alignedAlloc(u8, vk_helpers.texelBlockSize(vk_helpers.typeToFormat(F32x4)), @sizeOf(F32x4)));
        emissive.* = F32x4.new(0.5, 0.5, 0.5, std.math.nan(f32));
        const emissive_texture = try world.materials.textures.upload(&tc.vc, F32x4, allocator, &tc.encoder, tc.encoder.upload_allocator.getBufferSlice(emissive), vk.Extent2D { .width = 1, .height = 1 }, "");

        const material_handle = try world.materials.upload(&tc.vc, allocator, &tc.encoder, MaterialManager.Material {
            .normal = normal_texture,
            .emissive = emissive_texture,
            .bsdf = MaterialManager.PolymorphicBSDF {
                .lambert = MaterialManager.Lambert {
                    .color = albedo_texture,
                }
            }
        }, "white");

        _ = try world.accel.uploadInstance(&tc.vc, allocator, &tc.encoder, world.meshes, world.materials, Accel.Instance {
            .visible = true,
            .transform = Mat3x4.identity,
            .geometries = &[1]Accel.Geometry {
                .{
                    .material = material_handle,
                    .mesh = mesh_handle,
                }
            },
        });
    }

    var camera = Camera {};
    _ = try camera.appendLens(allocator, Camera.Lens {
        .origin = F32x3.new(0, 0, 0),
        .forward = F32x3.new(1, 0, 0),
        .up = F32x3.new(0, 0, 1),
        .vfov = std.math.pi / 3.0,
        .aperture = 0,
        .focus_distance = 1,
    });
    _ = try camera.appendSensor(&tc.vc, allocator, extent);

    var background = try Background.create(&tc.vc, allocator);
    var black = [4]f32 {0, 0, 0, 1};
    const image = Rgba2D {
        .ptr = @ptrCast(&black),
        .extent = .{
            .width = 1,
            .height = 1,
        }
    };
    try background.addBackground(&tc.vc, allocator, &tc.encoder, image, "black");

    var scene = Scene {
        .world = world,
        .camera = camera,
        .background = background,
    };
    defer scene.destroy(&tc.vc, allocator);

    var pipeline = try Pipeline.create(&tc.vc, allocator, &tc.encoder, .{ scene.world.materials.textures.descriptor_layout.handle, scene.world.constant_specta.descriptor_layout.handle }, .{
        .max_bounces = 1024,
        .env_samples_per_bounce = 0,
        .mesh_samples_per_bounce = 0,
    }, .{ scene.background.sampler });
    defer pipeline.destroy(&tc.vc);

    try tc.encoder.submitAndIdleUntilDone(&tc.vc);

    try tc.renderToOutput(&pipeline, &scene, 1024);
    try assertWhiteFurnaceImage(tc.output_buffer.slice);

    try tc.encoder.begin();
    // do that again but with mesh sampling
    const other_pipeline = try pipeline.recreate(&tc.vc, allocator, &tc.encoder, .{
        .max_bounces = 1024,
        .env_samples_per_bounce = 0,
        .mesh_samples_per_bounce = 1,
    });
    defer tc.vc.device.destroyPipeline(other_pipeline, null);
    try tc.encoder.submitAndIdleUntilDone(&tc.vc);

    try tc.renderToOutput(&pipeline, &scene, 1024);
    try assertWhiteFurnaceImage(tc.output_buffer.slice);
}

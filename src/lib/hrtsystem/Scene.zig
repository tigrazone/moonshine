const std = @import("std");
const vk = @import("vulkan");
const Gltf = @import("zgltf");

const engine = @import("../engine.zig");

const core = engine.core;
const VulkanContext = core.VulkanContext;
const Encoder = core.Encoder;

const Background = @import("./BackgroundManager.zig");
const World = @import("./World.zig");
const Camera = @import("./Camera.zig");

const exr = engine.fileformats.exr;

const Self = @This();

world: World,
background: Background,
camera: Camera,

// glTF doesn't correspond very well to the internal data structures here so this is very inefficient
// also very inefficient because it's written very inefficiently, can remove a lot of copying, but that's a problem for another time
// inspection bool specifies whether some buffers should be created with the `transfer_src_flag` for inspection
pub fn fromGltfExr(vc: *const VulkanContext, allocator: std.mem.Allocator, encoder: *Encoder, gltf_filepath: []const u8, skybox_filepath: []const u8, extent: vk.Extent2D) !Self {
    var gltf = Gltf.init(allocator);
    defer gltf.deinit();

    const buffer = try std.fs.cwd().readFileAllocOptions(
        allocator,
        gltf_filepath,
        std.math.maxInt(usize),
        null,
        4,
        null
    );
    defer allocator.free(buffer);
    try gltf.parse(buffer);

    const camera_create_info = try Camera.Lens.fromGltf(gltf);
    var camera = Camera {};
    errdefer camera.destroy(vc, allocator);
    _ = try camera.appendLens(allocator, camera_create_info);
    _ = try camera.appendSensor(vc, allocator, extent);

    var world = try World.fromGltf(vc, allocator, encoder, gltf, std.fs.path.dirname(gltf_filepath));
    errdefer world.destroy(vc, allocator);

    var background = try Background.create(vc, allocator);
    errdefer background.destroy(vc, allocator);
    {
        const skybox_image = try exr.helpers.Rgba2D.load(allocator, skybox_filepath);
        defer allocator.free(skybox_image.asSlice());
        try background.addBackground(vc, allocator, encoder, skybox_image, "exr");
    }

    return Self {
        .world = world,
        .background = background,
        .camera = camera,
    };
}

pub fn pushDescriptors(self: *const Self, sensor: u32, background: u32) engine.hrtsystem.pipeline.StandardBindings {
    return engine.hrtsystem.pipeline.StandardBindings {
        .tlas = self.world.accel.tlas_handle,
        .instances = self.world.accel.instances_device.handle,
        .world_to_instances = self.world.accel.world_to_instance_device.handle,
        .meshes = self.world.meshes.addresses_buffer.handle,
        .geometries = self.world.accel.geometries.handle,
        .material_values = self.world.materials.materials.handle,
        .triangle_power_image = .{ .view = self.world.accel.triangle_powers.view },
        .triangle_meta = self.world.accel.triangle_powers_meta.handle,
        .geometry_to_triangle_power_offset = self.world.accel.geometry_to_triangle_power_offset.handle,
        .emissive_triangle_count = self.world.accel.emissive_triangle_count.handle,
        .background_rgb_image = .{ .view = self.background.data.items[background].rgb_image.view },
        .background_luminance_image = .{ .view = self.background.data.items[background].luminance_image.view },
        .output_image = .{ .view = self.camera.sensors.items[sensor].image.view },
    };
}

pub fn destroy(self: *Self, vc: *const VulkanContext, allocator: std.mem.Allocator) void {
    self.world.destroy(vc, allocator);
    self.background.destroy(vc, allocator);
    self.camera.destroy(vc, allocator);
}

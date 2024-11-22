const std = @import("std");
const vk = @import("vulkan");
const Gltf = @import("zgltf");

const engine = @import("../engine.zig");
const core = engine.core;
const VulkanContext = core.VulkanContext;
const Encoder = core.Encoder;

const Sensor = core.Sensor;

const vector = @import("../vector.zig");
const F32x2 = vector.Vec2(f32);
const F32x3 = vector.Vec3(f32);
const F32x4 = vector.Vec4(f32);
const Mat3x4 = vector.Mat3x4(f32);

pub const Lens = extern struct {
    origin: F32x3,
    forward: F32x3,
    up: F32x3,
    vfov: f32, // radians
    vfov_tan: f32,
    aperture: f32,
    focus_distance: f32,
    u: F32x3,
    v: F32x3,

    pub fn fromGltf(gltf: Gltf) !Lens {
        // just use first camera found in nodes, if none found, use an arbitrary default
        const yfov, const transform = for (gltf.data.nodes.items) |node| {
            if (node.camera) |camera| {
                const yfov = gltf.data.cameras.items[camera].type.perspective.yfov;
                const mat = Gltf.getGlobalTransform(&gltf.data, node);
                // convert to Z-up
                const transform = Mat3x4.new(
                    F32x4.new(mat[0][0], mat[1][0], mat[2][0], mat[3][0]),
                    F32x4.new(mat[0][2], mat[1][2], mat[2][2], mat[3][2]),
                    F32x4.new(mat[0][1], mat[1][1], mat[2][1], mat[3][1]),
                );
                break .{ yfov, transform };
            }
        } else .{ std.math.pi / 6.0, Mat3x4.new(
            F32x4.new(1, 0, 0, 0),
            F32x4.new(0, 0, 1, 5), // looking at origin
            F32x4.new(0, 1, 0, 0),
        ) };

        const up = transform.mul_vec(F32x3.new(0.0, 1.0,  0.0)).unit();
        const w  = transform.mul_vec(F32x3.new(0.0, 0.0, -1.0)).unit();
        const u  = up.cross(w).unit();
        const v  = u.cross(w);

        return Lens {
            .origin = transform.mul_point(F32x3.new(0.0, 0.0, 0.0)),
            .forward = w,
            .up = up,
            .vfov = yfov,
            .vfov_tan = std.math.tan(yfov / 2),
            .aperture = 0.0,
            .focus_distance = 1.0,
            .u = u,
            .v = v,
        };
    }

    pub fn prepareCameraPreCalcs(self: Lens) Lens {
        var newLens : Lens = self;

        newLens.vfov_tan = std.math.tan(self.vfov / 2);
        newLens.u = self.up.cross(self.forward).unit();
        newLens.v = newLens.u.cross(self.forward);
        return newLens;
    }

    // should correspond to GPU-side generateRay
    pub fn directionFromUv(self: Lens, uv: F32x2, aspect: f32) F32x3 {
        const w = self.forward;
        const u = self.u;
        const v = self.v;

        const h = self.vfov_tan;
        const viewport_height = 2 * h * self.focus_distance;
        const viewport_width = aspect * viewport_height;

        const horizontal = u.mul_scalar(viewport_width);
        const vertical = v.mul_scalar(viewport_height);

        const lower_left_corner = self.origin.sub(horizontal.div_scalar(2)).sub(vertical.div_scalar(2)).add(w.mul_scalar(self.focus_distance));

        return (lower_left_corner.add(horizontal.mul_scalar(uv.x)).add(vertical.mul_scalar(uv.y)).sub(self.origin)).unit();
    }
};

sensors: std.ArrayListUnmanaged(Sensor) = .{},
lenses: std.ArrayListUnmanaged(Lens) = .{},

const Self = @This();

pub const SensorHandle = u32;
pub fn appendSensor(self: *Self, vc: *const VulkanContext, allocator: std.mem.Allocator, extent: vk.Extent2D) !SensorHandle {
    var buf: [32]u8 = undefined;
    const name = try std.fmt.bufPrintZ(&buf, "render {}", .{self.sensors.items.len});

    try self.sensors.append(allocator, try Sensor.create(vc, extent, name));
    return @intCast(self.sensors.items.len - 1);
}

pub const LensHandle = u32;
pub fn appendLens(self: *Self, allocator: std.mem.Allocator, lens0: Lens) !LensHandle {
    var lens = lens0;
    lens.u  = lens.up.cross(lens.forward).unit();
    lens.v  = lens.u.cross(lens.forward);
    lens.vfov_tan = std.math.tan(lens.vfov / 2);
    try self.lenses.append(allocator, lens);
    return @intCast(self.lenses.items.len - 1);
}

pub fn clearAllSensors(self: *Self) void {
    for (self.sensors.items) |*sensor| {
        sensor.clear();
    }
}

pub fn destroy(self: *Self, vc: *const VulkanContext, allocator: std.mem.Allocator) void {
    for (self.sensors.items) |*sensor| {
        sensor.destroy(vc);
    }
    self.sensors.deinit(allocator);
    self.lenses.deinit(allocator);
}

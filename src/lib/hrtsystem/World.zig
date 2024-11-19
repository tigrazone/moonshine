// a world contains:
// - meshes
// - materials
// - an acceleration structure/mesh heirarchy

const std = @import("std");
const vk = @import("vulkan");
const Gltf = @import("zgltf");
const zigimg = @import("zigimg");

const engine = @import("../engine.zig");
const core = engine.core;
const VulkanContext = core.VulkanContext;
const Encoder = core.Encoder;
const VkAllocator = core.Allocator;
const vk_helpers = core.vk_helpers;

const MaterialManager = engine.hrtsystem.MaterialManager;
const TextureManager = MaterialManager.TextureManager;

const MeshManager = engine.hrtsystem.MeshManager;
const Accel = engine.hrtsystem.Accel;
const ConstantSpectra = engine.hrtsystem.ConstantSpectra;

const vector = engine.vector;
const Mat3x4 = vector.Mat3x4(f32);
const F32x4 = vector.Vec4(f32);
const F32x3 = vector.Vec3(f32);
const F32x2 = vector.Vec2(f32);
const U32x3 = vector.Vec3(u32);
const U8x2 = vector.Vec2(u8);
const U8x4 = vector.Vec4(u8);

pub const Material = MaterialManager.Material;
pub const PolymorphicBSDF = MaterialManager.PolymorphicBSDF;
pub const Instance = Accel.Instance;
pub const Geometry = Accel.Geometry;

meshes: MeshManager,
materials: MaterialManager,

accel: Accel,

constant_specta: ConstantSpectra,

const Self = @This();

fn loadImage(allocator: std.mem.Allocator, image: Gltf.Image, gltf_directory: ?[]const u8) !zigimg.Image {
    if (image.data) |data| {
        return try zigimg.Image.fromMemory(allocator, data);
    } else if (image.uri) |uri| {
        const filepath = if (gltf_directory) |dir| try std.fs.path.join(allocator, &.{ dir, uri }) else uri;
        defer if (gltf_directory != null) allocator.free(filepath);
        return try zigimg.Image.fromFilePath(allocator, filepath);
    } else {
        return error.EmptyImage;
    }
}

// TODO: consider just uploading all textures upfront rather than as part of this function
fn gltfMaterialToMaterial(vc: *const VulkanContext, vk_allocator: *VkAllocator, allocator: std.mem.Allocator, encoder: *Encoder, gltf: Gltf, gltf_directory: ?[]const u8, gltf_material: Gltf.Material, textures: *TextureManager) !Material {
    // stuff that is in every material
    var material = blk: {
        var material: Material = undefined;
        material.normal = if (gltf_material.normal_texture) |texture| normal: {
            const image = gltf.data.images.items[gltf.data.textures.items[texture.index].source.?];

            // this gives us rgb --> need to convert to rg
            // theoretically gltf spec claims these values should already be linear
            var img = try loadImage(allocator, image, gltf_directory);
            std.debug.assert(img.pixels == .rgb24);
            defer img.deinit();

            const rg = try encoder.uploadAllocator().alloc(U8x2, img.pixels.len());
            for (img.pixels.rgb24, 0..) |pixel, i| {
                rg[i] = U8x2.new(pixel.r, pixel.g);
            }
            const debug_name = try std.fmt.allocPrintZ(allocator, "{s} normal", .{ gltf_material.name });
            defer allocator.free(debug_name);
            break :normal try textures.upload(vc, U8x2, vk_allocator, allocator, encoder, encoder.upload_allocator.getBufferSlice(rg), vk.Extent2D { .width = @intCast(img.width), .height = @intCast(img.height) }, debug_name);
        } else normal: {
            const rg: *F32x2 = @ptrCast(try encoder.uploadAllocator().alignedAlloc(u8, vk_helpers.texelBlockSize(vk_helpers.typeToFormat(F32x2)), @sizeOf(F32x2)));
            rg.* = Material.default_normal;
            break :normal try textures.upload(vc, F32x2, vk_allocator, allocator, encoder, encoder.upload_allocator.getBufferSlice(rg), vk.Extent2D { .width = 1, .height = 1 }, "default normal");
        };

        material.emissive = if (gltf_material.emissive_texture) |texture| emissive: {
            const image = gltf.data.images.items[gltf.data.textures.items[texture.index].source.?];

            var img = try loadImage(allocator, image, gltf_directory);
            defer img.deinit();

            // image may be rgba32 or rgb24
            const rgba = try encoder.uploadAllocator().alloc(U8x4, img.pixels.len());
            if (img.pixels == .rgb24) {
                for (img.pixels.rgb24, rgba) |src_pixel, *dst_pixel| {
                    dst_pixel.* = U8x4.new(src_pixel.r, src_pixel.g, src_pixel.b, std.math.maxInt(u8));
                }
            } else {
                for (img.pixels.rgba32, rgba) |src_pixel, *dst_pixel| {
                    dst_pixel.* = U8x4.new(src_pixel.r, src_pixel.g, src_pixel.b, src_pixel.a);
                }
            }

            const debug_name = try std.fmt.allocPrintZ(allocator, "{s} emissive", .{ gltf_material.name });
            defer allocator.free(debug_name);
            break :emissive try textures.upload(vc, U8x4, vk_allocator, allocator, encoder, encoder.upload_allocator.getBufferSlice(rgba), vk.Extent2D { .width = @intCast(img.width), .height = @intCast(img.height) }, debug_name);
        } else emissive: {
            const constant: *F32x4 = @ptrCast(try encoder.uploadAllocator().alignedAlloc(u8, vk_helpers.texelBlockSize(vk_helpers.typeToFormat(F32x4)), @sizeOf(F32x4)));
            constant.* = F32x4.new(gltf_material.emissive_factor[0], gltf_material.emissive_factor[1], gltf_material.emissive_factor[2], std.math.nan(f32)).mul_scalar(gltf_material.emissive_strength);
            const debug_name = try std.fmt.allocPrintZ(allocator, "{s} constant emissive {}", .{ gltf_material.name, constant });
            defer allocator.free(debug_name);
            break :emissive try textures.upload(vc, F32x4, vk_allocator, allocator, encoder, encoder.upload_allocator.getBufferSlice(constant), vk.Extent2D { .width = 1, .height = 1 }, debug_name);
        };

        break :blk material;
    };

    var standard_pbr: MaterialManager.StandardPBR = undefined;
    standard_pbr.ior = gltf_material.ior;

    if (gltf_material.transmission_factor == 1.0) {
        // infer cauchy's equation A, B constants assuming IOR
        // was measured at 560nm and material has dispersion of BK7
        const b = 0.00420; // BK7
        const b_nm = b * 1000000.0;  // μm2 to nm2
        const measured_ior_wavelength = 560.0;
        const a = standard_pbr.ior - b_nm / (measured_ior_wavelength * measured_ior_wavelength);
        material.bsdf = .{ .glass = .{
            .cauchy_a = a,
            .cauchy_b = b_nm,
        }};
        return material;
    }

    standard_pbr.color = if (gltf_material.metallic_roughness.base_color_texture) |texture| blk: {
        const image = gltf.data.images.items[gltf.data.textures.items[texture.index].source.?];

        var img = try loadImage(allocator, image, gltf_directory);
        defer img.deinit();

        // image may be rgba32 or rgb24
        const rgba = try encoder.uploadAllocator().alloc(U8x4, img.pixels.len());
        if (img.pixels == .rgb24) {
            for (img.pixels.rgb24, rgba) |src_pixel, *dst_pixel| {
                dst_pixel.* = U8x4.new(src_pixel.r, src_pixel.g, src_pixel.b, std.math.maxInt(u8));
            }
        } else {
            for (img.pixels.rgba32, rgba) |src_pixel, *dst_pixel| {
                dst_pixel.* = U8x4.new(src_pixel.r, src_pixel.g, src_pixel.b, src_pixel.a);
            }
        }

        const debug_name = try std.fmt.allocPrintZ(allocator, "{s} color", .{ gltf_material.name });
        defer allocator.free(debug_name);
        break :blk try textures.upload(vc, U8x4, vk_allocator, allocator, encoder, encoder.upload_allocator.getBufferSlice(rgba), vk.Extent2D { .width = @intCast(img.width), .height = @intCast(img.height) }, debug_name);
    } else blk: {
        const constant: *F32x4 = @ptrCast(try encoder.uploadAllocator().alignedAlloc(u8, vk_helpers.texelBlockSize(vk_helpers.typeToFormat(F32x4)), @sizeOf(F32x4)));
        constant.* = F32x4.new(gltf_material.metallic_roughness.base_color_factor[0], gltf_material.metallic_roughness.base_color_factor[1], gltf_material.metallic_roughness.base_color_factor[2], std.math.nan(f32));
        const debug_name = try std.fmt.allocPrintZ(allocator, "{s} constant color {}", .{ gltf_material.name, constant });
        defer allocator.free(debug_name);
        break :blk try textures.upload(vc, F32x4, vk_allocator, allocator, encoder, encoder.upload_allocator.getBufferSlice(constant), vk.Extent2D { .width = 1, .height = 1 }, debug_name);
    };

    if (gltf_material.metallic_roughness.metallic_roughness_texture) |texture| {
        const image = gltf.data.images.items[gltf.data.textures.items[texture.index].source.?];

        // this gives us rgb --> only need r (metallic) and g (roughness) channels
        // theoretically gltf spec claims these values should already be linear
        var img = try loadImage(allocator, image, gltf_directory);
        defer img.deinit();

        const metalness = try encoder.uploadAllocator().alloc(u8, img.pixels.len());
        const roughness = try encoder.uploadAllocator().alloc(u8, img.pixels.len());
        switch (img.pixels) {
            .rgb24 => |rgb| {
                for (rgb, metalness, roughness) |pixel, *r, *g| {
                    r.* = pixel.r;
                    g.* = pixel.g;
                }
            },
            .rgba32 => |rgba| {
                for (rgba, metalness, roughness) |pixel, *r, *g| {
                    r.* = pixel.r;
                    g.* = pixel.g;
                }
            },
            else => unreachable, // TODO
        }
        const debug_name_metalness = try std.fmt.allocPrintZ(allocator, "{s} metalness", .{ gltf_material.name });
        defer allocator.free(debug_name_metalness);
        standard_pbr.metalness = try textures.upload(vc, u8, vk_allocator, allocator, encoder, encoder.upload_allocator.getBufferSlice(metalness), vk.Extent2D { .width = @intCast(img.width), .height = @intCast(img.height) }, debug_name_metalness);
        const debug_name_roughness = try std.fmt.allocPrintZ(allocator, "{s} roughness", .{ gltf_material.name });
        defer allocator.free(debug_name_roughness);
        standard_pbr.metalness = try textures.upload(vc, u8, vk_allocator, allocator, encoder, encoder.upload_allocator.getBufferSlice(roughness), vk.Extent2D { .width = @intCast(img.width), .height = @intCast(img.height) }, debug_name_roughness);
        material.bsdf = .{ .standard_pbr = standard_pbr };
        return material;
    } else {
        if (gltf_material.metallic_roughness.metallic_factor == 0.0 and gltf_material.metallic_roughness.roughness_factor == 1.0) {
            // parse as lambert
            const lambert = MaterialManager.Lambert {
                .color = standard_pbr.color,
            };
            material.bsdf = .{ .lambert = lambert };
            return material;
        } else if (gltf_material.metallic_roughness.metallic_factor == 1.0 and gltf_material.metallic_roughness.roughness_factor == 0.0) {
            // parse as perfect mirror
            material.bsdf = .{ .perfect_mirror = {} };
            return material;
        } else {
            const debug_name_metalness = try std.fmt.allocPrintZ(allocator, "{s} constant metalness {}", .{ gltf_material.name, gltf_material.metallic_roughness.metallic_factor });
            defer allocator.free(debug_name_metalness);
            const metalness = try encoder.uploadAllocator().create(f32);
            metalness.* = gltf_material.metallic_roughness.metallic_factor;
            standard_pbr.metalness = try textures.upload(vc, f32, vk_allocator, allocator, encoder, encoder.upload_allocator.getBufferSlice(metalness), vk.Extent2D { .width = 1, .height = 1 }, debug_name_metalness);

            const debug_name_roughness = try std.fmt.allocPrintZ(allocator, "{s} constant roughness {}", .{ gltf_material.name, gltf_material.metallic_roughness.roughness_factor });
            defer allocator.free(debug_name_roughness);
            const roughness = try encoder.uploadAllocator().create(f32);
            roughness.* = gltf_material.metallic_roughness.metallic_factor;
            standard_pbr.roughness = try textures.upload(vc, f32, vk_allocator, allocator, encoder, encoder.upload_allocator.getBufferSlice(roughness), vk.Extent2D { .width = 1, .height = 1 }, debug_name_roughness);

            material.bsdf = .{ .standard_pbr = standard_pbr };
            return material;
        }
    }
}

// glTF doesn't correspond very well to the internal data structures here so this is very inefficient
// also very inefficient because it's written very inefficiently, can remove a lot of copying, but that's a problem for another time
pub fn fromGltf(vc: *const VulkanContext, vk_allocator: *VkAllocator, allocator: std.mem.Allocator, encoder: *Encoder, gltf: Gltf, gltf_directory: ?[]const u8) !Self {
    var materials = blk: {
        var materials = try MaterialManager.createEmpty(vc);
        errdefer materials.destroy(vc, allocator);

        for (gltf.data.materials.items) |material| {
            const mat = try gltfMaterialToMaterial(vc, vk_allocator, allocator, encoder, gltf, gltf_directory, material, &materials.textures);
            const namez = try allocator.dupeZ(u8, material.name);
            defer allocator.free(namez);
            _ = try materials.upload(vc, vk_allocator, allocator, encoder, mat, namez);
        }

        const default_material = try gltfMaterialToMaterial(vc, vk_allocator, allocator, encoder, gltf, gltf_directory, Gltf.Material {
            .name = "default",
        }, &materials.textures);
        _ = try materials.upload(vc, vk_allocator, allocator, encoder, default_material, "default");

        break :blk materials;
    };
    errdefer materials.destroy(vc, allocator);

    var objects = std.ArrayList(MeshManager.Mesh).init(allocator);
    defer objects.deinit();

    // go over heirarchy, finding meshes
    var instances = std.ArrayList(Instance).init(allocator);
    defer instances.deinit();
    defer for (instances.items) |instance| allocator.free(instance.geometries);

    const buffers = try allocator.alloc([]align(4) const u8, gltf.data.buffers.items.len);
    defer allocator.free(buffers);
    for (gltf.data.buffers.items, buffers) |src, *dst| {
        if (src.uri) |uri| {
            const bytes = try allocator.alignedAlloc(u8, 4, src.byte_length);
            const filepath = if (gltf_directory) |dir| try std.fs.path.join(allocator, &.{ dir, uri }) else uri;
            defer if (gltf_directory != null) allocator.free(filepath);
            _ = try std.fs.cwd().readFile(filepath, bytes);
            dst.* = bytes;
        } else {
            dst.* = gltf.glb_binary.?;
        }
    }
    defer for (buffers[(if (gltf.glb_binary != null) 1 else 0)..]) |buffer| allocator.free(buffer);

    for (gltf.data.nodes.items) |node| {
        if (node.mesh) |model_idx| {
            const mesh = gltf.data.meshes.items[model_idx];
            const geometries = try allocator.alloc(Geometry, mesh.primitives.items.len);
            for (mesh.primitives.items, geometries) |primitive, *geometry| {
                geometry.* = Geometry {
                    .mesh = @intCast(objects.items.len),
                    .material = @intCast(primitive.material orelse materials.material_count - 1),
                };
                // get indices
                const indices = if (primitive.indices) |indices_index| blk2: {
                    const accessor = gltf.data.accessors.items[indices_index];
                    const buffer = buffers[gltf.data.buffer_views.items[accessor.buffer_view.?].buffer];

                    break :blk2 switch (accessor.component_type) {
                        .unsigned_short => blk3: {
                            var indices = std.ArrayList(u16).init(allocator);
                            defer indices.deinit();

                            gltf.getDataFromBufferView(u16, &indices, accessor, buffer);

                            // convert to U32x3
                            const actual_indices = try encoder.uploadAllocator().alloc(U32x3, indices.items.len / 3);
                            for (actual_indices, 0..) |*index, i| {
                                index.* = U32x3.new(indices.items[i * 3 + 0], indices.items[i * 3 + 1], indices.items[i * 3 + 2]);
                            }
                            break :blk3 actual_indices;
                        },
                        .unsigned_integer => blk3: {
                            var indices = std.ArrayList(u32).init(encoder.uploadAllocator());
                            defer indices.deinit();

                            gltf.getDataFromBufferView(u32, &indices, accessor, buffer);

                            break :blk3 std.mem.bytesAsSlice(U32x3, std.mem.sliceAsBytes(try indices.toOwnedSlice()));
                        },
                        else => unreachable,
                    };
                } else null;
                errdefer if (indices) |nonnull| encoder.uploadAllocator().free(nonnull);

                const vertices = blk2: {
                    var positions = std.ArrayList(f32).init(encoder.uploadAllocator());
                    var texcoords = std.ArrayList(f32).init(encoder.uploadAllocator());
                    var normals = std.ArrayList(f32).init(encoder.uploadAllocator());

                    for (primitive.attributes.items) |attribute| {
                        switch (attribute) {
                            .position => |accessor_index| {
                                const accessor = gltf.data.accessors.items[accessor_index];
                                const buffer = buffers[gltf.data.buffer_views.items[accessor.buffer_view.?].buffer];
                                gltf.getDataFromBufferView(f32, &positions, accessor, buffer);
                            },
                            .texcoord => |accessor_index| {
                                const accessor = gltf.data.accessors.items[accessor_index];
                                const buffer = buffers[gltf.data.buffer_views.items[accessor.buffer_view.?].buffer];
                                gltf.getDataFromBufferView(f32, &texcoords, accessor, buffer);
                            },
                            .normal => |accessor_index| {
                                const accessor = gltf.data.accessors.items[accessor_index];
                                const buffer = buffers[gltf.data.buffer_views.items[accessor.buffer_view.?].buffer];
                                gltf.getDataFromBufferView(f32, &normals, accessor, buffer);
                            },
                            else => {},
                        }
                    }

                    const positions_slice = try positions.toOwnedSlice();
                    const texcoords_slice = try texcoords.toOwnedSlice();
                    const normals_slice = try normals.toOwnedSlice();

                    // TODO: remove ptrcast workaround below once ptrcast works on slices
                    break :blk2 .{
                        .positions = @as([*]F32x3, @ptrCast(positions_slice.ptr))[0..positions_slice.len / 3],
                        .texcoords = @as([*]F32x2, @ptrCast(texcoords_slice.ptr))[0..texcoords_slice.len / 2],
                        .normals = @as([*]F32x3, @ptrCast(normals_slice.ptr))[0..normals_slice.len / 3],
                    };
                };
                errdefer encoder.uploadAllocator().free(vertices.positions);
                errdefer encoder.uploadAllocator().free(vertices.texcoords);

                // get vertices
                try objects.append(MeshManager.Mesh {
                    .name = mesh.name,
                    .positions = encoder.upload_allocator.getBufferSlice(vertices.positions),
                    .texcoords = if (vertices.texcoords.len != 0) encoder.upload_allocator.getBufferSlice(vertices.texcoords) else null,
                    .normals = if (vertices.normals.len != 0) encoder.upload_allocator.getBufferSlice(vertices.normals) else null,
                    .indices = if (indices) |i| encoder.upload_allocator.getBufferSlice(i) else null,
                });
            }

            const mat = Gltf.getGlobalTransform(&gltf.data, node);
            // convert to Z-up
            try instances.append(Instance {
                .transform = Mat3x4.new(
                    F32x4.new(mat[0][0], mat[1][0], mat[2][0], mat[3][0]),
                    F32x4.new(mat[0][2], mat[1][2], mat[2][2], mat[3][2]),
                    F32x4.new(mat[0][1], mat[1][1], mat[2][1], mat[3][1]),
                ),
                .geometries = geometries,
            });
        }
    }

    var meshes = MeshManager {};
    errdefer meshes.destroy(vc, allocator);
    for (objects.items) |object| {
        _ = try meshes.upload(vc, vk_allocator, allocator, encoder, object);
    }

    var accel = try Accel.createEmpty(vc, vk_allocator, allocator, materials.textures.descriptor_layout, encoder);
    errdefer accel.destroy(vc, allocator);
    for (instances.items) |instance| {
        _ = try accel.uploadInstance(vc, vk_allocator, allocator, encoder, meshes, materials, instance);
    }

    return Self {
        .materials = materials,
        .meshes = meshes,
        .accel = accel,
        .constant_specta = try ConstantSpectra.create(vc, vk_allocator, encoder),
    };
}

pub fn createEmpty(vc: *const VulkanContext, vk_allocator: *VkAllocator, allocator: std.mem.Allocator, encoder: *Encoder) !Self {
    var materials = try MaterialManager.createEmpty(vc);
    errdefer materials.destroy(vc, allocator);

    return Self {
        .materials = materials,
        .meshes = .{},
        .accel = try Accel.createEmpty(vc, vk_allocator, allocator, materials.textures.descriptor_layout, encoder),
        .constant_specta = try ConstantSpectra.create(vc, vk_allocator, encoder),
    };
}

pub fn updateTransform(self: *Self, index: u32, new_transform: Mat3x4) void {
    self.accel.updateTransform(index, new_transform);
}

pub fn updateVisibility(self: *Self, index: u32, visible: bool) void {
    self.accel.updateVisibility(index, visible);
}

pub fn destroy(self: *Self, vc: *const VulkanContext, allocator: std.mem.Allocator) void {
    self.materials.destroy(vc, allocator);
    self.meshes.destroy(vc, allocator);
    self.accel.destroy(vc, allocator);
    self.constant_specta.destroy(vc);
}

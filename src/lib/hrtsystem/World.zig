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

pub const Material = MaterialManager.Material;
pub const PolymorphicBSDF = MaterialManager.PolymorphicBSDF;
pub const Instance = Accel.Instance;
pub const Geometry = Accel.Geometry;

meshes: MeshManager,
materials: MaterialManager,

accel: Accel,

constant_specta: ConstantSpectra,

const Self = @This();

// TODO: consider just uploading all textures upfront rather than as part of this function
fn gltfMaterialToMaterial(vc: *const VulkanContext, vk_allocator: *VkAllocator, allocator: std.mem.Allocator, encoder: Encoder, gltf: Gltf, gltf_material: Gltf.Material, textures: *TextureManager) !Material {
    // stuff that is in every material
    var material = blk: {
        var material: Material = undefined;
        material.normal = if (gltf_material.normal_texture) |texture| normal: {
            const image = gltf.data.images.items[gltf.data.textures.items[texture.index].source.?];
            std.debug.assert(std.mem.eql(u8, image.mime_type.?, "image/png"));

            // this gives us rgb --> need to convert to rg
            // theoretically gltf spec claims these values should already be linear
            var img = try zigimg.Image.fromMemory(allocator, image.data.?);
            defer img.deinit();

            var rg = try allocator.alloc(u8, img.pixels.len() * 2);
            defer allocator.free(rg);
            for (img.pixels.rgb24, 0..) |pixel, i| {
                rg[i * 2 + 0] = pixel.r;
                rg[i * 2 + 1] = pixel.g;
            }
            const debug_name = try std.fmt.allocPrintZ(allocator, "{s} normal", .{ gltf_material.name });
            defer allocator.free(debug_name);
            break :normal try textures.upload(vc, vk_allocator, allocator, encoder, TextureManager.Source {
                .raw = .{
                    .bytes = rg,
                    .extent = vk.Extent2D {
                        .width = @intCast(img.width),
                        .height = @intCast(img.height),
                    },
                    .format = .r8g8_unorm,
                },
            }, debug_name);
        } else try textures.upload(vc, vk_allocator, allocator, encoder, TextureManager.Source {
            .f32x2 = Material.default_normal,
        }, "default normal");

        material.emissive = if (gltf_material.emissive_texture) |texture| emissive: {
            const image = gltf.data.images.items[gltf.data.textures.items[texture.index].source.?];
            std.debug.assert(std.mem.eql(u8, image.mime_type.?, "image/png"));

            var img = try zigimg.Image.fromMemory(allocator, image.data.?);
            defer img.deinit();

            // image may be rgba32 or rgb24
            var rgba = if (img.pixels == .rgb24) try zigimg.color.PixelStorage.init(allocator, .rgba32, img.pixels.len()) else img.pixels;
            defer if (img.pixels == .rgb24) rgba.deinit(allocator);
            if (img.pixels == .rgb24) {
                for (img.pixels.rgb24, rgba.rgba32) |pixel, *rgba32| {
                    rgba32.* = zigimg.color.Rgba32.initRgba(pixel.r, pixel.g, pixel.b, std.math.maxInt(u8));
                }
            }

            const debug_name = try std.fmt.allocPrintZ(allocator, "{s} emissive", .{ gltf_material.name });
            defer allocator.free(debug_name);
            break :emissive try textures.upload(vc, vk_allocator, allocator, encoder, TextureManager.Source {
                .raw = .{
                    .bytes = rgba.asBytes(),
                    .extent = vk.Extent2D {
                        .width = @intCast(img.width),
                        .height = @intCast(img.height),
                    },
                    .format = .r8g8b8a8_srgb,
                },
            }, debug_name);
        } else emissive: {
            const constant = F32x3.new(gltf_material.emissive_factor[0], gltf_material.emissive_factor[1], gltf_material.emissive_factor[2]).mul_scalar(gltf_material.emissive_strength);
            const debug_name = try std.fmt.allocPrintZ(allocator, "{s} constant emissive {}", .{ gltf_material.name, constant });
            defer allocator.free(debug_name);
            break :emissive try textures.upload(vc, vk_allocator, allocator, encoder, TextureManager.Source {
                .f32x3 = constant,
            }, debug_name);
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
        std.debug.assert(std.mem.eql(u8, image.mime_type.?, "image/png"));

        var img = try zigimg.Image.fromMemory(allocator, image.data.?);
        defer img.deinit();

        // image may be rgba32 or rgb24
        var rgba = if (img.pixels == .rgb24) try zigimg.color.PixelStorage.init(allocator, .rgba32, img.pixels.len()) else img.pixels;
        defer if (img.pixels == .rgb24) rgba.deinit(allocator);
        if (img.pixels == .rgb24) {
            for (img.pixels.rgb24, rgba.rgba32) |pixel, *rgba32| {
                rgba32.* = zigimg.color.Rgba32.initRgba(pixel.r, pixel.g, pixel.b, std.math.maxInt(u8));
            }
        }

        const debug_name = try std.fmt.allocPrintZ(allocator, "{s} color", .{ gltf_material.name });
        defer allocator.free(debug_name);
        break :blk try textures.upload(vc, vk_allocator, allocator, encoder, TextureManager.Source {
            .raw = .{
                .bytes = rgba.asBytes(),
                .extent = vk.Extent2D {
                    .width = @intCast(img.width),
                    .height = @intCast(img.height),
                },
                .format = .r8g8b8a8_srgb,
            },
        }, debug_name);
    } else blk: {
        const constant = F32x3.new(gltf_material.metallic_roughness.base_color_factor[0], gltf_material.metallic_roughness.base_color_factor[1], gltf_material.metallic_roughness.base_color_factor[2]);
        const debug_name = try std.fmt.allocPrintZ(allocator, "{s} constant color {}", .{ gltf_material.name, constant });
        defer allocator.free(debug_name);
        break :blk try textures.upload(vc, vk_allocator, allocator, encoder, TextureManager.Source {
            .f32x3 = F32x3.new(gltf_material.metallic_roughness.base_color_factor[0], gltf_material.metallic_roughness.base_color_factor[1], gltf_material.metallic_roughness.base_color_factor[2])
        }, debug_name);
    };

    if (gltf_material.metallic_roughness.metallic_roughness_texture) |texture| {
        const image = gltf.data.images.items[gltf.data.textures.items[texture.index].source.?];
        std.debug.assert(std.mem.eql(u8, image.mime_type.?, "image/png"));

        // this gives us rgb --> only need r (metallic) and g (roughness) channels
        // theoretically gltf spec claims these values should already be linear
        var img = try zigimg.Image.fromMemory(allocator, image.data.?);
        defer img.deinit();

        const rs = try allocator.alloc(u8, img.pixels.len());
        defer allocator.free(rs);
        const gs = try allocator.alloc(u8, img.pixels.len());
        defer allocator.free(gs);
        for (img.pixels.rgb24, rs, gs) |pixel, *r, *g| {
            r.* = pixel.r;
            g.* = pixel.g;
        }
        const debug_name_metalness = try std.fmt.allocPrintZ(allocator, "{s} metalness", .{ gltf_material.name });
        defer allocator.free(debug_name_metalness);
        standard_pbr.metalness = try textures.upload(vc, vk_allocator, allocator, encoder, TextureManager.Source {
            .raw = .{
                .bytes = rs,
                .extent = vk.Extent2D {
                    .width = @intCast(img.width),
                    .height = @intCast(img.height),
                },
                .format = .r8_unorm,
            },
        }, debug_name_metalness);
        const debug_name_roughness = try std.fmt.allocPrintZ(allocator, "{s} roughness", .{ gltf_material.name });
        defer allocator.free(debug_name_roughness);
        standard_pbr.roughness = try textures.upload(vc, vk_allocator, allocator, encoder, TextureManager.Source {
            .raw = .{
                .bytes = gs,
                .extent = vk.Extent2D {
                    .width = @intCast(img.width),
                    .height = @intCast(img.height),
                },
                .format = .r8_unorm,
            },
        }, debug_name_roughness);
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
            standard_pbr.metalness = try textures.upload(vc, vk_allocator, allocator, encoder, TextureManager.Source {
                .f32x1 = gltf_material.metallic_roughness.metallic_factor,
            }, debug_name_metalness);
            const debug_name_roughness = try std.fmt.allocPrintZ(allocator, "{s} constant debug_name_roughness {}", .{ gltf_material.name, gltf_material.metallic_roughness.roughness_factor });
            defer allocator.free(debug_name_roughness);
            standard_pbr.roughness = try textures.upload(vc, vk_allocator, allocator, encoder, TextureManager.Source {
                .f32x1 = gltf_material.metallic_roughness.roughness_factor,
            }, debug_name_roughness);
            material.bsdf = .{ .standard_pbr = standard_pbr };
            return material;
        }
    }
}

// glTF doesn't correspond very well to the internal data structures here so this is very inefficient
// also very inefficient because it's written very inefficiently, can remove a lot of copying, but that's a problem for another time
// inspection bool specifies whether some buffers should be created with the `transfer_src_flag` for inspection
pub fn fromGlb(vc: *const VulkanContext, vk_allocator: *VkAllocator, allocator: std.mem.Allocator, encoder: Encoder, gltf: Gltf) !Self {
    var materials = blk: {
        var materials = try MaterialManager.createEmpty(vc);
        errdefer materials.destroy(vc, allocator);

        for (gltf.data.materials.items) |material| {
            const mat = try gltfMaterialToMaterial(vc, vk_allocator, allocator, encoder, gltf, material, &materials.textures);
            _ = try materials.upload(vc, vk_allocator, allocator, encoder, mat);
        }

        const default_material = try gltfMaterialToMaterial(vc, vk_allocator, allocator, encoder, gltf, Gltf.Material {
            .name = "default",
        }, &materials.textures);
        _ = try materials.upload(vc, vk_allocator, allocator, encoder, default_material);

        break :blk materials;
    };
    errdefer materials.destroy(vc, allocator);

    var objects = std.ArrayList(MeshManager.Mesh).init(allocator);
    defer objects.deinit();
    defer for (objects.items) |*object| object.destroy(allocator);

    // go over heirarchy, finding meshes
    var instances = std.ArrayList(Instance).init(allocator);
    defer instances.deinit();
    defer for (instances.items) |instance| allocator.free(instance.geometries);

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

                    break :blk2 switch (accessor.component_type) {
                        .unsigned_short => blk3: {
                            var indices = std.ArrayList(u16).init(allocator);
                            defer indices.deinit();

                            gltf.getDataFromBufferView(u16, &indices, accessor, gltf.glb_binary.?);

                            // convert to U32x3
                            const actual_indices = try allocator.alloc(U32x3, indices.items.len / 3);
                            for (actual_indices, 0..) |*index, i| {
                                index.* = U32x3.new(indices.items[i * 3 + 0], indices.items[i * 3 + 1], indices.items[i * 3 + 2]);
                            }
                            break :blk3 actual_indices;
                        },
                        .unsigned_integer => blk3: {
                            var indices = std.ArrayList(u32).init(allocator);
                            defer indices.deinit();

                            gltf.getDataFromBufferView(u32, &indices, accessor, gltf.glb_binary.?);

                            break :blk3 std.mem.bytesAsSlice(U32x3, std.mem.sliceAsBytes(try indices.toOwnedSlice()));
                        },
                        else => unreachable,
                    };
                } else null;
                errdefer if (indices) |nonnull| allocator.free(nonnull);

                const vertices = blk2: {
                    var positions = std.ArrayList(f32).init(allocator);
                    var texcoords = std.ArrayList(f32).init(allocator);
                    var normals = std.ArrayList(f32).init(allocator);

                    for (primitive.attributes.items) |attribute| {
                        switch (attribute) {
                            .position => |accessor_index| {
                                const accessor = gltf.data.accessors.items[accessor_index];
                                gltf.getDataFromBufferView(f32, &positions, accessor, gltf.glb_binary.?);
                            },
                            .texcoord => |accessor_index| {
                                const accessor = gltf.data.accessors.items[accessor_index];
                                gltf.getDataFromBufferView(f32, &texcoords, accessor, gltf.glb_binary.?);
                            },
                            .normal => |accessor_index| {
                                const accessor = gltf.data.accessors.items[accessor_index];
                                gltf.getDataFromBufferView(f32, &normals, accessor, gltf.glb_binary.?);
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
                errdefer allocator.free(vertices.positions);
                errdefer allocator.free(vertices.texcoords);

                // get vertices
                try objects.append(MeshManager.Mesh {
                    .positions = vertices.positions,
                    .texcoords = if (vertices.texcoords.len != 0) vertices.texcoords else null,
                    .normals = if (vertices.normals.len != 0) vertices.normals else null,
                    .indices = indices,
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

pub fn createEmpty(vc: *const VulkanContext, vk_allocator: *VkAllocator, allocator: std.mem.Allocator, encoder: Encoder) !Self {
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

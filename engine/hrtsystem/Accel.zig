const std = @import("std");
const vk = @import("vulkan");

const engine = @import("../engine.zig");
const VulkanContext = engine.core.VulkanContext;
const Commands = engine.core.Commands;
const VkAllocator = engine.core.Allocator;
const Image = engine.core.Image;
const vk_helpers = engine.core.vk_helpers;

const MeshManager = @import("./MeshManager.zig");
const MaterialManager = @import("./MaterialManager.zig");

const vector = @import("../vector.zig");
const Mat3x4 = vector.Mat3x4(f32);
const F32x3 = vector.Vec3(f32);

// "accel" perhaps the wrong name for this struct at this point, maybe "heirarchy" would be better
// the acceleration structure is the primary world heirarchy, and controls
// how all the meshes and materials fit together
//
// an acceleration structure has:
// - a list of instances
//
// each instance has:
// - a transform
// - a visible flag
// - a list of geometries
//
// each geometry (BLAS) has:
// - a mesh
// - a material

pub const Instance = struct {
    transform: Mat3x4, // transform of this instance
    visible: bool = true, // whether this instance is visible
    geometries: []const Geometry, // geometries in this instance
};

pub const Geometry = extern struct {
    mesh: u32, // idx of mesh that this geometry uses
    material: u32, // idx of material that this geometry uses
};

const BottomLevelAccels = std.MultiArrayList(struct {
    handle: vk.AccelerationStructureKHR,
    buffer: VkAllocator.DeviceBuffer(u8),
});

const TrianglePowerPipeline = engine.core.pipeline.Pipeline(.{ .shader_path = "hrtsystem/mesh_sampling/power.hlsl",
    .SpecConstants = extern struct {
        indexed_attributes: bool align(@alignOf(vk.Bool32)) = true,
        two_component_normal_texture: bool align(@alignOf(vk.Bool32)) = true,
    },
    .PushConstants = extern struct {
        instance_index: u32,
        geometry_index: u32,
        src_primitive_count: u32,
    },
    .PushSetBindings = struct {
        instances: vk.Buffer,
        world_to_instances: vk.Buffer,
        meshes: vk.Buffer,
        geometries: vk.Buffer,
        material_values: vk.Buffer,
        emissive_triangle_count: vk.Buffer,
        dst_power: engine.core.pipeline.StorageImage,
        dst_triangle_metadata: vk.Buffer,
    },
    .additional_descriptor_layout_count = 1,
});

const TrianglePowerFoldPipeline = engine.core.pipeline.Pipeline(.{ .shader_path = "hrtsystem/mesh_sampling/fold.hlsl",
    .PushSetBindings = struct {
        src_mip: engine.core.pipeline.SampledImage,
        dst_mip: engine.core.pipeline.StorageImage,
        geometry_to_triangle_power_offset: vk.Buffer,
        emissive_triangle_count: vk.Buffer,
    },
    .PushConstants = extern struct {
        geometry_index: u32,
        triangle_count: u32,
    },
});

const TriangleMetadata = extern struct {
    instance_index: u32,
    geometry_index: u32,
};

triangle_power_pipeline: TrianglePowerPipeline,
triangle_power_fold_pipeline: TrianglePowerFoldPipeline,
// TODO: this needs to be 2D and possibly aliased
triangle_powers_meta: VkAllocator.DeviceBuffer(TriangleMetadata),
triangle_powers: Image,
triangle_powers_mips: [std.math.log2(max_emissive_triangles) + 1]vk.ImageView,
geometry_to_triangle_power_offset: VkAllocator.DeviceBuffer(u32) = .{},
emissive_triangle_count: VkAllocator.DeviceBuffer(u32), // size 1

blases: BottomLevelAccels = .{},

instance_count: u32 = 0,
instances_device: VkAllocator.DeviceBuffer(vk.AccelerationStructureInstanceKHR) = .{},
instances_host: VkAllocator.HostBuffer(vk.AccelerationStructureInstanceKHR) = .{},
instances_address: vk.DeviceAddress = 0,

// keep track of inverse transform -- non-inverse we can get from instances_device
// transforms provided by shader only in hit/intersection shaders but we need them
// in raygen
// ray queries provide them in any shader which would be a benefit of using them
world_to_instance_device: VkAllocator.DeviceBuffer(Mat3x4) = .{},
world_to_instance_host: VkAllocator.HostBuffer(Mat3x4) = .{},

// flat jagged array for geometries --
// use instanceCustomIndex + GeometryID() here to get geometry
geometry_count: u24 = 0,
geometries: VkAllocator.DeviceBuffer(Geometry) = .{},

// tlas stuff
tlas_handle: vk.AccelerationStructureKHR = .null_handle,
tlas_buffer: VkAllocator.DeviceBuffer(u8) = .{},

tlas_update_scratch_buffer: VkAllocator.DeviceBuffer(u8) = .{},
tlas_update_scratch_address: vk.DeviceAddress = 0,

const Self = @This();

// TODO: resizable buffers
const max_instances = std.math.powi(u32, 2, 12) catch unreachable;
const max_geometries = std.math.powi(u32, 2, 12) catch unreachable;
const max_emissive_triangles = std.math.powi(u32, 2, 15) catch unreachable;

// lots of temp memory allocations here
// commands must be in recording state
// returns scratch buffers that must be kept alive until command is completed
fn makeBlases(vc: *const VulkanContext, vk_allocator: *VkAllocator, allocator: std.mem.Allocator, commands: *Commands, mesh_manager: MeshManager, geometries: []const []const Geometry, blases: *BottomLevelAccels) ![]const VkAllocator.OwnedDeviceBuffer {
    const build_geometry_infos = try allocator.alloc(vk.AccelerationStructureBuildGeometryInfoKHR, geometries.len);
    defer allocator.free(build_geometry_infos);
    defer for (build_geometry_infos) |build_geometry_info| allocator.free(build_geometry_info.p_geometries.?[0..build_geometry_info.geometry_count]);

    const scratch_buffers = try allocator.alloc(VkAllocator.OwnedDeviceBuffer, geometries.len);

    const build_infos = try allocator.alloc([*]vk.AccelerationStructureBuildRangeInfoKHR, geometries.len);
    defer allocator.free(build_infos);
    defer for (build_infos, build_geometry_infos) |build_info, build_geometry_info| allocator.free(build_info[0..build_geometry_info.geometry_count]);

    try blases.ensureUnusedCapacity(allocator, geometries.len);

    for (geometries, build_infos, build_geometry_infos, scratch_buffers) |list, *build_info, *build_geometry_info, *scratch_buffer| {
        const vk_geometries = try allocator.alloc(vk.AccelerationStructureGeometryKHR, list.len);

        build_geometry_info.* = vk.AccelerationStructureBuildGeometryInfoKHR {
            .type = .bottom_level_khr,
            .flags = .{ .prefer_fast_trace_bit_khr = true },
            .mode = .build_khr,
            .geometry_count = @intCast(vk_geometries.len),
            .p_geometries = vk_geometries.ptr,
            .scratch_data = undefined,
        };

        const primitive_counts = try allocator.alloc(u32, list.len);
        defer allocator.free(primitive_counts);

        build_info.* = (try allocator.alloc(vk.AccelerationStructureBuildRangeInfoKHR, list.len)).ptr;

        for (list, vk_geometries, primitive_counts, 0..) |geo, *geometry, *primitive_count, j| {
            const mesh = mesh_manager.meshes.get(geo.mesh);

            geometry.* = vk.AccelerationStructureGeometryKHR {
                .geometry_type = .triangles_khr,
                .flags = .{ .opaque_bit_khr = true },
                .geometry = .{
                    .triangles = .{
                        .vertex_format = .r32g32b32_sfloat,
                        .vertex_data = .{
                            .device_address = mesh.position_buffer.getAddress(vc),
                        },
                        .vertex_stride = @sizeOf(F32x3),
                        .max_vertex = @intCast(mesh.vertex_count - 1),
                        .index_type = .uint32,
                        .index_data = .{
                            .device_address = mesh.index_buffer.getAddress(vc),
                        },
                        .transform_data = .{
                            .device_address = 0,
                        }
                    }
                }
            };

            build_info.*[j] =  vk.AccelerationStructureBuildRangeInfoKHR {
                .primitive_count = @intCast(mesh.index_count),
                .primitive_offset = 0,
                .transform_offset = 0,
                .first_vertex = 0,
            };
            primitive_count.* = build_info.*[j].primitive_count;
        }

        const size_info = getBuildSizesInfo(vc, build_geometry_info, primitive_counts.ptr);

        scratch_buffer.* = try vk_allocator.createOwnedDeviceBuffer(vc, size_info.build_scratch_size, .{ .shader_device_address_bit = true, .storage_buffer_bit = true });
        errdefer scratch_buffer.destroy(vc);
        build_geometry_info.scratch_data.device_address = scratch_buffer.getAddress(vc);

        const buffer = try vk_allocator.createDeviceBuffer(vc, allocator, u8, size_info.acceleration_structure_size, .{ .acceleration_structure_storage_bit_khr = true, .shader_device_address_bit = true });
        errdefer buffer.destroy(vc);

        build_geometry_info.dst_acceleration_structure = try vc.device.createAccelerationStructureKHR(&.{
            .buffer = buffer.handle,
            .offset = 0,
            .size = size_info.acceleration_structure_size,
            .type = .bottom_level_khr,
        }, null);
        errdefer vc.device.destroyAccelerationStructureKHR(build_geometry_info.dst_acceleration_structure, null);

        blases.appendAssumeCapacity(.{
            .handle = build_geometry_info.dst_acceleration_structure,
            .buffer = buffer,
        });
    }

    commands.buffer.buildAccelerationStructuresKHR(@intCast(build_geometry_infos.len), build_geometry_infos.ptr, build_infos.ptr);

    return scratch_buffers;
}

pub fn createEmpty(vc: *const VulkanContext, vk_allocator: *VkAllocator, allocator: std.mem.Allocator, texture_descriptor_layout: MaterialManager.TextureManager.DescriptorLayout, commands: *Commands) !Self {
    var triangle_power_pipeline = try TrianglePowerPipeline.create(vc, allocator, .{}, .{}, .{ texture_descriptor_layout.handle });
    errdefer triangle_power_pipeline.destroy(vc);

    var triangle_power_fold_pipeline = try TrianglePowerFoldPipeline.create(vc, allocator, .{}, .{}, .{});
    errdefer triangle_power_fold_pipeline.destroy(vc);

    const triangle_powers_meta = try vk_allocator.createDeviceBuffer(vc, allocator, TriangleMetadata, max_emissive_triangles, .{ .storage_buffer_bit = true, .transfer_dst_bit = true });
    errdefer triangle_powers_meta.destroy(vc);

    const emissive_triangle_count = try vk_allocator.createDeviceBuffer(vc, allocator, u32, 1, .{ .storage_buffer_bit = true });
    errdefer emissive_triangle_count.destroy(vc);

    const triangle_powers = try Image.create(vc, vk_allocator, vk.Extent2D { .width = max_emissive_triangles, .height = 1 }, .{ .storage_bit = true, .sampled_bit = true, .transfer_dst_bit = true }, .r32_sfloat, true, "triangle powers");
    errdefer triangle_powers.destroy(vc);

    try commands.startRecording(vc);
    commands.buffer.pipelineBarrier2(&vk.DependencyInfo {
        .image_memory_barrier_count = 1,
        .p_image_memory_barriers = &[1]vk.ImageMemoryBarrier2 {
            .{
                .dst_stage_mask = .{ .clear_bit = true },
                .dst_access_mask = .{ .transfer_write_bit = true },
                .old_layout = .undefined,
                .new_layout = .transfer_dst_optimal,
                .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
                .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
                .image = triangle_powers.handle,
                .subresource_range = .{
                    .aspect_mask = .{ .color_bit = true },
                    .base_mip_level = 0,
                    .level_count = vk.REMAINING_MIP_LEVELS,
                    .base_array_layer = 0,
                    .layer_count = vk.REMAINING_ARRAY_LAYERS,
                },
            }
        },
    });
    commands.buffer.clearColorImage(triangle_powers.handle, .transfer_dst_optimal, &vk.ClearColorValue { .float_32 = .{ 0, 0, 0, 0 }}, 1, &[1]vk.ImageSubresourceRange{
        .{
            .aspect_mask = .{ .color_bit = true },
            .base_mip_level = 0,
            .level_count = vk.REMAINING_MIP_LEVELS,
            .base_array_layer = 0,
            .layer_count = vk.REMAINING_ARRAY_LAYERS,
        }
    });
    commands.buffer.fillBuffer(triangle_powers_meta.handle, 0, @sizeOf(u32), 0);
    commands.buffer.pipelineBarrier2(&vk.DependencyInfo {
        .image_memory_barrier_count = 1,
        .p_image_memory_barriers = &[1]vk.ImageMemoryBarrier2 {
            .{
                .src_stage_mask = .{ .clear_bit = true },
                .src_access_mask = .{ .transfer_write_bit = true },
                .old_layout = .transfer_dst_optimal,
                .new_layout = .shader_read_only_optimal,
                .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
                .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
                .image = triangle_powers.handle,
                .subresource_range = .{
                    .aspect_mask = .{ .color_bit = true },
                    .base_mip_level = 0,
                    .level_count = vk.REMAINING_MIP_LEVELS,
                    .base_array_layer = 0,
                    .layer_count = vk.REMAINING_ARRAY_LAYERS,
                },
            }
        },
    });
    try commands.submitAndIdleUntilDone(vc);

    var triangle_powers_mips: [std.math.log2(max_emissive_triangles) + 1]vk.ImageView = undefined;
    for (&triangle_powers_mips, 0..) |*view, level_index| {
        view.* = try vc.device.createImageView(&vk.ImageViewCreateInfo {
            .flags = .{},
            .image = triangle_powers.handle,
            .view_type = vk.ImageViewType.@"1d",
            .format = .r32_sfloat,
            .components = .{
                .r = .identity,
                .g = .identity,
                .b = .identity,
                .a = .identity,
            },
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = @intCast(level_index),
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = vk.REMAINING_ARRAY_LAYERS,
            },
        }, null);
    }

    return Self {
        .triangle_power_pipeline = triangle_power_pipeline,
        .triangle_power_fold_pipeline = triangle_power_fold_pipeline,
        .triangle_powers_meta = triangle_powers_meta,
        .triangle_powers = triangle_powers,
        .triangle_powers_mips = triangle_powers_mips,
        .emissive_triangle_count = emissive_triangle_count,
    };
}

// accel must not be in use
pub const Handle = u32;
pub fn uploadInstance(self: *Self, vc: *const VulkanContext, vk_allocator: *VkAllocator, allocator: std.mem.Allocator, commands: *Commands, mesh_manager: MeshManager, material_manager: MaterialManager, instance: Instance) !Handle {
    std.debug.assert(self.geometry_count + instance.geometries.len <= max_geometries);
    std.debug.assert(self.instance_count < max_instances);

    try commands.startRecording(vc);
    const scratch_buffers = try makeBlases(vc, vk_allocator, allocator, commands, mesh_manager, &.{ instance.geometries }, &self.blases);
    defer allocator.free(scratch_buffers);
    defer for (scratch_buffers) |scratch_buffer| scratch_buffer.destroy(vc);

    // update geometries flat jagged array
    {
        if (self.geometries.is_null()) {
            self.geometries = try vk_allocator.createDeviceBuffer(vc, allocator, Geometry, max_geometries, .{ .storage_buffer_bit = true, .transfer_dst_bit = true });
            try vk_helpers.setDebugName(vc, self.geometries.handle, "geometries");
            self.geometry_to_triangle_power_offset = try vk_allocator.createDeviceBuffer(vc, allocator, u32, max_geometries, .{ .storage_buffer_bit = true });
            try vk_helpers.setDebugName(vc, self.geometry_to_triangle_power_offset.handle, "geometry to triangle power offset");
        }

        commands.recordUpdateBuffer(Geometry, self.geometries, instance.geometries, self.geometry_count);
    }

    // upload instance
    {
        if (self.instances_device.is_null()) {
            self.instances_device = try vk_allocator.createDeviceBuffer(vc, allocator, vk.AccelerationStructureInstanceKHR, max_instances, .{ .shader_device_address_bit = true, .transfer_dst_bit = true, .acceleration_structure_build_input_read_only_bit_khr = true, .storage_buffer_bit = true });
            self.instances_host = try vk_allocator.createHostBuffer(vc, vk.AccelerationStructureInstanceKHR, max_instances, .{ .transfer_src_bit = true });
            try vk_helpers.setDebugName(vc, self.instances_device.handle, "instances");
            self.instances_address = self.instances_device.getAddress(vc);
        }

        const vk_instance = vk.AccelerationStructureInstanceKHR {
            .transform = vk.TransformMatrixKHR {
                .matrix = @bitCast(instance.transform),
            },
            .instance_custom_index_and_mask = .{
                .instance_custom_index = self.geometry_count,
                .mask = if (instance.visible) 0xFF else 0x00,
            },
            .instance_shader_binding_table_record_offset_and_flags = .{
                .instance_shader_binding_table_record_offset = 0,
                .flags = 0,
            },
            .acceleration_structure_reference = vc.device.getAccelerationStructureDeviceAddressKHR(&.{
                .acceleration_structure = self.blases.items(.handle)[self.blases.len - 1],
            }),
        };

        self.instances_host.data[self.instance_count] = vk_instance;

        commands.recordUpdateBuffer(vk.AccelerationStructureInstanceKHR, self.instances_device, &.{ vk_instance }, self.instance_count); // TODO: can copy
    }

    // upload world_to_instance matrix
    {
        if (self.world_to_instance_device.is_null()) {
            self.world_to_instance_device = try vk_allocator.createDeviceBuffer(vc, allocator, Mat3x4, max_instances, .{ .storage_buffer_bit = true, .transfer_dst_bit = true });
            self.world_to_instance_host = try vk_allocator.createHostBuffer(vc, Mat3x4, max_instances, .{ .transfer_src_bit = true });
        }

        self.world_to_instance_host.data[self.instance_count] = instance.transform.inverse_affine();

        commands.recordUpdateBuffer(Mat3x4, self.world_to_instance_device, &.{ instance.transform.inverse_affine() }, self.instance_count);
    }

    self.instance_count += 1;

    // update TLAS
    var geometry_info = vk.AccelerationStructureBuildGeometryInfoKHR {
        .type = .top_level_khr,
        .flags = .{ .prefer_fast_trace_bit_khr = true, .allow_update_bit_khr = true },
        .mode = .build_khr,
        .geometry_count = 1,
        .p_geometries = @ptrCast(&vk.AccelerationStructureGeometryKHR {
            .geometry_type = .instances_khr,
            .flags = .{ .opaque_bit_khr = true },
            .geometry = .{
                .instances = .{
                    .array_of_pointers = vk.FALSE,
                    .data = .{
                        .device_address = self.instances_address,
                    }
                }
            },
        }),
        .scratch_data = undefined,
    };

    const size_info = getBuildSizesInfo(vc, &geometry_info, @ptrCast(&self.instance_count));

    const scratch_buffer = try vk_allocator.createOwnedDeviceBuffer(vc, size_info.build_scratch_size, .{ .shader_device_address_bit = true, .storage_buffer_bit = true });
    defer scratch_buffer.destroy(vc);

    self.tlas_buffer.destroy(vc);
    self.tlas_buffer = try vk_allocator.createDeviceBuffer(vc, allocator, u8, size_info.acceleration_structure_size, .{ .acceleration_structure_storage_bit_khr = true, .shader_device_address_bit = true });

    vc.device.destroyAccelerationStructureKHR(self.tlas_handle, null);
    geometry_info.dst_acceleration_structure = try vc.device.createAccelerationStructureKHR(&.{
        .buffer = self.tlas_buffer.handle,
        .offset = 0,
        .size = size_info.acceleration_structure_size,
        .type = .top_level_khr,
    }, null);
    self.tlas_handle = geometry_info.dst_acceleration_structure;

    geometry_info.scratch_data.device_address = scratch_buffer.getAddress(vc);

    self.tlas_update_scratch_buffer.destroy(vc);
    self.tlas_update_scratch_buffer = try vk_allocator.createDeviceBuffer(vc, allocator, u8, size_info.update_scratch_size, .{ .shader_device_address_bit = true, .storage_buffer_bit = true });
    self.tlas_update_scratch_address = self.tlas_update_scratch_buffer.getAddress(vc);

    commands.buffer.buildAccelerationStructuresKHR(1, @ptrCast(&geometry_info), &[_][*]const vk.AccelerationStructureBuildRangeInfoKHR{ @ptrCast(&vk.AccelerationStructureBuildRangeInfoKHR {
        .primitive_count = @intCast(self.instance_count),
        .first_vertex = 0,
        .primitive_offset = 0,
        .transform_offset = 0,
    })});

    for (instance.geometries, 0..) |geometry, i| {
        const index_count = mesh_manager.meshes.get(geometry.mesh).index_count;

        // this mesh is too big to importance sample...
        // it may still emit without importance sampling, though
        //
        // TODO: technically this should check that the that total (in the whole scene) emissive triangle count is less than
        // the maximum number of emissive triangles, but we don't have access to that info on the host.
        // probably we will just get a GPU crash instead :(
        if (index_count > max_emissive_triangles) continue;

        commands.buffer.pipelineBarrier2(&vk.DependencyInfo {
            .image_memory_barrier_count = 1,
            .p_image_memory_barriers = &[1]vk.ImageMemoryBarrier2 {
                .{
                    .dst_stage_mask = .{ .compute_shader_bit = true },
                    .dst_access_mask = .{ .shader_write_bit = true },
                    .old_layout = .shader_read_only_optimal,
                    .new_layout = .general,
                    .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
                    .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
                    .image = self.triangle_powers.handle,
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

        self.triangle_power_pipeline.recordBindPipeline(commands.buffer);
        self.triangle_power_pipeline.recordBindAdditionalDescriptorSets(commands.buffer, .{ material_manager.textures.descriptor_set });
        self.triangle_power_pipeline.recordPushDescriptors(commands.buffer, .{
            .instances = self.instances_device.handle,
            .world_to_instances = self.world_to_instance_device.handle,
            .meshes = mesh_manager.addresses_buffer.handle,
            .geometries = self.geometries.handle,
            .material_values = material_manager.materials.handle,
            .emissive_triangle_count = self.emissive_triangle_count.handle,
            .dst_power = .{ .view = self.triangle_powers_mips[0] },
            .dst_triangle_metadata = self.triangle_powers_meta.handle,
        });
        self.triangle_power_pipeline.recordPushConstants(commands.buffer, .{
            .instance_index = @intCast(self.instance_count - 1),
            .geometry_index = @intCast(i),
            .src_primitive_count = index_count,
        });
        const shader_local_size = 32; // must be kept in sync with shader -- looks like HLSL doesn't support setting this via spec constants
        const dispatch_size = std.math.divCeil(u32, index_count, shader_local_size) catch unreachable;
        self.triangle_power_pipeline.recordDispatch(commands.buffer, .{ .width = dispatch_size, .height = 1, .depth = 1 });
        self.triangle_power_fold_pipeline.recordBindPipeline(commands.buffer);

        for (1..self.triangle_powers_mips.len) |dst_mip_level| {
            commands.buffer.pipelineBarrier2(&vk.DependencyInfo {
                .image_memory_barrier_count = 2,
                .p_image_memory_barriers = &[2]vk.ImageMemoryBarrier2 {
                    .{
                        .src_stage_mask = .{ .compute_shader_bit = true },
                        .src_access_mask = .{ .shader_write_bit = true },
                        .dst_stage_mask = .{ .compute_shader_bit = true },
                        .dst_access_mask = .{ .shader_read_bit = true },
                        .old_layout = .general,
                        .new_layout = .shader_read_only_optimal,
                        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
                        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
                        .image = self.triangle_powers.handle,
                        .subresource_range = .{
                            .aspect_mask = .{ .color_bit = true },
                            .base_mip_level = @intCast(dst_mip_level - 1),
                            .level_count = 1,
                            .base_array_layer = 0,
                            .layer_count = vk.REMAINING_ARRAY_LAYERS,
                        },
                    },
                    .{
                        .dst_stage_mask = .{ .compute_shader_bit = true },
                        .dst_access_mask = .{ .shader_read_bit = true },
                        .old_layout = .shader_read_only_optimal,
                        .new_layout = .general,
                        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
                        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
                        .image = self.triangle_powers.handle,
                        .subresource_range = .{
                            .aspect_mask = .{ .color_bit = true },
                            .base_mip_level = @intCast(dst_mip_level),
                            .level_count = 1,
                            .base_array_layer = 0,
                            .layer_count = vk.REMAINING_ARRAY_LAYERS,
                        },
                    }
                },
            });
            self.triangle_power_fold_pipeline.recordPushDescriptors(commands.buffer, .{
                .src_mip = .{ .view = self.triangle_powers_mips[dst_mip_level - 1] },
                .dst_mip = .{ .view = self.triangle_powers_mips[dst_mip_level] },
                .geometry_to_triangle_power_offset = self.geometry_to_triangle_power_offset.handle,
                .emissive_triangle_count = self.emissive_triangle_count.handle,
            });
            self.triangle_power_fold_pipeline.recordPushConstants(commands.buffer, .{
                .geometry_index = self.geometry_count + @as(u32, @intCast(i)),
                .triangle_count = index_count,
            });
            const dst_mip_size = std.math.pow(u32, 2, @intCast(self.triangle_powers_mips.len - dst_mip_level));
            const mip_dispatch_size = std.math.divCeil(u32, dst_mip_size, shader_local_size) catch unreachable;
            self.triangle_power_fold_pipeline.recordDispatch(commands.buffer, .{ .width = mip_dispatch_size, .height = 1, .depth = 1 });
        }

        commands.buffer.pipelineBarrier2(&vk.DependencyInfo {
            .image_memory_barrier_count = 1,
            .p_image_memory_barriers = &[1]vk.ImageMemoryBarrier2 {
                .{
                    .src_stage_mask = .{ .compute_shader_bit = true },
                    .src_access_mask = .{ .shader_write_bit = true },
                    .dst_stage_mask = .{ .compute_shader_bit = true },
                    .dst_access_mask = .{ .shader_read_bit = true },
                    .old_layout = .general,
                    .new_layout = .shader_read_only_optimal,
                    .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
                    .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
                    .image = self.triangle_powers.handle,
                    .subresource_range = .{
                        .aspect_mask = .{ .color_bit = true },
                        .base_mip_level = self.triangle_powers_mips.len - 1,
                        .level_count = 1,
                        .base_array_layer = 0,
                        .layer_count = vk.REMAINING_ARRAY_LAYERS,
                    },
                }
            },
        });
    }

    try commands.submitAndIdleUntilDone(vc);

    self.geometry_count += @intCast(instance.geometries.len);

    return @intCast(self.instance_count - 1);
}

// probably bad idea if you're changing many
// must recordRebuild to see changes
pub fn recordUpdateSingleTransform(self: *Self, command_buffer: VulkanContext.CommandBuffer, instance_idx: u32, new_transform: Mat3x4) void {
    const offset = @sizeOf(vk.AccelerationStructureInstanceKHR) * instance_idx + @offsetOf(vk.AccelerationStructureInstanceKHR, "transform");
    const offset_inverse = @sizeOf(Mat3x4) * instance_idx;
    const size = @sizeOf(vk.TransformMatrixKHR);
    command_buffer.updateBuffer(self.instances_device.handle, offset, size, &new_transform);
    command_buffer.updateBuffer(self.world_to_instance_device.handle, offset_inverse, size, &new_transform.inverse_affine());
    const barriers = [_]vk.BufferMemoryBarrier2 {
        .{
            .src_stage_mask = .{ .clear_bit = true }, // cmdUpdateBuffer seems to be clear for some reason
            .src_access_mask = .{ .transfer_write_bit = true },
            .dst_stage_mask = .{ .acceleration_structure_build_bit_khr = true },
            .dst_access_mask = .{ .acceleration_structure_read_bit_khr = true, .shader_storage_read_bit = true },
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .buffer = self.instances_device.handle,
            .offset = offset,
            .size = size,
        },
        .{
            .src_stage_mask = .{ .clear_bit = true }, // cmdUpdateBuffer seems to be clear for some reason
            .src_access_mask = .{ .transfer_write_bit = true },
            .dst_stage_mask = .{ .ray_tracing_shader_bit_khr = true },
            .dst_access_mask = .{ .shader_storage_read_bit = true },
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .buffer = self.world_to_instance_device.handle,
            .offset = offset_inverse,
            .size = size,
        },
    };
    command_buffer.pipelineBarrier2(&vk.DependencyInfo {
        .buffer_memory_barrier_count = barriers.len,
        .p_buffer_memory_barriers = &barriers,
    });
}

// TODO: get it working
pub fn updateVisibility(self: *Self, instance_idx: u32, visible: bool) void {
    self.instances_host.data[instance_idx].instance_custom_index_and_mask.mask = if (visible) 0xFF else 0x00;
}

// probably bad idea if you're changing many
pub fn recordUpdateSingleMaterial(self: Self, command_buffer: VulkanContext.CommandBuffer, geometry_idx: u32, new_material_idx: u32) void {
    const offset = @sizeOf(Geometry) * geometry_idx + @offsetOf(Geometry, "material");
    const size = @sizeOf(u32);
    command_buffer.updateBuffer(self.geometries.handle, offset, size, &new_material_idx);
    command_buffer.pipelineBarrier2(&vk.DependencyInfo {
        .buffer_memory_barrier_count = 1,
        .p_buffer_memory_barriers = @ptrCast(&vk.BufferMemoryBarrier2 {
            .src_stage_mask = .{ .clear_bit = true }, // cmdUpdateBuffer seems to be clear for some reason
            .src_access_mask = .{ .transfer_write_bit = true },
            .dst_stage_mask = .{ .ray_tracing_shader_bit_khr = true },
            .dst_access_mask = .{ .shader_storage_read_bit = true },
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .buffer = self.geometries.handle,
            .offset = offset,
            .size = size,
        }),
    });
}

pub fn recordRebuild(self: *Self, command_buffer: VulkanContext.CommandBuffer) !void {
    const geometry = vk.AccelerationStructureGeometryKHR {
        .geometry_type = .instances_khr,
        .flags = .{ .opaque_bit_khr = true },
        .geometry = .{
            .instances = .{
                .array_of_pointers = vk.FALSE,
                .data = .{
                    .device_address = self.instances_address,
                }
            }
        },
    };

    var geometry_info = vk.AccelerationStructureBuildGeometryInfoKHR {
        .type = .top_level_khr,
        .flags = .{ .prefer_fast_trace_bit_khr = true, .allow_update_bit_khr = true },
        .mode = .update_khr,
        .src_acceleration_structure = self.tlas_handle,
        .dst_acceleration_structure = self.tlas_handle,
        .geometry_count = 1,
        .p_geometries = @ptrCast(&geometry),
        .scratch_data = .{
            .device_address = self.tlas_update_scratch_address,
        },
    };

    const build_info = vk.AccelerationStructureBuildRangeInfoKHR {
        .primitive_count = self.instance_count,
        .first_vertex = 0,
        .primitive_offset = 0,
        .transform_offset = 0,
    };

    const build_info_ref = &build_info;

    command_buffer.buildAccelerationStructuresKHR(1, @ptrCast(&geometry_info), @ptrCast(&build_info_ref));

    const barriers = [_]vk.MemoryBarrier2 {
        .{
            .src_stage_mask = .{ .acceleration_structure_build_bit_khr = true },
            .src_access_mask = .{ .acceleration_structure_write_bit_khr = true },
            .dst_stage_mask = .{ .ray_tracing_shader_bit_khr = true },
            .dst_access_mask = .{ .acceleration_structure_read_bit_khr = true },
        }
    };
    command_buffer.pipelineBarrier2(&vk.DependencyInfo {
        .memory_barrier_count = barriers.len,
        .p_memory_barriers = &barriers,
    });
}

pub fn destroy(self: *Self, vc: *const VulkanContext, allocator: std.mem.Allocator) void {
    self.instances_device.destroy(vc);
    self.instances_host.destroy(vc);
    self.world_to_instance_device.destroy(vc);
    self.world_to_instance_host.destroy(vc);

    self.geometries.destroy(vc);

    self.triangle_powers.destroy(vc);
    self.triangle_powers_meta.destroy(vc);
    self.geometry_to_triangle_power_offset.destroy(vc);
    self.emissive_triangle_count.destroy(vc);

    self.triangle_power_pipeline.destroy(vc);
    self.triangle_power_fold_pipeline.destroy(vc);

    self.tlas_update_scratch_buffer.destroy(vc);

    for (self.triangle_powers_mips) |view| {
        vc.device.destroyImageView(view, null);
    }

    const blases_slice = self.blases.slice();
    const blases_handles = blases_slice.items(.handle);
    const blases_buffers = blases_slice.items(.buffer);

    for (0..self.blases.len) |i| {
        vc.device.destroyAccelerationStructureKHR(blases_handles[i], null);
        blases_buffers[i].destroy(vc);
    }
    self.blases.deinit(allocator);

    vc.device.destroyAccelerationStructureKHR(self.tlas_handle, null);
    self.tlas_buffer.destroy(vc);
}

fn getBuildSizesInfo(vc: *const VulkanContext, geometry_info: *const vk.AccelerationStructureBuildGeometryInfoKHR, max_primitive_count: [*]const u32) vk.AccelerationStructureBuildSizesInfoKHR {
    var size_info: vk.AccelerationStructureBuildSizesInfoKHR = undefined;
    size_info.s_type = .acceleration_structure_build_sizes_info_khr;
    size_info.p_next = null;
    vc.device.getAccelerationStructureBuildSizesKHR(.device_khr, geometry_info, max_primitive_count, &size_info);
    return size_info;
}

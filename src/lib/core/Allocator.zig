// vulkan allocator
// currently essentially just a passthrough allocator
// functions here are just utilites that help with stuff

const vk = @import("vulkan");
const std = @import("std");
const core = @import("../engine.zig").core;
const VulkanContext = core.VulkanContext;
const vk_helpers = core.vk_helpers;

const Allocator = @This();

const MemoryStorage = std.ArrayListUnmanaged(vk.DeviceMemory);

memory_type_properties: std.BoundedArray(vk.MemoryPropertyFlags, vk.MAX_MEMORY_TYPES),
memory: MemoryStorage,

pub fn create(vc: *const VulkanContext) Allocator {
    const properties = vc.instance.getPhysicalDeviceMemoryProperties(vc.physical_device.handle);

    var memory_type_properties = std.BoundedArray(vk.MemoryPropertyFlags, vk.MAX_MEMORY_TYPES).init(properties.memory_type_count) catch unreachable;

    for (properties.memory_types[0..properties.memory_type_count], memory_type_properties.slice()) |memory_type, *memory_type_property| {
        memory_type_property.* = memory_type.property_flags;
    }

    return Allocator {
        .memory_type_properties = memory_type_properties,
        .memory = MemoryStorage {},
    };
}

pub fn destroy(self: *Allocator, vc: *const VulkanContext, allocator: std.mem.Allocator) void {
    for (self.memory.items) |memory| {
        vc.device.freeMemory(memory, null);
    }
    self.memory.deinit(allocator);
}


pub fn findMemoryType(self: *const Allocator, type_filter: u32, properties: vk.MemoryPropertyFlags) !std.meta.Int(.unsigned, vk.MAX_MEMORY_TYPES) {
    return for (self.memory_type_properties.slice(), 0..) |memory_type_properties, i| {
        if (type_filter & (@as(u32, 1) << @intCast(i)) != 0 and memory_type_properties.contains(properties)) {
            break @intCast(i);
        }
    } else error.UnavailbleMemoryType;
}

fn createRawBuffer(self: *Allocator, vc: *const VulkanContext, size: vk.DeviceSize, usage: vk.BufferUsageFlags, properties: vk.MemoryPropertyFlags, buffer: *vk.Buffer, buffer_memory: *vk.DeviceMemory) !void {
    buffer.* = try vc.device.createBuffer(&.{
            .size = size,
            .usage = usage,
            .sharing_mode = vk.SharingMode.exclusive,
    }, null);
    errdefer vc.device.destroyBuffer(buffer.*, null);

    const mem_requirements = vc.device.getBufferMemoryRequirements(buffer.*);

    const allocate_info = vk.MemoryAllocateInfo {
        .allocation_size = mem_requirements.size,
        .memory_type_index = try self.findMemoryType(mem_requirements.memory_type_bits, properties),
        .p_next = if (usage.contains(.{ .shader_device_address_bit = true })) &vk.MemoryAllocateFlagsInfo {
            .device_mask = 0,
            .flags = .{ .device_address_bit = true },
            } else null,
    };

    buffer_memory.* = try vc.device.allocateMemory(&allocate_info, null);
    errdefer vc.device.freeMemory(buffer_memory.*, null);

    try vc.device.bindBufferMemory(buffer.*, buffer_memory.*, 0);
}

pub fn DeviceBuffer(comptime T: type) type {
    const type_info = @typeInfo(T);
    if (type_info == .@"struct" and type_info.@"struct".layout == .auto) @compileError("Struct layout of " ++ @typeName(T) ++ " must be specified explicitly, but is not");
    return struct {
        handle: vk.Buffer = .null_handle,

        const BufferSelf = @This();

        pub fn destroy(self: BufferSelf, vc: *const VulkanContext) void {
            vc.device.destroyBuffer(self.handle, null);
        }

        // must've been created with shader device address bit enabled
        pub fn getAddress(self: BufferSelf, vc: *const VulkanContext) vk.DeviceAddress {
            return if (self.handle == .null_handle) 0 else vc.device.getBufferDeviceAddress(&.{
                .buffer = self.handle,
            });
        }

        pub fn is_null(self: BufferSelf) bool {
            return self.handle == .null_handle;
        }

        pub fn sizeInBytes(self: BufferSelf) usize {
            return self.data.len * @sizeOf(T);
        }
    };
}

pub fn createDeviceBuffer(self: *Allocator, vc: *const VulkanContext, allocator: std.mem.Allocator, comptime T: type, count: vk.DeviceSize, usage: vk.BufferUsageFlags, name: [:0]const u8) !DeviceBuffer(T) {
    if (count == 0) return DeviceBuffer(T) {};

    var buffer: vk.Buffer = undefined;
    var memory: vk.DeviceMemory = undefined;
    try self.createRawBuffer(vc, @sizeOf(T) * count, usage, .{ .device_local_bit = true }, &buffer, &memory);

    try vk_helpers.setDebugName(vc.device, buffer, name);
    try vk_helpers.setDebugName(vc.device, memory, name);

    try self.memory.append(allocator, memory);

    return DeviceBuffer(T) {
        .handle = buffer,
    };
}

// buffer that owns it's own memory; good for temp buffers which will quickly be destroyed
pub const OwnedDeviceBuffer = struct {
    handle: vk.Buffer,
    memory: vk.DeviceMemory,

    pub fn destroy(self: OwnedDeviceBuffer, vc: *const VulkanContext) void {
        vc.device.destroyBuffer(self.handle, null);
        vc.device.freeMemory(self.memory, null);
    }

    // must've been created with shader device address bit enabled
    pub fn getAddress(self: OwnedDeviceBuffer, vc: *const VulkanContext) vk.DeviceAddress {
        return vc.device.getBufferDeviceAddress(&.{
            .buffer = self.handle,
        });
    }
};

pub fn createOwnedDeviceBuffer(self: *Allocator, vc: *const VulkanContext, size: vk.DeviceSize, usage: vk.BufferUsageFlags, name: [:0]const u8) !OwnedDeviceBuffer {
    var buffer: vk.Buffer = undefined;
    var memory: vk.DeviceMemory = undefined;
    try self.createRawBuffer(vc, size, usage, .{ .device_local_bit = true }, &buffer, &memory);

    try vk_helpers.setDebugName(vc.device, buffer, name);
    try vk_helpers.setDebugName(vc.device, memory, name);

    return OwnedDeviceBuffer {
        .handle = buffer,
        .memory = memory,
    };
}

pub fn HostBuffer(comptime T: type) type {
    return struct {
        handle: vk.Buffer = .null_handle,
        memory: vk.DeviceMemory = .null_handle,
        data: []T = &.{},

        const BufferSelf = @This();

        pub fn destroy(self: BufferSelf, vc: *const VulkanContext) void {
            if (self.handle != .null_handle) {
                vc.device.destroyBuffer(self.handle, null);
                vc.device.unmapMemory(self.memory);
                vc.device.freeMemory(self.memory, null);
            }
        }

        // must've been created with shader device address bit enabled
        pub fn getAddress(self: BufferSelf, vc: *const VulkanContext) vk.DeviceAddress {
            return vc.device.getBufferDeviceAddress(&.{
                .buffer = self.handle,
            });
        }

        pub fn toBytes(self: BufferSelf) HostBuffer(u8) {
            return HostBuffer(u8) {
                .handle = self.handle,
                .memory = self.memory,
                .data = @as([*]u8, @ptrCast(self.data))[0..self.data.len * @sizeOf(T)],
            };
        }

        pub fn sizeInBytes(self: BufferSelf) usize {
            return self.data.len * @sizeOf(T);
        }
    };
}

// count not in bytes, but number of T
pub fn createHostBuffer(self: *Allocator, vc: *const VulkanContext, comptime T: type, count: vk.DeviceSize, usage: vk.BufferUsageFlags) !HostBuffer(T) {
    if (count == 0) {
        return HostBuffer(T) {
            .handle = .null_handle,
            .memory = .null_handle,
            .data = &.{},
        };
    }

    var buffer: vk.Buffer = undefined;
    var memory: vk.DeviceMemory = undefined;
    const size = @sizeOf(T) * count;
    try self.createRawBuffer(vc, size, usage, .{ .host_visible_bit = true, .host_coherent_bit = true }, &buffer, &memory);

    const data = @as([*]T, @ptrCast(@alignCast((try vc.device.mapMemory(memory, 0, size, .{})).?)))[0..count];

    return HostBuffer(T) {
        .handle = buffer,
        .memory = memory,
        .data = data,
    };
}

fn sliceContainsPtr(container: []const u8, ptr: [*]const u8) bool {
    return @intFromPtr(ptr) >= @intFromPtr(container.ptr) and
        @intFromPtr(ptr) < (@intFromPtr(container.ptr) + container.len);
}

pub fn HostVisiblePageAllocator(comptime usage: vk.BufferUsageFlags, comptime required_memory_properties: vk.MemoryPropertyFlags) type {
    if (comptime !required_memory_properties.contains(.{ .host_visible_bit = true})) @compileError("HostVisiblePageAllocator must only be used to allocate host visible memory");

    return struct {
        const Self = @This();

        const MemoryTypeIndex = std.meta.Int(.unsigned, vk.MAX_MEMORY_TYPES);
        const Metadata = struct {
            // technically the pointer here is entirely redundant as it basically points to itself
            // but std.Treap doesn't let me make use of this
            slice: []const u8,
            memory: vk.DeviceMemory,
            buffer: vk.Buffer,
        };
        const Allocations = std.Treap(Metadata, struct {
            fn compare(a: Metadata, b: Metadata) std.math.Order {
                return std.math.order(@intFromPtr(a.slice.ptr), @intFromPtr(b.slice.ptr));
            }
        }.compare);

        device: VulkanContext.Device,
        memory_type_index: MemoryTypeIndex,
        allocations: Allocations = .{},

        pub fn init(vc: *const VulkanContext) Self {
            var memory_requirements = vk.MemoryRequirements2 {
                .memory_requirements = undefined,
            };
            vc.device.getDeviceBufferMemoryRequirements(&vk.DeviceBufferMemoryRequirements {
                .p_create_info = &vk.BufferCreateInfo {
                    .usage = usage,
                    .size = 0,
                    .sharing_mode = .exclusive,
                },
            }, &memory_requirements);

            const available_memory_properties = vc.instance.getPhysicalDeviceMemoryProperties(vc.physical_device.handle);
            const memory_type_index: MemoryTypeIndex = for (available_memory_properties.memory_types[0..available_memory_properties.memory_type_count], 0..) |available_properties, i| {
                if (memory_requirements.memory_requirements.memory_type_bits & (@as(u32, 1) << @intCast(i)) != 0 and available_properties.property_flags.contains(required_memory_properties)) {
                    break @intCast(i);
                }
            } else unreachable;

            return Self {
                .device = vc.device,
                .memory_type_index = memory_type_index,
                .allocations = Allocations {},
            };
        }

        pub fn allocator(self: *Self) std.mem.Allocator {
            return .{
                .ptr = self,
                .vtable = &.{
                    .alloc = alloc,
                    .resize = std.mem.Allocator.noResize,
                    .free = free,
                },
            };
        }

        fn alloc(ctx: *anyopaque, len: usize, alignment_log2: u8, ret_addr: usize) ?[*]u8 {
            _ = ret_addr;
            const self: *Self = @ptrCast(@alignCast(ctx));
            const required_alignment = @as(usize, 1) << @as(std.mem.Allocator.Log2Align, @intCast(alignment_log2));

            const worst_case_additional_memory_required = @max(@sizeOf(Allocations.Node), required_alignment);

            const total_len = len + worst_case_additional_memory_required;
            const allocate_info = vk.MemoryAllocateInfo {
                .allocation_size = total_len,
                .memory_type_index = self.memory_type_index,
            };

            const memory = self.device.allocateMemory(&allocate_info, null) catch return null;
            if (comptime @import("build_options").vk_validation) {
                var debug_name_buf: [128]u8 = undefined;
                const debug_name = std.fmt.bufPrintZ(&debug_name_buf, "[{}]align({})", .{ len, required_alignment }) catch |err| std.debug.panic("{s}", .{ @errorName(err) });
                vk_helpers.setDebugName(self.device, memory, debug_name) catch |err| std.debug.panic("{s}", .{ @errorName(err) });
            }

            const minimum_guaranteed_alignment = 64; // https://docs.vulkan.org/spec/latest/chapters/limits.html#limits-minmax may as well commuincate this to the compiler
            const ptr_unaligned: [*]align(minimum_guaranteed_alignment) u8 = @alignCast(@ptrCast(self.device.mapMemory(memory, 0, vk.WHOLE_SIZE, .{}) catch return null));
            const ptr_aligned = std.mem.alignPointer(ptr_unaligned + @sizeOf(Allocations.Node), required_alignment).?;

            const buffer = self.device.createBuffer(&.{
                .size = len,
                .usage = usage,
                .sharing_mode = .exclusive,
            }, null) catch return null;

            self.device.bindBufferMemory(buffer, memory, 0) catch return null;

            const key = Metadata {
                .slice = ptr_unaligned[0..total_len],
                .memory = memory,
                .buffer = buffer,
            };
            var entry = self.allocations.getEntryFor(key);
            entry.set(self.getNode(ptr_aligned));

            return ptr_aligned;
        }

        fn free(ctx: *anyopaque, buf: []u8, alignment_log2: u8, ret_addr: usize) void {
            _ = alignment_log2;
            _ = ret_addr;

            const self: *Self = @ptrCast(@alignCast(ctx));

            const node = self.getNode(buf.ptr);
            const buffer = node.key.buffer;
            const memory = node.key.memory;
            var entry = self.allocations.getEntryForExisting(node);
            entry.set(null);
            self.device.destroyBuffer(buffer, null);
            self.device.freeMemory(memory, null);
        }

        // for pointer returned by alloc
        // ptr must have come from this allocator
        fn getNode(_: Self, ptr: [*]u8) *Allocations.Node {
            return @ptrFromInt(@intFromPtr(ptr) - @sizeOf(Allocations.Node));
        }

        // for any pointer in allocation
        // ptr must have come from this allocator
        fn findNode(self: Self, ptr: [*]const u8) *Allocations.Node {
            var maybe_node = self.allocations.root;

            while (maybe_node) |node| {
                if (sliceContainsPtr(node.key.slice, ptr)) {
                    return node;
                } else {
                    const order = std.math.order(@intFromPtr(ptr), @intFromPtr(node.key.slice.ptr));
                    if (order == .eq) unreachable;
                    maybe_node = node.children[@intFromBool(order == .gt)];
                }
            }

            unreachable;
        }
    };
}

pub const UploadPageAllocator = HostVisiblePageAllocator(vk.BufferUsageFlags { .transfer_src_bit = true }, vk.MemoryPropertyFlags { .host_visible_bit = true, .host_coherent_bit = true });
pub const DownloadPageAllocator = HostVisiblePageAllocator(vk.BufferUsageFlags { .transfer_dst_bit = true }, vk.MemoryPropertyFlags { .host_visible_bit = true, .host_coherent_bit = true, .host_cached_bit = true });
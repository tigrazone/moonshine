const vk = @import("vulkan");
const std = @import("std");
const core = @import("../engine.zig").core;
const VulkanContext = core.VulkanContext;
const Encoder = core.Encoder;
const vk_helpers = core.vk_helpers;

const vk_map_memory_minimum_guaranteed_alignment = 64; // https://docs.vulkan.org/spec/latest/chapters/limits.html#limits-minmax may as well commuincate this to the compiler

fn createRawBuffer(vc: *const VulkanContext, size: vk.DeviceSize, usage: vk.BufferUsageFlags, properties: vk.MemoryPropertyFlags, name: [:0]const u8) !std.meta.Tuple(&.{ vk.Buffer, vk.DeviceMemory }) {
    const buffer = try vc.device.createBuffer(&.{
            .size = size,
            .usage = usage,
            .sharing_mode = vk.SharingMode.exclusive,
    }, null);
    errdefer vc.device.destroyBuffer(buffer, null);
    try vk_helpers.setDebugName(vc.device, buffer, name);

    const mem_requirements = vc.device.getBufferMemoryRequirements(buffer);

    const allocate_info = vk.MemoryAllocateInfo {
        .allocation_size = mem_requirements.size,
        .memory_type_index = try vc.findMemoryType(mem_requirements.memory_type_bits, properties),
        .p_next = if (usage.contains(.{ .shader_device_address_bit = true })) &vk.MemoryAllocateFlagsInfo {
            .device_mask = 0,
            .flags = .{ .device_address_bit = true },
            } else null,
    };

    const memory = try vc.device.allocateMemory(&allocate_info, null);
    errdefer vc.device.freeMemory(memory, null);
    try vk_helpers.setDebugName(vc.device, memory, name);

    try vc.device.bindBufferMemory(buffer, memory, 0);

    return .{ buffer, memory };
}

pub fn Buffer(comptime T: type, comptime memory_properties: vk.MemoryPropertyFlags, comptime usage: vk.BufferUsageFlags) type {
    const type_info = @typeInfo(T);
    if (type_info == .@"struct" and type_info.@"struct".layout == .auto) @compileError("Struct layout of " ++ @typeName(T) ++ " must be specified explicitly, but is not");

    const host_visible = memory_properties.contains(.{ .host_visible_bit = true });

    return struct {
        handle: vk.Buffer = .null_handle,
        memory: vk.DeviceMemory = .null_handle,
        slice: if (host_visible) []T else void = if (host_visible) &.{} else {},

        const Self = @This();

        pub fn create(vc: *const VulkanContext, count: vk.DeviceSize, name: [:0]const u8) !Self {
            if (count == 0) return Self {};

            const size =  @sizeOf(T) * count;
            const buffer, const memory = try createRawBuffer(vc, size, usage, memory_properties, name);

            const slice = if (host_visible) blk: {
                const ptr: [*]align(vk_map_memory_minimum_guaranteed_alignment) u8 = @alignCast(@ptrCast(try vc.device.mapMemory(memory, 0, vk.WHOLE_SIZE, .{})));
                break :blk @as([*]T, @ptrCast(ptr))[0..count];
            } else ({});

            return Self {
                .handle = buffer,
                .memory = memory,
                .slice = slice,
            };
        }

        pub fn destroy(self: Self, vc: *const VulkanContext) void {
            if (self.handle != .null_handle) {
                vc.device.destroyBuffer(self.handle, null);
                vc.device.freeMemory(self.memory, null);
            }
        }

        pub usingnamespace if (usage.contains(.{ .shader_device_address_bit = true })) struct {
            pub fn getAddress(self: Self, vc: *const VulkanContext) vk.DeviceAddress {
                return if (self.handle == .null_handle) 0 else vc.device.getBufferDeviceAddress(&.{
                    .buffer = self.handle,
                });
            }
        } else struct {};

        pub usingnamespace if (usage.contains(.{ .transfer_dst_bit = true })) struct {
            pub fn updateFrom(self: Self, encoder: *Encoder, dst_offset: vk.DeviceSize, src: []const T) void {
                const bytes = std.mem.sliceAsBytes(src);
                encoder.buffer.updateBuffer(self.handle, dst_offset * @sizeOf(T), bytes.len, src.ptr);
            }

            pub fn uploadFrom(self: Self, encoder: *Encoder, src: BufferSlice(T)) void {
                const region = vk.BufferCopy {
                    .src_offset = src.offset,
                    .dst_offset = 0,
                    .size = src.asBytes().len,
                };

                encoder.buffer.copyBuffer(src.handle, self.handle, 1, @ptrCast(&region));
            }
        } else struct {};

        pub fn isNull(self: Self) bool {
            return self.handle == .null_handle;
        }
    };
}

pub fn UploadBuffer(comptime T: type) type {
    return Buffer(T, vk.MemoryPropertyFlags { .host_visible_bit = true, .host_coherent_bit = true }, vk.BufferUsageFlags { .transfer_src_bit = true });
}

pub fn DownloadBuffer(comptime T: type) type {
    return Buffer(T, vk.MemoryPropertyFlags { .host_visible_bit = true, .host_coherent_bit = true, .host_cached_bit = true }, vk.BufferUsageFlags { .transfer_dst_bit = true });
}

pub fn DeviceBuffer(comptime T: type, comptime usage: vk.BufferUsageFlags) type {
    return Buffer(T, vk.MemoryPropertyFlags { .device_local_bit = true }, usage);
}

pub fn BufferSlice(comptime T: type) type {
    return struct {
        handle: vk.Buffer = .null_handle,
        offset: vk.DeviceSize = 0,
        len: vk.DeviceSize = 0,

        const Self = @This();

        pub fn asBytes(self: Self) BufferSlice(u8) {
            return BufferSlice(u8) {
                .handle = self.handle,
                .offset = self.offset,
                .len = @sizeOf(T) * self.len,
            };
        }
    };
}

fn sliceContainsPtr(container: []const u8, ptr: [*]const u8) bool {
    return @intFromPtr(ptr) >= @intFromPtr(container.ptr) and
        @intFromPtr(ptr) < (@intFromPtr(container.ptr) + container.len);
}

pub fn HostVisiblePageAllocator(comptime memory_properties: vk.MemoryPropertyFlags, comptime usage: vk.BufferUsageFlags) type {
    if (comptime !memory_properties.contains(.{ .host_visible_bit = true})) @compileError("HostVisiblePageAllocator must only be used to allocate host visible memory");

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
        required_alignment_log2: std.math.Log2Int(vk.DeviceSize),
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

            const memory_type_index: MemoryTypeIndex = vc.findMemoryType(memory_requirements.memory_requirements.memory_type_bits, memory_properties) catch unreachable;

            return Self {
                .device = vc.device,
                .memory_type_index = memory_type_index,
                .required_alignment_log2 = std.math.log2_int(vk.DeviceSize, memory_requirements.memory_requirements.alignment), // vulkan guarantees this is a power of two,
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

        pub fn getBufferSlice(self: Self, data: anytype) BufferSlice(@typeInfo(@TypeOf(data)).pointer.child) {
            const T = @typeInfo(@TypeOf(data)).pointer.child;

            const ptr = switch (@typeInfo(@TypeOf(data)).pointer.size) {
                .One => data,
                .Slice => data.ptr,
                else => comptime unreachable,
            };

            const len = switch (@typeInfo(@TypeOf(data)).pointer.size) {
                .One => 1,
                .Slice => data.len,
                else => comptime unreachable,
            };

            const node = self.findNode(@ptrCast(ptr));

            return BufferSlice(T) {
                .handle = node.key.buffer,
                .offset = @as([*]const u8, @ptrCast(ptr)) - node.key.slice.ptr,
                .len = len,
            };
        }

        fn alloc(ctx: *anyopaque, len: usize, alignment_log2: u8, ret_addr: usize) ?[*]u8 {
            _ = ret_addr;
            const self: *Self = @ptrCast(@alignCast(ctx));
            const required_alignment = @as(usize, 1) << @as(std.mem.Allocator.Log2Align, @intCast(alignment_log2));

            const worst_case_additional_memory_required = @max(@sizeOf(Allocations.Node), required_alignment);

            // > For a VkBuffer, the size memory requirement is never greater than the result of
            // > aligning VkBufferCreateInfo::size with the alignment memory requirement.
            const required_alignment_vk = @as(usize, 1) << @as(std.mem.Allocator.Log2Align, @intCast(self.required_alignment_log2));
            const total_len = std.mem.alignForward(usize, len + worst_case_additional_memory_required, required_alignment_vk);
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

            const ptr_unaligned: [*]align(vk_map_memory_minimum_guaranteed_alignment) u8 = @alignCast(@ptrCast(self.device.mapMemory(memory, 0, vk.WHOLE_SIZE, .{}) catch return null));
            const ptr_aligned = std.mem.alignPointer(ptr_unaligned + @sizeOf(Allocations.Node), required_alignment).?;

            const buffer = self.device.createBuffer(&.{
                .size = total_len,
                .usage = usage,
                .sharing_mode = .exclusive,
            }, null) catch return null;
            if (comptime @import("build_options").vk_validation) {
                var debug_name_buf: [128]u8 = undefined;
                const debug_name = std.fmt.bufPrintZ(&debug_name_buf, "[{}]align({})", .{ len, required_alignment }) catch |err| std.debug.panic("{s}", .{ @errorName(err) });
                vk_helpers.setDebugName(self.device, buffer, debug_name) catch |err| std.debug.panic("{s}", .{ @errorName(err) });
            }

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

pub const UploadPageAllocator = HostVisiblePageAllocator(vk.MemoryPropertyFlags { .host_visible_bit = true, .host_coherent_bit = true }, vk.BufferUsageFlags { .transfer_src_bit = true });
pub const DownloadPageAllocator = HostVisiblePageAllocator(vk.MemoryPropertyFlags { .host_visible_bit = true, .host_coherent_bit = true, .host_cached_bit = true }, vk.BufferUsageFlags { .transfer_dst_bit = true });
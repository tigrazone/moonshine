// imgui platform implementation

const vk = @import("vulkan");
const std = @import("std");
const fs = std.fs;

const engine = @import("../engine.zig");
const core = engine.core;
const VulkanContext = core.VulkanContext;
const Encoder = core.Encoder;
const vk_helpers = core.vk_helpers;

const Image = core.Image;
const DescriptorLayout = core.descriptor.DescriptorLayout;

const Swapchain = engine.displaysystem.Swapchain;
const Display = engine.displaysystem.Display;

const vector = engine.vector;
const F32x3 = vector.Vec3(f32);

const imgui = @import("./imgui.zig");
const Window = @import("../Window.zig");

pub const required_vulkan_functions = [_]vk.ApiInfo {
    .{
        .device_commands = .{
            .createGraphicsPipelines = true,
            .cmdBindVertexBuffers = true,
            .cmdBindIndexBuffer = true,
            .cmdSetViewport = true,
            .cmdSetScissor = true,
            .cmdDrawIndexed = true,
            .cmdBeginRendering = true,
            .cmdEndRendering = true,
        },
    },
};

const frames_in_flight = Display.frames_in_flight;
const Self = @This();

const VertexBuffer = core.mem.Buffer(imgui.DrawVert, .{ .host_visible_bit = true, .host_coherent_bit = true }, .{ .vertex_buffer_bit =  true });
const IndexBuffer = core.mem.Buffer(imgui.DrawIdx, .{ .host_visible_bit = true, .host_coherent_bit = true }, .{ .index_buffer_bit =  true });

extent: vk.Extent2D,

descriptor_set_layout: GuiDescriptorLayout,
pipeline_layout: vk.PipelineLayout,
pipeline: vk.Pipeline,

font_sampler: vk.Sampler,
font_image: Image,
font_image_set: vk.DescriptorSet,

vertex_buffers: [frames_in_flight]VertexBuffer,
index_buffers: [frames_in_flight]IndexBuffer,

views: std.BoundedArray(vk.ImageView, Swapchain.max_image_count),

pub fn create(vc: *const VulkanContext, swapchain: Swapchain, window: Window, extent: vk.Extent2D, encoder: *Encoder) !Self {
    if (imgui.getCurrentContext()) |_| @panic("cannot create more than one Gui");

    imgui.createContext();
    imgui.implGlfwInit(window);

    //store moonshine imgui ini on same folder with exe
    const ini_file = "moonshine.ini";

    // Defining this type made it easier for me to understand allocating a
    // string array.
    const String = []const u8;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    // See also std.fs.selfExeDirPathAlloc
    const exe_path = try std.fs.selfExePathAlloc(allocator);
    defer allocator.free(exe_path);
    const opt_exe_dir = std.fs.path.dirname(exe_path);

    if (opt_exe_dir) |exe_dir| {
        var paths = [_]String{ exe_dir, ini_file };

        const file_path = try fs.path.join(allocator, &paths);
        defer allocator.free(file_path);
        //std.debug.print("File location: {s}\n", .{file_path});

        imgui.getIO().IniFilename = try allocator.dupeZ(u8, file_path);
    }


    // load required vulkan state
    const font_sampler = try vc.device.createSampler(&.{
        .mag_filter = .linear,
        .min_filter = .linear,
        .mipmap_mode = .linear,
        .address_mode_u = .repeat,
        .address_mode_v = .repeat,
        .address_mode_w = .repeat,
        .min_lod = -1000,
        .max_lod = 1000,
        .anisotropy_enable = vk.FALSE,
        .max_anisotropy = 1.0,
        .compare_enable = vk.FALSE,
        .compare_op = .never,
        .border_color = .float_transparent_black,
        .unnormalized_coordinates = vk.FALSE,
        .mip_lod_bias = 0.0,
    }, null);

    const descriptor_set_layout = try GuiDescriptorLayout.create(vc, .{ font_sampler });

    const pipeline_layout = try vc.device.createPipelineLayout(&vk.PipelineLayoutCreateInfo {
        .set_layout_count = 1,
        .p_set_layouts = @ptrCast(&descriptor_set_layout.handle),
        .push_constant_range_count = 1,
        .p_push_constant_ranges = @ptrCast(&vk.PushConstantRange{
            .stage_flags = .{ .vertex_bit = true },
            .offset = 0,
            .size = @sizeOf(f32) * 4,
        }),
    }, null);

    const pipeline = blk: {
        const vert_module = try vc.device.createShaderModule(&.{
            .code_size = vert_spv.len * @sizeOf(u32),
            .p_code = &vert_spv,
        }, null);
        defer vc.device.destroyShaderModule(vert_module, null);
        const frag_module = try vc.device.createShaderModule(&.{
            .code_size = frag_spv.len * @sizeOf(u32),
            .p_code = &frag_spv,
        }, null);
        defer vc.device.destroyShaderModule(frag_module, null);

        const shader_stage_create_info = [_]vk.PipelineShaderStageCreateInfo{
            .{
                .module = vert_module,
                .stage = .{ .vertex_bit = true },
                .p_name = "main",
            },
            .{
                .module = frag_module,
                .stage = .{ .fragment_bit = true },
                .p_name = "main",
            },
        };
        const vertex_attribute_descriptions = [_]vk.VertexInputAttributeDescription{
            .{
                .location = 0,
                .binding = 0,
                .format = .r32g32_sfloat,
                .offset = @offsetOf(imgui.DrawVert, "pos"),
            },
            .{
                .location = 1,
                .binding = 0,
                .format = .r32g32_sfloat,
                .offset = @offsetOf(imgui.DrawVert, "uv"),
            },
            .{
                .location = 2,
                .binding = 0,
                .format = .r8g8b8a8_unorm,
                .offset = @offsetOf(imgui.DrawVert, "col"),
            },
        };
        const dynamic_states = [_]vk.DynamicState{ .viewport, .scissor };
        var pipeline: vk.Pipeline = undefined;
        _ = try vc.device.createGraphicsPipelines(.null_handle, 1, @ptrCast(&vk.GraphicsPipelineCreateInfo{
            .stage_count = shader_stage_create_info.len,
            .p_stages = &shader_stage_create_info,
            .p_vertex_input_state = &vk.PipelineVertexInputStateCreateInfo{
                .vertex_binding_description_count = 1,
                .p_vertex_binding_descriptions = @ptrCast(&vk.VertexInputBindingDescription{
                    .binding = 0,
                    .stride = @sizeOf(imgui.DrawVert),
                    .input_rate = .vertex,
                }),
                .vertex_attribute_description_count = vertex_attribute_descriptions.len,
                .p_vertex_attribute_descriptions = &vertex_attribute_descriptions,
            },
            .p_input_assembly_state = &vk.PipelineInputAssemblyStateCreateInfo{
                .topology = .triangle_list,
                .primitive_restart_enable = vk.FALSE,
            },
            .p_viewport_state = &vk.PipelineViewportStateCreateInfo{
                .viewport_count = 1,
                .scissor_count = 1,
            },
            .p_rasterization_state = &vk.PipelineRasterizationStateCreateInfo{
                .depth_clamp_enable = vk.FALSE,
                .rasterizer_discard_enable = vk.FALSE,
                .polygon_mode = .fill,
                .front_face = .counter_clockwise,
                .depth_bias_enable = 0.0,
                .depth_bias_constant_factor = 0.0,
                .depth_bias_clamp = 0.0,
                .depth_bias_slope_factor = 0.0,
                .line_width = 1.0,
            },
            .p_color_blend_state = &vk.PipelineColorBlendStateCreateInfo{
                .logic_op_enable = vk.FALSE,
                .logic_op = .clear,
                .attachment_count = 1,
                .p_attachments = @ptrCast(&vk.PipelineColorBlendAttachmentState{
                    .blend_enable = vk.TRUE,
                    .src_color_blend_factor = .src_alpha,
                    .dst_color_blend_factor = .one_minus_src_alpha,
                    .color_blend_op = .add,
                    .src_alpha_blend_factor = .one,
                    .dst_alpha_blend_factor = .one_minus_src_alpha,
                    .alpha_blend_op = .add,
                    .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
                }),
                .blend_constants = .{ 0.0, 0.0, 0.0, 0.0 },
            },
            .p_dynamic_state = &vk.PipelineDynamicStateCreateInfo{
                .dynamic_state_count = dynamic_states.len,
                .p_dynamic_states = &dynamic_states,
            },
            .p_multisample_state = &vk.PipelineMultisampleStateCreateInfo {
                .rasterization_samples = .{ .@"1_bit" = true },
                .min_sample_shading = 1.0,
                .sample_shading_enable = vk.FALSE,
                .alpha_to_coverage_enable = vk.FALSE,
                .alpha_to_one_enable = vk.FALSE,
            },
            .layout = pipeline_layout,
            .render_pass = .null_handle,
            .subpass = 0,
            .base_pipeline_index = 0,
            .p_next = &vk.PipelineRenderingCreateInfo{
                .view_mask = 0,
                .color_attachment_count = 1,
                .p_color_attachment_formats = @ptrCast(&swapchain.image_format),
                .depth_attachment_format = .undefined,
                .stencil_attachment_format = .undefined,
            },
        }), null, @ptrCast(&pipeline));
        break :blk pipeline;
    };

    const font_image = blk: {
        const tex_data = imgui.getTexDataAsAlpha8(imgui.getIO().Fonts);
        const image = try Image.create(vc, tex_data[1], .{ .transfer_dst_bit = true, .sampled_bit = true }, .r8_unorm, false, "imgui font");
        errdefer image.destroy(vc);

        const img_data = tex_data[0][0 .. tex_data[1].width * tex_data[1].height * @sizeOf(u8)];
        const staging_data = try encoder.uploadAllocator().dupe(u8, img_data);
        encoder.uploadDataToImage(u8, encoder.upload_allocator.getBufferSlice(staging_data), image.handle, tex_data[1], .shader_read_only_optimal);
        break :blk image;
    };

    const font_image_set = try descriptor_set_layout.allocate_set(vc, [_]vk.WriteDescriptorSet{
        .{
            .dst_set = undefined,
            .dst_binding = 0,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = .combined_image_sampler,
            .p_image_info = @ptrCast(&vk.DescriptorImageInfo{
                .sampler = undefined,
                .image_view = font_image.view,
                .image_layout = .read_only_optimal,
            }),
            .p_buffer_info = undefined,
            .p_texel_buffer_view = undefined,
        },
    });

    // sort of a stupid allocation strategy right now
    // create a pretty big buffer that should be good enough for most things, and ensure dear imgui doesn't want to use more at each frame
    // TODO: change this when the rest of the system becomes smart enough that this looks stupid in comparison
    var vertex_buffers: [frames_in_flight]VertexBuffer = undefined;
    for (&vertex_buffers) |*buffer| {
        buffer.* = try VertexBuffer.create(vc, std.math.maxInt(imgui.DrawIdx), "imgui vertex buffer");
    }
    var index_buffers: [frames_in_flight]IndexBuffer = undefined;
    for (&index_buffers) |*buffer| {
        buffer.* = try IndexBuffer.create(vc, std.math.maxInt(imgui.DrawIdx), "imgui index buffer");
    }

    var views = std.BoundedArray(vk.ImageView, Swapchain.max_image_count){
        .buffer = undefined,
        .len = swapchain.images.len,
    };
    inline for (&views.buffer, 0..) |*view, i| {
        if (i > views.len) {
            break;
        }
        view.* = try vc.device.createImageView(&vk.ImageViewCreateInfo{
            .image = swapchain.images.slice()[i],
            .view_type = .@"2d",
            .format = swapchain.image_format,
            .components = vk.ComponentMapping{
                .r = .identity,
                .g = .identity,
                .b = .identity,
                .a = .identity,
            },
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        }, null);
        try vk_helpers.setDebugName(vc.device, view.*, std.fmt.comptimePrint("swapchain image view {}", .{ i }));
    }

    return Self{
        .extent = extent,

        .descriptor_set_layout = descriptor_set_layout,
        .pipeline_layout = pipeline_layout,
        .pipeline = pipeline,

        .font_sampler = font_sampler,
        .font_image = font_image,
        .font_image_set = font_image_set,

        .vertex_buffers = vertex_buffers,
        .index_buffers = index_buffers,

        .views = views,
    };
}

pub fn resize(self: *Self, vc: *const VulkanContext, swapchain: Swapchain) !void {
    for (self.views.slice()) |view| vc.device.destroyImageView(view, null);

    for (self.views.slice(), 0..) |*view, i| {
        view.* = try vc.device.createImageView(&vk.ImageViewCreateInfo{
            .image = swapchain.images.slice()[i],
            .view_type = .@"2d",
            .format = swapchain.image_format,
            .components = vk.ComponentMapping{
                .r = .identity,
                .g = .identity,
                .b = .identity,
                .a = .identity,
            },
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        }, null);
    }
    self.extent = swapchain.extent;
}

pub fn destroy(self: *Self, vc: *const VulkanContext) void {
    for (self.views.slice()) |view| vc.device.destroyImageView(view, null);

    self.descriptor_set_layout.destroy(vc);
    vc.device.destroyPipelineLayout(self.pipeline_layout, null);
    vc.device.destroyPipeline(self.pipeline, null);

    vc.device.destroySampler(self.font_sampler, null);
    self.font_image.destroy(vc);

    for (self.vertex_buffers) |buffer| buffer.destroy(vc);
    for (self.index_buffers) |buffer| buffer.destroy(vc);

    imgui.implGlfwShutdown();
    imgui.destroyContext();
}

pub fn startFrame(self: *Self) void {
    _ = self; // ensure we're initialized
    imgui.implGlfwNewFrame();
    imgui.newFrame();
}

pub fn endFrame(self: *Self, command_buffer: VulkanContext.CommandBuffer, swapchain_image_index: usize, display_image_index: usize) void {
    imgui.render();
    const draw_data = imgui.getDrawData();

    // copy all imgui vertex/index data into our one big buffer
    const vertex_buffer = self.vertex_buffers[display_image_index];
    const index_buffer = self.index_buffers[display_image_index];
    std.debug.assert(draw_data.TotalVtxCount <= vertex_buffer.slice.len);
    std.debug.assert(draw_data.TotalIdxCount <= index_buffer.slice.len);
    if (draw_data.CmdListsCount > 0) {
        var vertex_offset: usize = 0;
        var index_offset: usize = 0;
        for (draw_data.CmdLists.Data[0..@intCast(draw_data.CmdListsCount)]) |cmd_list| {
            const vertex_count: usize = @intCast(cmd_list.*.VtxBuffer.Size);
            const index_count: usize = @intCast(cmd_list.*.IdxBuffer.Size);
            @memcpy(vertex_buffer.slice[vertex_offset..].ptr, cmd_list.*.VtxBuffer.Data[0..vertex_count]);
            @memcpy(index_buffer.slice[index_offset..].ptr, cmd_list.*.IdxBuffer.Data[0..index_count]);
            vertex_offset += vertex_count;
            index_offset += index_count;
        }
    } else return;

    command_buffer.beginRendering(&vk.RenderingInfo{
        .render_area = vk.Rect2D{
            .offset = vk.Offset2D{
                .x = 0.0,
                .y = 0.0,
            },
            .extent = self.extent,
        },
        .layer_count = 1,
        .view_mask = 0,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&vk.RenderingAttachmentInfo{
            .image_view = self.views.slice()[swapchain_image_index],
            .image_layout = .color_attachment_optimal,
            .resolve_mode = .{},
            .resolve_image_layout = .undefined,
            .load_op = .load,
            .store_op = .store,
            .clear_value = undefined,
        }),
    });
    command_buffer.bindPipeline(.graphics, self.pipeline);
    command_buffer.bindVertexBuffers(0, 1, @ptrCast(&vertex_buffer.handle), @ptrCast(&@as(vk.DeviceSize, 0)));
    command_buffer.bindIndexBuffer(index_buffer.handle, 0, .uint16);
    command_buffer.setViewport(0, 1, @ptrCast(&vk.Viewport{
        .x = 0,
        .y = 0,
        .width = @floatFromInt(self.extent.width),
        .height = @floatFromInt(self.extent.height),
        .min_depth = 0.0,
        .max_depth = 1.0,
    }));
    {
        const scale = [2]f32{
            2.0 / draw_data.DisplaySize.x,
            2.0 / draw_data.DisplaySize.y,
        };
        const translate = [2]f32{
            -1.0 - draw_data.DisplayPos.x * scale[0],
            -1.0 - draw_data.DisplayPos.y * scale[1],
        };
        command_buffer.pushConstants(self.pipeline_layout, .{ .vertex_bit = true }, 0, @sizeOf(f32) * 4, &std.mem.toBytes(.{ scale, translate }));
    }
    command_buffer.bindDescriptorSets(.graphics, self.pipeline_layout, 0, 1, @ptrCast(&self.font_image_set), 0, undefined);

    var global_idx_offset: u32 = 0;
    var global_vtx_offset: u32 = 0;
    for (draw_data.CmdLists.Data[0..@intCast(draw_data.CmdListsCount)]) |cmd_list| {
        for (cmd_list.*.CmdBuffer.Data[0..@intCast(cmd_list.*.CmdBuffer.Size)]) |cmd| {
            if (cmd.UserCallback) |_| @panic("todo");
            command_buffer.setScissor(0, 1, @ptrCast(&vk.Rect2D{
                .offset = vk.Offset2D {
                    .x = @intFromFloat(cmd.ClipRect.x),
                    .y = @intFromFloat(cmd.ClipRect.y),
                },
                .extent = vk.Extent2D {
                    .width = @as(u32, @intFromFloat(cmd.ClipRect.z)) - @as(u32, @intFromFloat(cmd.ClipRect.x)),
                    .height = @as(u32, @intFromFloat(cmd.ClipRect.w)) - @as(u32, @intFromFloat(cmd.ClipRect.y)),
                },
            }));
            command_buffer.drawIndexed(cmd.ElemCount, 1, global_idx_offset + cmd.IdxOffset, @intCast(global_vtx_offset + cmd.VtxOffset), 0);
        }
        global_idx_offset += @intCast(cmd_list.*.IdxBuffer.Size);
        global_vtx_offset += @intCast(cmd_list.*.VtxBuffer.Size);
    }
    command_buffer.endRendering();
}

pub const GuiDescriptorLayout = DescriptorLayout(&.{
    .{
        .descriptor_type = .combined_image_sampler,
        .descriptor_count = 1,
        .stage_flags = .{ .fragment_bit = true },
        .binding_flags = .{},
    },
}, .{}, 1, "Gui");

// glsl_shader.vert, compiled with:
// # glslangValidator -V -x -g0 -o glsl_shader.vert.u32 glsl_shader.vert
//
// #version 450 core
// layout(location = 0) in vec2 aPos;
// layout(location = 1) in vec2 aUV;
// layout(location = 2) in vec4 aColor;
// layout(push_constant) uniform uPushConstant { vec2 uScale; vec2 uTranslate; } pc;
//
// out gl_PerVertex { vec4 gl_Position; };
// layout(location = 0) out struct { vec4 Color; vec2 UV; } Out;
//
// void main() {
//     Out.Color = aColor;
//     Out.UV = aUV;
//     gl_Position = vec4(aPos * pc.uScale + pc.uTranslate, 0, 1);
// }
const vert_spv = [_]u32{
	0x07230203,0x00010000,0x0008000b,0x0000002e,0x00000000,0x00020011,0x00000001,0x0006000b,
	0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
	0x000a000f,0x00000000,0x00000004,0x6e69616d,0x00000000,0x0000000b,0x0000000f,0x00000015,
	0x0000001b,0x0000001c,0x00040047,0x0000000b,0x0000001e,0x00000000,0x00040047,0x0000000f,
	0x0000001e,0x00000002,0x00040047,0x00000015,0x0000001e,0x00000001,0x00050048,0x00000019,
	0x00000000,0x0000000b,0x00000000,0x00030047,0x00000019,0x00000002,0x00040047,0x0000001c,
	0x0000001e,0x00000000,0x00050048,0x0000001e,0x00000000,0x00000023,0x00000000,0x00050048,
	0x0000001e,0x00000001,0x00000023,0x00000008,0x00030047,0x0000001e,0x00000002,0x00020013,
	0x00000002,0x00030021,0x00000003,0x00000002,0x00030016,0x00000006,0x00000020,0x00040017,
	0x00000007,0x00000006,0x00000004,0x00040017,0x00000008,0x00000006,0x00000002,0x0004001e,
	0x00000009,0x00000007,0x00000008,0x00040020,0x0000000a,0x00000003,0x00000009,0x0004003b,
	0x0000000a,0x0000000b,0x00000003,0x00040015,0x0000000c,0x00000020,0x00000001,0x0004002b,
	0x0000000c,0x0000000d,0x00000000,0x00040020,0x0000000e,0x00000001,0x00000007,0x0004003b,
	0x0000000e,0x0000000f,0x00000001,0x00040020,0x00000011,0x00000003,0x00000007,0x0004002b,
	0x0000000c,0x00000013,0x00000001,0x00040020,0x00000014,0x00000001,0x00000008,0x0004003b,
	0x00000014,0x00000015,0x00000001,0x00040020,0x00000017,0x00000003,0x00000008,0x0003001e,
	0x00000019,0x00000007,0x00040020,0x0000001a,0x00000003,0x00000019,0x0004003b,0x0000001a,
	0x0000001b,0x00000003,0x0004003b,0x00000014,0x0000001c,0x00000001,0x0004001e,0x0000001e,
	0x00000008,0x00000008,0x00040020,0x0000001f,0x00000009,0x0000001e,0x0004003b,0x0000001f,
	0x00000020,0x00000009,0x00040020,0x00000021,0x00000009,0x00000008,0x0004002b,0x00000006,
	0x00000028,0x00000000,0x0004002b,0x00000006,0x00000029,0x3f800000,0x00050036,0x00000002,
	0x00000004,0x00000000,0x00000003,0x000200f8,0x00000005,0x0004003d,0x00000007,0x00000010,
	0x0000000f,0x00050041,0x00000011,0x00000012,0x0000000b,0x0000000d,0x0003003e,0x00000012,
	0x00000010,0x0004003d,0x00000008,0x00000016,0x00000015,0x00050041,0x00000017,0x00000018,
	0x0000000b,0x00000013,0x0003003e,0x00000018,0x00000016,0x0004003d,0x00000008,0x0000001d,
	0x0000001c,0x00050041,0x00000021,0x00000022,0x00000020,0x0000000d,0x0004003d,0x00000008,
	0x00000023,0x00000022,0x00050085,0x00000008,0x00000024,0x0000001d,0x00000023,0x00050041,
	0x00000021,0x00000025,0x00000020,0x00000013,0x0004003d,0x00000008,0x00000026,0x00000025,
	0x00050081,0x00000008,0x00000027,0x00000024,0x00000026,0x00050051,0x00000006,0x0000002a,
	0x00000027,0x00000000,0x00050051,0x00000006,0x0000002b,0x00000027,0x00000001,0x00070050,
	0x00000007,0x0000002c,0x0000002a,0x0000002b,0x00000028,0x00000029,0x00050041,0x00000011,
	0x0000002d,0x0000001b,0x0000000d,0x0003003e,0x0000002d,0x0000002c,0x000100fd,0x00010038 };

// glsl_shader.frag, compiled with:
// # glslangValidator -V -x -g0 -o glsl_shader.frag.u32 glsl_shader.frag
//
// #version 450 core
// layout(location = 0) out vec4 fColor;
// layout(set=0, binding=0) uniform sampler2D sTexture;
// layout(location = 0) in struct { vec4 Color; vec2 UV; } In;
// void main() {
//     fColor = In.Color * texture(sTexture, In.UV.st).rrrr;
// }
const frag_spv = [_]u32{
	0x07230203,0x00010000,0x0008000b,0x0000001f,0x00000000,0x00020011,0x00000001,0x0006000b,
	0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
	0x0007000f,0x00000004,0x00000004,0x6e69616d,0x00000000,0x00000009,0x0000000d,0x00030010,
	0x00000004,0x00000007,0x00040047,0x00000009,0x0000001e,0x00000000,0x00040047,0x0000000d,
	0x0000001e,0x00000000,0x00040047,0x00000016,0x00000022,0x00000000,0x00040047,0x00000016,
	0x00000021,0x00000000,0x00020013,0x00000002,0x00030021,0x00000003,0x00000002,0x00030016,
	0x00000006,0x00000020,0x00040017,0x00000007,0x00000006,0x00000004,0x00040020,0x00000008,
	0x00000003,0x00000007,0x0004003b,0x00000008,0x00000009,0x00000003,0x00040017,0x0000000a,
	0x00000006,0x00000002,0x0004001e,0x0000000b,0x00000007,0x0000000a,0x00040020,0x0000000c,
	0x00000001,0x0000000b,0x0004003b,0x0000000c,0x0000000d,0x00000001,0x00040015,0x0000000e,
	0x00000020,0x00000001,0x0004002b,0x0000000e,0x0000000f,0x00000000,0x00040020,0x00000010,
	0x00000001,0x00000007,0x00090019,0x00000013,0x00000006,0x00000001,0x00000000,0x00000000,
	0x00000000,0x00000001,0x00000000,0x0003001b,0x00000014,0x00000013,0x00040020,0x00000015,
	0x00000000,0x00000014,0x0004003b,0x00000015,0x00000016,0x00000000,0x0004002b,0x0000000e,
	0x00000018,0x00000001,0x00040020,0x00000019,0x00000001,0x0000000a,0x00050036,0x00000002,
	0x00000004,0x00000000,0x00000003,0x000200f8,0x00000005,0x00050041,0x00000010,0x00000011,
	0x0000000d,0x0000000f,0x0004003d,0x00000007,0x00000012,0x00000011,0x0004003d,0x00000014,
	0x00000017,0x00000016,0x00050041,0x00000019,0x0000001a,0x0000000d,0x00000018,0x0004003d,
	0x0000000a,0x0000001b,0x0000001a,0x00050057,0x00000007,0x0000001c,0x00000017,0x0000001b,
	0x0009004f,0x00000007,0x0000001d,0x0000001c,0x0000001c,0x00000000,0x00000000,0x00000000,
	0x00000000,0x00050085,0x00000007,0x0000001e,0x00000012,0x0000001d,0x0003003e,0x00000009,
	0x0000001e,0x000100fd,0x00010038 };

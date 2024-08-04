pub const Display = @import("./Display.zig");
pub const Swapchain = @import("./Swapchain.zig");
const metrics = @import("build_options").vk_metrics;

const vk = @import("vulkan");

pub const required_vulkan_functions = [_]vk.ApiInfo {
    .{
        .instance_commands = .{
            .destroySurfaceKHR = true,
            .getPhysicalDeviceSurfacePresentModesKHR = true,
            .getPhysicalDeviceSurfaceFormatsKHR = true,
            .getPhysicalDeviceSurfaceSupportKHR = true,
            .getPhysicalDeviceSurfaceCapabilitiesKHR = true,
        },
        .device_commands = .{
            .getSwapchainImagesKHR = true,
            .createSwapchainKHR = true,
            .acquireNextImage2KHR = true,
            .queuePresentKHR = true,
            .destroySwapchainKHR = true,
        },
    },
} ++ if (metrics) [_]vk.ApiInfo {
    .{
        .device_commands = .{
            .cmdWriteTimestamp2 = true,
        }
    }
} else [_]vk.ApiInfo {};

pub const required_device_extensions = [_][*:0]const u8{
    vk.extensions.khr_swapchain.name,
};

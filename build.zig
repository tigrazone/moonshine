const std = @import("std");

// TODO: useful error messages on missing system deps

pub fn build(b: *std.Build) !void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    const optimize = b.standardOptimizeOption(.{});

    // packages/libraries we'll need below
    const vulkan = makeVulkanModule(b, target);
    const glfw = try makeGlfwLibrary(b, target);
    const cimgui = makeCImguiLibrary(b, target, glfw);
    const tinyexr = makeTinyExrLibrary(b, target);
    const wuffs = makeWuffsLibrary(b, target);
    const default_engine_options = EngineOptions.fromCli(b);

    var compiles = std.ArrayList(*std.Build.Step.Compile).init(b.allocator);

    // TODO: make custom test runner parallel + share some state across tests
    try compiles.append(blk: {
        var engine_options = default_engine_options;
        engine_options.window = false;
        engine_options.gui = false;
        const engine = makeEngineModule(b, vulkan, engine_options, target);

        const tests = b.addTest(.{
            .name = "tests",
            .root_source_file = b.path("src/lib/tests.zig"),
            .test_runner = b.path("src/lib/test_runner.zig"),
            .target = target,
            .optimize = optimize,
        });
        tests.root_module.addImport("vulkan", vulkan);
        tests.root_module.addImport("engine", engine);

        break :blk tests;
    });

    // online exe
    try compiles.append(blk: {
        var engine_options = default_engine_options;
        engine_options.vk_metrics = true;
        //engine_options.shader_source = .load; // for hot shader reload
        engine_options.shader_source = .embed;
        const engine = makeEngineModule(b, vulkan, engine_options, target);
        const exe = b.addExecutable(.{
            .name = "online",
            .root_source_file = b.path("src/bin/online.zig"),
            .target = target,
            .optimize = optimize,
        });
        exe.root_module.addImport("vulkan", vulkan);
        exe.root_module.addImport("engine", engine);
        glfw.add(&exe.root_module);
        glfw.add(engine);
        tinyexr.add(&exe.root_module);
        tinyexr.add(engine);
        cimgui.add(&exe.root_module);
        cimgui.add(engine);
        wuffs.add(&exe.root_module);
        wuffs.add(engine);

        break :blk exe;
    });

    // offline exe
    try compiles.append(blk: {
        var engine_options = default_engine_options;
        engine_options.window = false;
        engine_options.gui = false;
        const engine = makeEngineModule(b, vulkan, engine_options, target);
        const exe = b.addExecutable(.{
            .name = "offline",
            .root_source_file = b.path("src/bin/offline.zig"),
            .target = target,
            .optimize = optimize,
        });
        exe.root_module.addImport("vulkan", vulkan);
        exe.root_module.addImport("engine", engine);
        tinyexr.add(&exe.root_module);
        tinyexr.add(engine);
        wuffs.add(&exe.root_module);
        wuffs.add(engine);

        break :blk exe;
    });

    // hydra shared lib
    if (target.result.os.tag == .linux) {
        var engine_options = default_engine_options;
        engine_options.window = false;
        engine_options.gui = false;
        engine_options.shader_source = .embed;
        const engine = makeEngineModule(b, vulkan, engine_options, target);

        // once https://github.com/ziglang/zig/issues/9698 lands
        // wont need to make own header
        const zig_lib = b.addSharedLibrary(.{
            .name = "moonshine",
            .root_source_file = b.path("src/bin/hydra/hydra.zig"),
            .target = target,
            .optimize = optimize,
            .pic = true,
        });
        zig_lib.root_module.addImport("vulkan", vulkan);
        zig_lib.root_module.addImport("engine", engine);
        zig_lib.linkLibC();
        try compiles.append(zig_lib);

        const lib = b.addSharedLibrary(.{
            .name = "hdMoonshine",
            .target = target,
            .optimize = optimize,
        });
        lib.addCSourceFiles(.{
           .files = &.{
                "hydra/rendererPlugin.cpp",
                "hydra/renderDelegate.cpp",
                "hydra/renderPass.cpp",
                "hydra/renderBuffer.cpp",
                "hydra/mesh.cpp",
                "hydra/camera.cpp",
                "hydra/instancer.cpp",
                "hydra/material.cpp",
            },
        });
        lib.linkLibrary(zig_lib);

        // options
        const usd_dir = b.option([]const u8, "usd-path", "Where your USD SDK is installed.") orelse "../USD";
        const tbb_dir = b.option([]const u8, "tbb-path", "Where your TBB is installed.");
        const usd_monolithic = b.option(bool, "usd-monolithic", "Whether USD was built monolithically.") orelse false;

        // link against usd produced libraries
        lib.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ usd_dir, "lib/" }) });
        if (usd_monolithic) {
            lib.linkSystemLibrary("usd_ms");
        } else {
            lib.linkSystemLibrary("usd_hd");
            lib.linkSystemLibrary("usd_sdr");
            lib.linkSystemLibrary("usd_hio");
        }

        // include headers necessary for usd
        lib.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ usd_dir, "include/" }) });
        if (tbb_dir) |dir| lib.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ dir, "include/" }) });
        lib.defineCMacro("TBB_USE_DEBUG", "0"); // not sure why we need this

        // might need python headers if USD built with python support
        {
            var out_code: u8 = undefined;
            var iter = std.mem.splitScalar(u8, b.runAllowFail(&.{ "python3-config", "--includes" }, &out_code, .Inherit) catch b.runAllowFail(&.{ "python-config", "--includes" }, &out_code, .Inherit) catch "", ' ');
            while (iter.next()) |include_dir| lib.addSystemIncludePath(.{ .cwd_relative = include_dir[2..] });
        }

        // deal with the fact that USD is not (supposed to be) compiled with clang
        // make nicer once https://github.com/ziglang/zig/issues/3936
        {
            // configure necessary gnu macros
            lib.defineCMacro("ARCH_HAS_GNU_STL_EXTENSIONS", null);

            // link against stdlibc++
            lib.addObjectFile(.{ .cwd_relative = std.mem.trim(u8, b.run(&.{ "g++", "-print-file-name=libstdc++.so" }), &std.ascii.whitespace) });

            // need stdlibc++ include directories
            // i've had to do some arcane magic to figure out what to do here,
            // and i'm not even convinced it'll work on any system other than mine
            var first = true;
            var iter = std.mem.splitScalar(u8, runAllowFailStderr(b, &.{ "g++", "-E", "-Wp,-v", "-xc++", "/dev/null" }) catch "", '\n');
            while (iter.next()) |include_dir| if (include_dir.len > 0 and include_dir[0] == ' ') {
                if (first) {
                    lib.addIncludePath(.{ .cwd_relative = include_dir[1..] });
                } else {
                    lib.addSystemIncludePath(.{ .cwd_relative = include_dir[1..] });
                }
                first = false;
            };
        }

        const step = b.step("hydra", "Build hydra delegate");

        const install = b.addInstallArtifact(lib, .{ .dest_sub_path = "hdMoonshine.so" });
        step.dependOn(&install.step);

        const write_pluginfo_json = b.addWriteFiles();
        const pluginfo_file = write_pluginfo_json.add("plugInfo.json",
            \\{
            \\    "Plugins": [
            \\        {
            \\            "Info": {
            \\                "Types": {
            \\                    "HdMoonshinePlugin": {
            \\                        "bases": [
            \\                            "HdRendererPlugin"
            \\                        ],
            \\                        "displayName": "Moonshine",
            \\                        "priority": 1
            \\                    }
            \\                }
            \\            },
            \\            "LibraryPath": "hdMoonshine.so",
            \\            "Name": "HdMoonshine",
            \\            "ResourcePath": ".",
            \\            "Root": ".",
            \\            "Type": "library"
            \\        }
            \\    ]
            \\}
        );
        const install_pluginfo_json = b.addInstallLibFile(pluginfo_file, "plugInfo.json");
        step.dependOn(&install_pluginfo_json.step);
    }

    // create run step for all exes
    for (compiles.items) |exe| {
        if (exe.kind == .lib or exe.kind == .obj) continue;
        const run = b.addRunArtifact(exe);
        if (b.args) |args| run.addArgs(args);

        const step_name = try std.fmt.allocPrint(b.allocator, "run-{s}", .{ exe.name });
        const step_description = try std.fmt.allocPrint(b.allocator, "Run {s}", .{ exe.name });
        const step = b.step(step_name, step_description);
        step.dependOn(&run.step);
    }

    // create install step for all compiles
    for (compiles.items) |compile| {
        const install = b.addInstallArtifact(compile, .{});

        const step_name = try std.fmt.allocPrint(b.allocator, "install-{s}", .{ compile.name });
        const step_description = try std.fmt.allocPrint(b.allocator, "Install {s}", .{ compile.name });
        const step = b.step(step_name, step_description);
        step.dependOn(&install.step);
    }

    // create check step that type-checks all compiles
    // probably does a bit more atm but what can you do
    const check_step = b.step("check", "Type check all");
    for (compiles.items) |compile| {
        check_step.dependOn(&compile.step);
    }
}

pub fn runAllowFailStderr(self: *std.Build, argv: []const []const u8) ![]u8 {
    const max_output_size = 400 * 1024;
    var child = std.process.Child.init(argv, self.allocator);
    child.stdin_behavior = .Ignore;
    child.stdout_behavior = .Ignore;
    child.stderr_behavior = .Pipe;
    child.env_map = &self.graph.env_map;

    try child.spawn();

    const stderr = child.stderr.?.reader().readAllAlloc(self.allocator, max_output_size) catch {
        return error.ReadFailure;
    };
    errdefer self.allocator.free(stderr);

    const term = try child.wait();
    switch (term) {
        .Exited => |code| {
            if (code != 0) return error.ExitCodeFailure;
            return stderr;
        },
        .Signal, .Stopped, .Unknown => {
            return error.ProcessTerminated;
        },
    }
}

const ShaderSource = enum {
    embed, // embed SPIRV shaders into binary at compile time
    load,  // dynamically load shader and compile to SPIRV at runtime (but also check build-time correctness)
};

pub const EngineOptions = struct {
    const rt_shader_args = [_][]const u8 { "-T", ShaderType.ray_tracing.dxcProfile() };
    const compute_shader_args = [_][]const u8 { "-T", ShaderType.compute.dxcProfile() };
    const stdout_shader_args = [_][]const u8{ "-Fo", "/dev/stdout" }; // TODO: windows

    vk_validation: bool = false,
    vk_metrics: bool = false,
    shader_source: ShaderSource = .embed,
    rt_shader_compile_cmd: []const []const u8 = &(base_shader_compile_cmd ++ rt_shader_args ++ stdout_shader_args),
    compute_shader_compile_cmd: []const []const u8 = &(base_shader_compile_cmd ++ compute_shader_args ++ stdout_shader_args),

    // modules
    hrtsystem: bool = true,
    window: bool = true,
    gui: bool = true,

    fn fromCli(b: *std.Build) EngineOptions {
        var options = EngineOptions {};

        if (b.option(bool, "vk-validation", "Enable vulkan validation")) |vk_validation| {
            options.vk_validation = vk_validation;
        }

        return options;
    }
};

fn makeEngineModule(b: *std.Build, vk: *std.Build.Module, options: EngineOptions, target: std.Build.ResolvedTarget) *std.Build.Module {
    const zgltf = b.dependency("zgltf", .{}).module("zgltf");

    // actual engine
    const build_options = b.addOptions();
    build_options.addOption(bool, "vk_validation", options.vk_validation);
    build_options.addOption(bool, "vk_metrics", options.vk_metrics);
    build_options.addOption(ShaderSource, "shader_source", if (target.result.os.tag == .linux) options.shader_source else .embed); // hot reload currently only supported on linux
    build_options.addOption([]const []const u8, "rt_shader_compile_cmd", options.rt_shader_compile_cmd);  // shader compilation command to use if shaders are to be loaded at runtime
    build_options.addOption([]const []const u8, "compute_shader_compile_cmd", options.compute_shader_compile_cmd);  // shader compilation command to use if shaders are to be loaded at runtime
    build_options.addOption(bool, "window", options.window);
    build_options.addOption(bool, "gui", options.gui);
    build_options.addOption(bool, "hrtsystem", options.hrtsystem);

    var imports = std.ArrayList(std.Build.Module.Import).init(b.allocator);

    imports.appendSlice(&.{
        .{
            .name = "vulkan",
            .module = vk,
        },
        .{
            .name = "zgltf",
            .module = zgltf,
        },
        .{
            .name = "build_options",
            .module = build_options.createModule(),
        },
    }) catch @panic("OOM");

    imports.appendSlice(&.{
        compileShader(b, .ray_tracing, "hrtsystem/input.hlsl"),
        compileShader(b, .ray_tracing, "hrtsystem/main_pt.hlsl"),
        compileShader(b, .ray_tracing, "hrtsystem/main_direct.hlsl"),
        compileShader(b, .compute, "hrtsystem/background/equirectangular_to_equal_area.hlsl"),
        compileShader(b, .compute, "hrtsystem/background/luminance.hlsl"),
        compileShader(b, .compute, "hrtsystem/background/fold.hlsl"),
        compileShader(b, .compute, "hrtsystem/mesh_sampling/power.hlsl"),
        compileShader(b, .compute, "hrtsystem/mesh_sampling/fold.hlsl"),
    }) catch @panic("OOM");

    const module = b.createModule(.{
        .root_source_file = b.path("src/lib/engine.zig"),
        .imports = imports.items,
    });

    module.link_libc = true; // always needed to load vulkan

    return module;
}

fn makeVulkanModule(b: *std.Build, target: std.Build.ResolvedTarget) *std.Build.Module {
    const vulkan_zig = b.dependency("vulkan_zig", .{});
    const vulkan_headers = b.dependency("vulkan_headers", .{});
    const vk_generate_cmd = b.addRunArtifact(vulkan_zig.artifact("vulkan-zig-generator"));
    vk_generate_cmd.addFileArg(vulkan_headers.path("registry/vk.xml"));
    const vk_zig = vk_generate_cmd.addOutputFileArg("vk.zig");
    return b.addModule("vulkan-zig", .{
        .root_source_file = vk_zig,
        .target = target,
        .optimize = .ReleaseFast,
    });
}

const CLibrary = struct {
    include_path: std.Build.LazyPath,
    library: *std.Build.Step.Compile,

    fn add(self: CLibrary, module: *std.Build.Module) void {
        module.linkLibrary(self.library);
        module.addIncludePath(self.include_path);
    }
};

fn makeCImguiLibrary(b: *std.Build, target: std.Build.ResolvedTarget, glfw: CLibrary) CLibrary {
    const cimgui = b.dependency("cimgui", .{});
    const imgui = b.dependency("imgui", .{});

    const lib = b.addStaticLibrary(.{
        .name = "cimgui",
        .target = target,
        .optimize = .ReleaseFast,
    });
    lib.linkLibCpp();
    lib.addCSourceFiles(.{
        .root = cimgui.path(""),
        .files = &.{
            "cimgui.cpp",
        }
    });
    lib.addIncludePath(imgui.path(""));
    lib.addCSourceFiles(.{
        .root = imgui.path(""),
        .files = &.{
            "imgui.cpp",
            "imgui_draw.cpp",
            "imgui_demo.cpp",
            "imgui_widgets.cpp",
            "imgui_tables.cpp",
            "backends/imgui_impl_glfw.cpp",
        }, .flags = &.{
            "-DGLFW_INCLUDE_NONE",
            "-DIMGUI_IMPL_API=extern \"C\"",
        }
    });
    lib.addIncludePath(glfw.include_path);

    return CLibrary {
        .include_path = cimgui.path(""),
        .library = lib,
    };
}

fn makeTinyExrLibrary(b: *std.Build, target: std.Build.ResolvedTarget) CLibrary {
    const tinyexr = b.dependency("tinyexr", .{});
    const miniz_path = "deps/miniz/";

    const lib = b.addStaticLibrary(.{
        .name = "tinyexr",
        .target = target,
        .optimize = .ReleaseFast,
    });
    lib.linkLibCpp();
    lib.addIncludePath(tinyexr.path(miniz_path));
    lib.addCSourceFiles(.{
        .root = tinyexr.path(""),
        .files = &.{
            "tinyexr.cc",
            miniz_path ++ "miniz.c",
        },
    });

    return CLibrary {
        .include_path = tinyexr.path(""),
        .library = lib,
    };
}

fn makeWuffsLibrary(b: *std.Build, target: std.Build.ResolvedTarget) CLibrary {
    const base = b.dependency("wuffs", .{});

    const lib = b.addStaticLibrary(.{
        .name = "wuffs",
        .target = target,
        .optimize = .ReleaseFast,
    });
    lib.linkLibC();
    lib.addCSourceFiles(.{
        .root = base.path(""),
        .files = &.{
            "release/c/wuffs-v0.4.c",
        },
        .flags = &.{
            "-DWUFFS_IMPLEMENTATION",
        }
    });

    return CLibrary {
        .include_path = base.path("release/c/"),
        .library = lib,
    };
}

fn makeGlfwLibrary(b: *std.Build, target: std.Build.ResolvedTarget) !CLibrary {
    const glfw = b.dependency("glfw", .{});
    const lib = b.addStaticLibrary(.{
        .name = "glfw",
        .target = target,
        .optimize = .ReleaseFast,
    });

    const build_wayland = b.option(bool, "wayland", "Support Wayland on Linux. (default: true)") orelse true;
    const build_x11 = b.option(bool, "x11", "Support X11 on Linux. (default: true)") orelse true;

    if (!build_wayland and !build_x11) return error.NoSelectedLinuxDisplayServerProtocol;

    if (target.result.os.tag == .linux and build_wayland) {
        const wayland_include_path = generateWaylandHeaders(b, glfw.path(""));
        lib.addIncludePath(wayland_include_path);
    }

    // collect source files
    const sources = blk: {
        var sources = std.ArrayList([]const u8).init(b.allocator);

        const source_path = "src/";

        const general_sources = [_][]const u8 {
            source_path ++ "context.c",
            source_path ++ "init.c",
            source_path ++ "input.c",
            source_path ++ "monitor.c",
            source_path ++ "vulkan.c",
            source_path ++ "window.c",
            source_path ++ "egl_context.c",
            source_path ++ "osmesa_context.c",
            source_path ++ "platform.c",
            source_path ++ "null_init.c",
            source_path ++ "null_window.c",
            source_path ++ "null_joystick.c",
            source_path ++ "null_monitor.c",
        };

        const linux_sources = [_][]const u8 {
            source_path ++ "posix_time.c",
            source_path ++ "posix_thread.c",
            source_path ++ "posix_module.c",
            source_path ++ "posix_poll.c",
            source_path ++ "xkb_unicode.c",
            source_path ++ "linux_joystick.c",
        };

        const x11_sources = [_][]const u8 {
            source_path ++ "x11_init.c",
            source_path ++ "x11_monitor.c",
            source_path ++ "x11_window.c",
            source_path ++ "glx_context.c",
        };

        const wayland_sources = [_][]const u8 {
            source_path ++ "wl_init.c",
            source_path ++ "wl_monitor.c",
            source_path ++ "wl_window.c",
        };

        const windows_sources = [_][]const u8 {
            source_path ++ "win32_thread.c",
            source_path ++ "wgl_context.c",
            source_path ++ "win32_init.c",
            source_path ++ "win32_monitor.c",
            source_path ++ "win32_time.c",
            source_path ++ "win32_joystick.c",
            source_path ++ "win32_window.c",
            source_path ++ "win32_module.c",
        };

        try sources.appendSlice(&general_sources);

        if (target.result.os.tag == .linux) {
            try sources.appendSlice(&linux_sources);
            if (build_wayland) try sources.appendSlice(&wayland_sources);
            if (build_x11) try sources.appendSlice(&x11_sources);
        } else if (target.result.os.tag == .windows) try sources.appendSlice(&windows_sources);

        break :blk sources.items;
    };

    const flags = blk: {
        var flags = std.ArrayList([]const u8).init(b.allocator);

        if (target.result.os.tag == .linux) {
            if (build_wayland) try flags.append("-D_GLFW_WAYLAND");
            if (build_x11) try flags.append("-D_GLFW_X11");
        } else if (target.result.os.tag == .windows) try flags.append("-D_GLFW_WIN32");

        break :blk flags.items;
    };

    lib.addCSourceFiles(.{
        .root = glfw.path(""),
        .files = sources,
        .flags = flags,
    });

    // link and include necessary deps
    lib.linkLibC();

    if (target.result.os.tag == .linux) {
        if (build_wayland) lib.linkSystemLibrary("wayland-client");
        if (build_x11) lib.linkSystemLibrary("X11");
    } else if (target.result.os.tag == .windows) lib.linkSystemLibrary("gdi32");

    return CLibrary {
        .include_path = glfw.path("include"),
        .library = lib,
    };
}

fn generateWaylandHeaders(b: *std.Build, path: std.Build.LazyPath) std.Build.LazyPath {
    const protocols_dir = path.path(b, "deps/wayland");

    const write_file_step = b.addWriteFiles();
    write_file_step.step.name = "Write Wayland headers";

    generateWaylandHeader(b, write_file_step, protocols_dir, "xdg-shell");
    generateWaylandHeader(b, write_file_step, protocols_dir, "xdg-decoration-unstable-v1");
    generateWaylandHeader(b, write_file_step, protocols_dir, "xdg-activation-v1");
    generateWaylandHeader(b, write_file_step, protocols_dir, "viewporter");
    generateWaylandHeader(b, write_file_step, protocols_dir, "relative-pointer-unstable-v1");
    generateWaylandHeader(b, write_file_step, protocols_dir, "pointer-constraints-unstable-v1");
    generateWaylandHeader(b, write_file_step, protocols_dir, "idle-inhibit-unstable-v1");
    generateWaylandHeader(b, write_file_step, protocols_dir, "fractional-scale-v1");
    generateWaylandHeader(b, write_file_step, protocols_dir, "wayland");

    return write_file_step.getDirectory();
}

fn generateWaylandHeader(b: *std.Build, write_file_step: *std.Build.Step.WriteFile, protocols_dir: std.Build.LazyPath, protocol_name: []const u8) void {
    const in_xml = protocols_dir.path(b, b.fmt("{s}.xml", .{ protocol_name }));

    const out_source_name = b.fmt("{s}-client-protocol-code.h", .{ protocol_name });
    const gen_private_code_step = b.addSystemCommand(&.{ "wayland-scanner", "private-code" });
    gen_private_code_step.addFileArg(in_xml);
    const out_source = gen_private_code_step.addOutputFileArg(out_source_name);
    _ = write_file_step.addCopyFile(out_source, out_source_name);

    const out_header_name = b.fmt("{s}-client-protocol.h", .{ protocol_name });
    const gen_client_header_step = b.addSystemCommand(&.{ "wayland-scanner", "client-header" });
    gen_client_header_step.addFileArg(in_xml);
    const out_header = gen_client_header_step.addOutputFileArg(out_header_name);
    _ = write_file_step.addCopyFile(out_header, out_header_name);
}

const base_shader_compile_cmd = [_][]const u8 {
    "dxc",
    "-HV", "2021",
    "-spirv",
    "-fspv-target-env=vulkan1.3",
    "-fvk-use-scalar-layout",
    "-Ges", // strict mode
    "-WX", // treat warnings as errors
};

const ShaderType = enum {
    compute,
    ray_tracing,

    fn dxcProfile(self: ShaderType) []const u8 {
        return switch (self) {
            .compute => "cs_6_7",
            .ray_tracing => "lib_6_7",
        };
    }
};

fn compileShader(b: *std.Build, shader_type: ShaderType, path: []const u8) std.Build.Module.Import {
    const input_file_path = b.path(b.pathJoin(&.{ "src/lib/shaders", path }));

    const get_dependendies = std.Build.Step.Run.create(b, b.fmt("get dependencies of {s}", .{ path }));
    get_dependendies.addArgs(&base_shader_compile_cmd);
    get_dependendies.addArg("-T");
    get_dependendies.addArg(shader_type.dxcProfile());
    get_dependendies.addFileArg(input_file_path);
    get_dependendies.addArg("-MF");
    _ = get_dependendies.addDepFileOutputArg(b.fmt("{s}.d", .{ path }));

    const compile_shader = std.Build.Step.Run.create(b, b.fmt("compile {s}", .{ path }));
    compile_shader.addArgs(&base_shader_compile_cmd);
    compile_shader.addArg("-T");
    compile_shader.addArg(shader_type.dxcProfile());
    compile_shader.addFileArg(input_file_path);
    compile_shader.addArg("-Fo");
    const spv_file = compile_shader.addOutputFileArg(b.fmt("{s}.spv", .{ path }));

    compile_shader.step.dependOn(&get_dependendies.step);
    compile_shader.dep_output_file = get_dependendies.argv.getLast().output_file;

    return std.Build.Module.Import {
        .name = path,
        .module = std.Build.Module.create(b, .{
            .root_source_file = spv_file,
        }),
    };
}

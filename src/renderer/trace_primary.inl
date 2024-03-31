#pragma once

#include <core.inl>
#include <renderer/core.inl>

#include <voxels/particles/voxel_particles.inl>

DAXA_DECL_TASK_HEAD_BEGIN(TraceDepthPrepassCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, render_depth_prepass_image)
DAXA_DECL_TASK_HEAD_END
struct TraceDepthPrepassComputePush {
    DAXA_TH_BLOB(TraceDepthPrepassCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(TracePrimaryCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_3D, blue_noise_vec2)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, debug_texture)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, render_depth_prepass_image)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, velocity_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
struct TracePrimaryComputePush {
    DAXA_TH_BLOB(TracePrimaryCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(R32D32Blit)
DAXA_TH_IMAGE_INDEX(FRAGMENT_SHADER_SAMPLED, REGULAR_2D, input_tex)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, output_tex)
DAXA_DECL_TASK_HEAD_END
struct R32D32BlitPush {
    DAXA_TH_BLOB(R32D32Blit, uses)
};

struct Aabb {
    daxa_f32vec3 minimum;
    daxa_f32vec3 maximum;
};
DAXA_DECL_BUFFER_PTR(Aabb)

DAXA_DECL_TASK_HEAD_BEGIN(TestRt)
DAXA_TH_BUFFER_PTR(RAY_TRACING_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_TLAS_PTR(RAY_TRACING_SHADER_READ, tlas)
DAXA_TH_BUFFER_PTR(RAY_TRACING_SHADER_READ, daxa_BufferPtr(Aabb), aabbs)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, g_buffer_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, velocity_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, vs_normal_image_id)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, depth_image_id)
DAXA_DECL_TASK_HEAD_END
#if defined(DAXA_RAY_TRACING) || defined(__cplusplus)
struct TestRtPush {
    daxa_TlasId tlas;
    DAXA_TH_BLOB(TestRt, uses)
};
#endif

#if defined(__cplusplus)

struct GbufferRenderer {
    GbufferDepth gbuffer_depth;

    void next_frame() {
        gbuffer_depth.next_frame();
    }

    auto render(GpuContext &gpu_context, VoxelWorldBuffers &voxel_buffers)
        -> std::pair<GbufferDepth &, daxa::TaskImageView> {
        gbuffer_depth.gbuffer = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::R32G32B32A32_UINT,
            .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
            .name = "gbuffer",
        });
        gbuffer_depth.geometric_normal = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::A2B10G10R10_UNORM_PACK32,
            .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
            .name = "normal",
        });

        gbuffer_depth.downscaled_view_normal = std::nullopt;
        gbuffer_depth.downscaled_depth = std::nullopt;

        gbuffer_depth.depth = PingPongImage{};
        auto [depth_image, prev_depth_image] = gbuffer_depth.depth.get(
            gpu_context,
            {
                .format = daxa::Format::D32_SFLOAT,
                .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::DEPTH_STENCIL_ATTACHMENT,
                .name = "depth_image",
            });

        gpu_context.frame_task_graph.use_persistent_image(depth_image);
        gpu_context.frame_task_graph.use_persistent_image(prev_depth_image);

        auto velocity_image = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
            .name = "velocity_image",
        });

        auto depth_prepass_image = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::R32_SFLOAT,
            .size = {gpu_context.render_resolution.x / PREPASS_SCL, gpu_context.render_resolution.y / PREPASS_SCL, 1},
            .name = "depth_prepass_image",
        });

        gpu_context.add(ComputeTask<TraceDepthPrepassCompute::Task, TraceDepthPrepassComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"trace_primary.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{TraceDepthPrepassCompute::AT.gpu_input, gpu_context.task_input_buffer}},
                VOXELS_BUFFER_USES_ASSIGN(TraceDepthPrepassCompute, voxel_buffers),
                daxa::TaskViewVariant{std::pair{TraceDepthPrepassCompute::AT.render_depth_prepass_image, depth_prepass_image}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TraceDepthPrepassComputePush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(TraceDepthPrepassCompute::AT.render_depth_prepass_image).ids[0]).value();
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
                ti.recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
            },
        });

        auto temp_depth_image = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::R32_SFLOAT,
            .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
            .name = "temp_depth_image",
        });

        gpu_context.add(ComputeTask<TracePrimaryCompute::Task, TracePrimaryComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"trace_primary.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{TracePrimaryCompute::AT.gpu_input, gpu_context.task_input_buffer}},
                VOXELS_BUFFER_USES_ASSIGN(TracePrimaryCompute, voxel_buffers),
                daxa::TaskViewVariant{std::pair{TracePrimaryCompute::AT.blue_noise_vec2, gpu_context.task_blue_noise_vec2_image}},
                daxa::TaskViewVariant{std::pair{TracePrimaryCompute::AT.debug_texture, gpu_context.task_debug_texture}},
                daxa::TaskViewVariant{std::pair{TracePrimaryCompute::AT.render_depth_prepass_image, depth_prepass_image}},
                daxa::TaskViewVariant{std::pair{TracePrimaryCompute::AT.g_buffer_image_id, gbuffer_depth.gbuffer}},
                daxa::TaskViewVariant{std::pair{TracePrimaryCompute::AT.velocity_image_id, velocity_image}},
                daxa::TaskViewVariant{std::pair{TracePrimaryCompute::AT.vs_normal_image_id, gbuffer_depth.geometric_normal}},
                daxa::TaskViewVariant{std::pair{TracePrimaryCompute::AT.depth_image_id, temp_depth_image}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TracePrimaryComputePush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(TracePrimaryCompute::AT.g_buffer_image_id).ids[0]).value();
                ti.recorder.set_pipeline(pipeline);
                // debug_utils::Console::add_log(fmt::format("0 {}, {}", image_info.size.x, image_info.size.y));
                set_push_constant(ti, push);
                // debug_utils::Console::add_log(fmt::format("1 {}, {}", image_info.size.x, image_info.size.y));
                // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
                ti.recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
            },
        });

        struct TestRtTaskInfo {
            daxa::TlasId tlas_id;
            uint32_t *raygen_shader_binding_table_offset;
        };

        gpu_context.add(RayTracingTask<TestRt::Task, TestRtPush, TestRtTaskInfo>{
            .compile_info = daxa::RayTracingPipelineCompileInfo{
                .ray_gen_infos = {daxa::ShaderCompileInfo{.source = daxa::ShaderFile{"trace_primary.rt.glsl"}}},
                .intersection_infos = {daxa::ShaderCompileInfo{.source = daxa::ShaderFile{"trace_primary.rt.glsl"}}},
                .any_hit_infos = {daxa::ShaderCompileInfo{.source = daxa::ShaderFile{"trace_primary.rt.glsl"}}},
                .closest_hit_infos = {daxa::ShaderCompileInfo{.source = daxa::ShaderFile{"trace_primary.rt.glsl"}}},
                .miss_hit_infos = {daxa::ShaderCompileInfo{.source = daxa::ShaderFile{"trace_primary.rt.glsl"}}},
                // Groups are in order of their shader indices.
                // NOTE: The order of the groups is important! raygen, miss, hit, callable
                .shader_groups_infos = {
                    daxa::RayTracingShaderGroupInfo{
                        .type = daxa::ShaderGroup::GENERAL,
                        .general_shader_index = 0,
                    },
                    daxa::RayTracingShaderGroupInfo{
                        .type = daxa::ShaderGroup::GENERAL,
                        .general_shader_index = 4,
                    },
                    daxa::RayTracingShaderGroupInfo{
                        .type = daxa::ShaderGroup::PROCEDURAL_HIT_GROUP,
                        .closest_hit_shader_index = 3,
                        .any_hit_shader_index = 2,
                        .intersection_shader_index = 1,
                    },
                },
                .push_constant_size = sizeof(TestRtPush),
                .name = "basic ray tracing pipeline",
            },
            .views = std::array{
                daxa::TaskViewVariant{std::pair{TestRt::AT.gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{TestRt::AT.tlas, voxel_buffers.task_tlas}},
                daxa::TaskViewVariant{std::pair{TestRt::AT.aabbs, voxel_buffers.aabb_buffer.task_resource}},
                daxa::TaskViewVariant{std::pair{TestRt::AT.g_buffer_image_id, gbuffer_depth.gbuffer}},
                daxa::TaskViewVariant{std::pair{TestRt::AT.velocity_image_id, velocity_image}},
                daxa::TaskViewVariant{std::pair{TestRt::AT.vs_normal_image_id, gbuffer_depth.geometric_normal}},
                daxa::TaskViewVariant{std::pair{TestRt::AT.depth_image_id, temp_depth_image}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RayTracingPipeline &pipeline, TestRtPush &push, TestRtTaskInfo const &info) {
                auto const image_info = ti.device.info_image(ti.get(TestRt::AT.g_buffer_image_id).ids[0]).value();
                ti.recorder.set_pipeline(pipeline);
                push.tlas = info.tlas_id;
                set_push_constant(ti, push);
                ti.recorder.trace_rays({.width = image_info.size.x, .height = image_info.size.y, .depth = 1});
            },
            .info = TestRtTaskInfo{
                .tlas_id = voxel_buffers.tlas,
            },
        });

        gpu_context.add(RasterTask<R32D32Blit::Task, R32D32BlitPush, NoTaskInfo>{
            .vert_source = daxa::ShaderFile{"FULL_SCREEN_TRIANGLE_VERTEX_SHADER"},
            .frag_source = daxa::ShaderFile{"R32_D32_BLIT"},
            .depth_test = daxa::DepthTestInfo{
                .depth_attachment_format = daxa::Format::D32_SFLOAT,
                .enable_depth_write = true,
                .depth_test_compare_op = daxa::CompareOp::ALWAYS,
            },
            .views = std::array{
                daxa::TaskViewVariant{std::pair{R32D32Blit::AT.input_tex, temp_depth_image}},
                daxa::TaskViewVariant{std::pair{R32D32Blit::AT.output_tex, depth_image}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::RasterPipeline &pipeline, R32D32BlitPush &push, NoTaskInfo const &) {
                auto render_image = ti.get(R32D32Blit::AT.output_tex).ids[0];
                auto const image_info = ti.device.info_image(render_image).value();
                auto renderpass_recorder = std::move(ti.recorder).begin_renderpass({
                    .depth_attachment = {{.image_view = ti.get(R32D32Blit::AT.output_tex).view_ids[0], .load_op = daxa::AttachmentLoadOp::CLEAR, .clear_value = std::array{0.0f, 0.0f, 0.0f, 0.0f}}},
                    .render_area = {.x = 0, .y = 0, .width = image_info.size.x, .height = image_info.size.y},
                });
                renderpass_recorder.set_pipeline(pipeline);
                set_push_constant(ti, renderpass_recorder, push);
                renderpass_recorder.draw({.vertex_count = 3});
                ti.recorder = std::move(renderpass_recorder).end_renderpass();
            },
        });

        debug_utils::DebugDisplay::add_pass({.name = "depth_prepass", .task_image_id = depth_prepass_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        debug_utils::DebugDisplay::add_pass({.name = "gbuffer", .task_image_id = gbuffer_depth.gbuffer, .type = DEBUG_IMAGE_TYPE_GBUFFER});
        debug_utils::DebugDisplay::add_pass({.name = "temp_depth_image", .task_image_id = temp_depth_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        debug_utils::DebugDisplay::add_pass({.name = "depth", .task_image_id = depth_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        debug_utils::DebugDisplay::add_pass({.name = "geometric_normal", .task_image_id = gbuffer_depth.geometric_normal, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        debug_utils::DebugDisplay::add_pass({.name = "velocity", .task_image_id = velocity_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        return {gbuffer_depth, velocity_image};
    }
};

#endif

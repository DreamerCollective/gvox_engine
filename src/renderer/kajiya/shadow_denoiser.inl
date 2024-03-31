#pragma once

#include <core.inl>
#include <renderer/core.inl>

DAXA_DECL_TASK_HEAD_BEGIN(ShadowBitPackCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, output_tex)
DAXA_DECL_TASK_HEAD_END
struct ShadowBitPackComputePush {
    daxa_f32vec4 input_tex_size;
    daxa_u32vec2 bitpacked_shadow_mask_extent;
    DAXA_TH_BLOB(ShadowBitPackCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(ShadowSpatialFilterCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, meta_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, geometric_normal_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, output_tex)
DAXA_DECL_TASK_HEAD_END
struct ShadowSpatialFilterComputePush {
    daxa_f32vec4 input_tex_size;
    daxa_u32vec2 bitpacked_shadow_mask_extent;
    daxa_u32 step_size;
    DAXA_TH_BLOB(ShadowSpatialFilterCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(ShadowTemporalFilterCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, shadow_mask_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, bitpacked_shadow_mask_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, prev_moments_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, prev_accum_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojection_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, output_moments_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, temporal_output_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, meta_output_tex)
DAXA_DECL_TASK_HEAD_END
struct ShadowTemporalFilterComputePush {
    daxa_f32vec4 input_tex_size;
    daxa_u32vec2 bitpacked_shadow_mask_extent;
    DAXA_TH_BLOB(ShadowTemporalFilterCompute, uses)
};

#if defined(__cplusplus)

struct ShadowDenoiser {
    PingPongImage ping_pong_moments_image;
    PingPongImage ping_pong_accum_image;

    void next_frame() {
        ping_pong_moments_image.swap();
        ping_pong_accum_image.swap();
    }

    auto denoise_shadow_mask(GpuContext &gpu_context, GbufferDepth const &gbuffer_depth, daxa::TaskImageView shadow_mask, daxa::TaskImageView reprojection_map) -> daxa::TaskImageView {
        auto bitpacked_shadow_mask_extent = daxa_u32vec2{(gpu_context.render_resolution.x + 7) / 8, (gpu_context.render_resolution.y + 3) / 4};

        auto bitpacked_shadows_image = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::R32_UINT,
            .size = {bitpacked_shadow_mask_extent.x, bitpacked_shadow_mask_extent.y, 1},
            .name = "bitpacked_shadows_image",
        });
        debug_utils::DebugDisplay::add_pass({.name = "shadow bitpack", .task_image_id = bitpacked_shadows_image, .type = DEBUG_IMAGE_TYPE_SHADOW_BITMAP});

        struct ShadowBitPackComputeInfo {
            daxa_u32vec2 bitpacked_shadow_mask_extent;
        };
        gpu_context.add(ComputeTask<ShadowBitPackCompute::Task, ShadowBitPackComputePush, ShadowBitPackComputeInfo>{
            .source = daxa::ShaderFile{"kajiya/shadow_denoiser.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{ShadowBitPackCompute::AT.gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{ShadowBitPackCompute::AT.input_tex, shadow_mask}},
                daxa::TaskViewVariant{std::pair{ShadowBitPackCompute::AT.output_tex, bitpacked_shadows_image}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, ShadowBitPackComputePush &push, ShadowBitPackComputeInfo const &info) {
                auto const image_info = ti.device.info_image(ti.get(ShadowBitPackCompute::AT.input_tex).ids[0]).value();
                push.input_tex_size = daxa_f32vec4{float(image_info.size.x), float(image_info.size.y), 0.0f, 0.0f};
                push.input_tex_size.z = 1.0f / push.input_tex_size.x;
                push.input_tex_size.w = 1.0f / push.input_tex_size.y;
                push.bitpacked_shadow_mask_extent = info.bitpacked_shadow_mask_extent;
                auto render_size = daxa_u32vec2{info.bitpacked_shadow_mask_extent.x * 2, info.bitpacked_shadow_mask_extent.y};
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                // assert((render_size.x % 8) == 0 && (render_size.y % 4) == 0);
                ti.recorder.dispatch({(render_size.x + 7) / 8, (render_size.y + 3) / 4});
            },
            .info = {
                .bitpacked_shadow_mask_extent = bitpacked_shadow_mask_extent,
            },
        });

        ping_pong_moments_image = PingPongImage{};
        auto [moments_image, prev_moments_image] = ping_pong_moments_image.get(
            gpu_context,
            {
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .name = "moments_image",
            });
        gpu_context.frame_task_graph.use_persistent_image(moments_image);
        gpu_context.frame_task_graph.use_persistent_image(prev_moments_image);

        ping_pong_accum_image = PingPongImage{};
        auto [accum_image, prev_accum_image] = ping_pong_accum_image.get(
            gpu_context,
            {
                .format = daxa::Format::R16G16_SFLOAT,
                .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .name = "accum_image",
            });
        gpu_context.frame_task_graph.use_persistent_image(accum_image);
        gpu_context.frame_task_graph.use_persistent_image(prev_accum_image);

        clear_task_images(gpu_context.device, std::array{prev_moments_image, prev_accum_image});

        auto spatial_input_image = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::R16G16_SFLOAT,
            .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
            .name = "spatial_input_image",
        });
        auto shadow_denoise_intermediary_1 = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::R16G16_SFLOAT,
            .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
            .name = "shadow_denoise_intermediary_1",
        });

        auto metadata_image = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::R32_UINT,
            .size = {bitpacked_shadow_mask_extent.x, bitpacked_shadow_mask_extent.y, 1},
            .name = "metadata_image",
        });

        gpu_context.add(ComputeTask<ShadowTemporalFilterCompute::Task, ShadowTemporalFilterComputePush, ShadowBitPackComputeInfo>{
            .source = daxa::ShaderFile{"kajiya/shadow_denoiser.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{ShadowTemporalFilterCompute::AT.gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{ShadowTemporalFilterCompute::AT.shadow_mask_tex, shadow_mask}},
                daxa::TaskViewVariant{std::pair{ShadowTemporalFilterCompute::AT.bitpacked_shadow_mask_tex, bitpacked_shadows_image}},
                daxa::TaskViewVariant{std::pair{ShadowTemporalFilterCompute::AT.prev_moments_tex, prev_moments_image}},
                daxa::TaskViewVariant{std::pair{ShadowTemporalFilterCompute::AT.prev_accum_tex, prev_accum_image}},
                daxa::TaskViewVariant{std::pair{ShadowTemporalFilterCompute::AT.reprojection_tex, reprojection_map}},
                daxa::TaskViewVariant{std::pair{ShadowTemporalFilterCompute::AT.output_moments_tex, moments_image}},
                daxa::TaskViewVariant{std::pair{ShadowTemporalFilterCompute::AT.temporal_output_tex, spatial_input_image}},
                daxa::TaskViewVariant{std::pair{ShadowTemporalFilterCompute::AT.meta_output_tex, metadata_image}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, ShadowTemporalFilterComputePush &push, ShadowBitPackComputeInfo const &info) {
                auto const image_info = ti.device.info_image(ti.get(ShadowTemporalFilterCompute::AT.output_moments_tex).ids[0]).value();
                auto const input_image_info = ti.device.info_image(ti.get(ShadowTemporalFilterCompute::AT.shadow_mask_tex).ids[0]).value();
                push.input_tex_size = daxa_f32vec4{float(input_image_info.size.x), float(input_image_info.size.y), 0.0f, 0.0f};
                push.input_tex_size.z = 1.0f / push.input_tex_size.x;
                push.input_tex_size.w = 1.0f / push.input_tex_size.y;
                push.bitpacked_shadow_mask_extent = info.bitpacked_shadow_mask_extent;
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
                ti.recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 3) / 4});
            },
            .info = {
                .bitpacked_shadow_mask_extent = bitpacked_shadow_mask_extent,
            },
        });

        struct ShadowSpatialFilterComputeInfo {
            daxa_u32 step_size;
            daxa_u32vec2 bitpacked_shadow_mask_extent;
        };
        auto shadow_spatial_task_callback = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, ShadowSpatialFilterComputePush &push, ShadowSpatialFilterComputeInfo const &info) {
            auto const image_info = ti.device.info_image(ti.get(ShadowSpatialFilterCompute::AT.output_tex).ids[0]).value();
            auto const input_image_info = ti.device.info_image(ti.get(ShadowSpatialFilterCompute::AT.input_tex).ids[0]).value();
            push.step_size = info.step_size;
            push.input_tex_size = daxa_f32vec4{float(input_image_info.size.x), float(input_image_info.size.y), 0.0f, 0.0f};
            push.input_tex_size.z = 1.0f / push.input_tex_size.x;
            push.input_tex_size.w = 1.0f / push.input_tex_size.y;
            push.bitpacked_shadow_mask_extent = info.bitpacked_shadow_mask_extent;
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
            ti.recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
        };

        gpu_context.add(ComputeTask<ShadowSpatialFilterCompute::Task, ShadowSpatialFilterComputePush, ShadowSpatialFilterComputeInfo>{
            .source = daxa::ShaderFile{"kajiya/shadow_denoiser.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{ShadowSpatialFilterCompute::AT.gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{ShadowSpatialFilterCompute::AT.input_tex, spatial_input_image}},
                daxa::TaskViewVariant{std::pair{ShadowSpatialFilterCompute::AT.meta_tex, metadata_image}},
                daxa::TaskViewVariant{std::pair{ShadowSpatialFilterCompute::AT.geometric_normal_tex, gbuffer_depth.geometric_normal}},
                daxa::TaskViewVariant{std::pair{ShadowSpatialFilterCompute::AT.depth_tex, gbuffer_depth.depth.current()}},
                daxa::TaskViewVariant{std::pair{ShadowSpatialFilterCompute::AT.output_tex, accum_image}},
            },
            .callback_ = shadow_spatial_task_callback,
            .info = {
                .step_size = 1,
                .bitpacked_shadow_mask_extent = bitpacked_shadow_mask_extent,
            },
        });

        gpu_context.add(ComputeTask<ShadowSpatialFilterCompute::Task, ShadowSpatialFilterComputePush, ShadowSpatialFilterComputeInfo>{
            .source = daxa::ShaderFile{"kajiya/shadow_denoiser.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{ShadowSpatialFilterCompute::AT.gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{ShadowSpatialFilterCompute::AT.input_tex, accum_image}},
                daxa::TaskViewVariant{std::pair{ShadowSpatialFilterCompute::AT.meta_tex, metadata_image}},
                daxa::TaskViewVariant{std::pair{ShadowSpatialFilterCompute::AT.geometric_normal_tex, gbuffer_depth.geometric_normal}},
                daxa::TaskViewVariant{std::pair{ShadowSpatialFilterCompute::AT.depth_tex, gbuffer_depth.depth.current()}},
                daxa::TaskViewVariant{std::pair{ShadowSpatialFilterCompute::AT.output_tex, shadow_denoise_intermediary_1}},
            },
            .callback_ = shadow_spatial_task_callback,
            .info = {
                .step_size = 2,
                .bitpacked_shadow_mask_extent = bitpacked_shadow_mask_extent,
            },
        });

        gpu_context.add(ComputeTask<ShadowSpatialFilterCompute::Task, ShadowSpatialFilterComputePush, ShadowSpatialFilterComputeInfo>{
            .source = daxa::ShaderFile{"kajiya/shadow_denoiser.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{ShadowSpatialFilterCompute::AT.gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{ShadowSpatialFilterCompute::AT.input_tex, shadow_denoise_intermediary_1}},
                daxa::TaskViewVariant{std::pair{ShadowSpatialFilterCompute::AT.meta_tex, metadata_image}},
                daxa::TaskViewVariant{std::pair{ShadowSpatialFilterCompute::AT.geometric_normal_tex, gbuffer_depth.geometric_normal}},
                daxa::TaskViewVariant{std::pair{ShadowSpatialFilterCompute::AT.depth_tex, gbuffer_depth.depth.current()}},
                daxa::TaskViewVariant{std::pair{ShadowSpatialFilterCompute::AT.output_tex, spatial_input_image}},
            },
            .callback_ = shadow_spatial_task_callback,
            .info = {
                .step_size = 4,
                .bitpacked_shadow_mask_extent = bitpacked_shadow_mask_extent,
            },
        });

        debug_utils::DebugDisplay::add_pass({.name = "shadow temporal", .task_image_id = moments_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        debug_utils::DebugDisplay::add_pass({.name = "shadow spatial0", .task_image_id = accum_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        debug_utils::DebugDisplay::add_pass({.name = "shadow spatial1", .task_image_id = shadow_denoise_intermediary_1, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        debug_utils::DebugDisplay::add_pass({.name = "shadow spatial2", .task_image_id = spatial_input_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        return spatial_input_image;
    }
};

#endif

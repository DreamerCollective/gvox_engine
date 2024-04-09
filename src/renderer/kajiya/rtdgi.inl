#pragma once

#include <core.inl>
#include <renderer/core.inl>
#include <renderer/kajiya/ircache.inl>

DAXA_DECL_TASK_HEAD_BEGIN(RtdgiFullresReprojectCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojection_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, output_tex)
DAXA_DECL_TASK_HEAD_END
struct RtdgiFullresReprojectComputePush {
    daxa_f32vec4 output_tex_size;
    DAXA_TH_BLOB(RtdgiFullresReprojectCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(RtdgiValidateCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, half_view_normal_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojected_gi_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, reservoir_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reservoir_ray_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_3D, blue_noise_vec2)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
IRCACHE_USE_BUFFERS(COMPUTE)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, CUBE, sky_cube_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance_lut)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, irradiance_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, ray_orig_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, rt_history_invalidity_out_tex)
DAXA_DECL_TASK_HEAD_END
struct RtdgiValidateComputePush {
    daxa_f32vec4 gbuffer_tex_size;
    DAXA_TH_BLOB(RtdgiValidateCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(RtdgiValidateRt)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(RAY_TRACING_SHADER_READ, daxa_BufferPtr(daxa_BufferPtr(BlasGeom)), geometry_pointers)
DAXA_TH_BUFFER_PTR(RAY_TRACING_SHADER_READ, daxa_BufferPtr(daxa_BufferPtr(VoxelBrickAttribs)), attribute_pointers)
DAXA_TH_BUFFER_PTR(RAY_TRACING_SHADER_READ, daxa_BufferPtr(daxa_f32vec3), blas_transforms)
DAXA_TH_TLAS_PTR(RAY_TRACING_SHADER_READ, tlas)
IRCACHE_USE_BUFFERS(RAY_TRACING)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_SAMPLED, REGULAR_2D, half_view_normal_tex)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_SAMPLED, REGULAR_2D, depth_tex)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_SAMPLED, REGULAR_2D, reprojected_gi_tex)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_STORAGE_READ_WRITE, REGULAR_2D, reservoir_tex)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_SAMPLED, REGULAR_2D, reservoir_ray_history_tex)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_SAMPLED, REGULAR_3D, blue_noise_vec2)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_SAMPLED, CUBE, sky_cube_tex)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_SAMPLED, REGULAR_2D, transmittance_lut)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_STORAGE_READ_WRITE, REGULAR_2D, irradiance_history_tex)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_STORAGE_READ_WRITE, REGULAR_2D, ray_orig_history_tex)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, rt_history_invalidity_out_tex)
DAXA_DECL_TASK_HEAD_END
struct RtdgiValidateRtPush {
    daxa_f32vec4 gbuffer_tex_size;
    DAXA_TH_BLOB(RtdgiValidateRt, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(RtdgiTraceCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, half_view_normal_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojected_gi_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojection_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_3D, blue_noise_vec2)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, CUBE, sky_cube_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance_lut)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
IRCACHE_USE_BUFFERS(COMPUTE)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, candidate_irradiance_out_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, candidate_normal_out_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, candidate_hit_out_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, rt_history_invalidity_in_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, rt_history_invalidity_out_tex)
DAXA_DECL_TASK_HEAD_END
struct RtdgiTraceComputePush {
    daxa_f32vec4 gbuffer_tex_size;
    DAXA_TH_BLOB(RtdgiTraceCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(RtdgiTraceRt)
DAXA_TH_BUFFER_PTR(RAY_TRACING_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(RAY_TRACING_SHADER_READ, daxa_BufferPtr(daxa_BufferPtr(BlasGeom)), geometry_pointers)
DAXA_TH_BUFFER_PTR(RAY_TRACING_SHADER_READ, daxa_BufferPtr(daxa_BufferPtr(VoxelBrickAttribs)), attribute_pointers)
DAXA_TH_BUFFER_PTR(RAY_TRACING_SHADER_READ, daxa_BufferPtr(daxa_f32vec3), blas_transforms)
DAXA_TH_TLAS_PTR(RAY_TRACING_SHADER_READ, tlas)
IRCACHE_USE_BUFFERS(RAY_TRACING)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_SAMPLED, REGULAR_2D, half_view_normal_tex)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_SAMPLED, REGULAR_2D, depth_tex)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_SAMPLED, REGULAR_2D, reprojected_gi_tex)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_SAMPLED, REGULAR_2D, reprojection_tex)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_SAMPLED, REGULAR_3D, blue_noise_vec2)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_SAMPLED, CUBE, sky_cube_tex)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_SAMPLED, REGULAR_2D, transmittance_lut)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, candidate_irradiance_out_tex)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, candidate_normal_out_tex)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, candidate_hit_out_tex)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_SAMPLED, REGULAR_2D, rt_history_invalidity_in_tex)
DAXA_TH_IMAGE_INDEX(RAY_TRACING_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, rt_history_invalidity_out_tex)
DAXA_DECL_TASK_HEAD_END
struct RtdgiTraceRtPush {
    daxa_f32vec4 gbuffer_tex_size;
    DAXA_TH_BLOB(RtdgiTraceRt, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(RtdgiValidityIntegrateCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojection_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, half_view_normal_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, half_depth_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, output_tex)
DAXA_DECL_TASK_HEAD_END
struct RtdgiValidityIntegrateComputePush {
    daxa_f32vec4 gbuffer_tex_size;
    daxa_f32vec4 output_tex_size;
    DAXA_TH_BLOB(RtdgiValidityIntegrateCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(RtdgiRestirTemporalCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, half_view_normal_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, candidate_radiance_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, candidate_normal_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, candidate_hit_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, radiance_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, ray_orig_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, ray_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reservoir_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojection_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, hit_normal_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, candidate_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, rt_invalidity_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, radiance_out_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, ray_orig_output_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, ray_output_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, hit_normal_output_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, reservoir_out_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, candidate_out_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, temporal_reservoir_packed_tex)
DAXA_DECL_TASK_HEAD_END
struct RtdgiRestirTemporalComputePush {
    daxa_f32vec4 gbuffer_tex_size;
    DAXA_TH_BLOB(RtdgiRestirTemporalCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(RtdgiRestirSpatialCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reservoir_input_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, bounced_radiance_input_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, half_view_normal_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, half_depth_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, half_ssao_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, temporal_reservoir_packed_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojected_gi_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, reservoir_output_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, bounced_radiance_output_tex)
DAXA_DECL_TASK_HEAD_END
struct RtdgiRestirSpatialComputePush {
    daxa_f32vec4 gbuffer_tex_size;
    daxa_f32vec4 output_tex_size;
    daxa_u32 spatial_reuse_pass_idx;
    // Only done in the last spatial resampling pass
    daxa_u32 perform_occlusion_raymarch;
    daxa_u32 occlusion_raymarch_importance_only;
    DAXA_TH_BLOB(RtdgiRestirSpatialCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(RtdgiRestirResolveCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_3D, blue_noise_vec2)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, radiance_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reservoir_input_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, gbuffer_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, half_view_normal_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, half_depth_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, ssao_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, candidate_radiance_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, candidate_hit_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, temporal_reservoir_packed_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, bounced_radiance_input_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, irradiance_output_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, rtdgi_debug_image)
DAXA_DECL_TASK_HEAD_END
struct RtdgiRestirResolveComputePush {
    daxa_f32vec4 gbuffer_tex_size;
    daxa_f32vec4 output_tex_size;
    DAXA_TH_BLOB(RtdgiRestirResolveCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(RtdgiTemporalFilterCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, variance_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojection_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, rt_history_invalidity_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, output_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, history_output_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, variance_history_output_tex)
DAXA_DECL_TASK_HEAD_END
struct RtdgiTemporalFilterComputePush {
    daxa_f32vec4 output_tex_size;
    daxa_f32vec4 gbuffer_tex_size;
    DAXA_TH_BLOB(RtdgiTemporalFilterCompute, uses)
};

DAXA_DECL_TASK_HEAD_BEGIN(RtdgiSpatialFilterCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, ssao_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, geometric_normal_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, output_tex)
DAXA_DECL_TASK_HEAD_END
struct RtdgiSpatialFilterComputePush {
    daxa_f32vec4 output_tex_size;
    DAXA_TH_BLOB(RtdgiSpatialFilterCompute, uses)
};

#if defined(__cplusplus)

struct ReprojectedRtdgi {
    daxa::TaskImageView reprojected_history_tex;
    daxa::TaskImageView temporal_output_tex;
};

struct RtdgiCandidates {
    daxa::TaskImageView candidate_radiance_tex;
    daxa::TaskImageView candidate_normal_tex;
    daxa::TaskImageView candidate_hit_tex;
};
struct RtdgiOutput {
    daxa::TaskImageView screen_irradiance_tex;
    RtdgiCandidates candidates;
};

struct RtdgiRenderer {
    PingPongImage temporal_radiance_tex;
    PingPongImage temporal_ray_orig_tex;
    PingPongImage temporal_ray_tex;
    PingPongImage pp_temporal_reservoir_tex;
    PingPongImage temporal_candidate_tex;

    PingPongImage temporal_invalidity_tex;

    PingPongImage temporal2_tex;
    PingPongImage temporal2_variance_tex;
    PingPongImage temporal_hit_normal_tex;

    daxa_u32 spatial_reuse_pass_count = 2;
    bool use_raytraced_reservoir_visibility = false;

    void next_frame() {
        temporal2_tex.swap();
        temporal_hit_normal_tex.swap();
        temporal_candidate_tex.swap();
        temporal_invalidity_tex.swap();
        temporal_radiance_tex.swap();
        temporal_ray_orig_tex.swap();
        temporal_ray_tex.swap();
        pp_temporal_reservoir_tex.swap();
        temporal2_variance_tex.swap();
    }

    auto reproject(GpuContext &gpu_context, daxa::TaskImageView reprojection_map) -> ReprojectedRtdgi {
        temporal2_tex = PingPongImage{};
        auto [temporal_output_tex, history_tex] = temporal2_tex.get(
            gpu_context,
            {
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .name = "temporal2_tex",
            });
        gpu_context.frame_task_graph.use_persistent_image(temporal_output_tex);
        gpu_context.frame_task_graph.use_persistent_image(history_tex);
        clear_task_images(gpu_context.device, std::array{temporal_output_tex, history_tex});

        auto reprojected_history_tex = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
            .name = "reprojected_history_tex",
        });

        gpu_context.add(ComputeTask<RtdgiFullresReprojectCompute::Task, RtdgiFullresReprojectComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"kajiya/rtdgi/fullres_reproject.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{RtdgiFullresReprojectCompute::AT.gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{RtdgiFullresReprojectCompute::AT.input_tex, history_tex}},
                daxa::TaskViewVariant{std::pair{RtdgiFullresReprojectCompute::AT.reprojection_tex, reprojection_map}},
                daxa::TaskViewVariant{std::pair{RtdgiFullresReprojectCompute::AT.output_tex, reprojected_history_tex}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, RtdgiFullresReprojectComputePush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(RtdgiFullresReprojectCompute::AT.output_tex).ids[0]).value();
                ti.recorder.set_pipeline(pipeline);
                push.output_tex_size = extent_inv_extent_2d(image_info);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
            },
        });
        debug_utils::DebugDisplay::add_pass({.name = "rtdgi reprojected history", .task_image_id = reprojected_history_tex, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        return ReprojectedRtdgi{
            reprojected_history_tex,
            temporal_output_tex,
        };
    }

    auto temporal(
        GpuContext &gpu_context,
        daxa::TaskImageView input_color,
        GbufferDepth &gbuffer_depth,
        daxa::TaskImageView reprojection_map,
        daxa::TaskImageView reprojected_history_tex,
        daxa::TaskImageView rt_history_invalidity_tex,
        daxa::TaskImageView temporal_output_tex) -> daxa::TaskImageView {
        temporal2_variance_tex = PingPongImage{};
        auto [temporal_variance_output_tex, variance_history_tex] = temporal2_variance_tex.get(
            gpu_context,
            {
                .format = daxa::Format::R16G16_SFLOAT,
                .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .name = "temporal2_variance_tex",
            });
        gpu_context.frame_task_graph.use_persistent_image(temporal_variance_output_tex);
        gpu_context.frame_task_graph.use_persistent_image(variance_history_tex);
        clear_task_images(gpu_context.device, std::array{temporal_variance_output_tex, variance_history_tex});

        auto temporal_filtered_tex = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
            .name = "temporal_filtered_tex",
        });

        gpu_context.add(ComputeTask<RtdgiTemporalFilterCompute::Task, RtdgiTemporalFilterComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"kajiya/rtdgi/temporal_filter.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{RtdgiTemporalFilterCompute::AT.gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{RtdgiTemporalFilterCompute::AT.input_tex, input_color}},
                daxa::TaskViewVariant{std::pair{RtdgiTemporalFilterCompute::AT.history_tex, reprojected_history_tex}},
                daxa::TaskViewVariant{std::pair{RtdgiTemporalFilterCompute::AT.variance_history_tex, variance_history_tex}},
                daxa::TaskViewVariant{std::pair{RtdgiTemporalFilterCompute::AT.reprojection_tex, reprojection_map}},
                daxa::TaskViewVariant{std::pair{RtdgiTemporalFilterCompute::AT.rt_history_invalidity_tex, rt_history_invalidity_tex}},
                daxa::TaskViewVariant{std::pair{RtdgiTemporalFilterCompute::AT.output_tex, temporal_filtered_tex}},
                daxa::TaskViewVariant{std::pair{RtdgiTemporalFilterCompute::AT.history_output_tex, temporal_output_tex}},
                daxa::TaskViewVariant{std::pair{RtdgiTemporalFilterCompute::AT.variance_history_output_tex, temporal_variance_output_tex}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, RtdgiTemporalFilterComputePush &push, NoTaskInfo const &) {
                auto const image_info = ti.device.info_image(ti.get(RtdgiTemporalFilterCompute::AT.reprojection_tex).ids[0]).value();
                auto const out_image_info = ti.device.info_image(ti.get(RtdgiTemporalFilterCompute::AT.history_output_tex).ids[0]).value();
                ti.recorder.set_pipeline(pipeline);
                push.gbuffer_tex_size = extent_inv_extent_2d(image_info);
                push.output_tex_size = extent_inv_extent_2d(out_image_info);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(out_image_info.size.x + 15) / 16, (out_image_info.size.y + 15) / 16});
            },
        });
        debug_utils::DebugDisplay::add_pass({.name = "rtdgi temporal filter", .task_image_id = temporal_filtered_tex, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        return temporal_filtered_tex;
    }

    auto spatial(
        GpuContext &gpu_context,
        daxa::TaskImageView input_color,
        GbufferDepth &gbuffer_depth,
        daxa::TaskImageView ssao_tex) -> daxa::TaskImageView {

        auto spatial_filtered_tex = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
            .name = "spatial_filtered_tex",
        });

        gpu_context.add(ComputeTask<RtdgiSpatialFilterCompute::Task, RtdgiSpatialFilterComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"kajiya/rtdgi/spatial_filter.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{RtdgiSpatialFilterCompute::AT.gpu_input, gpu_context.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{RtdgiSpatialFilterCompute::AT.input_tex, input_color}},
                daxa::TaskViewVariant{std::pair{RtdgiSpatialFilterCompute::AT.depth_tex, gbuffer_depth.depth.current()}},
                daxa::TaskViewVariant{std::pair{RtdgiSpatialFilterCompute::AT.ssao_tex, ssao_tex}},
                daxa::TaskViewVariant{std::pair{RtdgiSpatialFilterCompute::AT.geometric_normal_tex, gbuffer_depth.geometric_normal}},
                daxa::TaskViewVariant{std::pair{RtdgiSpatialFilterCompute::AT.output_tex, spatial_filtered_tex}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, RtdgiSpatialFilterComputePush &push, NoTaskInfo const &) {
                auto const out_image_info = ti.device.info_image(ti.get(RtdgiSpatialFilterCompute::AT.output_tex).ids[0]).value();
                ti.recorder.set_pipeline(pipeline);
                push.output_tex_size = extent_inv_extent_2d(out_image_info);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(out_image_info.size.x + 7) / 8, (out_image_info.size.y + 7) / 8});
            },
        });
        debug_utils::DebugDisplay::add_pass({.name = "rtdgi spatial filter", .task_image_id = spatial_filtered_tex, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        return spatial_filtered_tex;
    }

    auto render(
        GpuContext &gpu_context,
        ReprojectedRtdgi reprojected_rtdgi,
        GbufferDepth &gbuffer_depth,
        daxa::TaskImageView reprojection_map,
        daxa::TaskImageView sky_cube,
        daxa::TaskImageView transmittance_lut,
        IrcacheRenderState &ircache,
        VoxelWorldBuffers &voxel_buffers,
        daxa::TaskImageView ssao_tex) -> RtdgiOutput {
        auto [reprojected_history_tex, temporal_output_tex] = reprojected_rtdgi;
        auto half_ssao_tex = extract_downscaled_ssao(gpu_context, ssao_tex);
        debug_utils::DebugDisplay::add_pass({.name = "rtdgi downscaled ssao", .task_image_id = half_ssao_tex, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        auto gbuffer_half_res = daxa_u32vec2{(gpu_context.render_resolution.x + 1) / 2, (gpu_context.render_resolution.y + 1) / 2};

        temporal_hit_normal_tex = PingPongImage{};
        auto [hit_normal_output_tex, hit_normal_history_tex] = temporal_hit_normal_tex.get(
            gpu_context,
            {
                .format = daxa::Format::R8G8B8A8_UNORM,
                .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .name = "temporal_hit_normal_tex",
            });
        gpu_context.frame_task_graph.use_persistent_image(hit_normal_output_tex);
        gpu_context.frame_task_graph.use_persistent_image(hit_normal_history_tex);
        clear_task_images(gpu_context.device, std::array{hit_normal_output_tex, hit_normal_history_tex});

        temporal_candidate_tex = PingPongImage{};
        auto [candidate_output_tex, candidate_history_tex] = temporal_candidate_tex.get(
            gpu_context,
            {
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .name = "temporal_candidate_tex",
            });
        gpu_context.frame_task_graph.use_persistent_image(candidate_output_tex);
        gpu_context.frame_task_graph.use_persistent_image(candidate_history_tex);
        clear_task_images(gpu_context.device, std::array{candidate_output_tex, candidate_history_tex});

        auto candidate_radiance_tex = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
            .name = "candidate_radiance_tex",
        });
        auto candidate_normal_tex = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::R8G8B8A8_SNORM,
            .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
            .name = "candidate_normal_tex",
        });
        auto candidate_hit_tex = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
            .name = "candidate_hit_tex",
        });
        auto temporal_reservoir_packed_tex = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::R32G32B32A32_UINT,
            .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
            .name = "temporal_reservoir_packed_tex",
        });
        auto half_depth_tex = gbuffer_depth.get_downscaled_depth(gpu_context);

        debug_utils::DebugDisplay::add_pass({.name = "rtdgi downscaled depth", .task_image_id = half_depth_tex, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        temporal_invalidity_tex = PingPongImage{};
        auto [invalidity_output_tex, invalidity_history_tex] = temporal_invalidity_tex.get(
            gpu_context,
            {
                .format = daxa::Format::R16G16_SFLOAT,
                .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .name = "temporal_invalidity_tex",
            });
        gpu_context.frame_task_graph.use_persistent_image(invalidity_output_tex);
        gpu_context.frame_task_graph.use_persistent_image(invalidity_history_tex);
        clear_task_images(gpu_context.device, std::array{invalidity_output_tex, invalidity_history_tex});

        // auto [radiance_tex, temporal_reservoir_tex] =
        auto radiance_tex = daxa::TaskImageView{};
        auto temporal_reservoir_tex = daxa::TaskImageView{};
        {
            temporal_radiance_tex = PingPongImage{};
            auto [radiance_output_tex, radiance_history_tex] = temporal_radiance_tex.get(
                gpu_context,
                {
                    .format = daxa::Format::R16G16B16A16_SFLOAT,
                    .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                    .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
                    .name = "temporal_radiance_tex",
                });
            gpu_context.frame_task_graph.use_persistent_image(radiance_output_tex);
            gpu_context.frame_task_graph.use_persistent_image(radiance_history_tex);
            clear_task_images(gpu_context.device, std::array{radiance_output_tex, radiance_history_tex});

            temporal_ray_orig_tex = PingPongImage{};
            auto [ray_orig_output_tex, ray_orig_history_tex] = temporal_ray_orig_tex.get(
                gpu_context,
                {
                    .format = daxa::Format::R32G32B32A32_SFLOAT,
                    .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                    .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
                    .name = "temporal_ray_orig_tex",
                });
            gpu_context.frame_task_graph.use_persistent_image(ray_orig_output_tex);
            gpu_context.frame_task_graph.use_persistent_image(ray_orig_history_tex);
            clear_task_images(gpu_context.device, std::array{ray_orig_output_tex, ray_orig_history_tex});

            temporal_ray_tex = PingPongImage{};
            auto [ray_output_tex, ray_history_tex] = temporal_ray_tex.get(
                gpu_context,
                {
                    .format = daxa::Format::R16G16B16A16_SFLOAT,
                    .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                    .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
                    .name = "temporal_ray_tex",
                });
            gpu_context.frame_task_graph.use_persistent_image(ray_output_tex);
            gpu_context.frame_task_graph.use_persistent_image(ray_history_tex);
            clear_task_images(gpu_context.device, std::array{ray_output_tex, ray_history_tex});

            auto half_view_normal_tex = gbuffer_depth.get_downscaled_view_normal(gpu_context);

            auto rt_history_validity_pre_input_tex = gpu_context.frame_task_graph.create_transient_image({
                .format = daxa::Format::R8_UNORM,
                .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                .name = "rt_history_validity_pre_input_tex",
            });

            pp_temporal_reservoir_tex = PingPongImage{};
            auto [reservoir_output_tex, reservoir_history_tex] = pp_temporal_reservoir_tex.get(
                gpu_context,
                {
                    .format = daxa::Format::R32G32_UINT,
                    .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                    .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST,
                    .name = "temporal_reservoir_tex",
                });
            gpu_context.frame_task_graph.use_persistent_image(reservoir_output_tex);
            gpu_context.frame_task_graph.use_persistent_image(reservoir_history_tex);
            clear_task_images(gpu_context.device, std::array{reservoir_output_tex, reservoir_history_tex});

            auto use_hwrt = AppSettings::get<settings::Checkbox>("Graphics", "Use HWRT").value;

            if (!use_hwrt || true) {
                gpu_context.add(ComputeTask<RtdgiValidateCompute::Task, RtdgiValidateComputePush, NoTaskInfo>{
                    .source = daxa::ShaderFile{"kajiya/rtdgi/diffuse_validate.comp.glsl"},
                    .views = std::array{
                        daxa::TaskViewVariant{std::pair{RtdgiValidateCompute::AT.gpu_input, gpu_context.task_input_buffer}},
                        daxa::TaskViewVariant{std::pair{RtdgiValidateCompute::AT.half_view_normal_tex, half_view_normal_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiValidateCompute::AT.depth_tex, gbuffer_depth.depth.current()}},
                        daxa::TaskViewVariant{std::pair{RtdgiValidateCompute::AT.reprojected_gi_tex, reprojected_history_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiValidateCompute::AT.reservoir_tex, reservoir_history_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiValidateCompute::AT.reservoir_ray_history_tex, ray_history_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiValidateCompute::AT.blue_noise_vec2, gpu_context.task_blue_noise_vec2_image}},
                        // daxa::TaskViewVariant{std::pair{RtdgiValidateCompute::AT.reprojection_tex, reprojection_map}},
                        VOXELS_BUFFER_USES_ASSIGN(RtdgiValidateCompute, voxel_buffers),
                        IRCACHE_BUFFER_USES_ASSIGN(RtdgiValidateCompute, ircache),
                        daxa::TaskViewVariant{std::pair{RtdgiValidateCompute::AT.sky_cube_tex, sky_cube}},
                        daxa::TaskViewVariant{std::pair{RtdgiValidateCompute::AT.transmittance_lut, transmittance_lut}},
                        daxa::TaskViewVariant{std::pair{RtdgiValidateCompute::AT.irradiance_history_tex, radiance_history_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiValidateCompute::AT.ray_orig_history_tex, ray_orig_history_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiValidateCompute::AT.rt_history_invalidity_out_tex, rt_history_validity_pre_input_tex}},
                    },
                    .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, RtdgiValidateComputePush &push, NoTaskInfo const &) {
                        auto const image_info = ti.device.info_image(ti.get(RtdgiValidateCompute::AT.depth_tex).ids[0]).value();
                        auto const candidate_image_info = ti.device.info_image(ti.get(RtdgiValidateCompute::AT.reservoir_tex).ids[0]).value();
                        ti.recorder.set_pipeline(pipeline);
                        push.gbuffer_tex_size = extent_inv_extent_2d(image_info);
                        set_push_constant(ti, push);
                        ti.recorder.dispatch({(candidate_image_info.size.x + 7) / 8, (candidate_image_info.size.y + 7) / 8});
                    },
                });
            } else {
                gpu_context.add(RayTracingTask<RtdgiValidateRt::Task, RtdgiValidateRtPush, NoTaskInfo>{
                    .source = daxa::ShaderFile{"kajiya/rtdgi/diffuse_validate.rt.glsl"},
                    .views = std::array{
                        daxa::TaskViewVariant{std::pair{RtdgiValidateRt::AT.gpu_input, gpu_context.task_input_buffer}},
                        daxa::TaskViewVariant{std::pair{RtdgiValidateRt::AT.geometry_pointers, voxel_buffers.blas_geom_pointers.task_resource}},
                        daxa::TaskViewVariant{std::pair{RtdgiValidateRt::AT.attribute_pointers, voxel_buffers.blas_attr_pointers.task_resource}},
                        daxa::TaskViewVariant{std::pair{RtdgiValidateRt::AT.blas_transforms, voxel_buffers.blas_transforms.task_resource}},
                        daxa::TaskViewVariant{std::pair{RtdgiValidateRt::AT.tlas, voxel_buffers.task_tlas}},
                        IRCACHE_BUFFER_USES_ASSIGN(RtdgiValidateRt, ircache),
                        daxa::TaskViewVariant{std::pair{RtdgiValidateRt::AT.half_view_normal_tex, half_view_normal_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiValidateRt::AT.depth_tex, gbuffer_depth.depth.current()}},
                        daxa::TaskViewVariant{std::pair{RtdgiValidateRt::AT.reprojected_gi_tex, reprojected_history_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiValidateRt::AT.reservoir_tex, reservoir_history_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiValidateRt::AT.reservoir_ray_history_tex, ray_history_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiValidateRt::AT.blue_noise_vec2, gpu_context.task_blue_noise_vec2_image}},
                        // daxa::TaskViewVariant{std::pair{RtdgiValidateRt::AT.reprojection_tex, reprojection_map}},
                        daxa::TaskViewVariant{std::pair{RtdgiValidateRt::AT.sky_cube_tex, sky_cube}},
                        daxa::TaskViewVariant{std::pair{RtdgiValidateRt::AT.transmittance_lut, transmittance_lut}},
                        daxa::TaskViewVariant{std::pair{RtdgiValidateRt::AT.irradiance_history_tex, radiance_history_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiValidateRt::AT.ray_orig_history_tex, ray_orig_history_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiValidateRt::AT.rt_history_invalidity_out_tex, rt_history_validity_pre_input_tex}},
                    },
                    .callback_ = [](daxa::TaskInterface const &ti, daxa::RayTracingPipeline &pipeline, RtdgiValidateRtPush &push, NoTaskInfo const &) {
                        auto const image_info = ti.device.info_image(ti.get(RtdgiValidateRt::AT.depth_tex).ids[0]).value();
                        auto const candidate_image_info = ti.device.info_image(ti.get(RtdgiValidateRt::AT.reservoir_tex).ids[0]).value();
                        ti.recorder.set_pipeline(pipeline);
                        push.gbuffer_tex_size = extent_inv_extent_2d(image_info);
                        set_push_constant(ti, push);
                        ti.recorder.trace_rays({.width = candidate_image_info.size.x, .height = candidate_image_info.size.y, .depth = 1});
                    },
                });
            }

            debug_utils::DebugDisplay::add_pass({.name = "rtdgi validate", .task_image_id = rt_history_validity_pre_input_tex, .type = DEBUG_IMAGE_TYPE_DEFAULT});

            auto rt_history_validity_input_tex = gpu_context.frame_task_graph.create_transient_image({
                .format = daxa::Format::R8_UNORM,
                .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                .name = "rt_history_validity_input_tex",
            });

            if (!use_hwrt) {
                gpu_context.add(ComputeTask<RtdgiTraceCompute::Task, RtdgiTraceComputePush, NoTaskInfo>{
                    .source = daxa::ShaderFile{"kajiya/rtdgi/trace_diffuse.comp.glsl"},
                    .views = std::array{
                        daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::AT.gpu_input, gpu_context.task_input_buffer}},
                        daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::AT.half_view_normal_tex, half_view_normal_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::AT.depth_tex, gbuffer_depth.depth.current()}},
                        daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::AT.reprojected_gi_tex, reprojected_history_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::AT.reprojection_tex, reprojection_map}},
                        daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::AT.blue_noise_vec2, gpu_context.task_blue_noise_vec2_image}},
                        daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::AT.sky_cube_tex, sky_cube}},
                        daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::AT.transmittance_lut, transmittance_lut}},
                        VOXELS_BUFFER_USES_ASSIGN(RtdgiTraceCompute, voxel_buffers),
                        IRCACHE_BUFFER_USES_ASSIGN(RtdgiTraceCompute, ircache),
                        daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::AT.candidate_irradiance_out_tex, candidate_radiance_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::AT.candidate_normal_out_tex, candidate_normal_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::AT.candidate_hit_out_tex, candidate_hit_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::AT.rt_history_invalidity_in_tex, rt_history_validity_pre_input_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiTraceCompute::AT.rt_history_invalidity_out_tex, rt_history_validity_input_tex}},
                    },
                    .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, RtdgiTraceComputePush &push, NoTaskInfo const &) {
                        auto const image_info = ti.device.info_image(ti.get(RtdgiTraceCompute::AT.depth_tex).ids[0]).value();
                        auto const candidate_image_info = ti.device.info_image(ti.get(RtdgiTraceCompute::AT.candidate_hit_out_tex).ids[0]).value();
                        ti.recorder.set_pipeline(pipeline);
                        push.gbuffer_tex_size = extent_inv_extent_2d(image_info);
                        set_push_constant(ti, push);
                        ti.recorder.dispatch({(candidate_image_info.size.x + 7) / 8, (candidate_image_info.size.y + 7) / 8});
                    },
                });
            } else {
                gpu_context.add(RayTracingTask<RtdgiTraceRt::Task, RtdgiTraceRtPush, NoTaskInfo>{
                    .source = daxa::ShaderFile{"kajiya/rtdgi/trace_diffuse.rt.glsl"},
                    .views = std::array{
                        daxa::TaskViewVariant{std::pair{RtdgiTraceRt::AT.gpu_input, gpu_context.task_input_buffer}},
                        daxa::TaskViewVariant{std::pair{RtdgiTraceRt::AT.geometry_pointers, voxel_buffers.blas_geom_pointers.task_resource}},
                        daxa::TaskViewVariant{std::pair{RtdgiTraceRt::AT.attribute_pointers, voxel_buffers.blas_attr_pointers.task_resource}},
                        daxa::TaskViewVariant{std::pair{RtdgiTraceRt::AT.blas_transforms, voxel_buffers.blas_transforms.task_resource}},
                        daxa::TaskViewVariant{std::pair{RtdgiTraceRt::AT.tlas, voxel_buffers.task_tlas}},
                        IRCACHE_BUFFER_USES_ASSIGN(RtdgiTraceRt, ircache),
                        daxa::TaskViewVariant{std::pair{RtdgiTraceRt::AT.half_view_normal_tex, half_view_normal_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiTraceRt::AT.depth_tex, gbuffer_depth.depth.current()}},
                        daxa::TaskViewVariant{std::pair{RtdgiTraceRt::AT.reprojected_gi_tex, reprojected_history_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiTraceRt::AT.reprojection_tex, reprojection_map}},
                        daxa::TaskViewVariant{std::pair{RtdgiTraceRt::AT.blue_noise_vec2, gpu_context.task_blue_noise_vec2_image}},
                        daxa::TaskViewVariant{std::pair{RtdgiTraceRt::AT.sky_cube_tex, sky_cube}},
                        daxa::TaskViewVariant{std::pair{RtdgiTraceRt::AT.transmittance_lut, transmittance_lut}},
                        daxa::TaskViewVariant{std::pair{RtdgiTraceRt::AT.candidate_irradiance_out_tex, candidate_radiance_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiTraceRt::AT.candidate_normal_out_tex, candidate_normal_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiTraceRt::AT.candidate_hit_out_tex, candidate_hit_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiTraceRt::AT.rt_history_invalidity_in_tex, rt_history_validity_pre_input_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiTraceRt::AT.rt_history_invalidity_out_tex, rt_history_validity_input_tex}},
                    },
                    .callback_ = [](daxa::TaskInterface const &ti, daxa::RayTracingPipeline &pipeline, RtdgiTraceRtPush &push, NoTaskInfo const &) {
                        auto const image_info = ti.device.info_image(ti.get(RtdgiTraceRt::AT.depth_tex).ids[0]).value();
                        auto const candidate_image_info = ti.device.info_image(ti.get(RtdgiTraceRt::AT.candidate_hit_out_tex).ids[0]).value();
                        ti.recorder.set_pipeline(pipeline);
                        push.gbuffer_tex_size = extent_inv_extent_2d(image_info);
                        set_push_constant(ti, push);
                        ti.recorder.trace_rays({.width = candidate_image_info.size.x, .height = candidate_image_info.size.y, .depth = 1});
                    },
                });
            }

            debug_utils::DebugDisplay::add_pass({.name = "rtdgi trace", .task_image_id = candidate_radiance_tex, .type = DEBUG_IMAGE_TYPE_DEFAULT});

            gpu_context.add(ComputeTask<RtdgiValidityIntegrateCompute::Task, RtdgiValidityIntegrateComputePush, NoTaskInfo>{
                .source = daxa::ShaderFile{"kajiya/rtdgi/temporal_validity_integrate.comp.glsl"},
                .views = std::array{
                    daxa::TaskViewVariant{std::pair{RtdgiValidityIntegrateCompute::AT.gpu_input, gpu_context.task_input_buffer}},
                    daxa::TaskViewVariant{std::pair{RtdgiValidityIntegrateCompute::AT.input_tex, rt_history_validity_input_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiValidityIntegrateCompute::AT.history_tex, invalidity_history_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiValidityIntegrateCompute::AT.reprojection_tex, reprojection_map}},
                    daxa::TaskViewVariant{std::pair{RtdgiValidityIntegrateCompute::AT.half_view_normal_tex, half_view_normal_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiValidityIntegrateCompute::AT.half_depth_tex, half_depth_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiValidityIntegrateCompute::AT.output_tex, invalidity_output_tex}},
                },
                .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, RtdgiValidityIntegrateComputePush &push, NoTaskInfo const &) {
                    auto const image_info = ti.device.info_image(ti.get(RtdgiValidityIntegrateCompute::AT.reprojection_tex).ids[0]).value();
                    auto const output_image_info = ti.device.info_image(ti.get(RtdgiValidityIntegrateCompute::AT.output_tex).ids[0]).value();
                    ti.recorder.set_pipeline(pipeline);
                    push.gbuffer_tex_size = extent_inv_extent_2d(image_info);
                    push.output_tex_size = extent_inv_extent_2d(output_image_info);
                    set_push_constant(ti, push);
                    ti.recorder.dispatch({(output_image_info.size.x + 7) / 8, (output_image_info.size.y + 7) / 8});
                },
            });

            debug_utils::DebugDisplay::add_pass({.name = "rtdgi temporal validate", .task_image_id = invalidity_output_tex, .type = DEBUG_IMAGE_TYPE_DEFAULT});

            gpu_context.add(ComputeTask<RtdgiRestirTemporalCompute::Task, RtdgiRestirTemporalComputePush, NoTaskInfo>{
                .source = daxa::ShaderFile{"kajiya/rtdgi/restir_temporal.comp.glsl"},
                .views = std::array{
                    daxa::TaskViewVariant{std::pair{RtdgiRestirTemporalCompute::AT.gpu_input, gpu_context.task_input_buffer}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirTemporalCompute::AT.half_view_normal_tex, half_view_normal_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirTemporalCompute::AT.depth_tex, gbuffer_depth.depth.current()}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirTemporalCompute::AT.candidate_radiance_tex, candidate_radiance_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirTemporalCompute::AT.candidate_normal_tex, candidate_normal_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirTemporalCompute::AT.candidate_hit_tex, candidate_hit_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirTemporalCompute::AT.radiance_history_tex, radiance_history_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirTemporalCompute::AT.ray_orig_history_tex, ray_orig_history_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirTemporalCompute::AT.ray_history_tex, ray_history_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirTemporalCompute::AT.reservoir_history_tex, reservoir_history_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirTemporalCompute::AT.reprojection_tex, reprojection_map}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirTemporalCompute::AT.hit_normal_history_tex, hit_normal_history_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirTemporalCompute::AT.candidate_history_tex, candidate_history_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirTemporalCompute::AT.rt_invalidity_tex, invalidity_output_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirTemporalCompute::AT.radiance_out_tex, radiance_output_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirTemporalCompute::AT.ray_orig_output_tex, ray_orig_output_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirTemporalCompute::AT.ray_output_tex, ray_output_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirTemporalCompute::AT.hit_normal_output_tex, hit_normal_output_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirTemporalCompute::AT.reservoir_out_tex, reservoir_output_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirTemporalCompute::AT.candidate_out_tex, candidate_output_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirTemporalCompute::AT.temporal_reservoir_packed_tex, temporal_reservoir_packed_tex}},
                },
                .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, RtdgiRestirTemporalComputePush &push, NoTaskInfo const &) {
                    auto const image_info = ti.device.info_image(ti.get(RtdgiRestirTemporalCompute::AT.depth_tex).ids[0]).value();
                    auto const out_image_info = ti.device.info_image(ti.get(RtdgiRestirTemporalCompute::AT.radiance_out_tex).ids[0]).value();
                    ti.recorder.set_pipeline(pipeline);
                    push.gbuffer_tex_size = extent_inv_extent_2d(image_info);
                    set_push_constant(ti, push);
                    ti.recorder.dispatch({(out_image_info.size.x + 7) / 8, (out_image_info.size.y + 7) / 8});
                },
            });

            debug_utils::DebugDisplay::add_pass({.name = "restir temporal", .task_image_id = radiance_output_tex, .type = DEBUG_IMAGE_TYPE_DEFAULT});

            // return std::pair{radiance_output_tex, reservoir_output_tex};
            radiance_tex = radiance_output_tex;
            temporal_reservoir_tex = reservoir_output_tex;
        }

        auto irradiance_tex = daxa::TaskImageView{};
        {
            auto half_view_normal_tex = gbuffer_depth.get_downscaled_view_normal(gpu_context);

            auto reservoir_output_tex0 = gpu_context.frame_task_graph.create_transient_image({
                .format = daxa::Format::R32G32_UINT,
                .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                .name = "reservoir_output_tex0",
            });
            auto reservoir_output_tex1 = gpu_context.frame_task_graph.create_transient_image({
                .format = daxa::Format::R32G32_UINT,
                .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                .name = "reservoir_output_tex1",
            });

            auto bounced_radiance_output_tex0 = gpu_context.frame_task_graph.create_transient_image({
                .format = daxa::Format::B10G11R11_UFLOAT_PACK32,
                .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                .name = "bounced_radiance_output_tex0",
            });
            auto bounced_radiance_output_tex1 = gpu_context.frame_task_graph.create_transient_image({
                .format = daxa::Format::B10G11R11_UFLOAT_PACK32,
                .size = {gbuffer_half_res.x, gbuffer_half_res.y, 1},
                .name = "bounced_radiance_output_tex1",
            });

            auto reservoir_input_tex = temporal_reservoir_tex;
            auto bounced_radiance_input_tex = radiance_tex;

            for (uint32_t spatial_reuse_pass_idx = 0; spatial_reuse_pass_idx < this->spatial_reuse_pass_count; ++spatial_reuse_pass_idx) {
                auto perform_occulsion_raymarch =
                    (spatial_reuse_pass_idx + 1 == this->spatial_reuse_pass_count) ? 1u : 0u;

                auto occlusion_raymarch_importance_only =
                    (this->use_raytraced_reservoir_visibility) ? 1u : 0u;

                struct RestirSpatialTaskInfo {
                    daxa_u32 spatial_reuse_pass_idx;
                    daxa_u32 perform_occlusion_raymarch;
                    daxa_u32 occlusion_raymarch_importance_only;
                };

                gpu_context.add(ComputeTask<RtdgiRestirSpatialCompute::Task, RtdgiRestirSpatialComputePush, RestirSpatialTaskInfo>{
                    .source = daxa::ShaderFile{"kajiya/rtdgi/restir_spatial.comp.glsl"},
                    .views = std::array{
                        daxa::TaskViewVariant{std::pair{RtdgiRestirSpatialCompute::AT.gpu_input, gpu_context.task_input_buffer}},
                        daxa::TaskViewVariant{std::pair{RtdgiRestirSpatialCompute::AT.reservoir_input_tex, reservoir_input_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiRestirSpatialCompute::AT.bounced_radiance_input_tex, bounced_radiance_input_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiRestirSpatialCompute::AT.half_view_normal_tex, half_view_normal_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiRestirSpatialCompute::AT.half_depth_tex, half_depth_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiRestirSpatialCompute::AT.depth_tex, gbuffer_depth.depth.current()}},
                        daxa::TaskViewVariant{std::pair{RtdgiRestirSpatialCompute::AT.half_ssao_tex, half_ssao_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiRestirSpatialCompute::AT.temporal_reservoir_packed_tex, temporal_reservoir_packed_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiRestirSpatialCompute::AT.reprojected_gi_tex, reprojected_history_tex}},
                        daxa::TaskViewVariant{std::pair{RtdgiRestirSpatialCompute::AT.reservoir_output_tex, reservoir_output_tex0}},
                        daxa::TaskViewVariant{std::pair{RtdgiRestirSpatialCompute::AT.bounced_radiance_output_tex, bounced_radiance_output_tex0}},
                    },
                    .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, RtdgiRestirSpatialComputePush &push, RestirSpatialTaskInfo const &info) {
                        auto const image_info = ti.device.info_image(ti.get(RtdgiRestirSpatialCompute::AT.depth_tex).ids[0]).value();
                        auto const out_image_info = ti.device.info_image(ti.get(RtdgiRestirSpatialCompute::AT.reservoir_output_tex).ids[0]).value();
                        ti.recorder.set_pipeline(pipeline);
                        push.gbuffer_tex_size = extent_inv_extent_2d(image_info);
                        push.output_tex_size = extent_inv_extent_2d(image_info);
                        push.spatial_reuse_pass_idx = info.spatial_reuse_pass_idx;
                        push.perform_occlusion_raymarch = info.perform_occlusion_raymarch;
                        push.occlusion_raymarch_importance_only = info.occlusion_raymarch_importance_only;
                        set_push_constant(ti, push);
                        ti.recorder.dispatch({(out_image_info.size.x + 7) / 8, (out_image_info.size.y + 7) / 8});
                    },
                    .info = {
                        .spatial_reuse_pass_idx = spatial_reuse_pass_idx,
                        .perform_occlusion_raymarch = perform_occulsion_raymarch,
                        .occlusion_raymarch_importance_only = occlusion_raymarch_importance_only,
                    },
                });

                std::swap(reservoir_output_tex0, reservoir_output_tex1);
                std::swap(bounced_radiance_output_tex0, bounced_radiance_output_tex1);

                reservoir_input_tex = reservoir_output_tex1;
                bounced_radiance_input_tex = bounced_radiance_output_tex1;
            }
            debug_utils::DebugDisplay::add_pass({.name = "restir spatial", .task_image_id = bounced_radiance_input_tex, .type = DEBUG_IMAGE_TYPE_DEFAULT});

            auto irradiance_output_tex = gpu_context.frame_task_graph.create_transient_image({
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
                .name = "irradiance_output_tex",
            });

            auto rtdgi_debug_image = gpu_context.frame_task_graph.create_transient_image({
                .format = daxa::Format::R32G32B32A32_SFLOAT,
                .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
                .name = "rtdgi_debug_image",
            });

            gpu_context.add(ComputeTask<RtdgiRestirResolveCompute::Task, RtdgiRestirResolveComputePush, NoTaskInfo>{
                .source = daxa::ShaderFile{"kajiya/rtdgi/restir_resolve.comp.glsl"},
                .views = std::array{
                    daxa::TaskViewVariant{std::pair{RtdgiRestirResolveCompute::AT.gpu_input, gpu_context.task_input_buffer}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirResolveCompute::AT.blue_noise_vec2, gpu_context.task_blue_noise_vec2_image}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirResolveCompute::AT.radiance_tex, radiance_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirResolveCompute::AT.reservoir_input_tex, reservoir_input_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirResolveCompute::AT.gbuffer_tex, gbuffer_depth.gbuffer}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirResolveCompute::AT.depth_tex, gbuffer_depth.depth.current()}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirResolveCompute::AT.half_view_normal_tex, half_view_normal_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirResolveCompute::AT.half_depth_tex, half_depth_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirResolveCompute::AT.ssao_tex, ssao_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirResolveCompute::AT.candidate_radiance_tex, candidate_radiance_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirResolveCompute::AT.candidate_hit_tex, candidate_hit_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirResolveCompute::AT.temporal_reservoir_packed_tex, temporal_reservoir_packed_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirResolveCompute::AT.bounced_radiance_input_tex, bounced_radiance_input_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirResolveCompute::AT.irradiance_output_tex, irradiance_output_tex}},
                    daxa::TaskViewVariant{std::pair{RtdgiRestirResolveCompute::AT.rtdgi_debug_image, rtdgi_debug_image}},
                },
                .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, RtdgiRestirResolveComputePush &push, NoTaskInfo const &) {
                    auto const image_info = ti.device.info_image(ti.get(RtdgiRestirResolveCompute::AT.gbuffer_tex).ids[0]).value();
                    auto const out_image_info = ti.device.info_image(ti.get(RtdgiRestirResolveCompute::AT.irradiance_output_tex).ids[0]).value();
                    ti.recorder.set_pipeline(pipeline);
                    push.gbuffer_tex_size = extent_inv_extent_2d(image_info);
                    push.output_tex_size = extent_inv_extent_2d(image_info);
                    set_push_constant(ti, push);
                    ti.recorder.dispatch({(out_image_info.size.x + 7) / 8, (out_image_info.size.y + 7) / 8});
                },
            });
            debug_utils::DebugDisplay::add_pass({.name = "rtdgi debug", .task_image_id = rtdgi_debug_image, .type = DEBUG_IMAGE_TYPE_RTDGI_DEBUG});

            irradiance_tex = irradiance_output_tex;
            debug_utils::DebugDisplay::add_pass({.name = "restir resolve", .task_image_id = irradiance_tex, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        }

        auto filtered_tex = this->temporal(
            gpu_context,
            irradiance_tex,
            gbuffer_depth,
            reprojection_map,
            reprojected_history_tex,
            invalidity_output_tex,
            temporal_output_tex);

        filtered_tex = this->spatial(
            gpu_context,
            filtered_tex,
            gbuffer_depth,
            ssao_tex);

        return RtdgiOutput{
            .screen_irradiance_tex = filtered_tex,
            .candidates = {
                candidate_radiance_tex,
                candidate_normal_tex,
                candidate_hit_tex,
            },
        };
    }
};

#endif

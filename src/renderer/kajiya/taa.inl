#pragma once

#include <core.inl>
#include <application/input.inl>

#define TAA_WG_SIZE_X 16
#define TAA_WG_SIZE_Y 8

DAXA_DECL_TASK_HEAD_BEGIN(TaaReprojectCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojection_map)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, reprojected_history_img)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, closest_velocity_img)
DAXA_DECL_TASK_HEAD_END
struct TaaReprojectComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    DAXA_TH_BLOB(TaaReprojectCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(TaaFilterInputCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_image)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, filtered_input_img)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, filtered_input_deviation_img)
DAXA_DECL_TASK_HEAD_END
struct TaaFilterInputComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    DAXA_TH_BLOB(TaaFilterInputCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(TaaFilterHistoryCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojected_history_img)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, filtered_history_img)
DAXA_DECL_TASK_HEAD_END
struct TaaFilterHistoryComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    DAXA_TH_BLOB(TaaFilterHistoryCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(TaaInputProbCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_image)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, filtered_input_img)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, filtered_input_deviation_img)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojected_history_img)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, filtered_history_img)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojection_map)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, smooth_var_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, velocity_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, input_prob_img)
DAXA_DECL_TASK_HEAD_END
struct TaaInputProbComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    DAXA_TH_BLOB(TaaInputProbCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(TaaProbFilterCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_prob_img)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, prob_filtered1_img)
DAXA_DECL_TASK_HEAD_END
struct TaaProbFilterComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    DAXA_TH_BLOB(TaaProbFilterCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(TaaProbFilter2Compute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, prob_filtered1_img)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, prob_filtered2_img)
DAXA_DECL_TASK_HEAD_END
struct TaaProbFilter2ComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    DAXA_TH_BLOB(TaaProbFilter2Compute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(TaaCompute)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_image)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojected_history_img)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, reprojection_map)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, closest_velocity_img)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, velocity_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_image)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, smooth_var_history_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_prob_img)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, temporal_output_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, this_frame_output_img)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, smooth_var_output_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, temporal_velocity_output_tex)
DAXA_DECL_TASK_HEAD_END
struct TaaComputePush {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
    DAXA_TH_BLOB(TaaCompute, uses)
};

struct TaaPushCommon {
    daxa_f32vec2 input_tex_size;
    daxa_f32vec2 output_tex_size;
};

#if defined(__cplusplus)

struct TaaRenderer {
    PingPongImage ping_pong_taa_col_image;
    PingPongImage ping_pong_taa_vel_image;
    PingPongImage ping_pong_smooth_var_image;

    void next_frame() {
        ping_pong_taa_col_image.swap();
        ping_pong_taa_vel_image.swap();
        ping_pong_smooth_var_image.swap();
    }

    auto render(GpuContext &gpu_context, daxa::TaskImageView input_image, daxa::TaskImageView depth_image, daxa::TaskImageView reprojection_map) -> daxa::TaskImageView {
        ping_pong_taa_col_image = PingPongImage{};
        ping_pong_taa_vel_image = PingPongImage{};
        ping_pong_smooth_var_image = PingPongImage{};
        auto [temporal_output_tex, history_tex] = ping_pong_taa_col_image.get(
            gpu_context,
            {
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {gpu_context.output_resolution.x, gpu_context.output_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_SRC,
                .name = "taa_col",
            });
        auto [temporal_velocity_output_tex, velocity_history_tex] = ping_pong_taa_vel_image.get(
            gpu_context,
            {
                .format = daxa::Format::R16G16_SFLOAT,
                .size = {gpu_context.output_resolution.x, gpu_context.output_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_SRC,
                .name = "taa_vel",
            });
        auto [smooth_var_output_tex, smooth_var_history_tex] = ping_pong_smooth_var_image.get(
            gpu_context,
            {
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = {gpu_context.output_resolution.x, gpu_context.output_resolution.y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_SRC,
                .name = "smooth_var",
            });
        gpu_context.frame_task_graph.use_persistent_image(temporal_output_tex);
        gpu_context.frame_task_graph.use_persistent_image(history_tex);
        gpu_context.frame_task_graph.use_persistent_image(temporal_velocity_output_tex);
        gpu_context.frame_task_graph.use_persistent_image(velocity_history_tex);
        gpu_context.frame_task_graph.use_persistent_image(smooth_var_output_tex);
        gpu_context.frame_task_graph.use_persistent_image(smooth_var_history_tex);

        auto reprojected_history_img = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {gpu_context.output_resolution.x, gpu_context.output_resolution.y, 1},
            .name = "reprojected_history_img",
        });
        auto closest_velocity_img = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::R16G16_SFLOAT,
            .size = {gpu_context.output_resolution.x, gpu_context.output_resolution.y, 1},
            .name = "closest_velocity_img",
        });

        auto i_extent = daxa_f32vec2(static_cast<daxa_f32>(gpu_context.render_resolution.x), static_cast<daxa_f32>(gpu_context.render_resolution.y));
        auto o_extent = daxa_f32vec2(static_cast<daxa_f32>(gpu_context.output_resolution.x), static_cast<daxa_f32>(gpu_context.output_resolution.y));

        struct TaaTaskInfo {
            daxa_u32vec2 thread_count;
            daxa_f32vec2 input_tex_size;
            daxa_f32vec2 output_tex_size;
        };

        gpu_context.add(ComputeTask<TaaReprojectCompute::Task, TaaReprojectComputePush, TaaTaskInfo>{
            .source = daxa::ShaderFile{"kajiya/taa/reproject_history.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{TaaReprojectCompute::AT.gpu_input, gpu_context.task_input_buffer}},

                daxa::TaskViewVariant{std::pair{TaaReprojectCompute::AT.history_tex, history_tex}},
                daxa::TaskViewVariant{std::pair{TaaReprojectCompute::AT.reprojection_map, reprojection_map}},
                daxa::TaskViewVariant{std::pair{TaaReprojectCompute::AT.depth_image, depth_image}},

                daxa::TaskViewVariant{std::pair{TaaReprojectCompute::AT.reprojected_history_img, reprojected_history_img}},
                daxa::TaskViewVariant{std::pair{TaaReprojectCompute::AT.closest_velocity_img, closest_velocity_img}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaReprojectComputePush &push, TaaTaskInfo const &info) {
                push.input_tex_size = info.input_tex_size;
                push.output_tex_size = info.output_tex_size;
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(info.thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (info.thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y});
            },
            .info = {
                .thread_count = gpu_context.output_resolution,
                .input_tex_size = i_extent,
                .output_tex_size = o_extent,
            },
        });

        debug_utils::DebugDisplay::add_pass({.name = "taa reproject", .task_image_id = reprojected_history_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        auto filtered_input_img = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
            .name = "filtered_input_img",
        });
        auto filtered_input_deviation_img = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
            .name = "filtered_input_deviation_img",
        });

        gpu_context.add(ComputeTask<TaaFilterInputCompute::Task, TaaFilterInputComputePush, TaaTaskInfo>{
            .source = daxa::ShaderFile{"kajiya/taa/filter_input.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{TaaFilterInputCompute::AT.gpu_input, gpu_context.task_input_buffer}},

                daxa::TaskViewVariant{std::pair{TaaFilterInputCompute::AT.input_image, input_image}},
                daxa::TaskViewVariant{std::pair{TaaFilterInputCompute::AT.depth_image, depth_image}},

                daxa::TaskViewVariant{std::pair{TaaFilterInputCompute::AT.filtered_input_img, filtered_input_img}},
                daxa::TaskViewVariant{std::pair{TaaFilterInputCompute::AT.filtered_input_deviation_img, filtered_input_deviation_img}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaFilterInputComputePush &push, TaaTaskInfo const &info) {
                push.input_tex_size = info.input_tex_size;
                push.output_tex_size = info.output_tex_size;
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(info.thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (info.thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y});
            },
            .info = {
                .thread_count = gpu_context.render_resolution,
                .input_tex_size = i_extent,
                .output_tex_size = o_extent,
            },
        });

        debug_utils::DebugDisplay::add_pass({.name = "taa filter input", .task_image_id = filtered_input_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});
        debug_utils::DebugDisplay::add_pass({.name = "taa filter input deviation", .task_image_id = filtered_input_deviation_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        auto filtered_history_img = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
            .name = "filtered_history_img",
        });

        gpu_context.add(ComputeTask<TaaFilterHistoryCompute::Task, TaaFilterHistoryComputePush, TaaTaskInfo>{
            .source = daxa::ShaderFile{"kajiya/taa/filter_history.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{TaaFilterHistoryCompute::AT.gpu_input, gpu_context.task_input_buffer}},

                daxa::TaskViewVariant{std::pair{TaaFilterHistoryCompute::AT.reprojected_history_img, reprojected_history_img}},

                daxa::TaskViewVariant{std::pair{TaaFilterHistoryCompute::AT.filtered_history_img, filtered_history_img}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaFilterHistoryComputePush &push, TaaTaskInfo const &info) {
                push.input_tex_size = info.input_tex_size;
                push.output_tex_size = info.output_tex_size;
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(info.thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (info.thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y});
            },
            .info = {
                .thread_count = gpu_context.render_resolution,
                .input_tex_size = o_extent,
                .output_tex_size = i_extent,
            },
        });

        debug_utils::DebugDisplay::add_pass({.name = "taa filter history", .task_image_id = filtered_history_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        auto input_prob_img = [&]() {
            auto input_prob_img = gpu_context.frame_task_graph.create_transient_image({
                .format = daxa::Format::R16_SFLOAT,
                .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
                .name = "input_prob_img",
            });
            gpu_context.add(ComputeTask<TaaInputProbCompute::Task, TaaInputProbComputePush, TaaTaskInfo>{
                .source = daxa::ShaderFile{"kajiya/taa/input_prob.comp.glsl"},
                .views = std::array{
                    daxa::TaskViewVariant{std::pair{TaaInputProbCompute::AT.gpu_input, gpu_context.task_input_buffer}},

                    daxa::TaskViewVariant{std::pair{TaaInputProbCompute::AT.input_image, input_image}},
                    daxa::TaskViewVariant{std::pair{TaaInputProbCompute::AT.filtered_input_img, filtered_input_img}},
                    daxa::TaskViewVariant{std::pair{TaaInputProbCompute::AT.filtered_input_deviation_img, filtered_input_deviation_img}},
                    daxa::TaskViewVariant{std::pair{TaaInputProbCompute::AT.reprojected_history_img, reprojected_history_img}},
                    daxa::TaskViewVariant{std::pair{TaaInputProbCompute::AT.filtered_history_img, filtered_history_img}},
                    daxa::TaskViewVariant{std::pair{TaaInputProbCompute::AT.reprojection_map, reprojection_map}},
                    daxa::TaskViewVariant{std::pair{TaaInputProbCompute::AT.depth_image, depth_image}},
                    daxa::TaskViewVariant{std::pair{TaaInputProbCompute::AT.smooth_var_history_tex, smooth_var_history_tex}},
                    daxa::TaskViewVariant{std::pair{TaaInputProbCompute::AT.velocity_history_tex, velocity_history_tex}},

                    daxa::TaskViewVariant{std::pair{TaaInputProbCompute::AT.input_prob_img, input_prob_img}},
                },
                .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaInputProbComputePush &push, TaaTaskInfo const &info) {
                    push.input_tex_size = info.input_tex_size;
                    push.output_tex_size = info.output_tex_size;
                    ti.recorder.set_pipeline(pipeline);
                    set_push_constant(ti, push);
                    ti.recorder.dispatch({(info.thread_count.x + 15) / 16, (info.thread_count.y + 15) / 16});
                },
                .info = {
                    .thread_count = gpu_context.render_resolution,
                    .input_tex_size = i_extent,
                    .output_tex_size = o_extent,
                },
            });

            debug_utils::DebugDisplay::add_pass({.name = "taa input prob", .task_image_id = input_prob_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

            auto prob_filtered1_img = gpu_context.frame_task_graph.create_transient_image({
                .format = daxa::Format::R16_SFLOAT,
                .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
                .name = "prob_filtered1_img",
            });

            gpu_context.add(ComputeTask<TaaProbFilterCompute::Task, TaaProbFilterComputePush, TaaTaskInfo>{
                .source = daxa::ShaderFile{"kajiya/taa/filter_prob.comp.glsl"},
                .views = std::array{
                    daxa::TaskViewVariant{std::pair{TaaProbFilterCompute::AT.gpu_input, gpu_context.task_input_buffer}},

                    daxa::TaskViewVariant{std::pair{TaaProbFilterCompute::AT.input_prob_img, input_prob_img}},

                    daxa::TaskViewVariant{std::pair{TaaProbFilterCompute::AT.prob_filtered1_img, prob_filtered1_img}},
                },
                .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaProbFilterComputePush &push, TaaTaskInfo const &info) {
                    push.input_tex_size = info.input_tex_size;
                    push.output_tex_size = info.output_tex_size;
                    ti.recorder.set_pipeline(pipeline);
                    set_push_constant(ti, push);
                    ti.recorder.dispatch({(info.thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (info.thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y});
                },
                .info = {
                    .thread_count = gpu_context.render_resolution,
                    .input_tex_size = i_extent,
                    .output_tex_size = o_extent,
                },
            });

            debug_utils::DebugDisplay::add_pass({.name = "taa prob filter 1", .task_image_id = prob_filtered1_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

            auto prob_filtered2_img = gpu_context.frame_task_graph.create_transient_image({
                .format = daxa::Format::R16_SFLOAT,
                .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
                .name = "prob_filtered2_img",
            });

            gpu_context.add(ComputeTask<TaaProbFilter2Compute::Task, TaaProbFilter2ComputePush, TaaTaskInfo>{
                .source = daxa::ShaderFile{"kajiya/taa/filter_prob2.comp.glsl"},
                .views = std::array{
                    daxa::TaskViewVariant{std::pair{TaaProbFilter2Compute::AT.gpu_input, gpu_context.task_input_buffer}},

                    daxa::TaskViewVariant{std::pair{TaaProbFilter2Compute::AT.prob_filtered1_img, prob_filtered1_img}},

                    daxa::TaskViewVariant{std::pair{TaaProbFilter2Compute::AT.prob_filtered2_img, prob_filtered2_img}},
                },
                .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaProbFilter2ComputePush &push, TaaTaskInfo const &info) {
                    push.input_tex_size = info.input_tex_size;
                    push.output_tex_size = info.output_tex_size;
                    ti.recorder.set_pipeline(pipeline);
                    set_push_constant(ti, push);
                    ti.recorder.dispatch({(info.thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (info.thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y});
                },
                .info = {
                    .thread_count = gpu_context.render_resolution,
                    .input_tex_size = i_extent,
                    .output_tex_size = o_extent,
                },
            });

            debug_utils::DebugDisplay::add_pass({.name = "taa prob filter 2", .task_image_id = prob_filtered2_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

            return prob_filtered2_img;
        }();

        auto this_frame_output_img = gpu_context.frame_task_graph.create_transient_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = {gpu_context.output_resolution.x, gpu_context.output_resolution.y, 1},
            .name = "this_frame_output_img",
        });

        gpu_context.add(ComputeTask<TaaCompute::Task, TaaComputePush, TaaTaskInfo>{
            .source = daxa::ShaderFile{"kajiya/taa/taa.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{TaaCompute::AT.gpu_input, gpu_context.task_input_buffer}},

                daxa::TaskViewVariant{std::pair{TaaCompute::AT.input_image, input_image}},
                daxa::TaskViewVariant{std::pair{TaaCompute::AT.reprojected_history_img, reprojected_history_img}},
                daxa::TaskViewVariant{std::pair{TaaCompute::AT.reprojection_map, reprojection_map}},
                daxa::TaskViewVariant{std::pair{TaaCompute::AT.closest_velocity_img, closest_velocity_img}},
                daxa::TaskViewVariant{std::pair{TaaCompute::AT.velocity_history_tex, velocity_history_tex}},
                daxa::TaskViewVariant{std::pair{TaaCompute::AT.depth_image, depth_image}},
                daxa::TaskViewVariant{std::pair{TaaCompute::AT.smooth_var_history_tex, smooth_var_history_tex}},
                daxa::TaskViewVariant{std::pair{TaaCompute::AT.input_prob_img, input_prob_img}},

                daxa::TaskViewVariant{std::pair{TaaCompute::AT.temporal_output_tex, temporal_output_tex}},
                daxa::TaskViewVariant{std::pair{TaaCompute::AT.this_frame_output_img, this_frame_output_img}},
                daxa::TaskViewVariant{std::pair{TaaCompute::AT.smooth_var_output_tex, smooth_var_output_tex}},
                daxa::TaskViewVariant{std::pair{TaaCompute::AT.temporal_velocity_output_tex, temporal_velocity_output_tex}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TaaComputePush &push, TaaTaskInfo const &info) {
                push.input_tex_size = info.input_tex_size;
                push.output_tex_size = info.output_tex_size;
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(info.thread_count.x + (TAA_WG_SIZE_X - 1)) / TAA_WG_SIZE_X, (info.thread_count.y + (TAA_WG_SIZE_Y - 1)) / TAA_WG_SIZE_Y});
            },
            .info = {
                .thread_count = gpu_context.output_resolution,
                .input_tex_size = i_extent,
                .output_tex_size = o_extent,
            },
        });

        debug_utils::DebugDisplay::add_pass({.name = "taa", .task_image_id = this_frame_output_img, .type = DEBUG_IMAGE_TYPE_DEFAULT});

        return daxa::TaskImageView{this_frame_output_img};
    }
};

#endif

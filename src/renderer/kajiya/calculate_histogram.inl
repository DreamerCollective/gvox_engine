#pragma once

#include <core.inl>
#include <application/input.inl>

#define LUMINANCE_HISTOGRAM_BIN_COUNT 256
#define LUMINANCE_HISTOGRAM_MIN_LOG2 -10.0
#define LUMINANCE_HISTOGRAM_MAX_LOG2 +16.0

DAXA_DECL_TASK_HEAD_BEGIN(CalculateHistogramCompute, 3)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, input_tex)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(daxa_u32), output_buffer)
DAXA_DECL_TASK_HEAD_END
struct CalculateHistogramComputePush {
    daxa_u32vec2 input_extent;
    DAXA_TH_BLOB(CalculateHistogramCompute, uses)
};

#if defined(__cplusplus)

inline auto calculate_luminance_histogram(GpuContext &gpu_context, daxa::TaskImageView blur_pyramid, daxa::TaskBufferView dst_histogram, daxa_u32vec2 image_size, uint32_t &histogram_index) {
    image_size = {(image_size.x + 1) / 2, (image_size.y + 1) / 2};
    auto mip_count = ceil_log2(std::max(image_size.x, image_size.y)) - 1;

    auto input_mip_level = std::max(mip_count, 7u) - 7;

    auto hist_size = static_cast<uint32_t>(sizeof(uint32_t) * LUMINANCE_HISTOGRAM_BIN_COUNT);
    auto tmp_histogram = gpu_context.frame_task_graph.create_transient_buffer({
        .size = sizeof(uint32_t) * LUMINANCE_HISTOGRAM_BIN_COUNT,
        .name = "tmp_histogram",
    });

    gpu_context.frame_task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, tmp_histogram),
        },
        .task = [=](daxa::TaskInterface const &ti) {
            ti.recorder.clear_buffer({
                .buffer = ti.get(daxa::TaskBufferAttachmentIndex{0}).ids[0],
                .offset = 0,
                .size = hist_size,
                .clear_value = 0,
            });
        },
        .name = "clear histogram",
    });

    struct CalculateHistogramTaskInfo {
        daxa_u32 input_mip_level;
    };
    gpu_context.add(ComputeTask<CalculateHistogramCompute::Task, CalculateHistogramComputePush, CalculateHistogramTaskInfo>{
        .source = daxa::ShaderFile{"kajiya/calculate_histogram.comp.glsl"},
        .views = std::array{
            daxa::TaskViewVariant{std::pair{CalculateHistogramCompute::AT.gpu_input, gpu_context.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{CalculateHistogramCompute::AT.input_tex, blur_pyramid.view({.base_mip_level = input_mip_level, .level_count = 1})}},
            daxa::TaskViewVariant{std::pair{CalculateHistogramCompute::AT.output_buffer, tmp_histogram}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, CalculateHistogramComputePush &push, CalculateHistogramTaskInfo const &info) {
            auto const image_info = ti.device.info_image(ti.get(CalculateHistogramCompute::AT.input_tex).ids[0]).value();
            push.input_extent = {(image_info.size.x + ((1 << info.input_mip_level) - 1)) >> info.input_mip_level, (image_info.size.y + ((1 << info.input_mip_level) - 1)) >> info.input_mip_level};
            ti.recorder.set_pipeline(pipeline);
            set_push_constant(ti, push);
            // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
            ti.recorder.dispatch({(push.input_extent.x + 7) / 8, (push.input_extent.y + 7) / 8});
        },
        .info = {
            .input_mip_level = input_mip_level,
        },
    });

    gpu_context.frame_task_graph.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_READ, tmp_histogram),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, dst_histogram),
        },
        .task = [=, &histogram_index](daxa::TaskInterface const &ti) {
            ti.recorder.copy_buffer_to_buffer({
                .src_buffer = ti.get(daxa::TaskBufferAttachmentIndex{0}).ids[0],
                .dst_buffer = ti.get(daxa::TaskBufferAttachmentIndex{1}).ids[0],
                .dst_offset = histogram_index * hist_size,
                .size = hist_size,
            });
        },
        .name = "copy histogram",
    });
}

#endif

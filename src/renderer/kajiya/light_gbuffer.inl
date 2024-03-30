#pragma once

#include <core.inl>
#include <renderer/core.inl>

#include <renderer/kajiya/ircache.inl>

#define SHADING_MODE_DEFAULT 0
#define SHADING_MODE_NO_TEXTURES 1
#define SHADING_MODE_DIFFUSE_GI 2
#define SHADING_MODE_REFLECTIONS 3
#define SHADING_MODE_RTX_OFF 4
#define SHADING_MODE_IRCACHE 5

DAXA_DECL_TASK_HEAD_BEGIN(LightGbufferCompute, 11)
// DAXA_DECL_TASK_HEAD_BEGIN(LightGbufferCompute, 11 + IRCACHE_BUFFER_USE_N)
// IRCACHE_USE_BUFFERS()
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, gbuffer_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, shadow_mask_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, rtr_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, rtdgi_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, output_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, CUBE, ibl_cube)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, sky_lut)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance_lut)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_3D, ae_lut)
DAXA_DECL_TASK_HEAD_END
struct LightGbufferComputePush {
    daxa_f32vec4 output_tex_size;
    daxa_u32 debug_shading_mode;
    daxa_u32 debug_show_wrc;
    DAXA_TH_BLOB(LightGbufferCompute, uses)
};

#if defined(__cplusplus)

inline auto light_gbuffer(
    GpuContext &gpu_context,
    GbufferDepth &gbuffer_depth,
    daxa::TaskImageView shadow_mask,
    daxa::TaskImageView rtr,
    daxa::TaskImageView rtdgi,
    IrcacheRenderState &ircache,
    daxa::TaskImageView ibl_cube,
    daxa::TaskImageView sky_lut,
    daxa::TaskImageView transmittance_lut,
    daxa::TaskImageView ae_lut) -> daxa::TaskImageView {

    auto output_image = gpu_context.frame_task_graph.create_transient_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = {gpu_context.render_resolution.x, gpu_context.render_resolution.y, 1},
        .name = "composited_image",
    });

    struct LightGbufferComputeTaskInfo {
        daxa_u32 debug_shading_mode = SHADING_MODE_DEFAULT;
    };
    auto task_info = LightGbufferComputeTaskInfo{};
    auto do_global_illumination = AppSettings::get<settings::Checkbox>("Graphics", "global_illumination").value;
    if (!do_global_illumination) {
        task_info.debug_shading_mode = SHADING_MODE_RTX_OFF;
    }

    gpu_context.add(ComputeTask<LightGbufferCompute::Task, LightGbufferComputePush, LightGbufferComputeTaskInfo>{
        .source = daxa::ShaderFile{"kajiya/light_gbuffer.comp.glsl"},
        .views = std::array{
            // IRCACHE_BUFFER_USES_ASSIGN(LightGbufferCompute, ircache),
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::AT.gpu_input, gpu_context.task_input_buffer}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::AT.gbuffer_tex, gbuffer_depth.gbuffer}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::AT.depth_tex, gbuffer_depth.depth.current().view()}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::AT.shadow_mask_tex, shadow_mask}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::AT.rtr_tex, rtr}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::AT.rtdgi_tex, rtdgi}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::AT.output_tex, output_image}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::AT.ibl_cube, ibl_cube}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::AT.sky_lut, sky_lut}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::AT.transmittance_lut, transmittance_lut}},
            daxa::TaskViewVariant{std::pair{LightGbufferCompute::AT.ae_lut, ae_lut}},
        },
        .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, LightGbufferComputePush &push, LightGbufferComputeTaskInfo const &info) {
            auto const image_info = ti.device.info_image(ti.get(LightGbufferCompute::AT.gbuffer_tex).ids[0]).value();
            ti.recorder.set_pipeline(pipeline);
            push.debug_shading_mode = info.debug_shading_mode;
            push.debug_show_wrc = 0;
            push.output_tex_size = extent_inv_extent_2d(image_info);
            set_push_constant(ti, push);
            // assert((render_size.x % 8) == 0 && (render_size.y % 8) == 0);
            ti.recorder.dispatch({(image_info.size.x + 7) / 8, (image_info.size.y + 7) / 8});
        },
        .info = task_info,
    });

    debug_utils::DebugDisplay::add_pass({.name = "lit gbuffer", .task_image_id = output_image, .type = DEBUG_IMAGE_TYPE_DEFAULT});

    return output_image;
}

#endif
